import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import exposure, draw
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/counting_perineuronal_nets"))

from image_io import load_channels_tif, get_or_create_mip
from roi import load_rois, get_roi_json_path, points_in_rois, get_point_region_assignment, compute_roi_area_mm2
from datasets.patched_datasets import PatchedMultiImageDataset
from methods.detection.train_fn import predict_points
from methods.detection.transforms import ToTensor

def normalize_wfa_for_detection(w_raw, method="Ninguno (Raw)", gamma=1.0):
    w_f = w_raw.astype(np.float32)
    
    if "Percentil Robusto" in method:
        p_low = float(np.percentile(w_f, 1.0))
        p_high = float(np.percentile(w_f, 99.5))
        if p_high > p_low:
            w_norm = np.clip((w_f - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            w_norm = (w_f - w_f.min()) / (w_f.max() - w_f.min() + 1e-8)
            
    elif "Percentil Agresivo" in method:
        p_low = float(np.percentile(w_f, 0.5))
        p_high = float(np.percentile(w_f, 99.8))
        if p_high > p_low:
            w_norm = np.clip((w_f - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            w_norm = (w_f - w_f.min()) / (w_f.max() - w_f.min() + 1e-8)
            
    elif "CLAHE" in method:
        w_minmax = (w_f - w_f.min()) / (w_f.max() - w_f.min() + 1e-8)
        w_norm = exposure.equalize_adapthist(w_minmax, clip_limit=0.02).astype(np.float32)
        
    elif "Min-Max" in method:
        w_norm = (w_f - w_f.min()) / (w_f.max() - w_f.min() + 1e-8)
        
    else: # "Ninguno (Raw)"
        w_norm = (w_f - w_f.min()) / (w_f.max() - w_f.min() + 1e-8)

    if gamma != 1.0 and gamma > 0:
        w_norm = np.power(w_norm, gamma)
        
    return w_norm

def extract_patch(args):
    wfa_norm, r, c, half_sz = args
    H, W = wfa_norm.shape
    r_s, r_e = r - half_sz, r + half_sz
    c_s, c_e = c - half_sz, c + half_sz
    
    r_s_c, r_e_c = max(0, r_s), min(H, r_e)
    c_s_c, c_e_c = max(0, c_s), min(W, c_e)
    patch_raw = wfa_norm[r_s_c:r_e_c, c_s_c:c_e_c].astype(np.float32)
    
    py0 = max(0, -r_s)
    py1 = max(0, r_e - H)
    px0 = max(0, -c_s)
    px1 = max(0, c_e - W)
    patch = np.pad(patch_raw, ((py0, py1), (px0, px1)), mode='constant')
    if patch.shape != (64, 64):
        patch = cv2.resize(patch, (64, 64))
    return patch, (r, c)

def run_pipeline_on_file(tif_path, out_segm_dir, out_metrics_dir,
                         model_pv_obj, model_loc, model_score, device,
                         pv_filter_type, pv_diameter, pv_flow_threshold, pv_cellprob_threshold,
                         loc_threshold, score_threshold, tile_size, tile_overlap,
                         px_size, do_pv_segmentation, calib_data):
    
    fname = os.path.basename(tif_path)
    base_name, _ = os.path.splitext(fname)
    (p_raw, w_raw, d_raw, a_raw) = load_channels_tif(tif_path)

    m_dapi = np.zeros_like(d_raw, dtype=np.uint16)

    # 1. PV Segmentation (Cellpose)
    m_pv = np.zeros_like(m_dapi)
    if do_pv_segmentation and model_pv_obj is not None:
        in_pv = p_raw.copy()
        if pv_filter_type == "Otsu Global":
            t = threshold_otsu(in_pv)
            in_pv[in_pv < t] = 0
        elif pv_filter_type == "CLAHE (Adaptativo Local)":
            clahe = exposure.equalize_adapthist(in_pv, clip_limit=0.03)
            in_pv = (clahe * 65535).astype(np.uint16)
        m_pv, _, _ = model_pv_obj.eval(in_pv, diameter=pv_diameter, 
                                        flow_threshold=pv_flow_threshold, cellprob_threshold=pv_cellprob_threshold)

    # 2. PNN Detection (PNNloc + PNNscore)
    pnn_radius_um = float(calib_data.get('pnn_radius_um', 20.0))
    scale_factor = (px_size / 0.325) * (20.0 / pnn_radius_um)
    
    wfa_norm_method = calib_data.get('wfa_norm_method', 'Ninguno (Raw)')
    wfa_gamma = float(calib_data.get('wfa_gamma', 1.0))
    wfa_norm = normalize_wfa_for_detection(w_raw, method=wfa_norm_method, gamma=wfa_gamma)
    
    wfa_single_path = tif_path.replace('.tif', '_wfa_temp.tif')
    if wfa_norm_method == "Ninguno (Raw)" and wfa_gamma == 1.0:
        wfa_rgb = np.stack([w_raw, w_raw, w_raw], axis=0)
    else:
        max_v = float(w_raw.max()) if w_raw.max() > 0 else 65535.0
        w_scaled = (wfa_norm * max_v).astype(np.uint16)
        wfa_rgb = np.stack([w_scaled, w_scaled, w_scaled], axis=0)
        
    tiff.imwrite(wfa_single_path, wfa_rgb.astype(np.uint16), imagej=True)

    img_preds = []
    try:
        stride_size = max(64, tile_size - tile_overlap)
        dataset = PatchedMultiImageDataset.from_paths(
            [wfa_single_path],
            patch_size=tile_size,
            stride=stride_size,
            transforms=ToTensor()
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        try:
            model_loc.transform.image_mean = [0.485, 0.456, 0.406]
            model_loc.transform.image_std = [0.229, 0.224, 0.225]
        except Exception:
            pass

        cfg_loc = OmegaConf.create({
            'model': {'module': {'nms': 0.3, 'out_channels': 1}},
            'data': {'validation': {'target_params': {'side': 64}}}
        })
        
        df_locs = predict_points(dataloader, model_loc, device, loc_threshold, cfg_loc)
        
        if not df_locs.empty:
            for idx, row in df_locs.iterrows():
                img_preds.append((int(row['Y']), int(row['X'])))
    finally:
        if os.path.exists(wfa_single_path):
            try:
                os.remove(wfa_single_path)
            except Exception:
                pass

    coords_loc = []
    scores_score = []
    candidates_list = []
    
    if len(img_preds) > 0:
        half_sz = 32
        patch_args = [(wfa_norm, int(r), int(c), half_sz) for r, c in img_preds]
        patches = []
        valid_coords = []
        for p, coord in map(extract_patch, patch_args):
            patches.append(p)
            valid_coords.append(coord)
            
        if len(patches) > 0:
            patches_tensor = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                score_outputs = model_score(patches_tensor).squeeze(-1).cpu().numpy()
                if score_outputs.ndim == 0:
                    score_outputs = np.array([float(score_outputs)])
            
            for idx_c, (coord, sc) in enumerate(zip(valid_coords, score_outputs), start=1):
                sc_float = float(sc)
                candidates_list.append({
                    'id': idx_c,
                    'centroid_y': float(coord[0]),
                    'centroid_x': float(coord[1]),
                    'score': sc_float,
                    'is_confirmed': bool(sc_float >= score_threshold)
                })
                if sc_float >= score_threshold:
                    coords_loc.append(coord)
                    scores_score.append(sc_float)

    # 3. Morphometry & Ring Masks
    H, W = w_raw.shape
    m_wfa = np.zeros((H, W), dtype=np.uint16)
    m_ring = np.zeros((H, W), dtype=np.uint16)
    power_map = np.zeros((H, W), dtype=np.float32)
    
    soma_erosion_um = float(calib_data.get('soma_erosion_um', 2.0))
    soma_erosion_px = int(round(soma_erosion_um / px_size))
    
    pnn_radius_um = float(calib_data.get('pnn_radius_um', 20.0))
    pnn_radius_px = int(round(pnn_radius_um / px_size)) if pnn_radius_um > 0 else int(round(20.0 / px_size))

    r_batch = []
    
    for idx, ((r, c), sc) in enumerate(zip(coords_loc, scores_score), start=1):
        rr, cc = draw.disk((r, c), pnn_radius_px, shape=(H, W))
        m_wfa[rr, cc] = idx
        
        rr_inner, cc_inner = draw.disk((r, c), max(1, pnn_radius_px - soma_erosion_px), shape=(H, W))
        ring_mask = np.zeros((H, W), dtype=bool)
        ring_mask[rr, cc] = True
        ring_mask[rr_inner, cc_inner] = False
        m_ring[ring_mask] = idx
        
        wfa_sum = float(np.sum(w_raw[rr, cc]))
        wfa_mean = float(np.mean(w_raw[rr, cc])) if len(rr) > 0 else 0.0
        wfa_peri = float(np.mean(w_raw[ring_mask])) if np.sum(ring_mask) > 0 else 0.0
        
        power_map[rr, cc] += wfa_norm[rr, cc]
        
        pv_coloc_label = int(m_pv[r, c]) if m_pv[r, c] > 0 else None
        is_pv_coloc = bool(pv_coloc_label is not None)
        
        pv_area = 0.0
        pv_diameter = 0.0
        if is_pv_coloc:
            pv_mask = (m_pv == pv_coloc_label)
            pv_area = float(np.sum(pv_mask) * (px_size ** 2))
            pv_diameter = float(2 * np.sqrt(pv_area / np.pi))

        pnn_area = float(len(rr) * (px_size ** 2))
        pnn_diameter = float(2 * np.sqrt(pnn_area / np.pi))

        r_batch.append({
            'label': idx,
            'centroid_y': r,
            'centroid_x': c,
            'area_um2': pnn_area,
            'diameter_um': pnn_diameter,
            'wfa_sum_intensity': wfa_sum,
            'wfa_mean_intensity': wfa_mean,
            'wfa_pericellular_intensity': wfa_peri,
            'wfa_intensity_norm': wfa_mean / (np.mean(w_raw) + 1e-8),
            'wfa_pericellular_norm': wfa_peri / (np.mean(w_raw) + 1e-8),
            'pv_intensity_norm': float(np.mean(p_raw[rr, cc])) / (np.mean(p_raw) + 1e-8) if len(rr) > 0 else 0.0,
            'is_pnn_plus': True,
            'is_pv_plus': is_pv_coloc,
            'pv_label': pv_coloc_label if is_pv_coloc else -1,
            'cell_type': "PV+/PNN+" if is_pv_coloc else "PV-/PNN+",
            'pv_area_um2': pv_area,
            'pv_diameter_um': pv_diameter,
            'pnn_area_um2': pnn_area,
            'pnn_diameter_um': pnn_diameter,
            'score': sc
        })

    # Add PV+/PNN- interneurons
    pv_props = regionprops(m_pv)
    max_pnn_label = len(coords_loc)
    for pvp in pv_props:
        pv_lbl = pvp.label
        pr, pc = int(pvp.centroid[0]), int(pvp.centroid[1])
        if m_wfa[pr, pc] == 0:
            pv_area = float(pvp.area * (px_size ** 2))
            pv_diameter = float(pvp.equivalent_diameter_area * px_size)
            max_pnn_label += 1
            r_batch.append({
                'label': max_pnn_label,
                'centroid_y': pr,
                'centroid_x': pc,
                'area_um2': pv_area,
                'diameter_um': pv_diameter,
                'wfa_sum_intensity': float(np.sum(w_raw[m_pv == pv_lbl])),
                'wfa_mean_intensity': float(np.mean(w_raw[m_pv == pv_lbl])),
                'wfa_pericellular_intensity': 0.0,
                'wfa_intensity_norm': 0.0,
                'wfa_pericellular_norm': 0.0,
                'pv_intensity_norm': float(np.mean(p_raw[m_pv == pv_lbl])) / (np.mean(p_raw) + 1e-8),
                'is_pnn_plus': False,
                'is_pv_plus': True,
                'pv_label': pv_lbl,
                'cell_type': "PV+/PNN-",
                'pv_area_um2': pv_area,
                'pv_diameter_um': pv_diameter,
                'pnn_area_um2': 0.0,
                'pnn_diameter_um': 0.0,
                'score': 0.0
            })

    df_b = pd.DataFrame(r_batch)
    if df_b.empty:
        df_b = pd.DataFrame(columns=[
            'label', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um', 
            'wfa_sum_intensity', 'wfa_mean_intensity', 'wfa_pericellular_intensity',
            'wfa_intensity_norm', 'wfa_pericellular_norm', 'pv_intensity_norm',
            'is_pnn_plus', 'is_pv_plus', 'pv_label',
            'cell_type', 'pv_area_um2', 'pv_diameter_um', 'pnn_area_um2', 'pnn_diameter_um', 'score'
        ])

    # Annotate ROIs (Regiones A, B, C)
    roi_json_path = get_roi_json_path(out_metrics_dir, base_name)
    regions_dict = load_rois(roi_json_path)

    if not df_b.empty:
        df_b['is_in_roi'] = points_in_rois(df_b[['centroid_y', 'centroid_x']].values, regions_dict, target_region="ALL")
        df_b['roi_region'] = get_point_region_assignment(df_b[['centroid_y', 'centroid_x']].values, regions_dict)
    else:
        df_b['is_in_roi'] = False
        df_b['roi_region'] = "NONE"

    os.makedirs(out_metrics_dir, exist_ok=True)
    os.makedirs(out_segm_dir, exist_ok=True)

    csv_name = f"{base_name}_nuclei_metrics.csv"
    df_b.to_csv(os.path.join(out_metrics_dir, csv_name), index=False)

    # Save segmented Masks TIFF
    stk = np.stack([m_dapi.astype(np.uint16),
                    m_pv.astype(np.uint16),
                    m_wfa.astype(np.uint16),
                    m_ring.astype(np.uint16),
                    w_raw.astype(np.uint16)], axis=0)
                          
    segm_name = f"{base_name}_masks.tif"
    tiff.imwrite(os.path.join(out_segm_dir, segm_name),
                 stk, imagej=True,
                 metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                           'Labels': ['DAPI_Mask', 'PV_Mask', 'PNN_Mask', 'Pericellular_Ring_Mask', 'WFA_Raw']})

    # Save Power Heatmap Image
    power_map_scaled = np.clip(power_map * 255.0, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(power_map_scaled, cv2.COLORMAP_TURBO)
    heatmap_path = os.path.join(out_segm_dir, f"{base_name}_power_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_color)

    # Save Summary Metrics
    total_pv_segmentation = int(np.max(m_pv)) if np.max(m_pv) > 0 else 0
    pv_pnn_plus = int(sum(1 for r in r_batch if r['cell_type'] == "PV+/PNN+"))
    hollow_pnn_plus = int(sum(1 for r in r_batch if r['cell_type'] == "PV-/PNN+"))
    total_pnn_plus = int(sum(1 for r in r_batch if r['is_pnn_plus']))

    image_area_mm2 = float((H * px_size * W * px_size) / 1e6)
    pnn_density_mm2 = float(total_pnn_plus / image_area_mm2) if image_area_mm2 > 0 else 0.0
    
    df_pnn = df_b[df_b['is_pnn_plus'] == True] if not df_b.empty else pd.DataFrame()
    df_coloc = df_b[df_b['cell_type'] == "PV+/PNN+"] if not df_b.empty else pd.DataFrame()
    
    pnn_energy = float(df_pnn['wfa_pericellular_norm'].mean()) if not df_pnn.empty and 'wfa_pericellular_norm' in df_pnn.columns and pd.notna(df_pnn['wfa_pericellular_norm'].mean()) else 0.0
    coloc_energy = float(df_coloc['wfa_pericellular_norm'].mean()) if not df_coloc.empty and 'wfa_pericellular_norm' in df_coloc.columns and pd.notna(df_coloc['wfa_pericellular_norm'].mean()) else 0.0
    mean_pnn_pericellular_wfa_norm = pnn_energy
    
    diffuse_wfa = float(np.mean(w_raw[m_wfa == 0])) / (np.mean(w_raw) + 1e-8) if np.sum(m_wfa == 0) > 0 else 0.0
    pct_pv_surrounded = float((pv_pnn_plus / total_pv_segmentation * 100)) if total_pv_segmentation > 0 else 0.0

    summary = {
        "total_pv_segmentation": total_pv_segmentation,
        "total_pnn_plus": total_pnn_plus,
        "pv_pnn_plus": pv_pnn_plus,
        "hollow_pnn_plus": hollow_pnn_plus,
        "image_area_mm2": image_area_mm2,
        "pnn_density_mm2": pnn_density_mm2,
        "pnn_energy": pnn_energy,
        "coloc_energy": coloc_energy,
        "mean_pnn_pericellular_wfa_norm": mean_pnn_pericellular_wfa_norm,
        "diffuse_wfa_fluorescence": diffuse_wfa,
        "pct_pv_surrounded_by_pnn": pct_pv_surrounded,
        "pixel_size": px_size
    }

    json_path = os.path.join(out_metrics_dir, f"{base_name}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    cand_path = os.path.join(out_metrics_dir, f"{base_name}_candidates.json")
    with open(cand_path, 'w') as f:
        json.dump(candidates_list, f, indent=2)

    return summary
