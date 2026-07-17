import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage import exposure, draw
import scipy.ndimage as ndi
import pandas as pd
import sys
import torch
import concurrent.futures

# Add PNNloc / PNNscore directories to path
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/counting_perineuronal_nets"))

from datasets.patched_datasets import PatchedMultiImageDataset
from methods.detection.train_fn import predict_points
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from methods.detection.transforms import ToTensor

def get_or_create_mip(raw_path, px_size=1.0):
    if "data/processed/mips" in raw_path:
        return raw_path
        
    # Construct MIP path: replace 'data/raw' with 'data/processed/mips'
    mip_path = raw_path.replace("data/raw", "data/processed/mips")
    os.makedirs(os.path.dirname(mip_path), exist_ok=True)
    
    if not os.path.exists(mip_path):
        print(f"Creando MIP para {raw_path} -> {mip_path}")
        img = tiff.imread(raw_path)
        with tiff.TiffFile(raw_path) as tif:
            axes = tif.series[0].axes
            
        if 'Z' in axes and len(img.shape) >= 4:
            z_idx = axes.index('Z')
            img = np.max(img, axis=z_idx)
            axes = axes.replace('Z', '')
            
        if axes == 'YXC':
            img = np.transpose(img, (2, 0, 1))
            
        # Ensure it has exactly 4 channels (pad or slice)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        num_ch = img.shape[0]
        if num_ch < 4:
            padding = np.zeros((4 - num_ch, img.shape[1], img.shape[2]), dtype=img.dtype)
            img = np.concatenate([img, padding], axis=0)
        elif num_ch > 4:
            img = img[:4]
            
        tiff.imwrite(mip_path, img.astype(np.uint16), imagej=True,
                     metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX'})
                     
    return mip_path

def load_channels_tif(path):
    # If loading from raw directory, redirect to cached MIP
    if "data/raw" in path:
        px_size = 1.0
        try:
            if os.path.exists("experiment_config.json"):
                with open("experiment_config.json", 'r') as f:
                    cfg = json.load(f)
                    px_size = cfg.get("pixel_size_um", 1.0)
        except Exception:
            pass
        path = get_or_create_mip(path, px_size)
        
    img = tiff.imread(path)
    
    with tiff.TiffFile(path) as tif:
        axes = tif.series[0].axes
        
    if 'Z' in axes and len(img.shape) >= 4:
        z_idx = axes.index('Z')
        img = np.max(img, axis=z_idx)
        axes = axes.replace('Z', '')
        
    if axes == 'YXC':
        img = np.transpose(img, (2, 0, 1))
        
    # Expected (C, Y, X)
    agr = img[0, :, :] if img.shape[0] >= 1 else np.zeros_like(img[0])
    dapi = img[1, :, :] if img.shape[0] >= 2 else img[0, :, :]
    wfa = img[2, :, :] if img.shape[0] >= 3 else np.zeros_like(img[0])
    pv = img[3, :, :] if img.shape[0] >= 4 else np.zeros_like(img[0])
    
    return (pv, wfa, dapi, agr)

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
                         model_dapi, model_pv_obj, model_loc, model_score, device,
                         filter_type, diameter, flow_threshold, cellprob_threshold,
                         pv_filter_type, pv_diameter, pv_flow_threshold, pv_cellprob_threshold,
                         loc_threshold, score_threshold, tile_size, tile_overlap,
                         px_size, do_pv_segmentation, calib_data):
    
    fname = os.path.basename(tif_path)
    base_name, _ = os.path.splitext(fname)
    (p_raw, w_raw, d_raw, a_raw) = load_channels_tif(tif_path)

    # 1. DAPI Segmentation (Cellpose)
    in_dapi = d_raw.copy()
    if filter_type == "Otsu Global":
        t = threshold_otsu(in_dapi)
        in_dapi[in_dapi < t] = 0
    elif filter_type == "CLAHE (Adaptativo Local)":
        clahe = exposure.equalize_adapthist(in_dapi, clip_limit=0.03)
        in_dapi = (clahe * 65535).astype(np.uint16)

    m_dapi, _, _ = model_dapi.eval(in_dapi, diameter=diameter, 
                                    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

    # 2. PV Segmentation (Cellpose)
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

    # 3. PNN Detection (PNNloc + PNNscore)
    # Convert 16-bit WFA to 8-bit for neural network compatibility
    w_min, w_max = w_raw.min(), w_raw.max()
    if w_max > w_min:
        wfa_8bit = ((w_raw - w_min) / (w_max - w_min) * 255.0).astype(np.uint8)
    else:
        wfa_8bit = w_raw.astype(np.uint8)

    scale_factor = px_size / 0.325
    H, W = w_raw.shape
    
    if abs(scale_factor - 1.0) > 0.01:
        new_H = int(round(H * scale_factor))
        new_W = int(round(W * scale_factor))
        wfa_8bit_scaled = cv2.resize(wfa_8bit, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
    else:
        wfa_8bit_scaled = wfa_8bit.copy()

    # Write temporary file for PatchedMultiImageDataset
    temp_wfa = f"temp_wfa_{os.getpid()}_{base_name}.tif"
    tiff.imwrite(temp_wfa, wfa_8bit_scaled.astype(np.float32))
    
    candidates = []
    H_scaled, W_scaled = wfa_8bit_scaled.shape
    prob_map_scaled = np.zeros((H_scaled, W_scaled), dtype=np.float32)
    try:
        cfg_loc = OmegaConf.load("data/models/pnn_v2_fasterrcnn_640/.hydra/config.yaml")
        cfg_loc.model.module.nms = 0.3
        
        stride = tile_size - tile_overlap
        ds = PatchedMultiImageDataset.from_paths([temp_wfa], patch_size=tile_size, stride=stride, transforms=ToTensor())
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        
        locs = predict_points(dl, model_loc, device, loc_threshold, cfg_loc)
        
        for idx, row in locs.iterrows():
            cy, cx = float(row["Y"]), float(row["X"])
            score = float(row["score"])
            candidates.append({
                "centroid_y": cy,
                "centroid_x": cx,
                "prob_map_val": score
            })
            # Draw a small Gaussian peak on the scaled prob map (heatmap)
            cy_int, cx_int = int(cy), int(cx)
            r_g = 15
            y0, y1 = max(0, cy_int - r_g), min(H_scaled, cy_int + r_g + 1)
            x0, x1 = max(0, cx_int - r_g), min(W_scaled, cx_int + r_g + 1)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    dist2 = (y - cy)**2 + (x - cx)**2
                    g = np.exp(-dist2 / (2 * (5.0**2)))
                    prob_map_scaled[y, x] = max(prob_map_scaled[y, x], score * g)
                    
        # Resize probability map back to original dimensions and save as TIFF
        prob_map_orig = cv2.resize(prob_map_scaled, (W, H), interpolation=cv2.INTER_LINEAR)
        prob_map_file = os.path.join(out_segm_dir, f"{base_name}_prob_map.tif")
        tiff.imwrite(prob_map_file, prob_map_orig.astype(np.float32))
    finally:
        if os.path.exists(temp_wfa):
            os.remove(temp_wfa)

    # Inferencia PNNscore on scaled WFA
    wfa_norm = wfa_8bit_scaled.astype(np.float32) / 255.0
    
    half_sz = 32
    patch_args = [(wfa_norm, int(c["centroid_y"]), int(c["centroid_x"]), half_sz) for c in candidates]
    patches, valid_coords, valid_original_cands = [], [], []
    
    with concurrent.futures.ThreadPoolExecutor() as pool:
        for idx, result in enumerate(pool.map(extract_patch, patch_args)):
            if result is not None:
                patches.append(result[0])
                valid_coords.append(result[1])
                valid_original_cands.append(candidates[idx])
                
    scores = []
    BATCH = 512
    for i in range(0, len(patches), BATCH):
        batch_t = torch.tensor(np.stack(patches[i:i+BATCH]), dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            outputs = model_score(batch_t)
            outputs = torch.sigmoid(outputs)
            scores.extend(outputs.cpu().numpy().flatten().tolist())
            
    # Generate confirmed PNN labeled mask on original dimensions
    m_wfa = np.zeros((H, W), dtype=np.uint16)
    final_pnns = []
    pnn_id = 1
    
    for cand, (r, c), score in zip(valid_original_cands, valid_coords, scores):
        if score >= score_threshold:
            # Map coordinates back to original resolution
            r_orig = r / scale_factor
            c_orig = c / scale_factor
            rad_orig = 12.0 / scale_factor
            
            r_int, c_int = int(round(r_orig)), int(round(c_orig))
            rad_int = max(3, int(round(rad_orig)))
            
            r_min, r_max = max(0, r_int - rad_int), min(H, r_int + rad_int + 1)
            c_min, c_max = max(0, c_int - rad_int), min(W, c_int + rad_int + 1)
            
            y_grid, x_grid = np.ogrid[r_min - r_orig : r_max - r_orig, c_min - c_orig : c_max - c_orig]
            circle_mask = y_grid*y_grid + x_grid*x_grid <= (rad_orig ** 2)
            circle_mask = circle_mask[:(r_max-r_min), :(c_max-c_min)]
            
            m_wfa[r_min:r_max, c_min:c_max][circle_mask] = pnn_id
            
            local_wfa = w_raw[r_min:r_max, c_min:c_max]
            wfa_vals = local_wfa[circle_mask]
            mean_intensity = float(np.mean(wfa_vals)) if len(wfa_vals) > 0 else 0.0
            max_intensity = float(np.max(wfa_vals)) if len(wfa_vals) > 0 else 0.0
            area_um2 = np.sum(circle_mask) * (px_size ** 2)
            diam = 2.0 * rad_orig * px_size
            
            final_pnns.append({
                "pnn_id": pnn_id,
                "score": float(score),
                "centroid_y": float(r_orig),
                "centroid_x": float(c_orig),
                "area_um2": area_um2,
                "diameter_um": diam,
                "wfa_mean_intensity": mean_intensity,
                "wfa_max_intensity": max_intensity
            })
            pnn_id += 1

    # 4. Colocalization of PNN and PV
    pv_props = regionprops(m_pv)
    wfa_props = regionprops(m_wfa)
    
    wfa_to_pv = {}
    pv_to_wfa = {}
    for wfa_prop in wfa_props:
        wfa_label = wfa_prop.label
        wfa_mask = (m_wfa == wfa_label)
        
        pv_in_wfa = m_pv[wfa_mask]
        unique_pv, counts = np.unique(pv_in_wfa, return_counts=True)
        
        valid_idx = unique_pv > 0
        unique_pv = unique_pv[valid_idx]
        counts = counts[valid_idx]
        
        if len(unique_pv) > 0:
            best_idx = np.argmax(counts)
            best_pv_lbl = unique_pv[best_idx]
            
            wfa_to_pv[wfa_label] = best_pv_lbl
            pv_to_wfa[best_pv_lbl] = wfa_label

    r_batch = []
    matched_pv_labels = set(wfa_to_pv.values())
    
    # Process PNN+ detections (both PV+ and PV- hollow PNNs)
    for wfa_prop in wfa_props:
        wfa_label = wfa_prop.label
        wfa_mask = (m_wfa == wfa_label)
        
        pnn_info = next((p for p in final_pnns if p["pnn_id"] == wfa_label), {})
        w_cy, w_cx = wfa_prop.centroid
        w_area = wfa_prop.area * (px_size ** 2)
        w_diam = wfa_prop.equivalent_diameter_area * px_size
        
        pv_label = wfa_to_pv.get(wfa_label, None)
        if pv_label is not None:
            pv_prop = next((p for p in pv_props if p.label == pv_label), None)
            if pv_prop is not None:
                cy, cx = pv_prop.centroid
                pv_area = pv_prop.area * (px_size ** 2)
                pv_diameter = pv_prop.equivalent_diameter_area * px_size
            else:
                cy, cx = w_cy, w_cx
                pv_area = 0.0
                pv_diameter = 0.0
            cell_type = "PV+/PNN+"
            is_pv_plus = True
        else:
            cy, cx = w_cy, w_cx
            pv_area = 0.0
            pv_diameter = 0.0
            cell_type = "PV-/PNN+"
            is_pv_plus = False
            
        wfa_s = float(np.sum(w_raw[wfa_mask]))
        
        r_batch.append({
            'label': wfa_label,
            'centroid_y': w_cy,
            'centroid_x': w_cx,
            'area_um2': w_area,
            'diameter_um': w_diam,
            'wfa_sum_intensity': wfa_s,
            'is_pnn_plus': True,
            'is_pv_plus': is_pv_plus,
            'pv_label': pv_label if is_pv_plus else -1,
            'cell_type': cell_type,
            'pv_area_um2': pv_area,
            'pv_diameter_um': pv_diameter,
            'pnn_area_um2': w_area,
            'pnn_diameter_um': w_diam,
            'score': pnn_info.get("score", 0.0)
        })
        
    # Process PV+/PNN- cells (PV+ somas not associated with any PNN)
    max_wfa_label = int(np.max(m_wfa)) if np.max(m_wfa) > 0 else 0
    for pvp in pv_props:
        pv_label = pvp.label
        if pv_label in matched_pv_labels:
            continue
            
        pv_mask = (m_pv == pv_label)
        cy, cx = pvp.centroid
        pv_area = pvp.area * (px_size ** 2)
        pv_diameter = pvp.equivalent_diameter_area * px_size
        
        wfa_s = float(np.sum(w_raw[pv_mask]))
        unique_label = max_wfa_label + pv_label
        
        r_batch.append({
            'label': unique_label,
            'centroid_y': cy,
            'centroid_x': cx,
            'area_um2': pv_area,
            'diameter_um': pv_diameter,
            'wfa_sum_intensity': wfa_s,
            'is_pnn_plus': False,
            'is_pv_plus': True,
            'pv_label': pv_label,
            'cell_type': "PV+/PNN-",
            'pv_area_um2': pv_area,
            'pv_diameter_um': pv_diameter,
            'pnn_area_um2': 0.0,
            'pnn_diameter_um': 0.0,
            'score': 0.0
        })

    # Save CSV outputs
    df_b = pd.DataFrame(r_batch)
    if df_b.empty:
        df_b = pd.DataFrame(columns=[
            'label', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um', 
            'wfa_sum_intensity', 'is_pnn_plus', 'is_pv_plus', 'pv_label',
            'cell_type', 'pv_area_um2', 'pv_diameter_um', 'pnn_area_um2', 'pnn_diameter_um', 'score'
        ])
        
    csv_name = f"{base_name}_nuclei_metrics.csv"
    df_b.to_csv(os.path.join(out_metrics_dir, csv_name), index=False)

    # Save segmented Masks TIFF (4 channels: DAPI, PV, PNN_Mask, WFA_Raw)
    stk = np.stack([m_dapi.astype(np.uint16),
                    m_pv.astype(np.uint16),
                    m_wfa.astype(np.uint16),
                    w_raw.astype(np.uint16)], axis=0)
                          
    segm_name = f"{base_name}_masks.tif"
    tiff.imwrite(os.path.join(out_segm_dir, segm_name),
                 stk, imagej=True,
                 metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                           'Labels': ['DAPI_Mask', 'PV_Mask', 'PNN_Mask', 'WFA_Raw']})

    # DAPI-centric metrics (for colocalization reference)
    pnn_radius_um = float(calib_data.get('pnn_radius_um', 20.0))
    dapi_props = regionprops(m_dapi, intensity_image=w_raw)
    dapi_batch = []
    
    for db in dapi_props:
        cy, cx = db.centroid
        is_pv_coloc = bool(m_pv[int(cy), int(cx)] > 0)
        is_pnn = bool(m_wfa[int(cy), int(cx)] > 0)
        
        r_px = pnn_radius_um / px_size if pnn_radius_um > 0 else 20.0 / px_size
        rd, cd = draw.disk((cy, cx), r_px, shape=w_raw.shape)
        wfa_sum = float(np.sum(w_raw[rd, cd]))
        
        dapi_batch.append({
            'label': db.label,
            'centroid_y': cy,
            'centroid_x': cx,
            'area_um2': db.area * (px_size ** 2),
            'diameter_um': db.equivalent_diameter_area * px_size,
            'dapi_mean_intensity': float(db.intensity_mean),
            'wfa_sum_intensity': wfa_sum,
            'is_pnn_plus': is_pnn,
            'is_pv_plus': is_pv_coloc
        })
        
    df_dapi = pd.DataFrame(dapi_batch)
    if df_dapi.empty:
        df_dapi = pd.DataFrame(columns=[
            'label', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um',
            'dapi_mean_intensity', 'wfa_sum_intensity', 'is_pnn_plus', 'is_pv_plus'
        ])
    
    dapi_csv_name = f"{base_name}_dapi_metrics.csv"
    df_dapi.to_csv(os.path.join(out_metrics_dir, dapi_csv_name), index=False)

    # Save summary JSON
    total_pv_segmentation = int(np.max(m_pv)) if np.max(m_pv) > 0 else 0
    pv_pnn_plus = int(sum(1 for r in r_batch if r['cell_type'] == "PV+/PNN+"))
    pv_pnn_minus = int(sum(1 for r in r_batch if r['cell_type'] == "PV+/PNN-"))
    hollow_pnn_plus = int(sum(1 for r in r_batch if r['cell_type'] == "PV-/PNN+"))
    total_pnn_plus = int(sum(1 for r in r_batch if r['is_pnn_plus']))
    
    summary = {
        "total_dapi": int(np.max(m_dapi)) if np.max(m_dapi) > 0 else 0,
        "total_pv_segmentation": total_pv_segmentation,
        "pnn_plus": total_pnn_plus,
        "pnn_minus": pv_pnn_minus,
        "dapi_pv_coloc": pv_pnn_plus,
        "pv_pnn_plus": pv_pnn_plus,
        "pv_pnn_minus": pv_pnn_minus,
        "hollow_pnn_plus": hollow_pnn_plus,
        "total_pnn_plus": total_pnn_plus,
        "pixel_size": px_size
    }
    
    json_name = f"{base_name}_summary.json"
    with open(os.path.join(out_metrics_dir, json_name), 'w') as fs:
        json.dump(summary, fs, indent=4)
        
    return summary
