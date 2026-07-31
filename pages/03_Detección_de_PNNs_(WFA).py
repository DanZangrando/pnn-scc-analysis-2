import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
import torch
import concurrent.futures
from cellpose import models
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.measure import regionprops
import pandas as pd
import sys
import subprocess
from pipeline import load_channels_tif
from omegaconf import OmegaConf

# Page configuration
st.set_page_config(page_title="Paso 3: Detección de PNNs (WFA)", layout="wide")

# Add PNN models path
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/counting_perineuronal_nets"))

# Premium Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght=300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main-header {
        background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .sub-header { color: #a0aec0; font-size: 1.1rem; margin-bottom: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #00f2fe; }
    .img-caption { font-weight: bold; color: #00f2fe; margin-bottom: 5px; text-align: center; }
    hr { border: 0; height: 1px; background: linear-gradient(to right, transparent, #00f2fe, transparent); margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Paso 3: Detección de Redes Perineuronales (PNNs)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detección de PNNs en el canal WFA mediante los modelos Faster R-CNN (PNNloc) y ConvNet (PNNscore).</div>', unsafe_allow_html=True)

RAW_DIR = "data/raw"
if not os.path.exists(RAW_DIR) or not any(os.path.isdir(os.path.join(RAW_DIR, d)) for d in os.listdir(RAW_DIR) if not d.startswith('.')):
    RAW_DIR = "data/processed/mips"

SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

if not os.path.exists(RAW_DIR):
    st.error(f"No se encontró el directorio `{RAW_DIR}`.")
    st.stop()

# Load global configuration
calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)

# Sidebar - Data selection
st.sidebar.header("📁 Selección de Datos")
groups = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
if not groups:
    st.warning("No hay grupos en raw data.")
    st.stop()

selected_group = st.sidebar.selectbox("Grupo:", groups, key="p3_group")
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones en {selected_group}.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections, key="p3_section")
section_dir = os.path.join(group_dir, selected_section)

tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith('.tif')])
if not tif_files:
    st.warning("No hay archivos .tif en la sección.")
    st.stop()

selected_filename = st.sidebar.selectbox("Imagen:", tif_files, key="p3_file")
selected_path = os.path.join(section_dir, selected_filename)

SEGM_DIR = os.path.join(SEGM_BASE_DIR, selected_group, selected_section)
METRICS_DIR = os.path.join(METRICS_BASE_DIR, selected_group, selected_section)
os.makedirs(SEGM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load WFA channel
try:
    (pv_raw, wfa_raw, dapi_raw, agr_raw) = load_channels_tif(selected_path)
    wfa_disp = cv2.normalize(wfa_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")
    st.stop()

# Sidebar - PNNloc & PNNscore Parameters
st.sidebar.header("⚙️ Ajustes de Deep Learning (IA)")
loc_threshold = st.sidebar.slider("Umbral de Probabilidad (PNNloc)", 0.05, 0.90, float(calib_data.get('lupori_loc_threshold', 0.20)), step=0.05)
score_threshold = st.sidebar.slider("Umbral de Calificación (PNNscore)", 0.05, 1.0, float(calib_data.get('lupori_score_threshold', 0.30)), step=0.05)
min_peak_dist = st.sidebar.slider("Distancia mínima entre PNNs (px)", 10, 80, int(calib_data.get('lupori_min_peak_dist', 30)), step=5)
tile_size = st.sidebar.select_slider("Tamaño de tile (px)", options=[256, 512, 1024, 2048], value=int(calib_data.get('lupori_tile_size', 1024)))
tile_overlap = st.sidebar.slider("Overlap entre tiles (px)", 16, 128, int(calib_data.get('lupori_tile_overlap', 32)), step=16)

px_size = float(calib_data.get('pixel_size_um', 1.0))

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🧠 Detectar PNNs (WFA)", type="primary")

base_fn, _ = os.path.splitext(selected_filename)
seg_file = os.path.join(SEGM_DIR, f"{base_fn}_masks.tif")
csv_file = os.path.join(METRICS_DIR, f"{base_fn}_nuclei_metrics.csv")
json_file = os.path.join(METRICS_DIR, f"{base_fn}_summary.json")
candidates_file = os.path.join(METRICS_DIR, f"{base_fn}_candidates.json")

# Model caching
@st.cache_resource
def load_deep_models():
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # PNNloc
    from models.FasterRCNN import FasterRCNNWrapper
    model_loc = FasterRCNNWrapper(in_channels=1, out_channels=1, model_pretrained=False)
    ckpt_loc = "data/models/pnn_v2_fasterrcnn_640/best.pth"
    checkpoint_loc = torch.load(ckpt_loc, map_location=device)
    model_loc.load_state_dict(checkpoint_loc['model'])
    model_loc.to(device).eval()
    
    # PNNscore
    from models.ConvNet import ConvNet
    model_score = ConvNet(in_channels=1, num_classes=1)
    ckpt_score = "data/models/pnn_v2_scoring_rank_learning/best.pth"
    checkpoint_score = torch.load(ckpt_score, map_location=device)
    model_score.load_state_dict(checkpoint_score['model'])
    model_score.to(device).eval()
    
    return model_loc, model_score, device

def extract_patch_helper(args):
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

if run_btn:
    # Save parameters to global config
    calib_data.update({
        'lupori_loc_threshold': loc_threshold,
        'lupori_score_threshold': score_threshold,
        'lupori_min_peak_dist': min_peak_dist,
        'lupori_tile_size': tile_size,
        'lupori_tile_overlap': tile_overlap
    })
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    with st.spinner("Ejecutando detección de PNNs (PNNloc + PNNscore)..."):
        try:
            model_loc, model_score, device = load_deep_models()
            
            # WFA 8-bit normalization
            w_min, w_max = wfa_raw.min(), wfa_raw.max()
            if w_max > w_min:
                wfa_8bit = ((wfa_raw - w_min) / (w_max - w_min) * 255.0).astype(np.uint8)
            else:
                wfa_8bit = wfa_raw.astype(np.uint8)

            scale_factor = px_size / 0.325
            H, W = wfa_raw.shape
            
            if abs(scale_factor - 1.0) > 0.01:
                new_H = int(round(H * scale_factor))
                new_W = int(round(W * scale_factor))
                wfa_8bit_scaled = cv2.resize(wfa_8bit, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
            else:
                wfa_8bit_scaled = wfa_8bit.copy()

            # Write temporary WFA file
            temp_wfa = f"temp_wfa_step3_{os.getpid()}_{base_fn}.tif"
            tiff.imwrite(temp_wfa, wfa_8bit_scaled.astype(np.float32))
            
            candidates = []
            try:
                from datasets.patched_datasets import PatchedMultiImageDataset
                from methods.detection.train_fn import predict_points
                from torch.utils.data import DataLoader
                from methods.detection.transforms import ToTensor
                
                cfg_loc = OmegaConf.load("data/models/pnn_v2_fasterrcnn_640/.hydra/config.yaml")
                cfg_loc.model.module.nms = 0.3
                
                stride = tile_size - tile_overlap
                ds = PatchedMultiImageDataset.from_paths([temp_wfa], patch_size=tile_size, stride=stride, transforms=ToTensor())
                dl = DataLoader(ds, batch_size=1, shuffle=False)
                
                locs = predict_points(dl, model_loc, device, loc_threshold, cfg_loc)
                H_scaled, W_scaled = wfa_8bit_scaled.shape
                prob_map_scaled = np.zeros((H_scaled, W_scaled), dtype=np.float32)
                
                raw_candidates = []
                for idx, row in locs.iterrows():
                    raw_candidates.append({
                        "centroid_y": float(row["Y"]),
                        "centroid_x": float(row["X"]),
                        "prob_map_val": float(row["score"])
                    })
                    
                # Greedy spatial coordinate NMS using min_peak_dist
                if raw_candidates:
                    sorted_cands = sorted(raw_candidates, key=lambda x: x["prob_map_val"], reverse=True)
                    for c in sorted_cands:
                        cy, cx = c["centroid_y"], c["centroid_x"]
                        dup = False
                        for mc in candidates:
                            mcy, mcx = mc["centroid_y"], mc["centroid_x"]
                            if np.sqrt((cy - mcy)**2 + (cx - mcx)**2) < min_peak_dist:
                                dup = True
                                break
                        if not dup:
                            c["id"] = len(candidates) + 1
                            candidates.append(c)
                            
                for idx, c in enumerate(candidates):
                    cy, cx = c["centroid_y"], c["centroid_x"]
                    score = c["prob_map_val"]
                    
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
                            
                # Save probability map as TIFF (resized back to original dimensions)
                prob_map_orig = cv2.resize(prob_map_scaled, (W, H), interpolation=cv2.INTER_LINEAR)
                prob_map_file = os.path.join(SEGM_DIR, f"{base_fn}_prob_map.tif")
                tiff.imwrite(prob_map_file, prob_map_orig.astype(np.float32))
            finally:
                if os.path.exists(temp_wfa):
                    os.remove(temp_wfa)

            # Scoring candidates (PNNscore) on scaled WFA
            wfa_norm = wfa_8bit_scaled.astype(np.float32) / 255.0
            
            half_sz = 32
            patch_args = [(wfa_norm, int(c["centroid_y"]), int(c["centroid_x"]), half_sz) for c in candidates]
            patches, valid_coords, valid_original_cands = [], [], []
            
            with concurrent.futures.ThreadPoolExecutor() as pool:
                for idx, result in enumerate(pool.map(extract_patch_helper, patch_args)):
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
            updated_cands = []
            
            for cand, (r, c), score in zip(valid_original_cands, valid_coords, scores):
                # Map coordinates back to original resolution for JSON saving
                r_orig = r / scale_factor
                c_orig = c / scale_factor
                
                cand_updated = cand.copy()
                cand_updated["centroid_y"] = float(r_orig)
                cand_updated["centroid_x"] = float(c_orig)
                cand_updated["score"] = float(score)
                updated_cands.append(cand_updated)
                
                if score >= score_threshold:
                    rad_orig = 12.0 / scale_factor
                    
                    r_int, c_int = int(round(r_orig)), int(round(c_orig))
                    rad_int = max(3, int(round(rad_orig)))
                    
                    r_min, r_max = max(0, r_int - rad_int), min(H, r_int + rad_int + 1)
                    c_min, c_max = max(0, c_int - rad_int), min(W, c_int + rad_int + 1)
                    
                    y_grid, x_grid = np.ogrid[r_min - r_orig : r_max - r_orig, c_min - c_orig : c_max - c_orig]
                    circle_mask = y_grid*y_grid + x_grid*x_grid <= (rad_orig ** 2)
                    circle_mask = circle_mask[:(r_max-r_min), :(c_max-c_min)]
                    
                    m_wfa[r_min:r_max, c_min:c_max][circle_mask] = pnn_id
                    
                    local_wfa = wfa_raw[r_min:r_max, c_min:c_max]
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
            
            # Save candidates
            with open(candidates_file, "w") as f:
                json.dump(updated_cands, f, indent=2)

            # Load or create segmented stack TIFF
            m_dapi = np.zeros_like(m_wfa)
            m_pv = np.zeros_like(m_wfa)
            if os.path.exists(seg_file):
                try:
                    loaded = tiff.imread(seg_file)
                    m_dapi = loaded[0, :, :]
                    m_pv = loaded[1, :, :]
                except Exception:
                    pass
            
            stk = np.stack([m_dapi.astype(np.uint16),
                            m_pv.astype(np.uint16),
                            m_wfa.astype(np.uint16),
                            wfa_raw.astype(np.uint16)], axis=0)
                            
            tiff.imwrite(seg_file, stk, imagej=True,
                         metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                                   'Labels': ['DAPI_Mask', 'PV_Mask', 'PNN_Mask', 'WFA_Raw']})
                                   
            # Colocalization matching with existing PV mask
            pv_props = regionprops(m_pv) if np.max(m_pv) > 0 else []
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
            
            # Process PNNs
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
                    
                wfa_s = float(np.sum(wfa_raw[wfa_mask]))
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
                
            # Process PV+/PNN-
            max_wfa_label = int(np.max(m_wfa)) if np.max(m_wfa) > 0 else 0
            for pvp in pv_props:
                pv_label = pvp.label
                if pv_label in matched_pv_labels:
                    continue
                pv_mask = (m_pv == pv_label)
                cy, cx = pvp.centroid
                pv_area = pvp.area * (px_size ** 2)
                pv_diameter = pvp.equivalent_diameter_area * px_size
                wfa_s = float(np.sum(wfa_raw[pv_mask]))
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
                
            df_b = pd.DataFrame(r_batch)
            if df_b.empty:
                df_b = pd.DataFrame(columns=[
                    'label', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um', 
                    'wfa_sum_intensity', 'is_pnn_plus', 'is_pv_plus', 'pv_label',
                    'cell_type', 'pv_area_um2', 'pv_diameter_um', 'pnn_area_um2', 'pnn_diameter_um', 'score'
                ])
            df_b.to_csv(csv_file, index=False)
            
            # Save Summary JSON
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
            with open(json_file, 'w') as fs:
                json.dump(summary, fs, indent=4)
                
            st.success("🎉 ¡Detección de PNNs completada!")
            st.rerun()
        except Exception as e:
            st.error(f"Error durante la detección: {e}")

# Previsualización y Mapa de Potencia (Estilo Lupori et al.)
st.subheader(f"Muestra seleccionada: `{selected_filename}`")
v_tab1, v_tab2, v_tab3 = st.tabs([
    "🧠 Redes PNN Detectadas",
    "⭕ Máscaras Pericelulares (Anillos 4µm)",
    "🔥 Mapa de Calor de Potencia (Lupori Energy Map)"
])

heatmap_path = os.path.join(SEGM_DIR, f"{base_fn}_power_heatmap.png")

with v_tab1:
    col_prev1, col_prev2 = st.columns(2)
    with col_prev1:
        st.markdown('<p class="img-caption">Canal WFA Original</p>', unsafe_allow_html=True)
        st.image(wfa_disp, width="stretch", clamp=True, channels="GRAY")

    with col_prev2:
        st.markdown('<p class="img-caption">Redes PNN Detectadas (IA)</p>', unsafe_allow_html=True)
        has_pnn_mask = False
        if os.path.exists(seg_file):
            try:
                loaded_masks = tiff.imread(seg_file)
                num_ch = loaded_masks.shape[0] if len(loaded_masks.shape) == 3 else 1
                m_pnn_mask = loaded_masks[2, :, :] if num_ch >= 3 else np.zeros_like(wfa_raw)
                if np.max(m_pnn_mask) > 0:
                    overlay = label2rgb(m_pnn_mask, image=wfa_disp, bg_label=0, alpha=0.4, image_alpha=1.0)
                    st.image(overlay, width="stretch", clamp=True)
                    has_pnn_mask = True
            except Exception as e:
                st.error(f"Error al cargar máscara segmentada: {e}")
                
        if not has_pnn_mask:
            st.info("👈 Ajusta los parámetros y presiona '🧠 Detectar PNNs (WFA)' para ver los resultados.")

with v_tab2:
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown('<p class="img-caption">Canal WFA + Máscara PNN</p>', unsafe_allow_html=True)
        if os.path.exists(seg_file):
            try:
                loaded_masks = tiff.imread(seg_file)
                num_ch = loaded_masks.shape[0] if len(loaded_masks.shape) == 3 else 1
                m_pnn_mask = loaded_masks[2, :, :] if num_ch >= 3 else np.zeros_like(wfa_raw)
                if np.max(m_pnn_mask) > 0:
                    overlay_pnn = label2rgb(m_pnn_mask, image=wfa_disp, bg_label=0, alpha=0.5, image_alpha=0.9)
                    st.image(overlay_pnn, width="stretch", clamp=True)
            except Exception:
                pass
    with col_r2:
        st.markdown('<p class="img-caption">Zona de Muestreo de Potencia: Anillos Pericelulares (4µm)</p>', unsafe_allow_html=True)
        if os.path.exists(seg_file):
            try:
                loaded_masks = tiff.imread(seg_file)
                num_ch = loaded_masks.shape[0] if len(loaded_masks.shape) == 3 else 1
                m_ring_mask = loaded_masks[3, :, :] if num_ch >= 4 else np.zeros_like(wfa_raw)

                if np.max(m_ring_mask) > 0:
                    overlay_ring = label2rgb(m_ring_mask, image=wfa_disp, bg_label=0, alpha=0.6, image_alpha=0.9)
                    st.image(overlay_ring, width="stretch", clamp=True)
                else:
                    st.info("Ejecuta la detección de PNNs para generar los anillos pericelulares de muestreo.")
            except Exception as e:
                st.warning(f"Información de anillos pericelulares no disponible: {e}")

with v_tab3:
    st.markdown('<p class="img-caption">Mapa de Calor de Potencia Pericelular (Energy / Pericellular WFA Intensity Heatmap - Lupori et al.)</p>', unsafe_allow_html=True)
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, caption="Mapa en pseudocolor TURBO (Lupori et al. 2023) representando la Potencia/Intensidad Pericelular de WFA.", width="stretch")
    else:
        st.info("El mapa de calor de potencia se genera automáticamente al ejecutar '🧠 Detectar PNNs (WFA)' o el procesamiento batch.")


# Inspector de Candidatos
if os.path.exists(candidates_file):
    try:
        with open(candidates_file, 'r') as f:
            cands_data = json.load(f)
            
        if len(cands_data) > 0:
            st.divider()
            st.subheader("🔍 Inspector de Candidatos (PNNscore)")
            st.write("Selecciona una PNN candidata de la lista para ver su parche 64x64 evaluado:")
            
            c_sel, c_score = st.columns([2, 1])
            with c_sel:
                cand_ids = [c["id"] for c in cands_data]
                selected_cand_id = st.selectbox(
                    "Seleccionar Candidato por ID:",
                    cand_ids,
                    index=0,
                    format_func=lambda cid: f"ID: {cid} (Y: {cands_data[cid-1]['centroid_y']:.0f}, X: {cands_data[cid-1]['centroid_x']:.0f}) - Score: {cands_data[cid-1].get('score', 0.0):.4f}"
                )
            
            selected_cand = cands_data[selected_cand_id - 1]
            cy, cx = selected_cand["centroid_y"], selected_cand["centroid_x"]
            score = selected_cand.get("score", 0.0)
            is_confirmed = score >= score_threshold
            
            with c_score:
                st.metric(
                    "Confianza PNNscore (IA)", 
                    f"{score:.4f}", 
                    delta="Aceptado" if is_confirmed else "Descartado",
                    delta_color="normal" if is_confirmed else "inverse"
                )
                
            H, W = wfa_raw.shape
            cy_int, cx_int = int(cy), int(cx)
            scale_factor = px_size / 0.325
            half_sz = max(16, int(round(64.0 / scale_factor)))
            y0, y1 = max(0, cy_int - half_sz), min(H, cy_int + half_sz)
            x0, x1 = max(0, cx_int - half_sz), min(W, cx_int + half_sz)
            
            patch_wfa = wfa_raw[y0:y1, x0:x1]
            wfa_min, wfa_max = patch_wfa.min(), patch_wfa.max()
            if wfa_max > wfa_min:
                patch_wfa_8bit = ((patch_wfa - wfa_min) / (wfa_max - wfa_min) * 255.0).astype(np.uint8)
            else:
                patch_wfa_8bit = patch_wfa.astype(np.uint8)
                
            wfa_rgb = cv2.cvtColor(patch_wfa_8bit, cv2.COLOR_GRAY2RGB)
            ctr_y = cy_int - y0
            ctr_x = cx_int - x0
            
            # Draw box
            box_half = max(3, int(round(12.0 / scale_factor)))
            box_color = (0, 255, 0) if is_confirmed else (255, 0, 0)
            cv2.rectangle(wfa_rgb, (ctr_x - box_half, ctr_y - box_half), (ctr_x + box_half, ctr_y + box_half), box_color, 1)
            cv2.circle(wfa_rgb, (ctr_x, ctr_y), 2, (0, 255, 255), -1)
            
            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(wfa_rgb, caption="Cuadro verde/rojo indica clasificación PNNscore en WFA.", width="stretch")
            with col_info:
                st.markdown(f"""
                * **ID Candidato:** {selected_cand_id}
                * **Centroide (Y, X):** `({cy:.1f}, {cx:.1f})` px
                * **Estado:** {"✅ Aprobado (PNN+)" if is_confirmed else "❌ Descartado"}
                """)
    except Exception as e:
        st.warning(f"Error en el inspector de candidatos: {e}")

# Inspección en Napari (siempre disponible si existe el archivo de máscaras)
if os.path.exists(seg_file):
    st.divider()
    st.markdown("### 🖥️ Inspección Visual en Napari")
    st.write("Visualiza la imagen con los canales biológicos originales y las máscaras segmentadas acumuladas hasta el momento.")
    if st.button("🧪 Abrir en Napari", type="primary", key="p3_btn_napari"):
        cmd = [sys.executable, "napari_viewer.py", "--path", seg_file, "--pixel_size", str(px_size), "--step", "wfa"]
        try:
            env = os.environ.copy()
            env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
            subprocess.Popen(cmd, env=env)
            st.success("✅ Visor Napari lanzado con éxito.")
        except Exception as e:
            st.error(f"Error al lanzar Napari: {e}")

# Métricas
if os.path.exists(csv_file):
    try:
        df_b = pd.read_csv(csv_file)
        if 'score' not in df_b.columns:
            df_b['score'] = 0.0
        df_pnn = df_b[df_b['is_pnn_plus'] == True]
        if not df_pnn.empty:
            st.divider()
            st.subheader("📊 Descriptores y Métricas Lupori (PNN+)")
            
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            c_m1.metric("Redes PNN+ Detectadas (IA)", f"{len(df_pnn)}")
            c_m2.metric("Área Promedio PNN (µm²)", f"{df_pnn['pnn_area_um2'].mean():.2f}")
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as fs:
                    summary_data = json.load(fs)
                
                c_m3.metric("Potencia PNN (Energy)", f"{summary_data.get('pnn_energy', 0.0):.2f}")
                c_m4.metric("Densidad PNN (PNNs/mm²)", f"{summary_data.get('pnn_density_mm2', 0.0):.1f}")
                
                st.info(f"⚡ **Métricas Lupori et al. (2023):**  \n"
                        f"* **Potencia PNN (Energy):** `{summary_data.get('pnn_energy', 0.0):.2f}`  \n"
                        f"* **Potencia Coexpresión (PV+/PNN+ Energy):** `{summary_data.get('coloc_energy', 0.0):.2f}`  \n"
                        f"* **Intensidad WFA Circundante (Ring Norm 0-1):** `{summary_data.get('mean_pnn_pericellular_wfa_norm', 0.0):.4f}`  \n"
                        f"* **Fluorescencia Difusa WFA:** `{summary_data.get('diffuse_wfa_fluorescence', 0.0):.4f}`  \n"
                        f"* **Coexpresión:** {summary_data.get('pct_pv_surrounded_by_pnn', 0.0):.1f}% de las células PV+ están rodeadas por PNN+ ({summary_data.get('pv_pnn_plus', 0)} / {summary_data.get('total_pv_segmentation', 0)}).")
            else:
                c_m3.metric("Confianza Promedio", f"{df_pnn['score'].mean():.4f}")
            
            st.markdown("### Tabla de Métricas de PNN+ (incluye Intensidad Circundante y Normalizada):")
            cols_to_show = [c for c in ['label', 'cell_type', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um', 'wfa_mean_intensity', 'wfa_pericellular_intensity', 'wfa_pericellular_norm', 'score'] if c in df_pnn.columns]
            st.dataframe(df_pnn[cols_to_show].head(100))
    except Exception as e:
        st.warning(f"Error al cargar descriptores de PNN+: {e}")

