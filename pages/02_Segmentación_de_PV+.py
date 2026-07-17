import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.measure import regionprops
import pandas as pd
import sys
import subprocess
from pipeline import load_channels_tif

# Page configuration
st.set_page_config(page_title="Paso 2: Segmentación de PV+", layout="wide")

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

st.markdown('<div class="main-header">🧪 Paso 2: Segmentación de Interneuronas PV+</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ajuste de parámetros y segmentación interactiva de somas Parvalbúmina+ (PV+) usando Cellpose.</div>', unsafe_allow_html=True)

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

selected_group = st.sidebar.selectbox("Grupo:", groups, key="p2_group")
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones en {selected_group}.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections, key="p2_section")
section_dir = os.path.join(group_dir, selected_section)

tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith('.tif')])
if not tif_files:
    st.warning("No hay archivos .tif en la sección.")
    st.stop()

selected_filename = st.sidebar.selectbox("Imagen:", tif_files, key="p2_file")
selected_path = os.path.join(section_dir, selected_filename)

SEGM_DIR = os.path.join(SEGM_BASE_DIR, selected_group, selected_section)
METRICS_DIR = os.path.join(METRICS_BASE_DIR, selected_group, selected_section)
os.makedirs(SEGM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load channels
try:
    (pv_raw, wfa_raw, dapi_raw, agr_raw) = load_channels_tif(selected_path)
    pv_disp = cv2.normalize(pv_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")
    st.stop()

# Sidebar - Cellpose PV Parameters
st.sidebar.header("⚙️ Ajustes de Cellpose PV")
use_gpu = st.sidebar.checkbox("Usar GPU (PyTorch)", value=True, key="p2_gpu")
filter_options = ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"]
pv_def_filter = calib_data.get('pv_cellpose_filter_type', "Ninguno")
pv_filter = st.selectbox("Filtro previo PV", filter_options, index=filter_options.index(pv_def_filter) if pv_def_filter in filter_options else 0, key="p2_filter")
pv_diam = st.number_input("Diámetro de PV (px)", value=float(calib_data.get('pv_cellpose_diameter', 30.0)), key="p2_diam", step=1.0)
pv_flow = st.slider("Flow Threshold PV", 0.0, 1.0, float(calib_data.get('pv_cellpose_flow_threshold', 0.4)), key="p2_flow")
pv_prob = st.slider("Cell Prob Threshold PV", -6.0, 6.0, float(calib_data.get('pv_cellpose_cellprob_threshold', 0.0)), key="p2_prob")

px_size = float(calib_data.get('pixel_size_um', 1.0))

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔬 Segmentar Canal PV+", type="primary", use_container_width=True)

base_fn, _ = os.path.splitext(selected_filename)
seg_file = os.path.join(SEGM_DIR, f"{base_fn}_masks.tif")
csv_file = os.path.join(METRICS_DIR, f"{base_fn}_nuclei_metrics.csv")
json_file = os.path.join(METRICS_DIR, f"{base_fn}_summary.json")

if run_btn:
    # Save parameters to global configuration
    calib_data.update({
        'pv_cellpose_filter_type': pv_filter,
        'pv_cellpose_diameter': pv_diam,
        'pv_cellpose_flow_threshold': pv_flow,
        'pv_cellpose_cellprob_threshold': pv_prob,
        'do_pv_segmentation': True
    })
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    with st.spinner("Ejecutando segmentación Cellpose para PV..."):
        try:
            in_pv = pv_raw.copy()
            if pv_filter == "Otsu Global":
                t = threshold_otsu(in_pv)
                in_pv[in_pv < t] = 0
            elif pv_filter == "CLAHE (Adaptativo Local)":
                clahe = exposure.equalize_adapthist(in_pv, clip_limit=0.03)
                in_pv = (clahe * 65535).astype(np.uint16)
                
            model_pv = models.CellposeModel(gpu=use_gpu)
            m_pv, _, _ = model_pv.eval(in_pv, diameter=pv_diam, 
                                        flow_threshold=pv_flow, cellprob_threshold=pv_prob)
            
            # Load or create segmented stack TIFF
            m_dapi = np.zeros_like(m_pv)
            m_wfa = np.zeros_like(m_pv)
            if os.path.exists(seg_file):
                try:
                    loaded = tiff.imread(seg_file)
                    num_ch = loaded.shape[0] if len(loaded.shape) == 3 else 1
                    m_dapi = loaded[0, :, :] if num_ch >= 1 else np.zeros_like(m_pv)
                    if num_ch >= 5:
                        m_wfa = loaded[4, :, :]
                    elif num_ch >= 3:
                        m_wfa = loaded[2, :, :]
                    else:
                        m_wfa = np.zeros_like(m_pv)
                except Exception:
                    pass
            
            stk = np.stack([m_dapi.astype(np.uint16),
                            m_pv.astype(np.uint16),
                            m_wfa.astype(np.uint16),
                            wfa_raw.astype(np.uint16)], axis=0)
                                
            tiff.imwrite(seg_file, stk, imagej=True,
                         metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                                   'Labels': ['DAPI_Mask', 'PV_Mask', 'PNN_Mask', 'WFA_Raw']})
                                   
            # Compute new PV metrics and do colocalization with existing PNN mask
            pv_props = regionprops(m_pv)
            wfa_props = regionprops(m_wfa) if np.max(m_wfa) > 0 else []
            
            # Colocalization matching
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
            
            # Load old CSV score if exists
            old_scores = {}
            if os.path.exists(csv_file):
                try:
                    df_old = pd.read_csv(csv_file)
                    for _, row in df_old.iterrows():
                        if row.get('is_pnn_plus') == True:
                            old_scores[int(row['label'])] = float(row.get('score', 0.0))
                except Exception:
                    pass
            
            # Process PNNs (WFA circles)
            for wfa_prop in wfa_props:
                wfa_label = wfa_prop.label
                wfa_mask = (m_wfa == wfa_label)
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
                    'score': old_scores.get(wfa_label, 0.0)
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
                
            st.success("🎉 ¡Segmentación completada con éxito!")
            st.rerun()
        except Exception as e:
            st.error(f"Error durante la segmentación: {e}")

# Previsualización
st.subheader(f"Muestra seleccionada: `{selected_filename}`")
col_prev1, col_prev2 = st.columns(2)

with col_prev1:
    st.markdown('<p class="img-caption">Canal PV Original</p>', unsafe_allow_html=True)
    st.image(pv_disp, width="stretch", clamp=True, channels="GRAY")

with col_prev2:
    st.markdown('<p class="img-caption">Somas PV+ Segmentados</p>', unsafe_allow_html=True)
    has_pv_mask = False
    if os.path.exists(seg_file):
        try:
            loaded_masks = tiff.imread(seg_file)
            m_pv = loaded_masks[1, :, :]
            if np.max(m_pv) > 0:
                overlay = label2rgb(m_pv, image=pv_disp, bg_label=0, alpha=0.4, image_alpha=1.0)
                st.image(overlay, width="stretch", clamp=True)
                has_pv_mask = True
        except Exception as e:
            st.error(f"Error al cargar máscara segmentada: {e}")
            
    if not has_pv_mask:
        st.info("👈 Ajusta los parámetros y presiona '🔬 Segmentar Canal PV+' para ver los resultados.")

# Inspección en Napari (siempre disponible si existe el archivo de máscaras)
if os.path.exists(seg_file):
    st.divider()
    st.markdown("### 🖥️ Inspección Visual en Napari")
    st.write("Visualiza la imagen con los canales biológicos originales y las máscaras segmentadas acumuladas hasta el momento.")
    if st.button("🧪 Abrir en Napari", type="primary", key="p2_btn_napari"):
        cmd = [sys.executable, "napari_viewer.py", "--path", seg_file, "--pixel_size", str(px_size), "--step", "pv"]
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
        df_pv = df_b[df_b['is_pv_plus'] == True]
        if not df_pv.empty:
            st.divider()
            st.subheader("📊 Descriptores de Interneuronas PV+")
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Interneuronas PV+ Segmentadas", f"{len(df_pv)}")
            c_m2.metric("Área Promedio Soma (µm²)", f"{df_pv['pv_area_um2'].mean():.2f}")
            c_m3.metric("Diámetro Promedio Soma (µm)", f"{df_pv['pv_diameter_um'].mean():.2f}")
            
            # If PNNs exist, show details
            if os.path.exists(json_file):
                with open(json_file, 'r') as fs:
                    summary_data = json.load(fs)
                st.write(f"De las interneuronas PV+, **{summary_data.get('pv_pnn_plus', 0)}** tienen red perineuronal (PV+/PNN+) y **{summary_data.get('pv_pnn_minus', 0)}** no tienen red (PV+/PNN-).")
            
            st.markdown("### Tabla de Métricas de PV+:")
            st.dataframe(df_pv.head(100), use_container_width=True)
    except Exception as e:
        st.warning(f"Error al cargar los descriptores: {e}")
