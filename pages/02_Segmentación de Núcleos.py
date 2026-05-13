import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.color import label2rgb
import pandas as pd
from pipeline import run_pipeline_on_file, load_channels_tif

st.set_page_config(page_title="Segmentación de Núcleos", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #bb86fc; }
    div[data-testid="stMetricLabel"] { color: #e0e0e0; }
    .img-caption { font-weight: bold; color: #bb86fc; margin-bottom: 5px; text-align: center; }
    hr { border: 0; height: 1px; background: linear-gradient(to right, transparent, #bb86fc, transparent); margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Página 2: Segmentación de Núcleos (DAPI + PV)")

# --- Paths ---
RAW_DIR = "data/raw"
SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

if not os.path.exists(RAW_DIR):
    st.error(f"No se encontró `{RAW_DIR}`.")
    st.stop()

# Get groups
groups = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
if not groups:
    st.warning("No hay grupos en `data/raw`.")
    st.stop()

st.sidebar.header("📁 Selección de Datos")
selected_group = st.sidebar.selectbox("Grupo:", groups)
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones (IPSI/CONTRA) en `{selected_group}`.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections)
section_dir = os.path.join(group_dir, selected_section)

tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith('.tif')])
if not tif_files:
    st.warning(f"No hay imágenes `.TIF` en `{selected_section}`.")
    st.stop()

selected_filename = st.sidebar.selectbox("Archivo:", tif_files)
selected_path = os.path.join(section_dir, selected_filename)

SEGM_DIR = os.path.join(SEGM_BASE_DIR, selected_group, selected_section)
METRICS_DIR = os.path.join(METRICS_BASE_DIR, selected_group, selected_section)
os.makedirs(SEGM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# --- Load Config ---
calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)
    st.sidebar.success("✅ Configuración Global Cargada")
else:
    st.sidebar.warning("⚠️ No se encontró configuración global.")

# Load Channels for visualization
try:
    (pv_raw, wfa_raw, dapi_raw, agr_raw) = load_channels_tif(selected_path)
    dapi_disp = cv2.normalize(dapi_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    pv_disp = cv2.normalize(pv_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    wfa_disp = cv2.normalize(wfa_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")
    st.stop()

st.subheader(f"Muestra: {selected_filename}")

c_col1, c_col2 = st.columns([2, 1])
with c_col1:
    view_mode = st.radio("Capa a visualizar:", ["Núcleos (DAPI)", "Parvalbúmina (PV+)", "Redes (PNN+)"], horizontal=True)

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="img-caption">Canal de Referencia</p>', unsafe_allow_html=True)
    if view_mode == "Núcleos (DAPI)":
        st.image(dapi_disp, use_container_width=True, clamp=True, channels="GRAY")
    elif view_mode == "Parvalbúmina (PV+)":
        st.image(pv_disp, use_container_width=True, clamp=True, channels="GRAY")
    else:
        st.image(wfa_disp, use_container_width=True, clamp=True, channels="GRAY")

# --- Configuration (can edit here and save to global) ---
st.sidebar.header("⚙️ Configuración (Preview)")
use_gpu = st.sidebar.checkbox("Usar GPU (PyTorch)", value=True)

with st.sidebar.expander("Parámetros Cellpose (DAPI)"):
    filter_type = st.selectbox("Filtro", ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"], index=0)
    diameter = st.number_input("Diámetro", value=float(calib_data.get('cellpose_diameter', 30.0)))
    flow_threshold = st.slider("Flow Threshold", 0.0, 1.0, float(calib_data.get('cellpose_flow_threshold', 0.4)))
    cellprob_threshold = st.slider("Cellprob", -6.0, 6.0, float(calib_data.get('cellpose_cellprob_threshold', 0.1)))

with st.sidebar.expander("Parámetros Cellpose (PV)"):
    do_pv = st.checkbox("Activar segmentación PV", value=True)
    pv_filter_type = st.selectbox("Filtro PV", ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"], index=0)
    pv_diameter = st.number_input("Diámetro PV", value=float(calib_data.get('pv_cellpose_diameter', 30.0)))
    pv_flow_threshold = st.slider("Flow Threshold PV", 0.0, 1.0, float(calib_data.get('pv_cellpose_flow_threshold', 0.4)))
    pv_cellprob_threshold = st.slider("Cellprob PV", -6.0, 6.0, float(calib_data.get('pv_cellpose_cellprob_threshold', 0.0)))

with st.sidebar.expander("Parámetros PNN"):
    pnn_radius_um = st.number_input("Radio PNN (µm)", value=float(calib_data.get('pnn_radius_um', 20.0)))
    pnn_threshold = st.number_input("Umbral WFA", value=float(calib_data.get('pnn_intensity_threshold', 500000.0)))
    pnn_exclusion_dist_um = st.number_input("Exclusión (µm)", value=float(calib_data.get('pnn_exclusion_distance_um', 15.0)))

px_size = calib_data.get('pixel_size_um', 1.0)

st.sidebar.divider()
st.sidebar.info("Para procesar todas las imágenes, utiliza el botón de 'Batch' en la página Principal.")
if st.sidebar.button("🔬 Previsualizar Segmentación", type="primary", use_container_width=True):
    with st.spinner("Ejecutando Pipeline para esta imagen..."):
        try:
            model_dapi = models.CellposeModel(gpu=use_gpu, model_type="nuclei")
            model_pv = models.CellposeModel(gpu=use_gpu, model_type="nuclei") if do_pv else None
            
            run_pipeline_on_file(
                tif_path=selected_path,
                out_segm_dir=SEGM_DIR,
                out_metrics_dir=METRICS_DIR,
                model_dapi=model_dapi,
                model_pv_obj=model_pv,
                filter_type=filter_type, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                pv_filter_type=pv_filter_type, pv_diameter=pv_diameter, pv_flow_threshold=pv_flow_threshold, pv_cellprob_threshold=pv_cellprob_threshold,
                pnn_radius_um=pnn_radius_um, pnn_threshold=pnn_threshold, pnn_exclusion_dist_um=pnn_exclusion_dist_um,
                px_size=px_size, do_pv_segmentation=do_pv, calib_data=calib_data
            )
            st.sidebar.success("Segmentación de prueba finalizada.")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.markdown('<p class="img-caption">Visualización de Resultados (Máscaras)</p>', unsafe_allow_html=True)
    seg_file = os.path.join(SEGM_DIR, selected_filename.replace('.TIF', '_segmented.tif').replace('.tif', '_segmented.tif'))
    csv_file = os.path.join(METRICS_DIR, selected_filename.replace('.TIF', '_nuclei_metrics.csv').replace('.tif', '_nuclei_metrics.csv'))
    
    current_masks = None
    base_img = dapi_disp
    
    if os.path.exists(seg_file):
        loaded = tiff.imread(seg_file)
        if view_mode == "Núcleos (DAPI)":
            current_masks = loaded[4, :, :]
            base_img = dapi_disp
        elif view_mode == "Parvalbúmina (PV+)":
            current_masks = loaded[5, :, :]
            base_img = pv_disp
        else:
            current_masks = loaded[6, :, :]
            base_img = wfa_disp
                
    if current_masks is not None:
        overlay = label2rgb(np.squeeze(current_masks), image=base_img, bg_label=0, alpha=0.4, image_alpha=1)
        st.image(overlay, use_container_width=True, clamp=True)
    else:
        st.info("👈 Procesa la muestra para previsualizar los resultados.")

st.divider()
if os.path.exists(csv_file):
    df_metrics = pd.read_csv(csv_file)
    st.subheader("📊 Dashboard de Resultados (Vista Previa)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total DAPI (Núcleos)", len(df_metrics))
    m2.metric("Total PV+", df_metrics['is_pv_plus'].sum() if 'is_pv_plus' in df_metrics.columns else 0)
    pnn_count = df_metrics['is_pnn_plus'].sum() if 'is_pnn_plus' in df_metrics.columns else 0
    m3.metric("PNN+ (Redes)", pnn_count)
    coloc = df_metrics[(df_metrics.get('is_pnn_plus', False) == True) & (df_metrics.get('is_pv_plus', False) == True)].shape[0] if 'is_pv_plus' in df_metrics.columns else 0
    m4.metric("Coloc (PV+/PNN+)", coloc)
    st.dataframe(df_metrics.head(20))
    
    st.markdown("### 🖥️ Inspección Visual en QuPath")
    if st.button("🌌 Abrir Imagen Segmentada en QuPath", type="primary"):
        from subprocess import Popen, check_output
        def open_q(p):
            exe = st.session_state.get('qupath_path', r"C:\Users\danie\AppData\Local\QuPath-0.7.0\QuPath-0.7.0.exe")
            win = check_output(['wslpath', '-w', p]).decode('utf-8').strip()
            Popen(['powershell.exe', '-Command', f"& '{exe}' '{win}'"])
        open_q(seg_file)
        st.success("Abriendo QuPath...")
