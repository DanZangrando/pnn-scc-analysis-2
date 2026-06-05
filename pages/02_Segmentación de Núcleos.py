import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
import pandas as pd
from pipeline import run_pipeline_on_file, load_channels_tif

# Page configuration
st.set_page_config(page_title="Segmentación de Núcleos", layout="wide")

# Custom CSS for a dark, premium aesthetic
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #00f2fe; }
    div[data-testid="stMetricLabel"] { color: #e0e0e0; }
    .img-caption { font-weight: bold; color: #00f2fe; margin-bottom: 5px; text-align: center; }
    hr { border: 0; height: 1px; background: linear-gradient(to right, transparent, #00f2fe, transparent); margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Paso 2: Segmentación y Colocalización de Núcleos (DAPI / PV)")
st.write("Ajusta los parámetros de Cellpose para segmentar núcleos DAPI e interneuronas PV+, y analiza su solapamiento y asociación con PNN.")

RAW_DIR = "data/raw"
if not os.path.exists(RAW_DIR) or not any(os.path.isdir(os.path.join(RAW_DIR, d)) for d in os.listdir(RAW_DIR) if not d.startswith('.')):
    RAW_DIR = "data/processed/mips"

SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

if not os.path.exists(RAW_DIR):
    st.error(f"No se encontró `{RAW_DIR}` ni `data/processed/mips`.")
    st.stop()

# Group selection
groups = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
if not groups:
    st.warning(f"No hay grupos en `{RAW_DIR}`.")
    st.stop()

st.sidebar.header("📁 Selección de Datos")
selected_group = st.sidebar.selectbox("Grupo:", groups, key="p2_group_select")
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones en `{selected_group}`.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections, key="p2_section_select")
section_dir = os.path.join(group_dir, selected_section)

tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith('.tif')])
if not tif_files:
    st.warning(f"No hay imágenes `.TIF` en `{selected_section}`.")
    st.stop()

selected_filename = st.sidebar.selectbox("Archivo:", tif_files, key="p2_file_select")
selected_path = os.path.join(section_dir, selected_filename)

SEGM_DIR = os.path.join(SEGM_BASE_DIR, selected_group, selected_section)
METRICS_DIR = os.path.join(METRICS_BASE_DIR, selected_group, selected_section)
os.makedirs(SEGM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load global configuration
calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)
    st.sidebar.success("✅ Configuración Global Cargada")
else:
    st.sidebar.warning("⚠️ No se encontró configuración global.")

# Load channels
try:
    (pv_raw, wfa_raw, dapi_raw, agr_raw) = load_channels_tif(selected_path)
    dapi_disp = cv2.normalize(dapi_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    pv_disp = cv2.normalize(pv_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    wfa_disp = cv2.normalize(wfa_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")
    st.stop()

st.subheader(f"Muestra: {selected_filename}")

col_v, _ = st.columns([2, 1])
with col_v:
    view_mode = st.radio("Capa a previsualizar:", ["Núcleos (DAPI)", "Células PV+ (Parvalbúmina)", "PNN+ con PV+ (Ocupadas)", "PNN+ Huecas (Sin PV+)", "Somas PNN (WFA Cellpose)", "PV+ con PNN (Ocupadas)"], horizontal=True, key="p2_view_mode")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="img-caption">Canal Original</p>', unsafe_allow_html=True)
    if view_mode == "Núcleos (DAPI)":
        st.image(dapi_disp, width="stretch", clamp=True, channels="GRAY")
    elif view_mode == "Células PV+ (Parvalbúmina)" or view_mode == "PV+ con PNN (Ocupadas)":
        st.image(pv_disp, width="stretch", clamp=True, channels="GRAY")
    else:
        st.image(wfa_disp, width="stretch", clamp=True, channels="GRAY")

# --- Configuration ---
st.sidebar.header("⚙️ Configuración Cellpose")
use_gpu = st.sidebar.checkbox("Usar GPU (PyTorch)", value=True, key="p2_use_gpu")

with st.sidebar.expander("🧬 Parámetros Cellpose (DAPI)"):
    filter_options = ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"]
    dapi_def_filter = calib_data.get('cellpose_filter_type', "CLAHE (Adaptativo Local)")
    dapi_filter = st.selectbox("Filtro DAPI", filter_options, index=filter_options.index(dapi_def_filter) if dapi_def_filter in filter_options else 0, key="p2_dapi_filter")
    dapi_diam = st.number_input("Diámetro Núcleo DAPI (px)", value=float(calib_data.get('cellpose_diameter', 30.0)), key="p2_dapi_diam")
    dapi_flow = st.slider("Flow Threshold DAPI", 0.0, 1.0, float(calib_data.get('cellpose_flow_threshold', 0.4)), key="p2_dapi_flow")
    dapi_prob = st.slider("Cellprob Threshold DAPI", -6.0, 6.0, float(calib_data.get('cellpose_cellprob_threshold', 0.1)), key="p2_dapi_prob")

with st.sidebar.expander("🧪 Parámetros Cellpose (PV)"):
    do_pv = st.checkbox("Activar segmentación PV", value=calib_data.get('do_pv_segmentation', True), key="p2_do_pv")
    pv_def_filter = calib_data.get('pv_cellpose_filter_type', "Ninguno")
    pv_filter_type = st.selectbox("Filtro PV", filter_options, index=filter_options.index(pv_def_filter) if pv_def_filter in filter_options else 0, key="p2_pv_filter")
    pv_diameter = st.number_input("Diámetro Soma PV (px)", value=float(calib_data.get('pv_cellpose_diameter', 30.0)), key="p2_pv_diam")
    pv_flow_threshold = st.slider("Flow Threshold PV", 0.0, 1.0, float(calib_data.get('pv_cellpose_flow_threshold', 0.4)), key="p2_pv_flow")
    pv_cellprob_threshold = st.slider("Cellprob Threshold PV", -6.0, 6.0, float(calib_data.get('pv_cellpose_cellprob_threshold', 0.0)), key="p2_pv_prob")

with st.sidebar.expander("🕸️ Parámetros PNN"):
    st.markdown("La detección PV+/PNN+ se realiza por colocalización espacial directa (máxima superficie). No se requiere anillo de expansión.")
    pv_expansion_dist_um = 0.0
    pnn_radius_um = 0.0
    pnn_threshold = 0.0
    pnn_exclusion_dist_um = 0.0

with st.sidebar.expander("🕸️ Parámetros Cellpose (WFA)"):
    do_wfa_cp = st.checkbox("Activar segmentación WFA (Cellpose)", value=calib_data.get('do_wfa_cellpose', True), key="p2_do_wfa_cp")
    wfa_cp_def_filter = calib_data.get('wfa_cellpose_filter_type', "Ninguno")
    wfa_cp_filter = st.selectbox("Filtro WFA", filter_options, index=filter_options.index(wfa_cp_def_filter) if wfa_cp_def_filter in filter_options else 0, key="p2_wfa_cp_filter")
    wfa_cp_diam = st.number_input("Diámetro PNN WFA (px)", value=float(calib_data.get('wfa_cellpose_diameter', 30.0)), key="p2_wfa_cp_diam")
    wfa_cp_flow = st.slider("Flow Threshold WFA", 0.0, 1.0, float(calib_data.get('wfa_cellpose_flow_threshold', 0.4)), key="p2_wfa_cp_flow")
    wfa_cp_prob = st.slider("Cellprob Threshold WFA", -6.0, 6.0, float(calib_data.get('wfa_cellpose_cellprob_threshold', 0.0)), key="p2_wfa_cp_prob")

px_size = calib_data.get('pixel_size_um', 1.0)

st.sidebar.divider()
if st.sidebar.button("🔬 Segmentar y Previsualizar Núcleos", type="primary", use_container_width=True, key="p2_btn_segment"):
    # Save parameters globally
    calib_data.update({
        'cellpose_filter_type': dapi_filter,
        'cellpose_diameter': dapi_diam,
        'cellpose_flow_threshold': dapi_flow,
        'cellpose_cellprob_threshold': dapi_prob,
        'do_pv_segmentation': do_pv,
        'pv_cellpose_filter_type': pv_filter_type,
        'pv_cellpose_diameter': pv_diameter,
        'pv_cellpose_flow_threshold': pv_flow_threshold,
        'pv_cellpose_cellprob_threshold': pv_cellprob_threshold,
        'pnn_radius_um': pnn_radius_um,
        'pv_expansion_dist_um': pv_expansion_dist_um,
        'pnn_intensity_threshold': pnn_threshold,
        'pnn_exclusion_distance_um': pnn_exclusion_dist_um,
        'do_wfa_cellpose': do_wfa_cp,
        'wfa_cellpose_filter_type': wfa_cp_filter,
        'wfa_cellpose_diameter': wfa_cp_diam,
        'wfa_cellpose_flow_threshold': wfa_cp_flow,
        'wfa_cellpose_cellprob_threshold': wfa_cp_prob
    })
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    with st.spinner("Ejecutando segmentación de núcleos..."):
        try:
            model_dapi = models.CellposeModel(gpu=use_gpu)
            model_pv = models.CellposeModel(gpu=use_gpu) if do_pv else None
            
            run_pipeline_on_file(
                tif_path=selected_path,
                out_segm_dir=SEGM_DIR,
                out_metrics_dir=METRICS_DIR,
                model_dapi=model_dapi,
                model_pv_obj=model_pv,
                filter_type=dapi_filter, diameter=dapi_diam, flow_threshold=dapi_flow, cellprob_threshold=dapi_prob,
                pv_filter_type=pv_filter_type, pv_diameter=pv_diameter, pv_flow_threshold=pv_flow_threshold, pv_cellprob_threshold=pv_cellprob_threshold,
                pv_expansion_dist_um=pv_expansion_dist_um, pnn_threshold=pnn_threshold, pnn_exclusion_dist_um=pnn_exclusion_dist_um,
                px_size=px_size, do_pv_segmentation=do_pv, calib_data=calib_data
            )
            st.sidebar.success("Segmentación de prueba finalizada.")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.markdown('<p class="img-caption">Máscara Segmentada</p>', unsafe_allow_html=True)
    base_fn, _ = os.path.splitext(selected_filename)
    seg_file = os.path.join(SEGM_DIR, f"{base_fn}_masks.tif")
    csv_file = os.path.join(METRICS_DIR, f"{base_fn}_nuclei_metrics.csv")
    dapi_csv_file = os.path.join(METRICS_DIR, f"{base_fn}_dapi_metrics.csv")
    json_file = os.path.join(METRICS_DIR, f"{base_fn}_summary.json")
    
    current_masks = None
    base_img = dapi_disp
    
    if os.path.exists(seg_file):
        loaded = tiff.imread(seg_file)
        pv_mask = loaded[1, :, :] if loaded.shape[0] >= 2 else np.zeros_like(loaded[0])
        wfa_mask = loaded[3, :, :] if loaded.shape[0] == 4 else (loaded[4, :, :] if loaded.shape[0] >= 5 else np.zeros_like(loaded[0]))

        if view_mode == "Núcleos (DAPI)":
            current_masks = loaded[0, :, :]
            base_img = dapi_disp
        elif view_mode == "Células PV+ (Parvalbúmina)":
            current_masks = pv_mask
            base_img = pv_disp
        elif view_mode == "Somas PNN (WFA Cellpose)":
            current_masks = wfa_mask
            base_img = wfa_disp
        elif view_mode == "PNN+ con PV+ (Ocupadas)":
            # Find WFA labels overlapping with pv_mask > 0
            wfa_labels = np.unique(wfa_mask)
            wfa_labels = wfa_labels[wfa_labels > 0]
            wfa_ocupadas = np.zeros_like(wfa_mask)
            for wfa_lbl in wfa_labels:
                submask = (wfa_mask == wfa_lbl)
                if np.any(pv_mask[submask] > 0):
                    wfa_ocupadas[submask] = wfa_lbl
            current_masks = wfa_ocupadas
            base_img = wfa_disp
        elif view_mode == "PNN+ Huecas (Sin PV+)":
            # Find WFA labels NOT overlapping with pv_mask > 0
            wfa_labels = np.unique(wfa_mask)
            wfa_labels = wfa_labels[wfa_labels > 0]
            wfa_huecas = np.zeros_like(wfa_mask)
            for wfa_lbl in wfa_labels:
                submask = (wfa_mask == wfa_lbl)
                if not np.any(pv_mask[submask] > 0):
                    wfa_huecas[submask] = wfa_lbl
            current_masks = wfa_huecas
            base_img = wfa_disp
        elif view_mode == "PV+ con PNN (Ocupadas)":
            # Find unique PV labels that overlap with wfa_mask > 0
            overlapping_labels = np.unique(pv_mask[wfa_mask > 0])
            overlapping_labels = overlapping_labels[overlapping_labels > 0]
            current_masks = np.where(np.isin(pv_mask, overlapping_labels), pv_mask, 0)
            base_img = pv_disp
                
    if current_masks is not None:
        overlay = label2rgb(np.squeeze(current_masks), image=base_img, bg_label=0, alpha=0.4, image_alpha=1)
        st.image(overlay, width="stretch", clamp=True)
    else:
        st.info("👈 Presiona '🔬 Segmentar y Previsualizar Núcleos' en la barra lateral para ver los resultados.")

st.divider()

if os.path.exists(json_file):
    with open(json_file, 'r') as jf:
        summary_data = json.load(jf)
        
    st.subheader("📊 Descriptores de Colocalización Celular (PV / PNN)")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("Total Somas PV+", f"{summary_data.get('total_pv_segmentation', 0)}")
    m2.metric("PV+ con PNN (Ocupadas)", f"{summary_data.get('pv_pnn_plus', 0)}")
    m3.metric("PV+ sin PNN (Sin Red)", f"{summary_data.get('pv_pnn_minus', 0)}")
    m4.metric("PNN+ Huecas (Sin PV+)", f"{summary_data.get('hollow_pnn_plus', 0)}")
    m5.metric("Total Redes PNN+", f"{summary_data.get('total_pnn_plus', 0)}")
    
    st.write("")
    st.write("**Tabla de Métricas Celulares (PV-Céntrico):**")
    if os.path.exists(csv_file):
        df_pv = pd.read_csv(csv_file)
        st.dataframe(df_pv.head(50), use_container_width=True)
            
    st.markdown("### 🖥️ Inspección Visual en Napari")
    if st.button("🧪 Abrir Imagen Segmentada en Napari", type="primary", key="p2_btn_napari"):
        import subprocess
        import sys
        
        cmd = [sys.executable, "napari_viewer.py", "--path", seg_file, "--pixel_size", str(px_size)]
        try:
            env = os.environ.copy()
            env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
            subprocess.Popen(cmd, env=env)
            st.success("✅ Visor Napari lanzado con éxito.")
        except Exception as e:
            st.error(f"Error al lanzar Napari: {e}")
