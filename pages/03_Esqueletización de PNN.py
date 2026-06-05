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

st.set_page_config(page_title="Esqueletización de PNN", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #bb86fc; }
    div[data-testid="stMetricLabel"] { color: #e0e0e0; }
    .img-caption { font-weight: bold; color: #bb86fc; margin-bottom: 5px; text-align: center; }
    hr { border: 0; height: 1px; background: linear-gradient(to right, transparent, #bb86fc, transparent); margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

st.title("🕸️ Paso 3: Esqueletización y Partición de PNN")
st.write("Ajusta los parámetros de binarización y morfología del esqueleto de WFA para representar fielmente las redes perineuronales.")

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
selected_group = st.sidebar.selectbox("Grupo:", groups)
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones en `{selected_group}`.")
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
    wfa_disp = cv2.normalize(wfa_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")
    st.stop()

st.subheader(f"Muestra: {selected_filename}")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="img-caption">Canal WFA (PNN) Original</p>', unsafe_allow_html=True)
    st.image(wfa_disp, width="stretch", clamp=True, channels="GRAY")

# --- Configuration ---
st.sidebar.header("⚙️ Configuración del Esqueleto")
use_gpu = st.sidebar.checkbox("Usar GPU (PyTorch)", value=True)

with st.sidebar.expander("Binarización de WFA"):
    thresh_methods = ["Automático (Otsu)", "Manual"]
    def_method = calib_data.get('pnn_wfa_threshold_method', "Automático (Otsu)")
    pnn_wfa_threshold_method = st.selectbox("Método Umbralado WFA", thresh_methods, index=thresh_methods.index(def_method) if def_method in thresh_methods else 0)
    pnn_wfa_manual_threshold = st.number_input("Umbral WFA Manual", value=float(calib_data.get('pnn_wfa_manual_threshold', 10000.0)))
    max_pnn_distance_um = st.number_input("Distancia máx. Voronoi (µm)", value=float(calib_data.get('max_pnn_distance_um', 20.0)))
    pnn_gaussian_sigma = st.slider("Suavizado Gaussiano WFA (Sigma)", 0.0, 3.0, float(calib_data.get('pnn_gaussian_sigma', 1.0)), step=0.1)

with st.sidebar.expander("Modificaciones de Esqueleto"):
    pnn_connect_fragments = st.checkbox("Conectar fragmentos cercanos", value=bool(calib_data.get('pnn_connect_fragments', False)))
    pnn_connection_radius_um = st.number_input("Radio de conexión (µm)", value=float(calib_data.get('pnn_connection_radius_um', 1.0)), step=0.1)
    pnn_pruning_min_voxels = st.number_input("Umbral de Poda (vóxeles)", value=int(calib_data.get('pnn_pruning_min_voxels', 0)), step=1)
    pnn_filter_by_nucleus = st.checkbox("Filtrar por conectividad al soma", value=bool(calib_data.get('pnn_filter_by_nucleus', False)))

with st.sidebar.expander("Clasificación PV+/PNN+"):
    st.markdown("La detección PV+/PNN+ se realiza por colocalización espacial directa (máxima superficie). No se requiere anillo de expansión.")
    pv_expansion_dist_um = 0.0
    pnn_threshold = 0.0
    pnn_exclusion_dist_um = 0.0

px_size = calib_data.get('pixel_size_um', 1.0)
dapi_filter = calib_data.get('cellpose_filter_type', 'Ninguno')
dapi_diam = calib_data.get('cellpose_diameter', 30.0)
dapi_flow = calib_data.get('cellpose_flow_threshold', 0.4)
dapi_prob = calib_data.get('cellpose_cellprob_threshold', 0.1)

pv_filter_type = calib_data.get('pv_cellpose_filter_type', 'Ninguno')
pv_diameter = calib_data.get('pv_cellpose_diameter', 15.0)
pv_flow_threshold = calib_data.get('pv_cellpose_flow_threshold', 0.4)
pv_cellprob_threshold = calib_data.get('pv_cellpose_cellprob_threshold', 0.0)

st.sidebar.divider()
if st.sidebar.button("🔬 Previsualizar Esqueleto y Partición", type="primary", use_container_width=True):
    # Save skeleton configurations globally
    calib_data.update({
        'pnn_wfa_threshold_method': pnn_wfa_threshold_method,
        'pnn_wfa_manual_threshold': pnn_wfa_manual_threshold,
        'max_pnn_distance_um': max_pnn_distance_um,
        'pnn_gaussian_sigma': pnn_gaussian_sigma,
        'pnn_connect_fragments': pnn_connect_fragments,
        'pnn_connection_radius_um': pnn_connection_radius_um,
        'pnn_pruning_min_voxels': pnn_pruning_min_voxels,
        'pnn_filter_by_nucleus': pnn_filter_by_nucleus,
        'pv_expansion_dist_um': pv_expansion_dist_um,
        'pnn_intensity_threshold': pnn_threshold,
        'pnn_exclusion_distance_um': pnn_exclusion_dist_um
    })
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    with st.spinner("Ejecutando pipeline de esqueletonización..."):
        try:
            model_dapi = models.CellposeModel(gpu=use_gpu)
            model_pv = models.CellposeModel(gpu=use_gpu)
            
            run_pipeline_on_file(
                tif_path=selected_path,
                out_segm_dir=SEGM_DIR,
                out_metrics_dir=METRICS_DIR,
                model_dapi=model_dapi,
                model_pv_obj=model_pv,
                filter_type=dapi_filter, diameter=dapi_diam, flow_threshold=dapi_flow, cellprob_threshold=dapi_prob,
                pv_filter_type=pv_filter_type, pv_diameter=pv_diameter, pv_flow_threshold=pv_flow_threshold, pv_cellprob_threshold=pv_cellprob_threshold,
                pv_expansion_dist_um=pv_expansion_dist_um, pnn_threshold=pnn_threshold, pnn_exclusion_dist_um=pnn_exclusion_dist_um,
                px_size=px_size, do_pv_segmentation=True, calib_data=calib_data
            )
            st.sidebar.success("Procesamiento de esqueleto finalizado.")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.markdown('<p class="img-caption">Esqueleto de Voronoi (sobre Huecos de PNN)</p>', unsafe_allow_html=True)
    base_fn, _ = os.path.splitext(selected_filename)
    seg_file = os.path.join(SEGM_DIR, f"{base_fn}_masks.tif")
    csv_file = os.path.join(METRICS_DIR, f"{base_fn}_nuclei_metrics.csv")
    
    skeleton_mask = None
    if os.path.exists(seg_file):
        loaded = tiff.imread(seg_file)
        if loaded.shape[0] >= 3:
            skeleton_mask = loaded[2, :, :] # Channel 2: Skeleton Mask
            wfa_mask = loaded[3, :, :] if loaded.shape[0] == 4 else (loaded[4, :, :] if loaded.shape[0] >= 5 else np.zeros_like(loaded[0]))
            
    if skeleton_mask is not None:
        # Create an overlay showing both the WFA mask as solid and the skeleton in red
        overlay = label2rgb(np.squeeze(wfa_mask), image=wfa_disp, bg_label=0, alpha=0.3, image_alpha=1)
        # Add red skeleton pixels
        ys, xs = np.where(skeleton_mask > 0)
        for y, x in zip(ys, xs):
            overlay[y, x] = [255, 0, 0] # Draw red pixels
        st.image(overlay, width="stretch", clamp=True)
    else:
        st.info("👈 Presiona '🔬 Previsualizar Esqueleto y Partición' en la barra lateral para ver los resultados.")

st.divider()

if os.path.exists(csv_file):
    df_metrics = pd.read_csv(csv_file)
    if not df_metrics.empty:
        total_pnn = df_metrics['is_pnn_plus'].sum()
        pct_pnn = (total_pnn / len(df_metrics)) * 100
        mean_len = df_metrics[df_metrics['is_pnn_plus'] == True]['skel_total_length_um'].mean()
        mean_branches = df_metrics[df_metrics['is_pnn_plus'] == True]['skel_branches_count'].mean()
        
        st.subheader("📊 Métricas de Esqueletos PNN")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Células PNN+ Totales", f"{total_pnn}")
        m2.metric("% Células PNN+ en PV+", f"{pct_pnn:.1f}%")
        m3.metric("Longitud Esqueleto Promedio", f"{mean_len:.2f} µm")
        m4.metric("Ramas Promedio", f"{mean_branches:.1f}")
        
        st.dataframe(df_metrics[['label', 'is_pnn_plus', 'skel_total_length_um', 'skel_branches_count', 'skel_endpoints_count', 'skel_junctions_count', 'skel_tortuosity_mean']].head(20), use_container_width=True)
        
        st.markdown("### 🖥️ Inspección Visual en Napari")
        if st.button("🧪 Abrir Imagen Segmentada en Napari", type="primary"):
            import subprocess
            import sys
            
            cmd = [sys.executable, "napari_viewer.py", "--path", seg_file, "--pixel_size", str(px_size)]
            try:
                env = os.environ.copy()
                env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
                subprocess.Popen(cmd, env=env)
                st.success("✅ Visor Napari lanzado.")
            except Exception as e:
                st.error(f"Error al lanzar Napari: {e}")
