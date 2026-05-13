import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, disk, binary_dilation
from skimage.filters import threshold_otsu
from skan import Skeleton, summarize
from skimage.measure import regionprops

st.set_page_config(page_title="Análisis de Esqueletos PNN", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #bb86fc; }
    </style>
""", unsafe_allow_html=True)

st.title("🕸️ Página 3: Análisis Morfológico de PNN (Esqueletos)")

# --- Paths ---
SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

if not os.path.exists(SEGM_BASE_DIR):
    st.error("No hay imágenes segmentadas disponibles. Procesa imágenes en la Página 2 (o en lote) primero.")
    st.stop()

# Get groups
groups = sorted([d for d in os.listdir(SEGM_BASE_DIR) if os.path.isdir(os.path.join(SEGM_BASE_DIR, d))])
if not groups:
    st.warning("No hay grupos procesados.")
    st.stop()

st.sidebar.header("📁 Selección de Datos")
selected_group = st.sidebar.selectbox("Grupo:", groups)

group_dir = os.path.join(SEGM_BASE_DIR, selected_group)
sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])

if not sections:
    st.warning(f"No hay secciones (IPSI/CONTRA) en `{selected_group}`.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections)

SEGM_DIR = os.path.join(group_dir, selected_section)
METRICS_DIR = os.path.join(METRICS_BASE_DIR, selected_group, selected_section)

# --- Load Config ---
calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)
    px_size = calib_data.get('pixel_size_um', 1.0)
else:
    st.warning("⚠️ Sin configuración global encontrada.")
    px_size = 1.0

# --- Data Loading ---
segmented_files = []
if os.path.exists(SEGM_DIR):
    segmented_files = sorted([f for f in os.listdir(SEGM_DIR) if f.endswith('_segmented.tif')])

if not segmented_files:
    st.error(f"No hay imágenes segmentadas en `{SEGM_DIR}`.")
    st.stop()

selected_file = st.sidebar.selectbox("Archivo Segmentado:", segmented_files)
base_name = selected_file.replace('_segmented.tif', '')

csv_path = os.path.join(METRICS_DIR, f"{base_name}_nuclei_metrics.csv")
seg_path = os.path.join(SEGM_DIR, selected_file)

if not os.path.exists(csv_path):
    st.error(f"No se encontró el archivo de métricas: {csv_path}")
    st.stop()

df_metrics = pd.read_csv(csv_path)

@st.cache_data
def load_seg_image(path):
    return tiff.imread(path)

img_stack = load_seg_image(seg_path)
# 0: AGR, 1: DAPI, 2: WFA, 3: PV, 4: DAPI_Mask, 5: PV_Mask, 6: PNN_Mask
if img_stack.shape[0] < 7:
    st.error("La imagen segmentada no tiene los 7 canales esperados.")
    st.stop()
    
wfa_full = img_stack[2]
dapi_mask = img_stack[4]
pv_mask = img_stack[5]
pnn_mask_full = img_stack[6]

# --- Filter Cells ---
st.sidebar.header("🎯 Filtros de Células")
show_only_pnn = st.sidebar.checkbox("Solo PNN+", value=True)

filtered_df = df_metrics.copy()
if show_only_pnn and 'is_pnn_plus' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['is_pnn_plus'] == True]

if filtered_df.empty:
    st.warning("No hay células que cumplan los filtros seleccionados.")
    st.stop()

st.sidebar.info(f"Células seleccionadas: {len(filtered_df)}")

# --- Cell Selector ---
selected_label = st.sidebar.selectbox("ID de Célula (Label):", filtered_df['label'].values)
cell_data = df_metrics[df_metrics['label'] == selected_label].iloc[0]

# --- PNN Skeletonization Options ---
st.sidebar.divider()
st.sidebar.header("🛠️ Ajustes de Esqueleto")
crop_size = st.sidebar.slider("Tamaño de Recorte (px)", 50, 300, 150)
pnn_threshold_method = st.sidebar.selectbox("Método de Umbralado PNN", ["Automático (Otsu)", "Manual"])
manual_thresh = 0
if pnn_threshold_method == "Manual":
    manual_thresh = st.sidebar.slider("Umbral WFA Manual", 0, 65535, 10000)

# --- Processing Current Cell ---
cx, cy = int(cell_data['centroid_x']), int(cell_data['centroid_y'])
half = crop_size // 2

# Safeguard crop
y1, y2 = max(0, cy-half), min(wfa_full.shape[0], cy+half)
x1, x2 = max(0, cx-half), min(wfa_full.shape[1], cx+half)

wfa_crop = wfa_full[y1:y2, x1:x2]
mask_crop = pv_mask[y1:y2, x1:x2] == selected_label

# Thresholding WFA for PNN structure
if pnn_threshold_method == "Automático (Otsu)":
    try:
        thresh = threshold_otsu(wfa_crop[wfa_crop > 0])
    except:
        thresh = 1000
else:
    thresh = manual_thresh

pnn_binary = wfa_crop > thresh

# Optional: clean up binary (remove internal nucleus area)
pnn_structure = pnn_binary.copy()
# Remove the internal PV soma area to skeletonize the outer ring
pnn_structure[mask_crop] = 0

# Skeletonize
skel = skeletonize(pnn_structure)

# Analysis with skan
skel_metrics = None
total_length = 0
n_branches = 0
avg_branch_len = 0

if np.any(skel):
    try:
        # Create Skeleton object (spacing in micrometers)
        sk_obj = Skeleton(skel, spacing=px_size)
        summary = summarize(sk_obj)
        if not summary.empty:
            total_length = summary['branch-distance'].sum()
            n_branches = len(summary)
            avg_branch_len = summary['branch-distance'].mean()
            skel_metrics = summary
    except Exception as e:
        st.error(f"Error en skan: {e}")

# --- Display ---
st.subheader(f"Análisis de la Célula #{selected_label}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Original WFA (Zoom)**")
    # Normalize for display
    disp_wfa = cv2.normalize(wfa_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    st.image(disp_wfa, width="stretch")

with col2:
    st.markdown("**Morfología (Binarizada)**")
    st.image(pnn_structure.astype(np.uint8)*255, width="stretch")

with col3:
    st.markdown("**Esqueleto (Skan)**")
    # Plotting skeleton overlay
    fig, ax = plt.subplots()
    ax.imshow(cv2.normalize(wfa_crop, None, 0, 255, cv2.NORM_MINMAX), cmap='gray')
    ys, xs = np.where(skel)
    ax.scatter(xs, ys, s=1, c='red')
    ax.axis('off')
    st.pyplot(fig)

st.divider()

# --- Metrics Display ---
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.metric("Longitud Total del Esqueleto", f"{total_length:.2f} µm")
with m_col2:
    st.metric("Número de Ramas", n_branches)
with m_col3:
    st.metric("Longitud Media de Rama", f"{avg_branch_len:.2f} µm")

if skel_metrics is not None:
    with st.expander("📊 Ver detalle de ramas (skan summary)"):
        st.dataframe(skel_metrics)

# --- Analysis Conclusion ---
st.info(f"""
💡 **Interpretación:** 
- Una longitud total mayor indica una red más densa y madura.
- El número de ramas refleja la complejidad estructural de la PNN.
- Este análisis individual permite validar la integridad de la red alrededor de la neurona PV+.
""")
