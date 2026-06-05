import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
import pandas as pd
import subprocess
import sys
from pipeline import load_channels_tif

# Page configuration
st.set_page_config(page_title="Resumen y Análisis de PNN", layout="wide")

# Custom CSS for premium styling (dark mode, glassmorphism, glowing metrics)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMarkdown h1 {
        color: #00f2fe;
        text-align: center;
        background: linear-gradient(120deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        margin-bottom: 25px;
    }
    .metric-card {
        background: rgba(30, 33, 48, 0.5);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.6);
    }
    .metric-title {
        color: #e0e0e0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #00f2fe;
        font-size: 1.8rem;
        font-weight: bold;
        font-family: 'Outfit', sans-serif;
    }
    .metric-unit {
        font-size: 0.9rem;
        color: #888888;
        margin-left: 2px;
    }
    .pnn-status-plus {
        color: #00ff88;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    .pnn-status-minus {
        color: #ff4a4a;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 74, 74, 0.3);
    }
    .img-caption {
        font-weight: bold;
        color: #00f2fe;
        margin-bottom: 8px;
        text-align: center;
    }
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #00f2fe, transparent);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Paso 5: Resumen y Análisis de PNN por Célula")
st.write("Inspecciona detalladamente los parámetros morfológicos, de intensidad y el esqueleto de cada célula/red perineuronal individualmente.")

RAW_BASE_DIR = "data/raw"
if not os.path.exists(RAW_BASE_DIR) or not any(os.path.isdir(os.path.join(RAW_BASE_DIR, d)) for d in os.listdir(RAW_BASE_DIR) if not d.startswith('.')):
    RAW_BASE_DIR = "data/processed/mips"

SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

# Load global configuration
calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)
    px_size = calib_data.get('pixel_size_um', 1.0)
else:
    px_size = 1.0

# Sidebar selections
if not os.path.exists(RAW_BASE_DIR):
    st.error(f"No se encontró el directorio `{RAW_BASE_DIR}` ni `data/processed/mips`.")
    st.stop()

groups = sorted([d for d in os.listdir(RAW_BASE_DIR) if os.path.isdir(os.path.join(RAW_BASE_DIR, d))])
if not groups:
    st.warning(f"No hay grupos experimentales en `{RAW_BASE_DIR}`.")
    st.stop()

st.sidebar.header("📁 Selección de Muestra")
selected_group = st.sidebar.selectbox("Grupo Experimental:", groups, key="p5_group_select")
group_dir = os.path.join(RAW_BASE_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones en `{selected_group}`.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección (IPSI/CONTRA):", sections, key="p5_section_select")
raw_section_dir = os.path.join(RAW_BASE_DIR, selected_group, selected_section)

raw_files = sorted([f for f in os.listdir(raw_section_dir) if f.lower().endswith('.tif')])
if not raw_files:
    st.error("No se detectaron archivos TIF en la sección seleccionada.")
    st.stop()

selected_raw_file = st.sidebar.selectbox("Muestra TIF:", raw_files, key="p5_file_select")
base_name, _ = os.path.splitext(selected_raw_file)

seg_path = os.path.join(SEGM_BASE_DIR, selected_group, selected_section, f"{base_name}_masks.tif")
csv_path = os.path.join(METRICS_BASE_DIR, selected_group, selected_section, f"{base_name}_nuclei_metrics.csv")
raw_path = os.path.join(raw_section_dir, selected_raw_file)

if not os.path.exists(seg_path) or not os.path.exists(csv_path):
    st.warning("⚠️ Muestra no segmentada todavía. Por favor corre la segmentación y esqueletización en los pasos anteriores.")
    st.stop()

# Load metrics and stack
df_metrics = pd.read_csv(csv_path)
if df_metrics.empty:
    st.error("No se encontraron células segmentadas en los archivos de métricas.")
    st.stop()

def load_segmented_stack(path):
    return tiff.imread(path)

try:
    img_stack = load_segmented_stack(seg_path)
    # Load raw PV and WFA channels from the original raw file
    pv_raw, wfa_raw, _, _ = load_channels_tif(raw_path)
except Exception as e:
    st.error(f"Error cargando imágenes: {e}")
    st.stop()

# Filter options for the cell dropdown
# --- Target Entity Selection ---
st.sidebar.header("🎯 Objetivo de Estudio")
analysis_target = st.sidebar.radio(
    "Selecciona la entidad a analizar:",
    ["Redes Perineuronales (PNN+)", "Células Parvalbúmina (PV+)"],
    key="p5_analysis_target"
)

st.sidebar.subheader("🎯 Filtros de Clasificación")
if analysis_target == "Redes Perineuronales (PNN+)":
    filter_type = st.sidebar.selectbox(
        "Filtrar redes:",
        ["Todas las redes PNN+", "Sólo Ocupadas (PNN+/PV+)", "Sólo Huecas (PNN+/PV-)"],
        key="p5_filter_pnn"
    )
    df_target = df_metrics[df_metrics['is_pnn_plus'] == True]
    if filter_type == "Sólo Ocupadas (PNN+/PV+)":
        df_filtered = df_target[df_target['cell_type'] == "PV+/PNN+"]
    elif filter_type == "Sólo Huecas (PNN+/PV-)":
        df_filtered = df_target[df_target['cell_type'] == "PV-/PNN+"]
    else:
        df_filtered = df_target

    if df_filtered.empty:
        st.info("No hay redes PNN+ que coincidan con el filtro.")
        st.stop()

    cell_options = []
    for idx, row in df_filtered.iterrows():
        lbl = int(row['label'])
        c_type = "PNN+/PV+" if row['cell_type'] == "PV+/PNN+" else "PNN+/PV-"
        cell_options.append((lbl, f"Red #{lbl} ({c_type})"))
else:
    filter_type = st.sidebar.selectbox(
        "Filtrar células PV+:",
        ["Todas las células PV+", "Sólo Contenidas (PV+/PNN+)", "Sólo Sin Red (PV+/PNN-)"],
        key="p5_filter_pv"
    )
    df_target = df_metrics[df_metrics['is_pv_plus'] == True]
    if filter_type == "Sólo Contenidas (PV+/PNN+)":
        df_filtered = df_target[df_target['cell_type'] == "PV+/PNN+"]
    elif filter_type == "Sólo Sin Red (PV+/PNN-)":
        df_filtered = df_target[df_target['cell_type'] == "PV+/PNN-"]
    else:
        df_filtered = df_target

    if df_filtered.empty:
        st.info("No hay células PV+ que coincidan con el filtro.")
        st.stop()

    cell_options = []
    for idx, row in df_filtered.iterrows():
        lbl = int(row['label'])
        pv_lbl = int(row['pv_label']) if row['pv_label'] > 0 else lbl
        c_type = "PV+/PNN+" if row['cell_type'] == "PV+/PNN+" else "PV+/PNN-"
        cell_options.append((lbl, f"Célula PV+ #{pv_lbl} ({c_type})"))

selected_cell_tuple = st.sidebar.selectbox(
    "Selecciona una Célula/Red (ID Label):",
    cell_options,
    format_func=lambda x: x[1],
    key="p5_cell_select"
)
selected_label = selected_cell_tuple[0]

# Retrieve specific cell data
cell_data = df_metrics[df_metrics['label'] == selected_label].iloc[0]
c_type = cell_data['cell_type']

# Extract PV+ and PNN+ metrics with backward compatibility
has_split_metrics = 'pv_area_um2' in cell_data

if has_split_metrics:
    pv_area_val = float(cell_data.get('pv_area_um2', 0.0))
    pv_diam_val = float(cell_data.get('pv_diameter_um', 0.0))
    pnn_area_val = float(cell_data.get('pnn_area_um2', 0.0))
    pnn_diam_val = float(cell_data.get('pnn_diameter_um', 0.0))
else:
    # Fallback to old behavior using area_um2 and diameter_um
    if cell_data['is_pv_plus']:
        pv_area_val = float(cell_data.get('area_um2', 0.0))
        pv_diam_val = float(cell_data.get('diameter_um', 0.0))
        pnn_area_val = 0.0
        pnn_diam_val = 0.0
    else:
        pv_area_val = 0.0
        pv_diam_val = 0.0
        pnn_area_val = float(cell_data.get('area_um2', 0.0))
        pnn_diam_val = float(cell_data.get('diameter_um', 0.0))

# Convert type representation to clear string
if c_type == "PV+/PNN+":
    display_type = "PNN+/PV+ (Ocupada)" if analysis_target == "Redes Perineuronales (PNN+)" else "PV+/PNN+ (Contenida en PNN)"
elif c_type == "PV-/PNN+":
    display_type = "PNN+/PV- (Hueca)"
else:
    display_type = "PV+/PNN- (Sin Red)"

# --- UI Layout: Metrics Dashboard ---
st.subheader(f"📊 Ficha de Métricas: {display_type} (Label: {selected_label})")

c1, c2, c3, c4 = st.columns(4)

if analysis_target == "Redes Perineuronales (PNN+)":
    # 1. Hueco de PNN+ (Sujeto)
    with c1:
        pnn_soma_text = f"{pnn_area_val:.1f} µm²"
        pnn_sub = f"Diámetro Hueco: {pnn_diam_val:.1f} µm | ID Red: {int(cell_data.get('label'))}"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Hueco de PNN+ (Sujeto)</div>
                <div class="metric-value">{pnn_soma_text}</div>
                <div style="font-size:0.85rem; color:#888888; margin-top:5px;">{pnn_sub}</div>
            </div>
            """, unsafe_allow_html=True)
            
    # 2. Estado de PV+ / Ocupación
    with c2:
        if cell_data['is_pv_plus']:
            status_class = "pnn-status-plus"
            status_text = "Ocupada (PV+)"
            pv_lbl_val = int(cell_data.get('pv_label', selected_label)) if cell_data.get('pv_label', -1) > 0 else int(selected_label)
            pv_sub = f"Soma PV+: {pv_area_val:.1f} µm² | ID PV: {pv_lbl_val}"
        else:
            status_class = "pnn-status-minus"
            status_text = "Hueca (PV-)"
            pv_sub = "Sin interneurona PV+ en el hueco"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estado de Ocupación PV</div>
                <div class="metric-value {status_class}">{status_text}</div>
                <div style="font-size:0.85rem; color:#888888; margin-top:5px;">{pv_sub}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Células Parvalbúmina (PV+)
    # 1. Soma PV+ (Sujeto)
    with c1:
        pv_soma_text = f"{pv_area_val:.1f} µm²"
        pv_lbl_val = int(cell_data.get('pv_label', selected_label)) if cell_data.get('pv_label', -1) > 0 else int(selected_label)
        pv_sub = f"Diámetro PV: {pv_diam_val:.1f} µm | ID PV: {pv_lbl_val}"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Soma PV+ (Sujeto)</div>
                <div class="metric-value">{pv_soma_text}</div>
                <div style="font-size:0.85rem; color:#888888; margin-top:5px;">{pv_sub}</div>
            </div>
            """, unsafe_allow_html=True)
            
    # 2. Estado / Presencia de Red PNN+
    with c2:
        if cell_data['is_pnn_plus']:
            status_class = "pnn-status-plus"
            status_text = "Contenida (PNN+)"
            pnn_sub = f"Hueco PNN: {pnn_area_val:.1f} µm² | ID Red: {int(cell_data.get('label'))}"
        else:
            status_class = "pnn-status-minus"
            status_text = "Sin Red (PNN-)"
            pnn_sub = "Sin red perineuronal alrededor"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estado de Red PNN</div>
                <div class="metric-value {status_class}">{status_text}</div>
                <div style="font-size:0.85rem; color:#888888; margin-top:5px;">{pnn_sub}</div>
            </div>
            """, unsafe_allow_html=True)

# 3. Espesor de la red
with c3:
    if cell_data['is_pnn_plus']:
        thick_val = f"{cell_data.get('skel_mean_thickness_um', 0.0):.2f}<span class=\"metric-unit\">µm</span>"
        thick_sub = f"Max Espesor: {cell_data.get('skel_max_thickness_um', 0.0):.2f} µm"
    else:
        thick_val = "N/A"
        thick_sub = "Sin red asociada"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Espesor de la Red</div>
            <div class="metric-value">{thick_val}</div>
            <div style="font-size:0.85rem; color:#888888; margin-top:5px;">{thick_sub}</div>
        </div>
        """, unsafe_allow_html=True)

# 4. Intensidad Local WFA
with c4:
    if cell_data['is_pnn_plus']:
        int_val = f"{cell_data.get('skel_mean_intensity', 0.0):,.0f}"
        int_sub = f"Suma Vecindad: {cell_data.get('skel_neighborhood_wfa_sum', 0.0):,.0f}"
    else:
        int_val = "N/A"
        int_sub = "Sin red asociada"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Intensidad Local WFA</div>
            <div class="metric-value">{int_val}</div>
            <div style="font-size:0.85rem; color:#888888; margin-top:5px;">{int_sub}</div>
        </div>
        """, unsafe_allow_html=True)

if cell_data['is_pnn_plus']:
    st.write("")
    c_t1, c_t2, c_t3, c_t4 = st.columns(4)
    with c_t1:
        st.metric("Longitud del Esqueleto", f"{cell_data.get('skel_total_length_um', 0.0):.2f} µm")
    with c_t2:
        st.metric("Número de Ramas (Branches)", f"{int(cell_data.get('skel_branches_count', 0))}")
    with c_t3:
        st.metric("Puntos de Conexión / Extremos", f"{int(cell_data.get('skel_junctions_count', 0))} / {int(cell_data.get('skel_endpoints_count', 0))}")
    with c_t4:
        st.metric("Tortuosidad Promedio", f"{cell_data.get('skel_tortuosity_mean', 1.0):.3f}")
else:
    st.write("")
    st.info("ℹ️ Esta célula PV+ no tiene una red perineuronal (PNN+) asociada, por lo que no presenta esqueleto ni métricas morfológicas de red.")

st.divider()

# --- Crop Overlays Generation ---
st.subheader("🖼️ Previsualización del Recorte (Crops 2D)")

crop_size = 200
cx, cy = int(cell_data['centroid_x']), int(cell_data['centroid_y'])
half = crop_size // 2

h, w = pv_raw.shape[0], pv_raw.shape[1]
y1, y2 = max(0, cy - half), min(h, cy + half)
x1, x2 = max(0, cx - half), min(w, cx + half)

# Extract original channels and segmentations
wfa_raw_crop = wfa_raw[y1:y2, x1:x2]
pv_raw_crop = pv_raw[y1:y2, x1:x2]

# Segment stack mask slices:
# 0: DAPI_Mask, 1: PV_Mask, 2: PNN_Skeleton_Mask, 3: WFA_Cellpose_Mask
pv_mask_crop = img_stack[1, y1:y2, x1:x2]
skel_mask_crop = img_stack[2, y1:y2, x1:x2]
wfa_mask_crop = img_stack[3, y1:y2, x1:x2] if img_stack.shape[0] == 4 else img_stack[4, y1:y2, x1:x2]

def get_pv_soma_overlay(pv, pv_mask, wfa_mask, cell_type, pv_label, wfa_label):
    norm = cv2.normalize(pv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = np.stack([norm, norm, norm], axis=-1)
    
    # 1. Overlay PV soma in Cyan if present
    if "PV+" in cell_type and pv_label > 0:
        mask = (pv_mask == pv_label)
        if np.any(mask):
            rgb[mask] = (rgb[mask] * 0.65 + np.array([0, 242, 254]) * 0.35).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (0, 242, 254), 2)
            
    # 2. Overlay WFA Cellpose soma in Magenta outline (PNN hole)
    if "PNN+" in cell_type and wfa_label > 0:
        wfa_soma = (wfa_mask == wfa_label)
        if np.any(wfa_soma):
            w_contours, _ = cv2.findContours(wfa_soma.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, w_contours, -1, (255, 0, 255), 1)
            
    return rgb

def get_wfa_skeleton_overlay(wfa, skel, label):
    norm = cv2.normalize(wfa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = np.stack([norm, norm, norm], axis=-1)
    mask = (skel == label)
    if np.any(mask):
        dilated_mask = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8))
        rgb[dilated_mask > 0] = [255, 40, 40]
    return rgb

def get_combined_overlay(pv, wfa, pv_mask, skel, wfa_mask, cell_type, pv_label, wfa_label):
    pv_norm = cv2.normalize(pv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    wfa_norm = cv2.normalize(wfa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combined RGB background: Magenta PV and Green WFA
    rgb = np.zeros((pv.shape[0], pv.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = pv_norm
    rgb[:, :, 1] = wfa_norm
    rgb[:, :, 2] = pv_norm
    
    # 1. Outline PV Soma Mask in Cyan (0, 242, 254)
    if "PV+" in cell_type and pv_label > 0:
        soma_mask = (pv_mask == pv_label)
        if np.any(soma_mask):
            contours, _ = cv2.findContours(soma_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (0, 242, 254), 1)
            
    # 2. Outline WFA Cellpose Soma (PNN hole) in Magenta (255, 0, 255)
    if "PNN+" in cell_type and wfa_label > 0:
        w_soma = (wfa_mask == wfa_label)
        if np.any(w_soma):
            w_contours, _ = cv2.findContours(w_soma.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, w_contours, -1, (255, 0, 255), 1)
        
    # 3. Superimpose Skeleton in Red (255, 50, 50)
    if "PNN+" in cell_type and wfa_label > 0:
        skel_mask = (skel == wfa_label)
        if np.any(skel_mask):
            dil = cv2.dilate(skel_mask.astype(np.uint8), np.ones((2, 2), np.uint8))
            rgb[dil > 0] = [255, 50, 50]
        
    return rgb

# Map parameters for overlays
pv_lbl = int(cell_data.get('pv_label', -1))
wfa_lbl = selected_label if "PNN+" in c_type else -1

overlay_pv = get_pv_soma_overlay(pv_raw_crop, pv_mask_crop, wfa_mask_crop, c_type, pv_lbl, wfa_lbl)
overlay_skel = get_wfa_skeleton_overlay(wfa_raw_crop, skel_mask_crop, wfa_lbl)
overlay_comb = get_combined_overlay(pv_raw_crop, wfa_raw_crop, pv_mask_crop, skel_mask_crop, wfa_mask_crop, c_type, pv_lbl, wfa_lbl)

col_img1, col_img2, col_img3 = st.columns(3)

with col_img1:
    st.markdown('<p class="img-caption">1. Somas Celulares (Soma PV Cian, Hueco PNN Magenta)</p>', unsafe_allow_html=True)
    st.image(overlay_pv, width='stretch', clamp=True)

with col_img2:
    st.markdown('<p class="img-caption">2. Canal WFA + Esqueleto de PNN (Rojo)</p>', unsafe_allow_html=True)
    st.image(overlay_skel, width='stretch', clamp=True)

with col_img3:
    st.markdown('<p class="img-caption">3. Fusión RGB Completa</p>', unsafe_allow_html=True)
    st.image(overlay_comb, width='stretch', clamp=True)
    st.caption("Leyenda: PV Soma (Cyan), PNN Soma/Hole (Magenta), Esqueleto PNN (Línea Roja).")

st.divider()

# --- Napari Interactive Viewer launching ---
st.subheader("🖥️ Exploración Completa Interactiva")
st.write("¿Deseas ver esta muestra completa en alta definición y ajustar capas en Napari?")

if st.button("🧪 Abrir en Napari", type="primary", key="p5_btn_napari"):
    cmd = [sys.executable, "napari_viewer.py", "--path", seg_path, "--pixel_size", str(px_size)]
    try:
        env = os.environ.copy()
        env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
        subprocess.Popen(cmd, env=env)
        st.success("✅ Visor Napari lanzado con éxito.")
    except Exception as e:
        st.error(f"Error al lanzar Napari: {e}")
