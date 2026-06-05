import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import distance_transform_edt, gaussian_filter
from pipeline import load_channels_tif

st.set_page_config(page_title="Medición de Intensidad y Diámetro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMarkdown h1 {
        color: #00ffcc;
        background: linear-gradient(120deg, #00ffcc 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }
    .info-box {
        background: rgba(0, 255, 204, 0.07);
        border: 1px solid rgba(0, 255, 204, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 18px;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .legend-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 5px 0;
        font-size: 0.88rem;
    }
    .legend-dot {
        width: 14px; height: 14px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; color: #00ffcc; }
    div[data-testid="stMetricLabel"] { color: #aaaaaa; font-size: 0.82rem; }
    .img-caption { font-weight: bold; color: #00ffcc; margin-bottom: 6px;
                   text-align: center; font-size: 0.95rem; }
    hr { border: 0; height: 1px;
         background: linear-gradient(to right, transparent, #00ffcc, transparent);
         margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

st.title("📏 Paso 4: Morfología e Intensidad del Esqueleto de PNN")

# --- Paths ---
RAW_BASE_DIR = "data/raw"
if not os.path.exists(RAW_BASE_DIR) or not any(
        os.path.isdir(os.path.join(RAW_BASE_DIR, d))
        for d in os.listdir(RAW_BASE_DIR) if not d.startswith('.')):
    RAW_BASE_DIR = "data/processed/mips"

SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)
    px_size = calib_data.get('pixel_size_um', 1.0)
else:
    px_size = 1.0

# --- Sidebar: Sample Selection ---
groups = sorted([d for d in os.listdir(RAW_BASE_DIR)
                 if os.path.isdir(os.path.join(RAW_BASE_DIR, d))])
if not groups:
    st.warning("No hay grupos experimentales disponibles.")
    st.stop()

st.sidebar.header("📁 Selección de Muestra")
selected_group = st.sidebar.selectbox("Grupo:", groups, key="p4_group")
group_dir = os.path.join(RAW_BASE_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir)
                   if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning("No hay secciones.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections, key="p4_section")
raw_section_dir = os.path.join(RAW_BASE_DIR, selected_group, selected_section)
raw_files = sorted([f for f in os.listdir(raw_section_dir) if f.lower().endswith('.tif')])

if not raw_files:
    st.error("No hay imágenes TIF.")
    st.stop()

selected_raw_file = st.sidebar.selectbox("Archivo TIF:", raw_files, key="p4_file")
base_name, _ = os.path.splitext(selected_raw_file)

seg_path = os.path.join(SEGM_BASE_DIR, selected_group, selected_section, f"{base_name}_masks.tif")
csv_path = os.path.join(METRICS_BASE_DIR, selected_group, selected_section, f"{base_name}_nuclei_metrics.csv")
raw_path = os.path.join(raw_section_dir, selected_raw_file)

# --- Sidebar: Run Button ---
st.sidebar.divider()
st.sidebar.header("🚀 Procesamiento")
use_gpu = st.sidebar.checkbox("Usar GPU (PyTorch)", value=True, key="p4_use_gpu")
if st.sidebar.button("🔬 Ejecutar/Actualizar Análisis", type="primary", use_container_width=True):
    with st.spinner("Ejecutando pipeline…"):
        try:
            from cellpose import models
            model_dapi = models.CellposeModel(gpu=use_gpu)
            model_pv = models.CellposeModel(gpu=use_gpu)
            from pipeline import run_pipeline_on_file
            run_pipeline_on_file(
                tif_path=raw_path,
                out_segm_dir=os.path.dirname(seg_path),
                out_metrics_dir=os.path.dirname(csv_path),
                model_dapi=model_dapi, model_pv_obj=model_pv,
                filter_type=calib_data.get('cellpose_filter_type', 'Ninguno'),
                diameter=calib_data.get('cellpose_diameter', 30.0),
                flow_threshold=calib_data.get('cellpose_flow_threshold', 0.4),
                cellprob_threshold=calib_data.get('cellpose_cellprob_threshold', 0.1),
                pv_filter_type=calib_data.get('pv_cellpose_filter_type', 'Ninguno'),
                pv_diameter=calib_data.get('pv_cellpose_diameter', 15.0),
                pv_flow_threshold=calib_data.get('pv_cellpose_flow_threshold', 0.4),
                pv_cellprob_threshold=calib_data.get('pv_cellpose_cellprob_threshold', 0.0),
                pv_expansion_dist_um=0.0, pnn_threshold=0.0, pnn_exclusion_dist_um=0.0,
                px_size=px_size, do_pv_segmentation=True, calib_data=calib_data
            )
            st.sidebar.success("✅ Análisis completado.")
            st.rerun()
        except Exception as e:
            st.error(f"Error al ejecutar: {e}")

if not os.path.exists(seg_path) or not os.path.exists(csv_path):
    st.warning("⚠️ Muestra no procesada todavía. Presiona **'Ejecutar/Actualizar Análisis'** en el sidebar o completa los pasos anteriores.")
    st.stop()

# --- Load Data ---
df_metrics = pd.read_csv(csv_path)
img_stack = tiff.imread(seg_path)

try:
    pv_raw, wfa_raw, _, _ = load_channels_tif(raw_path)
except Exception as e:
    st.error(f"Error cargando imagen raw: {e}")
    st.stop()

pv_mask_full = img_stack[1]
skeleton_full = img_stack[2]
wfa_mask_full = img_stack[3] if img_stack.shape[0] == 4 else img_stack[4]

# --- Filter: Only PNN+ networks ---
df_pnn = df_metrics[df_metrics['is_pnn_plus'] == True].copy()
if df_pnn.empty:
    st.info("No hay redes PNN+ procesadas en esta muestra.")
    st.stop()

# Label options with type indicator
def label_option(row):
    if row['cell_type'] == 'PV+/PNN+':
        return f"Red #{int(row['label'])} — Ocupada (PV+/PNN+)"
    else:
        return f"Red #{int(row['label'])} — Hueca (PNN+/PV-)"

label_opts = {row['label']: label_option(row) for _, row in df_pnn.iterrows()}

st.sidebar.divider()
st.sidebar.header("🔬 Red a Inspeccionar")
selected_label = st.sidebar.selectbox(
    "Seleccionar Red PNN+:",
    list(label_opts.keys()),
    format_func=lambda x: label_opts[x],
    key="p4_label_select"
)

cell_data = df_metrics[df_metrics['label'] == selected_label].iloc[0]
c_type = cell_data['cell_type']
is_occupied = (c_type == 'PV+/PNN+')

# --- Contextual explanation ---
st.markdown("""
<div class="info-box">
<b>¿Qué se mide en este paso?</b><br>
Cada <b>red perineuronal (PNN+)</b> es un anillo de proteoglicanos que rodea a las neuronas. 
Detectamos su forma mediante Cellpose sobre el <b>canal WFA</b> (lectina que marca las redes).
Sobre cada máscara WFA, construimos un <b>esqueleto topológico</b>: una línea central que recorre
el anillo de la red. En cada punto de ese esqueleto medimos:<br>
• <b>Espesor local (µm)</b>: radio hasta el borde exterior de la señal WFA × 2. Cuanto mayor, más gruesa la red en ese punto.<br>
• <b>Intensidad WFA local</b>: brillo del canal WFA en cada punto del esqueleto. Mayor intensidad → red más densa en ese punto.<br><br>
La <b>colocalización con PV+</b> se determina por si hay un soma de interneurona PV (canal PV)
dentro del hueco WFA detectado por Cellpose.
</div>
""", unsafe_allow_html=True)

# --- Header with classification badge ---
badge_color = "#00ff88" if is_occupied else "#aaaaaa"
badge_text = "🟢 Ocupada — PV+/PNN+" if is_occupied else "⚪ Hueca — PNN+/PV-"
st.markdown(f"""
<h3 style="margin-bottom:6px;">
  Red #{int(selected_label)} &nbsp;
  <span style="font-size:0.85rem; font-weight:600; color:{badge_color};
               background:rgba(0,0,0,0.3); padding:3px 10px; border-radius:20px;
               border:1px solid {badge_color};">{badge_text}</span>
</h3>
""", unsafe_allow_html=True)

# --- KPIs ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Espesor Promedio", f"{cell_data.get('skel_mean_thickness_um', 0.0):.2f} µm",
          help="Diámetro medio de la red medido a lo largo del esqueleto (EDT × 2 × px_size)")
k2.metric("Espesor Máximo", f"{cell_data.get('skel_max_thickness_um', 0.0):.2f} µm",
          help="Punto más grueso de la red en todo el esqueleto")
k3.metric("Intensidad Media WFA", f"{cell_data.get('skel_mean_intensity', 0.0):,.0f}",
          help="Brillo promedio del canal WFA en píxeles del esqueleto")
k4.metric("Longitud del Esqueleto", f"{cell_data.get('skel_total_length_um', 0.0):.1f} µm",
          help="Longitud total acumulada de todas las ramas del esqueleto")
k5.metric("Nº de Ramas", f"{int(cell_data.get('skel_branches_count', 0))}",
          help="Cuántas ramas individuales componen el esqueleto de esta red")

st.divider()

# --- Build crop around the selected network ---
crop_size = 200
cx, cy = int(cell_data['centroid_x']), int(cell_data['centroid_y'])
half = crop_size // 2
H, W = wfa_raw.shape[0], wfa_raw.shape[1]
y1, y2 = max(0, cy - half), min(H, cy + half)
x1, x2 = max(0, cx - half), min(W, cx + half)

wfa_crop = wfa_raw[y1:y2, x1:x2].astype(np.float32)
pv_crop  = pv_raw[y1:y2, x1:x2].astype(np.float32)
skel_crop = (skeleton_full[y1:y2, x1:x2] == selected_label)
wfa_mask_crop = (wfa_mask_full[y1:y2, x1:x2] == selected_label)
pv_mask_crop  = pv_mask_full[y1:y2, x1:x2]

# Build EDT from WFA mask (not binarization) for display consistency
pnn_gaussian_sigma = float(calib_data.get('pnn_gaussian_sigma', 1.0))
wfa_proc = wfa_crop.copy()
if pnn_gaussian_sigma > 0:
    wfa_proc = gaussian_filter(wfa_proc, sigma=pnn_gaussian_sigma)

# Use WFA mask directly for EDT (more stable than rebinarizing)
edt_crop = distance_transform_edt(wfa_mask_crop)

ys, xs = np.where(skel_crop)

if len(ys) == 0:
    st.warning("⚠️ No hay esqueleto calculado para esta red en este recorte. Ejecuta el Paso 3 (Esqueletización) primero.")
    st.stop()

radii = edt_crop[ys, xs]
diameters_um = radii * 2.0 * px_size
intensities_wfa = wfa_crop[ys, xs]

# -------------------------------------------------------------------
# VISUAL OVERLAYS (3 panels side by side)
# -------------------------------------------------------------------
st.subheader("🖼️ Visualización del Esqueleto sobre la Red WFA")

# Normalize helper
def norm8(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

# Colorize skeleton by thickness (jet colormap)
def colorize_skeleton(ys, xs, values, shape, cmap_name='jet'):
    """Returns RGB image of colored skeleton dots."""
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = values.min(), values.max()
    norm_vals = (values - vmin) / (vmax - vmin + 1e-9)
    skel_rgb = np.zeros((*shape, 3), dtype=np.uint8)
    for y, x, v in zip(ys, xs, norm_vals):
        r, g, b, _ = cmap(v)
        skel_rgb[y, x] = (int(r*255), int(g*255), int(b*255))
    # Dilate for visibility
    kernel = np.ones((3, 3), np.uint8)
    skel_rgb = cv2.dilate(skel_rgb, kernel)
    return skel_rgb

import matplotlib.pyplot as plt  # ensure imported

# Panel 1: WFA background + WFA mask outline + Skeleton colored by thickness
wfa_n = norm8(wfa_crop)
panel1 = cv2.cvtColor(wfa_n, cv2.COLOR_GRAY2RGB)
# WFA mask outline in cyan
mask_border = cv2.Canny(wfa_mask_crop.astype(np.uint8) * 255, 100, 200)
panel1[mask_border > 0] = [0, 220, 220]
# PV soma outline if occupied
if is_occupied:
    pv_lbl = int(cell_data.get('pv_label', -1))
    if pv_lbl > 0:
        pv_soma_mask = (pv_mask_crop == pv_lbl).astype(np.uint8) * 255
        pv_border = cv2.Canny(pv_soma_mask, 100, 200)
        panel1[pv_border > 0] = [255, 80, 255]  # magenta
# Skeleton colored by diameter
skel_colored_thick = colorize_skeleton(ys, xs, diameters_um, wfa_crop.shape)
skel_mask_bool = np.any(skel_colored_thick > 0, axis=2)
panel1[skel_mask_bool] = skel_colored_thick[skel_mask_bool]

# Panel 2: PV background + WFA mask outline + Skeleton colored by WFA intensity
pv_n = norm8(pv_crop)
panel2 = cv2.cvtColor(pv_n, cv2.COLOR_GRAY2RGB)
panel2[mask_border > 0] = [0, 220, 220]
skel_colored_int = colorize_skeleton(ys, xs, intensities_wfa, wfa_crop.shape, 'magma')
skel_mask_bool2 = np.any(skel_colored_int > 0, axis=2)
panel2[skel_mask_bool2] = skel_colored_int[skel_mask_bool2]

# Panel 3: Fused WFA+PV channels + full overlay
fused = np.zeros((*wfa_crop.shape, 3), dtype=np.uint8)
fused[:, :, 0] = norm8(pv_crop)    # Red → PV
fused[:, :, 1] = norm8(wfa_crop)   # Green → WFA
fused[mask_border > 0] = [0, 220, 220]
if is_occupied and pv_lbl > 0:
    fused[pv_border > 0] = [255, 80, 255]
# Skeleton in bright red on fused
for y, x in zip(ys, xs):
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny2, nx2 = y + dy, x + dx
            if 0 <= ny2 < fused.shape[0] and 0 <= nx2 < fused.shape[1]:
                fused[ny2, nx2] = [255, 50, 50]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="img-caption">1. Canal WFA + Esqueleto coloreado por Espesor</p>', unsafe_allow_html=True)
    st.image(panel1, clamp=True, use_container_width=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#888; text-align:center; margin-top:-8px;">
    <span style="color:#00dcdc;">━</span> Borde hueco WFA &nbsp;|&nbsp;
    <span style="color:#ff50ff;">━</span> Borde soma PV+ &nbsp;|&nbsp;
    <b>Esqueleto</b>: azul→fino · rojo→grueso
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown('<p class="img-caption">2. Canal PV + Esqueleto coloreado por Intensidad WFA</p>', unsafe_allow_html=True)
    st.image(panel2, clamp=True, use_container_width=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#888; text-align:center; margin-top:-8px;">
    <span style="color:#00dcdc;">━</span> Borde hueco WFA &nbsp;|&nbsp;
    <b>Esqueleto</b>: oscuro→baja intensidad · amarillo→alta intensidad
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown('<p class="img-caption">3. Fusión PV (rojo) + WFA (verde) + Esqueleto</p>', unsafe_allow_html=True)
    st.image(fused, clamp=True, use_container_width=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#888; text-align:center; margin-top:-8px;">
    <span style="color:#ff3232;">━</span> Esqueleto &nbsp;|&nbsp;
    <span style="color:#00dcdc;">━</span> Hueco WFA &nbsp;|&nbsp;
    <span style="color:#ff50ff;">━</span> Soma PV+
    </div>""", unsafe_allow_html=True)

st.divider()

# -------------------------------------------------------------------
# PROFILE PLOTS: Thickness and Intensity along skeleton (arc-length)
# -------------------------------------------------------------------
st.subheader("📈 Perfil a lo largo del Esqueleto")

# Approximate arc-length order: sort by angle from centroid
skel_cx = np.mean(xs)
skel_cy = np.mean(ys)
angles = np.arctan2(ys - skel_cy, xs - skel_cx)
order = np.argsort(angles)

arc_x = np.arange(len(order)) * px_size  # approximate arc in µm

fig_profile = go.Figure()
fig_profile.add_trace(go.Scatter(
    x=arc_x, y=diameters_um[order],
    mode='lines', name='Espesor local (µm)',
    line=dict(color='#00ffcc', width=2),
    fill='tozeroy', fillcolor='rgba(0,255,204,0.08)'
))
fig_profile.update_layout(
    template='plotly_dark',
    xaxis_title='Posición en Esqueleto (µm, orden angular)',
    yaxis_title='Espesor local (µm)',
    title='Variación del Espesor a lo largo del Anillo de PNN',
    height=300,
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig_profile, use_container_width=True)

fig_int_profile = go.Figure()
fig_int_profile.add_trace(go.Scatter(
    x=arc_x, y=intensities_wfa[order],
    mode='lines', name='Intensidad WFA',
    line=dict(color='#bb86fc', width=2),
    fill='tozeroy', fillcolor='rgba(187,134,252,0.08)'
))
fig_int_profile.update_layout(
    template='plotly_dark',
    xaxis_title='Posición en Esqueleto (µm, orden angular)',
    yaxis_title='Intensidad WFA (u.a.)',
    title='Variación de Intensidad WFA a lo largo del Esqueleto',
    height=300,
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig_int_profile, use_container_width=True)

st.divider()

# -------------------------------------------------------------------
# HISTOGRAMS
# -------------------------------------------------------------------
st.subheader("📊 Distribución de Valores")
col_h1, col_h2 = st.columns(2)

with col_h1:
    fig_h1 = px.histogram(
        x=diameters_um, nbins=20,
        labels={'x': 'Espesor local (µm)', 'y': 'Frecuencia'},
        template='plotly_dark', color_discrete_sequence=['#00ffcc'],
        title='Distribución de Espesor de la Red'
    )
    fig_h1.add_vline(x=diameters_um.mean(), line_dash='dash', line_color='white',
                     annotation_text=f"Media: {diameters_um.mean():.2f} µm",
                     annotation_position="top right")
    fig_h1.update_layout(height=280, margin=dict(t=40, b=30))
    st.plotly_chart(fig_h1, use_container_width=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#aaa; text-align:center;">
    El espesor se calcula como 2 × distancia euclidiana (EDT) desde cada punto del esqueleto
    al borde de la máscara WFA Cellpose, convertido a µm.
    </div>""", unsafe_allow_html=True)

with col_h2:
    fig_h2 = px.histogram(
        x=intensities_wfa, nbins=20,
        labels={'x': 'Intensidad WFA (u.a.)', 'y': 'Frecuencia'},
        template='plotly_dark', color_discrete_sequence=['#bb86fc'],
        title='Distribución de Intensidad WFA en el Esqueleto'
    )
    fig_h2.add_vline(x=intensities_wfa.mean(), line_dash='dash', line_color='white',
                     annotation_text=f"Media: {intensities_wfa.mean():,.0f}",
                     annotation_position="top right")
    fig_h2.update_layout(height=280, margin=dict(t=40, b=30))
    st.plotly_chart(fig_h2, use_container_width=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#aaa; text-align:center;">
    Brillo del canal WFA leído directamente en cada píxel del esqueleto.
    Refleja la densidad de proteoglicanos en esa posición de la red.
    </div>""", unsafe_allow_html=True)

st.divider()

# -------------------------------------------------------------------
# NAPARI
# -------------------------------------------------------------------
st.subheader("🧪 Exploración Interactiva en Napari")
st.markdown("""
<div class="info-box">
<b>En Napari verás las siguientes capas:</b><br>
<div class="legend-row"><span class="legend-dot" style="background:#0000ff;"></span> <b>01 - DAPI</b> — Núcleos de todas las células</div>
<div class="legend-row"><span class="legend-dot" style="background:#00ff00;"></span> <b>02 - WFA</b> — Señal de la lectina WFA (marca las redes perineuronales)</div>
<div class="legend-row"><span class="legend-dot" style="background:#aaaaaa;"></span> <b>03 - PV</b> — Señal de Parvalbúmina (marca las interneuronas)</div>
<div class="legend-row"><span class="legend-dot" style="background:#ff8800; border-radius:3px;"></span> <b>04 - Máscara DAPI</b> — Etiquetas de segmentación de núcleos DAPI (oculta por defecto)</div>
<div class="legend-row"><span class="legend-dot" style="background:#ff44ff; border-radius:3px;"></span> <b>05 - Máscara PV</b> — Somas PV+ segmentados por Cellpose en el canal PV</div>
<div class="legend-row"><span class="legend-dot" style="background:#00cc88; border-radius:3px;"></span> <b>06 - PNN+ Ocupadas</b> — Huecos WFA que contienen un soma PV+ (PNN+/PV+)</div>
<div class="legend-row"><span class="legend-dot" style="background:#ffaa00; border-radius:3px;"></span> <b>07 - PNN+ Huecas</b> — Huecos WFA sin soma PV+ (PNN+/PV-)</div>
<div class="legend-row"><span class="legend-dot" style="background:#ff4444; border-radius:3px;"></span> <b>08 - Esqueleto PNN</b> — Esqueleto topológico de cada hueco WFA (un label por red)</div>
</div>
""", unsafe_allow_html=True)

if st.button("🔭 Abrir en Napari", type="primary", key="p4_btn_napari"):
    import subprocess
    import sys
    cmd = [sys.executable, "napari_viewer.py", "--path", seg_path, "--pixel_size", str(px_size)]
    try:
        env = os.environ.copy()
        env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
        subprocess.Popen(cmd, env=env)
        st.success("✅ Visor Napari lanzado. Usa las capas del panel izquierdo para activar/desactivar canales.")
    except Exception as e:
        st.error(f"Error al lanzar Napari: {e}")
