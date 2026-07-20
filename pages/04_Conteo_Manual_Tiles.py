import streamlit as st
import os
import json
import numpy as np
import tifffile as tiff
import cv2
import pandas as pd
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from streamlit_image_coordinates import streamlit_image_coordinates

# Import the standard channel loader from pipeline
from pipeline import load_channels_tif

# Page configuration
st.set_page_config(
    page_title="Conteo Manual por Tiles 🔢🧠",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght=300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main-header {
        background: linear-gradient(120deg, #ff4b4b 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .sub-header { color: #a0aec0; font-size: 1.05rem; margin-bottom: 1.2rem; }
    .grand-total { font-size: 2rem; font-weight: 700; color: #f6e05e; margin-top: 8px; }
    .stButton>button { font-weight: bold; border-radius: 8px; }
    .stats-card {
        background-color: #1a202c;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4facfe;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── Constants & Helper Functions ─────────────────────────────────────────────
CONFIG_PATH = "experiment_config.json"
SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
COUNTS_FILE = os.path.join(SEGM_BASE_DIR, "tile_counts.json")
os.makedirs(SEGM_BASE_DIR, exist_ok=True)

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

config = load_config()

def _jload(p):
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _jsave(p, d):
    with open(p, "w") as f:
        json.dump(d, f, indent=2)

def get_observer_slug(obs_name):
    import re
    s = obs_name.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')

# ─── Image Discovery ──────────────────────────────────────────────────────────
raw_data_path = "data/raw"
if not os.path.exists(raw_data_path) or not any(os.path.isdir(os.path.join(raw_data_path, d)) for d in os.listdir(raw_data_path) if not d.startswith('.')):
    raw_data_path = "data/processed/mips"

all_tasks = []
if os.path.exists(raw_data_path):
    groups = sorted([d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d)) and not d.startswith('.')])
    for g in groups:
        g_path = os.path.join(raw_data_path, g)
        sections = sorted([d for d in os.listdir(g_path) if os.path.isdir(os.path.join(g_path, d)) and not d.startswith('.')])
        for s in sections:
            s_path = os.path.join(g_path, s)
            files = sorted([f for f in os.listdir(s_path) if f.lower().endswith('.tif')])
            for f in files:
                all_tasks.append((g, s, f, os.path.join(s_path, f)))

if not all_tasks:
    st.warning("⚠️ No se encontraron imágenes `.TIF` en la estructura de `data/raw` o `data/processed/mips`.")
    st.stop()

# ─── Load Counts Database & Session State ─────────────────────────────────────
for _sk, _sv in [("tc_task_idx", 0), ("tc_tile_idx", 0), ("tc_click_version", 0)]:
    if _sk not in st.session_state:
        st.session_state[_sk] = _sv

raw_counts = _jload(COUNTS_FILE)
if not raw_counts:
    raw_counts = {"Observador 1": {}}
elif any("___" in k for k in raw_counts.keys()):
    raw_counts = {"Observador 1": raw_counts}
    _jsave(COUNTS_FILE, raw_counts)

if "tc_counts" not in st.session_state:
    st.session_state["tc_counts"] = raw_counts

counts_db = st.session_state["tc_counts"]
observers_list = list(counts_db.keys())

if "selected_observer" not in st.session_state:
    st.session_state["selected_observer"] = observers_list[0] if observers_list else "Observador 1"

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.title("🔢 Configuración de Tiles")
st.sidebar.markdown("---")
st.sidebar.subheader("👥 Observador / Experto")

selected_obs = st.sidebar.selectbox(
    "Observador activo:",
    observers_list,
    index=observers_list.index(st.session_state["selected_observer"]) if st.session_state["selected_observer"] in observers_list else 0,
    key="tc_active_observer_select"
)
if selected_obs != st.session_state["selected_observer"]:
    st.session_state["selected_observer"] = selected_obs
    st.rerun()

new_obs_name = st.sidebar.text_input("Agregar nuevo observador:")
if st.sidebar.button("➕ Agregar Observador", width="stretch"):
    new_obs_cleaned = new_obs_name.strip()
    if new_obs_cleaned and new_obs_cleaned not in counts_db:
        counts_db[new_obs_cleaned] = {}
        st.session_state["tc_counts"] = counts_db
        _jsave(COUNTS_FILE, counts_db)
        st.session_state["selected_observer"] = new_obs_cleaned
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🔲 Grilla y Dimensiones")
grid_n = st.sidebar.slider("Filas (N)", 2, 12, int(config.get("tile_grid_n", 5)), 1)
grid_m = st.sidebar.slider("Columnas (M)", 2, 12, int(config.get("tile_grid_m", 5)), 1)
sz_mosaic = st.sidebar.slider("Tamaño selector mosaico (px)", 60, 250, int(config.get("tile_mosaic_size", 120)), 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visualización de Canales")
st.sidebar.caption("🔴 **WFA (PNN) en Rojo** para concordancia con el resto del proyecto.")

channel_list = [
    "Solo WFA (Rojo)",
    "WFA (Rojo) + DAPI (Azul)",
    "WFA (Rojo) + PV (Verde)",
    "WFA + PNNscore (Detección IA)",
    "WFA + PNNloc (Mapa de Probabilidad)",
    "WFA + DAPI + PV (RGB)"
]
default_ch = config.get("tile_channel_mode", "Solo WFA (Rojo)")
channel_idx = channel_list.index(default_ch) if default_ch in channel_list else 0
channel_mode = st.sidebar.selectbox("Canales:", channel_list, index=channel_idx)

p_low = st.sidebar.slider("Percentil bajo (Fondo)", 0, 10, int(config.get("tile_p_low", 0)))
p_high = st.sidebar.slider("Percentil alto (Brillo)", 90, 100, int(config.get("tile_p_high", 100)))
show_borders = st.sidebar.checkbox("Bordes del tile", value=bool(config.get("tile_show_grid", True)))
gamma = st.sidebar.slider("Gamma", 0.1, 4.0, float(config.get("tile_gamma", 1.0)), 0.1,
                          help="Valores > 1 hacen la imagen más brillante.")
tile_width = st.sidebar.slider("Ancho de visualización tile (px)", 400, 2000, int(config.get("tile_width", 950)), 50)

# Persist config settings
for _k, _v in {
    "tile_grid_n": grid_n, "tile_grid_m": grid_m,
    "tile_channel_mode": channel_mode, "tile_p_low": p_low,
    "tile_p_high": p_high, "tile_show_grid": show_borders,
    "tile_gamma": gamma, "tile_width": tile_width,
    "tile_mosaic_size": sz_mosaic
}.items():
    if config.get(_k) != _v:
        config[_k] = _v
        try:
            with open(CONFIG_PATH, "w") as _f:
                json.dump(config, _f, indent=4)
        except Exception:
            pass

total_tiles = grid_n * grid_m
task_idx = max(0, min(st.session_state["tc_task_idx"], len(all_tasks) - 1))
tile_idx = max(0, min(st.session_state["tc_tile_idx"], total_tiles - 1))

active_counts = counts_db.get(selected_obs, {})
sel_group, sel_section, sel_filename, sel_file_path = all_tasks[task_idx]
base_name, _ = os.path.splitext(sel_filename)
px_size = float(config.get("pixel_size_um", 1.0))

# Paths for automatic detection outputs
auto_csv_path = os.path.join(METRICS_BASE_DIR, sel_group, sel_section, f"{base_name}_nuclei_metrics.csv")
prob_map_path = os.path.join(SEGM_BASE_DIR, sel_group, sel_section, f"{base_name}_prob_map.tif")
has_auto_metrics = os.path.exists(auto_csv_path)

# ─── Tile Count Key & Helpers ─────────────────────────────────────────────────
def ckey(g, s, f, n, m):
    clean_f = f.lower().replace(".tif", "").replace(".tiff", "").strip()
    grid_str = f"{n}x{m}"
    rel_key = f"{g}/{s}/{clean_f}___{grid_str}"
    return rel_key

def get_entry(g, s, f, n, m, t):
    k = ckey(g, s, f, n, m)
    raw = active_counts.get(k, {}).get(str(t), None)
    if isinstance(raw, dict):
        if "clicks" not in raw:
            raw["clicks"] = []
        return raw
    if isinstance(raw, (int, float)):
        return {"manual": int(raw), "fp": 0, "fn": 0, "clicks": []}
    return {"manual": 0, "fp": 0, "fn": 0, "clicks": []}

def set_entry(g, s, f, n, m, t, entry):
    k = ckey(g, s, f, n, m)
    if k not in active_counts:
        active_counts[k] = {}
    active_counts[k][str(t)] = entry
    counts_db[selected_obs] = active_counts
    st.session_state["tc_counts"] = counts_db
    _jsave(COUNTS_FILE, counts_db)

def tiles_done(g, s, f, n, m):
    k = ckey(g, s, f, n, m)
    return {int(kt): vt for kt, vt in active_counts.get(k, {}).items()}

# ─── Channel Image Loader ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando imagen de 4 canales...", max_entries=3)
def load_channels_image(path, mtime=0.0):
    # Call the exact same loading/MIP function as in pipeline.py
    pv, wfa, dapi, agr = load_channels_tif(path)
    # Stack them back in a known fixed order for coordinate index access:
    # 0 = AGR, 1 = DAPI, 2 = WFA, 3 = PV
    return np.stack([agr, dapi, wfa, pv], axis=0).astype(np.uint16)

# ─── Global Contrast Normalization (Avoids the Tile-level Mosaic Effect) ──────
@st.cache_data(show_spinner="Calculando contraste de la imagen completa...", max_entries=3)
def get_global_percentiles(path, p_lo, p_hi, mtime=0.0):
    img_4ch = load_channels_image(path, mtime)
    percentiles = []
    for c in range(4):
        ch = img_4ch[c]
        if np.max(ch) > 0:
            lo = np.percentile(ch, p_lo)
            hi = np.percentile(ch, p_hi)
        else:
            lo, hi = 0.0, 1.0
        percentiles.append((lo, hi))
    return percentiles

@st.cache_data(show_spinner="Cargando detecciones automáticas (IA)...", max_entries=3)
def load_auto_centroids(csv_path, mtime=0.0):
    try:
        df = pd.read_csv(csv_path)
        if "is_pnn_plus" in df.columns:
            df = df[df["is_pnn_plus"] == True]
        if "centroid_y" in df.columns and "centroid_x" in df.columns:
            return [(float(r["centroid_y"]), float(r["centroid_x"])) for _, r in df.iterrows()]
    except Exception:
        pass
    return []

# ─── Tile Rendering Functions ────────────────────────────────────────────────
def tile_bounds(H, W, n, m, tidx):
    row = tidx // m
    col = tidx % m
    return int(row * H / n), int((row + 1) * H / n), int(col * W / m), int((col + 1) * W / m)

def make_tile_image(img_4ch, n, m, tidx, mode, global_pcts, borders, gamma=1.0, prob_map_file=None, auto_centroids=None):
    # img_4ch: (4, H, W) -> [AGR, DAPI, WFA, PV]
    H, W = img_4ch.shape[1], img_4ch.shape[2]
    r0, r1, c0, c1 = tile_bounds(H, W, n, m, tidx)
    crop_4ch = img_4ch[:, r0:r1, c0:c1].astype(np.float32)
    
    agr_crop = crop_4ch[0]
    dapi_crop = crop_4ch[1]
    wfa_crop = crop_4ch[2]  # WFA Channel (Red Perineuronal)
    pv_crop = crop_4ch[3]   # PV Channel (Parvalbumin)
    
    # 1. Normalize each channel globally using precomputed global image percentiles
    # [0=AGR, 1=DAPI, 2=WFA, 3=PV]
    lo_w, hi_w = global_pcts[2]
    wfa_norm = np.clip((wfa_crop - lo_w) / (hi_w - lo_w) * 255.0 if hi_w > lo_w else wfa_crop, 0, 255)
    
    lo_d, hi_d = global_pcts[1]
    dapi_norm = np.clip((dapi_crop - lo_d) / (hi_d - lo_d) * 255.0 if hi_d > lo_d else dapi_crop, 0, 255)
    
    lo_p, hi_p = global_pcts[3]
    pv_norm = np.clip((pv_crop - lo_p) / (hi_p - lo_p) * 255.0 if hi_p > lo_p else pv_crop, 0, 255)
    
    rgb = np.zeros((r1 - r0, c1 - c0, 3), dtype=np.float32)
    
    # 2. Assign channels to correct colors (WFA is ALWAYS Red, DAPI is Blue, PV is Green)
    if mode == "Solo WFA (Rojo)":
        rgb[:, :, 0] = wfa_norm
    elif mode == "WFA (Rojo) + DAPI (Azul)":
        rgb[:, :, 0] = wfa_norm
        rgb[:, :, 2] = dapi_norm
    elif mode == "WFA (Rojo) + PV (Verde)":
        rgb[:, :, 0] = wfa_norm
        rgb[:, :, 1] = pv_norm
    elif mode == "WFA + DAPI + PV (RGB)":
        rgb[:, :, 0] = wfa_norm
        rgb[:, :, 1] = pv_norm
        rgb[:, :, 2] = dapi_norm
    elif mode in ("WFA + PNNloc (Mapa de Probabilidad)", "WFA + PNNscore (Detección IA)"):
        rgb[:, :, 0] = wfa_norm
    else:
        rgb[:, :, 0] = wfa_norm

    # Apply probability map overlay if in Mapa mode
    if mode == "WFA + PNNloc (Mapa de Probabilidad)" and prob_map_file and os.path.exists(prob_map_file):
        try:
            prob_full = tiff.imread(prob_map_file)
            prob_crop = prob_full[r0:r1, c0:c1].astype(np.float32)
            rgb[:, :, 1] = np.clip(prob_crop * 255.0, 0.0, 255.0)
        except Exception:
            pass

    # Gamma correction
    if gamma != 1.0:
        rgb = np.clip(((rgb / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255)

    rgb = rgb.astype(np.uint8)

    # Draw automatic IA centroids as cyan/yellow circles if in Detección IA mode
    if mode == "WFA + PNNscore (Detección IA)" and auto_centroids:
        for cy, cx in auto_centroids:
            rx = int(cx - c0)
            ry = int(cy - r0)
            cv2.circle(rgb, (rx, ry), 12, (0, 255, 255), 2)
            cv2.circle(rgb, (rx, ry), 2, (0, 255, 255), -1)

    if borders:
        h_t, w_t = rgb.shape[:2]
        cv2.rectangle(rgb, (0, 0), (w_t - 1, h_t - 1), (255, 200, 0), 3)

    return rgb

@st.cache_data(show_spinner=False)
def get_cached_tile(path, n, m, tidx, mode, global_pcts, borders, gamma, mtime=0.0, prob_map_file=None, auto_centroids=None):
    img_4ch = load_channels_image(path, mtime)
    return make_tile_image(img_4ch, n, m, tidx, mode, global_pcts, borders, gamma, prob_map_file, auto_centroids)

@st.cache_data(show_spinner=False)
def get_tile_thumbnail(path, n, m, tidx, sz=120, mtime=0.0):
    img_4ch = load_channels_image(path, mtime)
    H, W = img_4ch.shape[1], img_4ch.shape[2]
    r0, r1, c0, c1 = tile_bounds(H, W, n, m, tidx)
    
    # Obtain global percentiles (using default 1-99 for preview)
    global_pcts = get_global_percentiles(path, 1, 99, mtime)
    
    wfa_crop = img_4ch[2, r0:r1, c0:c1].astype(np.float32)
    dapi_crop = img_4ch[1, r0:r1, c0:c1].astype(np.float32)
    
    lo_w, hi_w = global_pcts[2]
    wfa_norm = np.clip((wfa_crop - lo_w) / (hi_w - lo_w) * 255.0 if hi_w > lo_w else wfa_crop, 0, 255)
    
    lo_d, hi_d = global_pcts[1]
    dapi_norm = np.clip((dapi_crop - lo_d) / (hi_d - lo_d) * 255.0 if hi_d > lo_d else dapi_crop, 0, 255)
    
    rgb = np.zeros((r1 - r0, c1 - c0, 3), dtype=np.float32)
    rgb[:, :, 0] = wfa_norm
    rgb[:, :, 2] = dapi_norm
    
    # Upscale/Resize using high-quality cubic interpolation for crisp thumbnails
    return cv2.resize(rgb.astype(np.uint8), (sz, sz), interpolation=cv2.INTER_CUBIC)

def make_mosaic(path, n, m, current_tidx, counts_db, g, s, f, sz=120):
    mosaic = np.zeros((n * sz, m * sz, 3), dtype=np.uint8)
    done_f = tiles_done(g, s, f, n, m)
    path_mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    for tidx in range(n * m):
        row = tidx // m
        col = tidx % m
        thumb = get_tile_thumbnail(path, n, m, tidx, sz, path_mtime).copy()
        y0 = row * sz
        x0 = col * sz
        mosaic[y0:y0 + sz, x0:x0 + sz] = thumb
        if tidx == current_tidx:
            cv2.rectangle(mosaic, (x0, y0), (x0 + sz - 1, y0 + sz - 1), (255, 165, 0), 4)
        elif tidx in done_f:
            entry_m = done_f[tidx]
            manual = entry_m["manual"] if isinstance(entry_m, dict) else int(entry_m)
            clr = (80, 200, 120) if manual > 0 else (180, 90, 20)
            cv2.rectangle(mosaic, (x0, y0), (x0 + sz - 1, y0 + sz - 1), clr, 3)
            font_scale = max(0.5, sz / 150.0)
            cv2.putText(mosaic, str(manual), (x0 + 8, y0 + int(sz * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, clr, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(mosaic, (x0, y0), (x0 + sz - 1, y0 + sz - 1), (80, 80, 80), 1)
    return mosaic

def generate_stitched_evidence_image(path, g, s, f, n, m, counts_db):
    img_4ch = load_channels_image(path)
    H, W = img_4ch.shape[1], img_4ch.shape[2]
    
    global_pcts = get_global_percentiles(path, 1, 99)
    
    wfa_ch = img_4ch[2].astype(np.float32)
    dapi_ch = img_4ch[1].astype(np.float32)
    
    lo_w, hi_w = global_pcts[2]
    norm_wfa = np.clip((wfa_ch - lo_w) / (hi_w - lo_w) * 255.0 if hi_w > lo_w else wfa_ch, 0, 255).astype(np.uint8)
    
    lo_d, hi_d = global_pcts[1]
    norm_dapi = np.clip((dapi_ch - lo_d) / (hi_d - lo_d) * 255.0 if hi_d > lo_d else dapi_ch, 0, 255).astype(np.uint8)
    
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 0] = norm_wfa
    rgb[:, :, 2] = norm_dapi
    
    # Draw white grid lines
    for r in range(1, n):
        y = int(r * H / n)
        cv2.line(rgb, (0, y), (W, y), (255, 255, 255), 3)
    for c in range(1, m):
        x = int(c * W / m)
        cv2.line(rgb, (x, 0), (x, H), (255, 255, 255), 3)
        
    done_f = tiles_done(g, s, f, n, m)
    for t_str, entry_e in done_f.items():
        tidx = int(t_str)
        r0, r1, c0, c1 = tile_bounds(H, W, n, m, tidx)
        clicks_e = entry_e.get("clicks", []) if isinstance(entry_e, dict) else []
        for idx_c, pt in enumerate(clicks_e):
            cx, cy = pt
            gx = c0 + cx
            gy = r0 + cy
            cv2.circle(rgb, (gx, gy), 15, (255, 200, 0), 3)
            cv2.circle(rgb, (gx, gy), 3, (255, 200, 0), -1)
            label = f"{tidx+1}-{idx_c+1}"
            cv2.putText(rgb, label, (gx + 20, gy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb, label, (gx + 20, gy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return rgb

# ─── Load Active Image Data ───────────────────────────────────────────────────
file_mtime = os.path.getmtime(sel_file_path) if os.path.exists(sel_file_path) else 0.0
csv_mtime = os.path.getmtime(auto_csv_path) if os.path.exists(auto_csv_path) else 0.0

try:
    img_4ch = load_channels_image(sel_file_path, file_mtime)
    global_pcts = get_global_percentiles(sel_file_path, p_low, p_high, file_mtime)
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")
    st.stop()

H, W = img_4ch.shape[1], img_4ch.shape[2]
auto_centroids = load_auto_centroids(auto_csv_path, csv_mtime) if has_auto_metrics else []

done_tiles = tiles_done(sel_group, sel_section, sel_filename, grid_n, grid_m)
entry = get_entry(sel_group, sel_section, sel_filename, grid_n, grid_m, tile_idx)
clicks = entry.get("clicks", [])
manual_count = len(clicks) if len(clicks) > 0 else int(entry.get("manual", 0))

r0, r1, c0, c1 = tile_bounds(H, W, grid_n, grid_m, tile_idx)
tile_auto_centroids = [(cy, cx) for cy, cx in auto_centroids if r0 <= cy < r1 and c0 <= cx < c1]

# Make crop tile image (at original crop resolution)
tile_img = get_cached_tile(
    sel_file_path, grid_n, grid_m, tile_idx, channel_mode,
    global_pcts, show_borders, gamma, file_mtime,
    prob_map_path, tile_auto_centroids
)

orig_h, orig_w = tile_img.shape[:2]
target_w = tile_width
target_h = int(orig_h * target_w / orig_w)

# High-quality interpolation upscaling (Lanczos4) to display sharp image details!
tile_img_upscaled = cv2.resize(tile_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

n_auto = len(tile_auto_centroids)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Paso 4: Conteo Manual por Tiles y Validación de IA 🔢🧠</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Valida la detección de PNNs en el canal WFA (rojo) con contraste global uniforme y resolución interpolada de alta calidad.</div>', unsafe_allow_html=True)

# ─── Top Selection & Information ──────────────────────────────────────────────
top_l, top_r = st.columns([2, 3])

with top_l:
    with st.container(border=True):
        st.subheader("📁 Muestra Seleccionada")
        new_task_idx = st.selectbox(
            "Seleccionar imagen del experimento:",
            range(len(all_tasks)),
            format_func=lambda i: f"{all_tasks[i][0]} / {all_tasks[i][1]} / {all_tasks[i][2]}",
            index=task_idx,
            key="tc_task_sel"
        )
        if new_task_idx != task_idx:
            st.session_state["tc_task_idx"] = new_task_idx
            st.session_state["tc_tile_idx"] = 0
            st.rerun()

        area_mm2 = (H * W * (px_size ** 2)) / 1_000_000
        tile_w_um = (W / grid_m) * px_size
        tile_h_um = (H / grid_n) * px_size
        
        st.caption(f"📏 Resolución original: **{W}×{H} px** ({area_mm2:.2f} mm²) · Tamaño Tile: **{tile_w_um:.0f}×{tile_h_um:.0f} µm**")
        
        pct = len(done_tiles) / total_tiles if total_tiles else 0
        st.progress(pct, text=f"Progreso de revisión de tiles: {len(done_tiles)} / {total_tiles}")
        
        img_manual_total = sum(v.get("manual", 0) if isinstance(v, dict) else int(v) for v in done_tiles.values()) if done_tiles else 0
        
        c1m, c2m = st.columns(2)
        c1m.metric("PNN Manual (esta imagen)", img_manual_total)
        c2m.metric("PNN IA (esta imagen)", len(auto_centroids) if has_auto_metrics else "—")
        if not has_auto_metrics:
            st.info("ℹ️ Ejecuta el Paso 3 (Detección de PNNs) para ver la superposición automática de la IA.")

with top_r:
    with st.container(border=True):
        st.subheader("🗺️ Mosaico Selector de Tiles")
        st.caption(f"Haz clic sobre cualquier cuadrícula para activarla. Tamaño visual: `{grid_m * sz_mosaic}px`")
        
        mosaic = make_mosaic(sel_file_path, grid_n, grid_m, tile_idx, active_counts, sel_group, sel_section, sel_filename, sz=sz_mosaic)
        mosaic_pil = Image.fromarray(mosaic)
        
        mosaic_coords = streamlit_image_coordinates(
            mosaic_pil,
            width=grid_m * sz_mosaic,
            key=f"mosaic_{sel_group}_{sel_section}_{sel_filename}_{grid_n}_{grid_m}_{sz_mosaic}"
        )
        
        if mosaic_coords:
            mx, my = mosaic_coords["x"], mosaic_coords["y"]
            col_clicked = int(mx // sz_mosaic)
            row_clicked = int(my // sz_mosaic)
            clicked_tidx = row_clicked * grid_m + col_clicked
            if 0 <= clicked_tidx < total_tiles and clicked_tidx != tile_idx:
                set_entry(sel_group, sel_section, sel_filename, grid_n, grid_m, tile_idx,
                          {"manual": manual_count, "fp": 0, "fn": 0, "clicks": clicks})
                st.session_state["tc_tile_idx"] = clicked_tidx
                st.session_state["tc_click_version"] += 1
                st.session_state["last_processed_click"] = None
                st.rerun()

        st.caption("🟠 Actual · 🟢 Contado (>0) · 🟤 Contado (0) · ⬜ Pendiente")

# ─── Tracker Session State ────────────────────────────────────────────────────
tile_state_key = (sel_group, sel_section, sel_filename, tile_idx)
if st.session_state.get("last_tile_key") != tile_state_key:
    st.session_state["last_tile_key"] = tile_state_key
    st.session_state["last_processed_click"] = None

# Draw existing manual clicks onto the UPSCALED image (scaled coordinates)
drawn_img = tile_img_upscaled.copy()
clicks = entry.get("clicks", [])
for idx_c, pt in enumerate(clicks):
    cx_orig, cy_orig = pt
    # Map original coordinates to upscaled display coordinates
    cx = int(cx_orig * target_w / orig_w)
    cy = int(cy_orig * target_h / orig_h)
    
    cv2.circle(drawn_img, (cx, cy), 15, (255, 200, 0), 3)
    cv2.circle(drawn_img, (cx, cy), 3, (255, 200, 0), -1)
    cv2.putText(drawn_img, str(idx_c + 1), (cx + 18, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(drawn_img, str(idx_c + 1), (cx + 18, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

pil_img = Image.fromarray(drawn_img)

# Navigation & click action callbacks
def nav_prev():
    set_entry(sel_group, sel_section, sel_filename, grid_n, grid_m, tile_idx,
              {"manual": len(clicks), "fp": 0, "fn": 0, "clicks": clicks})
    st.session_state["tc_tile_idx"] = max(0, tile_idx - 1)
    st.session_state["tc_click_version"] += 1
    st.session_state["last_processed_click"] = None

def nav_next():
    set_entry(sel_group, sel_section, sel_filename, grid_n, grid_m, tile_idx,
              {"manual": len(clicks), "fp": 0, "fn": 0, "clicks": clicks})
    st.session_state["tc_tile_idx"] = min(total_tiles - 1, tile_idx + 1)
    st.session_state["tc_click_version"] += 1
    st.session_state["last_processed_click"] = None

def click_undo():
    if clicks:
        clicks.pop()
        entry["clicks"] = clicks
        entry["manual"] = len(clicks)
        set_entry(sel_group, sel_section, sel_filename, grid_n, grid_m, tile_idx, entry)
        st.session_state["tc_click_version"] += 1
        st.session_state["last_processed_click"] = None

def click_clear():
    entry["clicks"] = []
    entry["manual"] = 0
    set_entry(sel_group, sel_section, sel_filename, grid_n, grid_m, tile_idx, entry)
    st.session_state["tc_click_version"] += 1
    st.session_state["last_processed_click"] = None

def render_coords_component():
    coords = streamlit_image_coordinates(
        pil_img,
        width=target_w,
        key=f"coords_{sel_group}_{sel_section}_{sel_filename}_{tile_idx}_v{st.session_state['tc_click_version']}"
    )
    if coords:
        click_id = (coords["x"], coords["y"])
        if st.session_state.get("last_processed_click") != click_id:
            st.session_state["last_processed_click"] = click_id
            disp_w = coords.get("width")
            if disp_w and disp_w > 0:
                # Map back the click from upscaled coordinates to original crop resolution coordinate space
                cx = int(coords["x"] * orig_w / disp_w)
                cy = int(coords["y"] * orig_h / (disp_w * target_h / target_w))
                
                is_dup = any(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < 12 for px, py in clicks)
                if not is_dup:
                    clicks.append([cx, cy])
                    entry["clicks"] = clicks
                    entry["manual"] = len(clicks)
                    set_entry(sel_group, sel_section, sel_filename, grid_n, grid_m, tile_idx, entry)
                    st.rerun()

st.markdown("---")

# ─── Tile Interactive Canvas Section ──────────────────────────────────────────
tile_row = tile_idx // grid_m
tile_col = tile_idx % grid_m
st.markdown(f"### 🔍 **Tile {tile_idx + 1} / {total_tiles}** &nbsp;—&nbsp; Fila {tile_row + 1}, Columna {tile_col + 1}")

with st.container(border=True):
    render_coords_component()
    
    st.markdown("---")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1.5, 1.2, 1.2, 1.2, 1.2])
    with ctrl_col1:
        st.markdown(f"### 🧠 Conteo Tile: **{manual_count}**")
    with ctrl_col2:
        st.button("↩️ Deshacer", width="stretch", on_click=click_undo, disabled=(len(clicks) == 0), key="btn_undo")
    with ctrl_col3:
        st.button("🗑️ Limpiar", width="stretch", on_click=click_clear, disabled=(len(clicks) == 0), key="btn_clear")
    with ctrl_col4:
        st.button("⬅️ Anterior", width="stretch", on_click=nav_prev, disabled=(tile_idx == 0), key="btn_prev")
    with ctrl_col5:
        st.button("Siguiente ➡️", width="stretch", on_click=nav_next, disabled=(tile_idx == total_tiles - 1), key="btn_next")

    st.markdown("---")
    
    with st.expander("📸 Copiar / Descargar Evidencia de este Tile (Resolución Original)"):
        # Draw on original resolution for evidence download
        orig_drawn = tile_img.copy()
        for idx_c, pt in enumerate(clicks):
            cx, cy = pt
            cv2.circle(orig_drawn, (cx, cy), 10, (255, 200, 0), 2)
            cv2.circle(orig_drawn, (cx, cy), 2, (255, 200, 0), -1)
            cv2.putText(orig_drawn, str(idx_c + 1), (cx + 12, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        st.image(orig_drawn, caption=f"Evidencia Tile {tile_idx+1} (Original {orig_w}x{orig_h} px)")
        buf = io.BytesIO()
        Image.fromarray(orig_drawn).save(buf, format="TIFF")
        st.download_button(
            label="📥 Descargar TIF de este Tile",
            data=buf.getvalue(),
            file_name=f"{base_name}_tile_{tile_idx+1}.tif",
            mime="image/tiff",
            width="stretch"
        )
        
    r0t, r1t, c0t, c1t = tile_bounds(H, W, grid_n, grid_m, tile_idx)
    col_info1, col_info2 = st.columns(2)
    col_info1.caption(f"Región: {(r1t-r0t)*px_size:.0f}×{(c1t-c0t)*px_size:.0f} µm")
    if has_auto_metrics:
        col_info2.caption(f"🎯 **{n_auto}** PNNs auto-detectadas por PNNscore en este tile")

# ─── Tabs: Results Summary & Statistical Validation Test ──────────────────────
st.markdown("---")
tab_summary, tab_stats = st.tabs(["📊 Resumen de Conteos del Experimento", "🧪 Test Estadístico de Validación (Manual vs IA)"])

with tab_summary:
    rows = []
    grand_manual = 0
    grand_auto = 0
    
    for g, s, f, path in all_tasks:
        done_f = tiles_done(g, s, f, grid_n, grid_m)
        img_manual = sum(v.get("manual", 0) if isinstance(v, dict) else int(v) for v in done_f.values())
        grand_manual += img_manual
        
        auto_csv = os.path.join(METRICS_BASE_DIR, g, s, f"{os.path.splitext(f)[0]}_nuclei_metrics.csv")
        img_auto = 0
        has_auto_img = False
        if os.path.exists(auto_csv):
            try:
                df_a = pd.read_csv(auto_csv)
                if "is_pnn_plus" in df_a.columns:
                    img_auto = int(df_a["is_pnn_plus"].sum())
                    has_auto_img = True
            except Exception:
                pass
        
        if has_auto_img:
            grand_auto += img_auto
                
        n_counted = len(done_f)
        rows.append({
            "Grupo": g,
            "Sección": s,
            "Imagen": f,
            "Tiles Revisados": f"{n_counted}/{total_tiles}",
            "PNN Conteo Manual": img_manual,
            "PNN Conteo IA (PNNscore)": img_auto if has_auto_img else "—",
            "Diferencia (Manual - IA)": (img_manual - img_auto) if has_auto_img else "—",
            "Estado": "✅ Completo" if n_counted == total_tiles else ("🔄 En proceso" if n_counted > 0 else "⏳ Pendiente")
        })
        
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown(f'<div class="grand-total">Total Experimento — Manual: {grand_manual} | IA: {grand_auto}</div>', unsafe_allow_html=True)

    if active_counts:
        st.markdown("---")
        st.markdown("### 📥 Exportar Resultados de Conteo Manual")
        
        obs_slug = get_observer_slug(selected_obs)
        c_exp1, c_exp2 = st.columns(2)
        
        with c_exp1:
            export_rows = []
            for g, s, f, path in all_tasks:
                done_f = tiles_done(g, s, f, grid_n, grid_m)
                for tidx_e, v_e in done_f.items():
                    entry_e = v_e if isinstance(v_e, dict) else {"manual": int(v_e), "fp": 0, "fn": 0}
                    export_rows.append({
                        "grupo": g,
                        "seccion": s,
                        "imagen": f,
                        "tile": tidx_e,
                        "manual": entry_e.get("manual", 0),
                        "grilla": f"{grid_n}x{grid_m}"
                    })
            if export_rows:
                df_exp = pd.DataFrame(export_rows)
                st.download_button(
                    "📊 Exportar CSV Completo de Conteo Manual",
                    data=df_exp.to_csv(index=False).encode("utf-8"),
                    file_name=f"tile_counts_{grid_n}x{grid_m}_{obs_slug}.csv",
                    mime="text/csv",
                    width="stretch"
                )
                
        with c_exp2:
            ev_key = f"ev_bytes_{base_name}_{obs_slug}"
            if ev_key not in st.session_state:
                if st.button("🖼️ Generar Evidencia Completa (Stitched TIF)", width="stretch"):
                    with st.spinner("Generando imagen stitched completa con grilla y conteos..."):
                        try:
                            evidence_rgb = generate_stitched_evidence_image(sel_file_path, sel_group, sel_section, sel_filename, grid_n, grid_m, active_counts)
                            buf = io.BytesIO()
                            tiff.imwrite(buf, evidence_rgb)
                            st.session_state[ev_key] = buf.getvalue()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al generar la imagen de evidencia: {e}")
            else:
                st.download_button(
                    label="📥 Descargar Imagen de Evidencia Completa (TIF)",
                    data=st.session_state[ev_key],
                    file_name=f"{base_name}_conteo_manual_completo_{obs_slug}.tif",
                    mime="image/tiff",
                    width="stretch"
                )
                if st.button("🔄 Limpiar / Regenerar Evidencia", width="stretch"):
                    del st.session_state[ev_key]
                    st.rerun()

with tab_stats:
    st.subheader("🧪 Análisis Estadístico de Concordancia (Manual vs IA)")
    st.markdown("Evalúa la precisión y sesgo entre el conteo manual realizado por el observador y las detecciones automáticas de la red neuronal.")

    val_data = []
    for g, s, f, path in all_tasks:
        done_f = tiles_done(g, s, f, grid_n, grid_m)
        if not done_f:
            continue
            
        img_manual = sum(v.get("manual", 0) if isinstance(v, dict) else int(v) for v in done_f.values())
        
        auto_csv = os.path.join(METRICS_BASE_DIR, g, s, f"{os.path.splitext(f)[0]}_nuclei_metrics.csv")
        if os.path.exists(auto_csv):
            try:
                df_a = pd.read_csv(auto_csv)
                if "is_pnn_plus" in df_a.columns:
                    img_auto = int(df_a["is_pnn_plus"].sum())
                    val_data.append({
                        "sample": f"{g}/{s}/{f}",
                        "group": g,
                        "section": s,
                        "filename": f,
                        "manual": img_manual,
                        "auto": img_auto,
                        "diff": img_manual - img_auto,
                        "abs_diff": abs(img_manual - img_auto)
                    })
            except Exception:
                pass

    if len(val_data) < 3:
        st.info("ℹ️ Se necesitan al menos 3 imágenes procesadas con detección de IA (Paso 3) y conteo manual para calcular los tests estadísticos completos.")
    else:
        df_v = pd.DataFrame(val_data)
        
        # 1. Metric Cards
        y_manual = df_v["manual"].values
        y_auto = df_v["auto"].values
        
        r_pearson, p_pearson = stats.pearsonr(y_manual, y_auto)
        r_spearman, p_spearman = stats.spearmanr(y_manual, y_auto)
        mae = np.mean(np.abs(y_manual - y_auto))
        mean_bias = np.mean(y_manual - y_auto)
        
        # Paired test
        try:
            stat_w, p_wilcoxon = stats.wilcoxon(y_manual, y_auto)
        except Exception:
            p_wilcoxon = np.nan

        c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
        c_stat1.metric("Correlación Pearson (r)", f"{r_pearson:.4f}", f"p-val = {p_pearson:.4f}")
        c_stat2.metric("Correlación Spearman (rₛ)", f"{r_spearman:.4f}", f"p-val = {p_spearman:.4f}")
        c_stat3.metric("Error Absoluto Medio (MAE)", f"{mae:.2f} PNNs")
        c_stat4.metric("Sesgo Medio (Manual - IA)", f"{mean_bias:+.2f} PNNs")

        st.markdown("---")
        
        # 2. Scatter Plot with Identity Line (y = x)
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("#### 📈 Regresión: Conteo Manual vs Conteo IA")
            
            fig_scat = px.scatter(
                df_v, x="auto", y="manual", color="group", hover_data=["filename"],
                labels={"auto": "Conteo Automático (IA)", "manual": "Conteo Manual"},
                template="plotly_dark",
                title="Concordancia Muestra por Muestra"
            )
            
            # Identity line y = x
            max_val = max(max(y_manual), max(y_auto)) * 1.1
            fig_scat.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Línea de Identidad (y = x)",
                line=dict(color="rgba(255, 255, 255, 0.5)", dash="dash")
            ))
            st.plotly_chart(fig_scat)

        with col_plot2:
            st.markdown("#### 📉 Gráfico de Bland-Altman (Sesgo vs Promedio)")
            
            means = (y_manual + y_auto) / 2.0
            diffs = y_manual - y_auto
            sd_diff = np.std(diffs)
            
            fig_ba = go.Figure()
            fig_ba.add_trace(go.Scatter(
                x=means, y=diffs, mode="markers",
                marker=dict(size=10, color="#4facfe"),
                text=df_v["filename"],
                name="Muestras"
            ))
            
            # Bias and limits of agreement
            fig_ba.add_hline(y=mean_bias, line_color="#ff4b4b", annotation_text=f"Sesgo Medio: {mean_bias:+.2f}")
            fig_ba.add_hline(y=mean_bias + 1.96 * sd_diff, line_dash="dash", line_color="orange", annotation_text=f"+1.96 SD: {mean_bias + 1.96 * sd_diff:+.2f}")
            fig_ba.add_hline(y=mean_bias - 1.96 * sd_diff, line_dash="dash", line_color="orange", annotation_text=f"-1.96 SD: {mean_bias - 1.96 * sd_diff:+.2f}")
            
            fig_ba.update_layout(
                title="Bland-Altman Agreement",
                xaxis_title="Promedio [(Manual + IA) / 2]",
                yaxis_title="Diferencia [Manual - IA]",
                template="plotly_dark"
            )
            st.plotly_chart(fig_ba)

        # Conclusion Text Box
        st.markdown("---")
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Conclusión de Validación Estadística:")
        
        if p_pearson < 0.05 and r_pearson > 0.85:
            st.success(f"✅ **Excelente Concordancia:** Existe una correlación lineal alta y significativa ($r = {r_pearson:.4f}, p = {p_pearson:.4e}$) entre el conteo manual y el modelo automático.")
        elif p_pearson < 0.05:
            st.info(f"ℹ️ **Concordancia Moderada:** Se observa correlación significativa ($r = {r_pearson:.4f}, p = {p_pearson:.4e}$).")
        else:
            st.warning(f"⚠️ **Sin Correlación Significativa:** ($r = {r_pearson:.4f}, p = {p_pearson:.4f}$). Revisa si los parámetros de la red en el Paso 3 requieren ajuste.")

        if not np.isnan(p_wilcoxon):
            if p_wilcoxon > 0.05:
                st.success(f"✅ **Sin Sesgo Sistemático (Wilcoxon $p = {p_wilcoxon:.4f}$):** No hay diferencias estadísticamente significativas entre las mediciones del observador manual y la IA.")
            else:
                st.warning(f"⚠️ **Diferencia Sistemática (Wilcoxon $p = {p_wilcoxon:.4f}$):** Existe un sesgo promedio de `{mean_bias:+.2f}` PNNs entre el observador y la IA.")
        st.markdown('</div>', unsafe_allow_html=True)
