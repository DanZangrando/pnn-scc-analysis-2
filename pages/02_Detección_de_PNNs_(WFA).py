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
pnn_pkg_path = os.path.abspath("src/counting_perineuronal_nets")
if pnn_pkg_path not in sys.path:
    sys.path.insert(0, pnn_pkg_path)

src_path = os.path.abspath("src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import importlib
from image_io import load_channels_tif, get_or_create_mip
from ai_models import load_models
import pipeline_runner
try:
    importlib.reload(pipeline_runner)
except Exception:
    pass

run_pipeline_on_file = pipeline_runner.run_pipeline_on_file
normalize_wfa_for_detection = getattr(pipeline_runner, 'normalize_wfa_for_detection', None)

if normalize_wfa_for_detection is None:
    def normalize_wfa_for_detection(w_raw, method="Ninguno (Raw)", gamma=1.0):
        w_f = w_raw.astype(np.float32)
        if "Percentil Robusto" in method:
            p_low, p_high = float(np.percentile(w_f, 1.0)), float(np.percentile(w_f, 99.5))
            w_norm = np.clip((w_f - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
        elif "Percentil Agresivo" in method:
            p_low, p_high = float(np.percentile(w_f, 0.5)), float(np.percentile(w_f, 99.8))
            w_norm = np.clip((w_f - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
        elif "CLAHE" in method:
            w_minmax = (w_f - w_f.min()) / (w_f.max() - w_f.min() + 1e-8)
            w_norm = exposure.equalize_adapthist(w_minmax, clip_limit=0.02).astype(np.float32)
        else:
            w_norm = (w_f - w_f.min()) / (w_f.max() - w_f.min() + 1e-8)
        if gamma != 1.0 and gamma > 0:
            w_norm = np.power(w_norm, gamma)
        return w_norm

from omegaconf import OmegaConf

st.set_page_config(page_title="Paso 3: Detección de PNNs (WFA)", layout="wide")

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

st.markdown('<div class="main-header">🧠 Paso 2: Detección de Redes Perineuronales (PNNs)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detección de PNNs en el canal WFA mediante los modelos Faster R-CNN (PNNloc) y ConvNet (PNNscore).</div>', unsafe_allow_html=True)

RAW_DIR = "data/processed/mips"
if not os.path.exists(RAW_DIR) or not any(os.path.isdir(os.path.join(RAW_DIR, d)) for d in os.listdir(RAW_DIR) if not d.startswith('.')):
    RAW_DIR = "data/raw"

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
    st.warning("No hay grupos de imágenes disponibles.")
    st.stop()

selected_group = st.sidebar.selectbox("Grupo:", groups, key="p3_group")
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones en {selected_group}.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections, key="p3_section")
section_dir = os.path.join(group_dir, selected_section)

tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith(('.tif', '.czi'))])
if not tif_files:
    st.warning("No hay archivos de imagen (.tif o .czi) en la sección.")
    st.stop()

selected_filename = st.sidebar.selectbox("Imagen:", tif_files, key="p3_file")
selected_path = os.path.join(section_dir, selected_filename)

if selected_filename.lower().endswith('.czi') or "data/raw" in selected_path:
    from image_io import get_or_create_mip
    try:
        selected_path = get_or_create_mip(selected_path, float(calib_data.get('pixel_size_um', 0.8913)))
    except Exception as e:
        st.warning(f"Generando MIP: {e}")

if st.sidebar.button("🎯 Abrir / Dibujar ROIs en Napari", type="primary", key="p3_btn_roi"):
    cmd = [sys.executable, "src/napari_viewer.py", "--path", selected_path, "--pixel_size", str(calib_data.get('pixel_size_um', 0.8913)), "--edit-roi"]
    try:
        env = os.environ.copy()
        env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
        subprocess.Popen(cmd, env=env)
        st.sidebar.success("✅ Visor Napari lanzado.")
    except Exception as e:
        st.sidebar.error(f"Error abriendo Napari: {e}")

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
default_pnn_rad_p2 = float(calib_data.get('pnn_radius_um', 20.0))
if default_pnn_rad_p2 < 1.0:
    default_pnn_rad_p2 = 20.0

pnn_radius_um = st.sidebar.number_input(
    "Tamaño / Radio Esperado PNN (µm)",
    value=default_pnn_rad_p2,
    min_value=1.0, max_value=100.0, step=1.0,
    help="Radio físico de la red perineuronal en micras. Ajusta la escala de los parches para PNNloc (Faster R-CNN) y PNNscore (ConvNet) y el tamaño de los anillos pericelulares."
)
loc_threshold = st.sidebar.slider(
    "Umbral de Probabilidad (PNNloc)", 
    0.01, 0.90, 
    float(calib_data.get('lupori_loc_threshold', 0.05)), 
    step=0.01,
    help="Umbral de detección de Faster R-CNN. Valores más bajos detectan más candidatos de PNNs."
)

accept_all_pnnloc = st.sidebar.checkbox(
    "⚡ Aceptar todas las detecciones de PNNloc (Omitir filtro PNNscore)",
    value=bool(calib_data.get('accept_all_pnnloc', False)),
    help="Si se activa, se aceptan todas las redes encontradas por Faster R-CNN sin descartar ninguna por puntaje de ConvNet."
)

if accept_all_pnnloc:
    score_threshold = -999.0
    st.sidebar.info("💡 Modo 'Aceptar Todo' activado: no se filtrará ninguna red por puntaje.")
else:
    default_score_val = float(calib_data.get('lupori_score_threshold', -1.0))
    if default_score_val < -3.0 or default_score_val > 3.0:
        default_score_val = -1.0
    score_threshold = st.sidebar.slider(
        "Umbral de Calificación (PNNscore)", 
        -3.0, 3.0, 
        default_score_val, 
        step=0.05,
        help="El modelo ConvNet genera puntuaciones continuas (logits). Valores negativos como -0.8 o -1.0 permiten aceptar PNNs con tinción más tenue o heterogénea."
    )

min_peak_dist = st.sidebar.slider("Distancia mínima entre PNNs (px)", 10, 80, int(calib_data.get('lupori_min_peak_dist', 30)), step=5)
tile_size = st.sidebar.select_slider("Tamaño de tile (px)", options=[256, 512, 640, 1024, 2048], value=int(calib_data.get('lupori_tile_size', 640)))
tile_overlap = st.sidebar.slider("Overlap entre tiles (px)", 16, 128, int(calib_data.get('lupori_tile_overlap', 32)), step=16)
soma_erosion_um = st.sidebar.slider("Erosión de Soma (µm)", 0.0, 4.0, float(calib_data.get('soma_erosion_um', 1.5)), step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🌟 Realce / Normalización WFA (IA)")
wfa_norm_options = [
    "Ninguno (Raw)",
    "Percentil Robusto (1-99.5%)",
    "CLAHE (Adaptativo Local)",
    "Percentil Agresivo (0.5-99.8%)",
    "Min-Max Estándar (0-1)"
]
default_wfa_norm = calib_data.get('wfa_norm_method', "Ninguno (Raw)")
idx_wfa_norm = wfa_norm_options.index(default_wfa_norm) if default_wfa_norm in wfa_norm_options else 0

wfa_norm_method = st.sidebar.selectbox(
    "Preprocesamiento WFA (Inferencia IA):",
    wfa_norm_options,
    index=idx_wfa_norm,
    help="Aplica normalización o estiramiento de contraste al canal WFA exclusivamente durante la inferencia de la IA. Las intensidades biológicas de la imagen original se conservan intactas."
)

wfa_gamma = st.sidebar.slider(
    "Ajuste Gamma / Contraste (IA)",
    0.5, 2.0, float(calib_data.get('wfa_gamma', 1.0)), step=0.1,
    help="Valores < 1.0 aumentan la visibilidad de PNNs tenues; valores > 1.0 suprimen el fondo difuso."
)

px_size = float(calib_data.get('pixel_size_um', 1.0))

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🧠 Detectar PNNs (WFA)", type="primary")

base_fn, _ = os.path.splitext(selected_filename)
seg_file = os.path.join(SEGM_DIR, f"{base_fn}_masks.tif")
csv_file = os.path.join(METRICS_DIR, f"{base_fn}_nuclei_metrics.csv")
json_file = os.path.join(METRICS_DIR, f"{base_fn}_summary.json")
candidates_file = os.path.join(METRICS_DIR, f"{base_fn}_candidates.json")

if run_btn:
    calib_data.update({
        'pnn_radius_um': pnn_radius_um,
        'lupori_loc_threshold': loc_threshold,
        'lupori_score_threshold': score_threshold,
        'lupori_min_peak_dist': min_peak_dist,
        'lupori_tile_size': tile_size,
        'lupori_tile_overlap': tile_overlap,
        'soma_erosion_um': soma_erosion_um,
        'wfa_norm_method': wfa_norm_method,
        'wfa_gamma': wfa_gamma
    })
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)

    with st.spinner("Ejecutando detección unificada de PNNs (PNNloc + PNNscore)..."):
        try:
            model_pv, model_loc, model_score, device = load_models()
            summary = run_pipeline_on_file(
                tif_path=selected_path,
                out_segm_dir=SEGM_DIR,
                out_metrics_dir=METRICS_DIR,
                model_pv_obj=model_pv,
                model_loc=model_loc,
                model_score=model_score,
                device=device,
                pv_filter_type=calib_data.get("pv_cellpose_filter_type", "Ninguno"),
                pv_diameter=float(calib_data.get("pv_cellpose_diameter", 30.0)),
                pv_flow_threshold=float(calib_data.get("pv_cellpose_flow_threshold", 0.4)),
                pv_cellprob_threshold=float(calib_data.get("pv_cellpose_cellprob_threshold", 0.0)),
                loc_threshold=loc_threshold,
                score_threshold=score_threshold,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                px_size=px_size,
                do_pv_segmentation=calib_data.get("do_pv_segmentation", True),
                calib_data=calib_data
            )
            st.success("🎉 ¡Detección de PNNs completada con éxito!")
            st.rerun()
        except Exception as e:
            st.error(f"Error durante la detección de PNNs: {e}")

# Previsualización y Mapa de Potencia (Estilo Lupori et al.)
st.subheader(f"Muestra seleccionada: `{selected_filename}`")
v_tab1, v_tab2, v_tab3 = st.tabs([
    "🧠 Redes PNN Detectadas",
    "⭕ Máscaras Pericelulares (Anillos 4µm)",
    "🔥 Mapa de Calor de Potencia (Lupori Energy Map)"
])

heatmap_path = os.path.join(SEGM_DIR, f"{base_fn}_power_heatmap.png")

with v_tab1:
    show_norm_prev = st.checkbox("🔍 Previsualizar Canal WFA con Ajuste (como lo ve la IA)", value=False, key="wfa_show_norm_prev")
    col_prev1, col_prev2 = st.columns(2)
    with col_prev1:
        if show_norm_prev:
            w_norm_arr = normalize_wfa_for_detection(wfa_raw, method=wfa_norm_method, gamma=wfa_gamma)
            w_disp_shown = (w_norm_arr * 255.0).astype(np.uint8)
            st.markdown(f'<p class="img-caption">Canal WFA Preprocesado ({wfa_norm_method}, γ={wfa_gamma})</p>', unsafe_allow_html=True)
        else:
            w_disp_shown = wfa_disp
            st.markdown('<p class="img-caption">Canal WFA Original (Raw)</p>', unsafe_allow_html=True)
            
        st.image(w_disp_shown, width="stretch", clamp=True, channels="GRAY")

    with col_prev2:
        st.markdown('<p class="img-caption">Redes PNN Detectadas (IA)</p>', unsafe_allow_html=True)
        has_pnn_mask = False
        if os.path.exists(seg_file):
            try:
                loaded_masks = tiff.imread(seg_file)
                num_ch = loaded_masks.shape[0] if len(loaded_masks.shape) == 3 else 1
                m_pnn_mask = loaded_masks[2, :, :] if num_ch >= 3 else np.zeros_like(wfa_raw)
                n_detected_pnn = int(np.max(m_pnn_mask))
                if n_detected_pnn > 0:
                    overlay = label2rgb(m_pnn_mask, image=wfa_disp, bg_label=0, alpha=0.4, image_alpha=1.0)
                    st.image(overlay, width="stretch", clamp=True)
                    st.caption(f"🎯 **{n_detected_pnn}** redes PNN detectadas por la IA en esta imagen.")
                    has_pnn_mask = True
                else:
                    st.warning("⚠️ La IA procesó la imagen pero no detectó ninguna PNN con los umbrales actuales.")
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
cands_data = []
if os.path.exists(candidates_file) and os.path.getsize(candidates_file) > 2:
    try:
        with open(candidates_file, 'r') as f:
            cands_data = json.load(f)
    except Exception:
        cands_data = []

if not cands_data and os.path.exists(csv_file):
    try:
        df_c = pd.read_csv(csv_file)
        if not df_c.empty and 'centroid_y' in df_c.columns:
            for idx_r, row in df_c[df_c['is_pnn_plus'] == True].iterrows():
                cands_data.append({
                    'id': int(row.get('label', idx_r + 1)),
                    'centroid_y': float(row['centroid_y']),
                    'centroid_x': float(row['centroid_x']),
                    'score': float(row.get('score', 1.0)),
                    'is_confirmed': True
                })
    except Exception:
        pass

if len(cands_data) > 0:
    try:
        st.divider()
        st.subheader("🔍 Inspector de Candidatos (PNNscore)")
        st.write("Selecciona una PNN candidata de la lista para ver su parche evaluado:")
        
        c_sel, c_score = st.columns([2, 1])
        with c_sel:
            cand_map = {c["id"]: c for c in cands_data}
            cand_ids = list(cand_map.keys())
            selected_cand_id = st.selectbox(
                "Seleccionar Candidato por ID:",
                cand_ids,
                index=0,
                format_func=lambda cid: f"ID: {cid} (Y: {cand_map[cid]['centroid_y']:.0f}, X: {cand_map[cid]['centroid_x']:.0f}) - Score: {cand_map[cid].get('score', 0.0):.4f}"
            )
        
        selected_cand = cand_map[selected_cand_id]
        cy, cx = selected_cand["centroid_y"], selected_cand["centroid_x"]
        score = selected_cand.get("score", 0.0)
        prob_calib = 1.0 / (1.0 + np.exp(-score)) * 100.0
        is_confirmed = score >= score_threshold
        
        with c_score:
            st.metric(
                "Confianza PNNscore (IA)", 
                f"{score:.3f} ({prob_calib:.1f}%)", 
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
        cmd = [sys.executable, "src/napari_viewer.py", "--path", seg_file, "--pixel_size", str(px_size), "--step", "wfa"]
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

