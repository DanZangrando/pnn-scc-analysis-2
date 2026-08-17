import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import sys
import subprocess

sys.path.append(os.path.abspath("src"))
from image_io import get_or_create_mip, extract_czi_pixel_size, load_channels_tif
from roi import load_rois, save_rois, get_roi_json_path
from ai_models import load_models
from pipeline_runner import run_pipeline_on_file

st.set_page_config(
    page_title="PNN SSC Analysis — Inicio & ROIs Napari",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for Premium Modern Aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght=300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    
    .hero-header {
        background: linear-gradient(120deg, #8a2be2 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.8rem; font-weight: 700; margin-bottom: 0.2rem;
    }
    .hero-sub { color: #a0aec0; font-size: 1.15rem; margin-bottom: 1.5rem; }
    
    .card-box {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .mip-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 2px solid #38bdf8;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 25px;
    }

    .roi-highlight-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 2px solid #6366f1;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #00f2fe; }
    </style>
    """, unsafe_allow_html=True)

# Path Constants
CONFIG_PATH = "experiment_config.json"
RAW_DATA_PATH = "data/raw" if os.path.exists("data/raw") else "data/processed/mips"
SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=4)

calib_data = load_config()

# Main Header
st.markdown('<div class="hero-header">PNN SSC Analysis 🧠🔬</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Cuantificación automatizada de Redes Perineuronales (PNNs), Interneuronas PV+ y ROIs Multicapa (Regiones A, B, C)</div>', unsafe_allow_html=True)

# Build sample tasks index
all_tasks = []
groups = []
if os.path.exists(RAW_DATA_PATH):
    groups = sorted([d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d)) and not d.startswith('.')])
    for g in groups:
        g_path = os.path.join(RAW_DATA_PATH, g)
        sections = [d for d in os.listdir(g_path) if os.path.isdir(os.path.join(g_path, d)) and not d.startswith('.')]
        for s in sections:
            s_path = os.path.join(g_path, s)
            files = [f for f in os.listdir(s_path) if f.lower().endswith(('.tif', '.czi'))]
            for f in files:
                all_tasks.append((g, s, f, os.path.join(s_path, f)))

# Sidebar Configuration Parameters
st.sidebar.title("⚙️ Parámetros Globales")

px_size_default = float(calib_data.get('pixel_size_um', 0.8913))
px_size = st.sidebar.number_input(
    "Calibración Activa (µm/px)",
    value=px_size_default,
    format="%.4f", step=0.005,
    help="Resolución física del microscopio. Se actualiza automáticamente al detectar metadatos CZI."
)

filter_options = ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"]

with st.sidebar.expander("🧪 Parámetros Cellpose (PV)"):
    do_pv = st.checkbox("Activar segmentación PV", value=calib_data.get('do_pv_segmentation', True))
    pv_def_filter = calib_data.get('pv_cellpose_filter_type', "Ninguno")
    pv_filter = st.selectbox("Filtro PV", filter_options, index=filter_options.index(pv_def_filter) if pv_def_filter in filter_options else 0)
    pv_diam = st.number_input("Diámetro PV (px)", value=float(calib_data.get('pv_cellpose_diameter', 30.0)), step=1.0)
    pv_flow = st.slider("Flow Threshold (PV)", 0.0, 1.0, float(calib_data.get('pv_cellpose_flow_threshold', 0.4)))
    pv_prob = st.slider("Cell Prob Threshold (PV)", -6.0, 6.0, float(calib_data.get('pv_cellpose_cellprob_threshold', 0.0)))

with st.sidebar.expander("🧠 Parámetros PNNloc / PNNscore"):
    default_pnn_rad = float(calib_data.get('pnn_radius_um', 20.0))
    if default_pnn_rad < 1.0:
        default_pnn_rad = 20.0
    pnn_radius_um = st.number_input("Radio Esperado PNN (µm)", value=default_pnn_rad, min_value=1.0, max_value=100.0, step=1.0, help="Radio físico de la red perineuronal en micras. Ajusta el escalado de parches para PNNloc y PNNscore.")
    loc_threshold = st.slider("Umbral Prob (PNNloc)", 0.01, 0.90, float(calib_data.get('lupori_loc_threshold', 0.05)), step=0.01)
    default_score_val = float(calib_data.get('lupori_score_threshold', -1.0))
    if default_score_val < -3.0 or default_score_val > 3.0:
        default_score_val = -1.0
    score_threshold = st.slider("Umbral Score (PNNscore)", -3.0, 3.0, default_score_val, step=0.05, help="Puntuación continua (logit) del modelo ConvNet. Valores entre -1.5 y 0.0 rescatan PNNs tenues.")
    min_peak_dist = st.slider("Distancia mín PNNs (px)", 10, 80, int(calib_data.get('lupori_min_peak_dist', 60)), step=5)
    tile_size = st.select_slider("Tamaño tile (px)", options=[256, 512, 640, 1024, 2048], value=int(calib_data.get('lupori_tile_size', 640)))
    tile_overlap = st.slider("Overlap tiles (px)", 16, 128, int(calib_data.get('lupori_tile_overlap', 64)), step=16)

    wfa_norm_options = [
        "Ninguno (Raw)",
        "Percentil Robusto (1-99.5%)",
        "CLAHE (Adaptativo Local)",
        "Percentil Agresivo (0.5-99.8%)",
        "Min-Max Estándar (0-1)"
    ]
    default_wfa_norm = calib_data.get('wfa_norm_method', "Ninguno (Raw)")
    idx_wfa_norm = wfa_norm_options.index(default_wfa_norm) if default_wfa_norm in wfa_norm_options else 0
    wfa_norm_method = st.selectbox("Realce / Normalización WFA (IA)", wfa_norm_options, index=idx_wfa_norm, help="Normaliza o realza el canal WFA durante la inferencia de la IA sin alterar los valores de la imagen original.")
    wfa_gamma = st.slider("Ajuste Gamma (IA)", 0.5, 2.0, float(calib_data.get('wfa_gamma', 1.0)), step=0.1)

if st.sidebar.button("💾 Guardar Configuración Global"):
    calib_data.update({
        'pixel_size_um': px_size,
        'pnn_radius_um': pnn_radius_um,
        'do_pv_segmentation': do_pv,
        'pv_cellpose_filter_type': pv_filter,
        'pv_cellpose_diameter': pv_diam,
        'pv_cellpose_flow_threshold': pv_flow,
        'pv_cellpose_cellprob_threshold': pv_prob,
        'lupori_loc_threshold': loc_threshold,
        'lupori_score_threshold': score_threshold,
        'lupori_min_peak_dist': min_peak_dist,
        'lupori_tile_size': tile_size,
        'lupori_tile_overlap': tile_overlap,
        'wfa_norm_method': wfa_norm_method,
        'wfa_gamma': wfa_gamma,
        'channels': ["AGR", "DAPI", "WFA", "PV"]
    })
    save_config(calib_data)
    st.sidebar.success("✅ Configuración guardada correctamente.")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: SELECCIÓN DE MUESTRA, GESTIÓN DE MIPS Y CALIBRACIÓN AUTOMÁTICA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mip-card">
<h3 style="margin-top:0; color:#38bdf8;">🖼️ 1. Selección de Muestra, Gestión de MIPs y Calibración Física</h3>
<p style="color:#cbd5e1; margin-bottom:15px;">Selecciona una muestra del experimento. La aplicación trabajará siempre con su <b>MIP proyectado de 4 canales</b> (.tif). La escala de calibración física (µm/px) se detecta automáticamente de los metadatos de la imagen.</p>
</div>
""", unsafe_allow_html=True)

if groups:
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        selected_group = st.selectbox("Grupo:", groups, key="sel_group")
    with col_sel2:
        g_dir = os.path.join(RAW_DATA_PATH, selected_group)
        sections = sorted([d for d in os.listdir(g_dir) if os.path.isdir(os.path.join(g_dir, d))])
        selected_section = st.selectbox("Sección:", sections, key="sel_section")
    with col_sel3:
        s_dir = os.path.join(g_dir, selected_section)
        files = sorted([f for f in os.listdir(s_dir) if f.lower().endswith(('.tif', '.czi'))])
        selected_file = st.selectbox("Imagen Muestra:", files, key="sel_file")

    if selected_file:
        base_fn, _ = os.path.splitext(selected_file)
        raw_file_path = os.path.join('data/raw', selected_group, selected_section, selected_file)
        mip_file_path = os.path.join('data/processed/mips', selected_group, selected_section, f"{base_fn}.tif")

        # Detect CZI XML Calibration
        detected_px_size = 0.8913
        if os.path.exists(raw_file_path) and raw_file_path.lower().endswith('.czi'):
            czi_px = extract_czi_pixel_size(raw_file_path)
            if czi_px > 0:
                detected_px_size = czi_px
        
        if abs(detected_px_size - px_size) > 1e-4:
            calib_data['pixel_size_um'] = float(detected_px_size)
            save_config(calib_data)
            px_size = float(detected_px_size)

        mip_exists = os.path.exists(mip_file_path)
        mips_exist_count = sum(1 for g, s, f, _ in all_tasks
                               if os.path.exists(os.path.join("data/processed/mips", g, s, f"{os.path.splitext(f)[0]}.tif")))

        # Layout with clean vertical alignment
        c_status1, c_status2, c_status3 = st.columns([2.2, 1, 1], vertical_alignment="center")
        
        with c_status1:
            if mip_exists:
                st.success(f"✅ **MIP Proyectado Listo:** `{mip_file_path}`\n\n📏 **Calibración CZI Detectada:** `{px_size:.4f} µm/px`")
            else:
                st.warning(f"⚠️ **MIP Pendiente:** No se ha generado la proyección Z de 4 canales para `{selected_file}`.")
        
        with c_status2:
            if st.button("🖼️ Generar MIP Muestra", type="primary" if not mip_exists else "secondary", use_container_width=True, key="btn_gen_single_mip"):
                with st.spinner("Proyectando Z-stack de 4 canales a 16-bit..."):
                    try:
                        mip_file_path = get_or_create_mip(raw_file_path, px_size, force_recreate=True)
                        st.success(f"✅ ¡MIP generado correctamente para {base_fn}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generando MIP: {e}")

        with c_status3:
            if st.button("📦 Generar MIPs Experimento", type="secondary", use_container_width=True, key="btn_gen_all_mips"):
                if not all_tasks:
                    st.error("No se encontraron muestras en `data/raw`.")
                else:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    success_mips = 0
                    for idx, (g, s, f, raw_p) in enumerate(all_tasks):
                        status_text.text(f"Generando MIP [{idx+1}/{len(all_tasks)}]: {g}/{s}/{f}")
                        try:
                            get_or_create_mip(raw_p, px_size, force_recreate=True)
                            success_mips += 1
                        except Exception as e:
                            st.error(f"Error en {f}: {e}")
                        progress_bar.progress((idx + 1) / len(all_tasks))
                    status_text.text(f"✅ ¡Pre-procesamiento MIP completado! {success_mips}/{len(all_tasks)} imágenes proyectadas.")
                    st.success(f"✅ Se han generado {success_mips} archivos MIP (.tif) correctamente.")
                    st.rerun()

        st.info(f"ℹ️ **Calibración Global:** Resoluciones detectadas: **{px_size:.4f} µm/px** | MIPs listos en todo el experimento: **{mips_exist_count}** / **{len(all_tasks)}**")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: VISOR INTERACTIVO Y EDICIÓN DE ROIS EN NAPARI
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="roi-highlight-box">
<h3 style="margin-top:0; color:#818cf8;">🎨 2. Herramienta de ROIs y Visor Interactivo Napari</h3>
<p style="color:#cbd5e1; margin-bottom:15px;">Dibuja, edita y gestiona polígonos sobre 3 capas de regiones independientes (<b>Región A</b>, <b>Región B</b> y <b>Región C</b>) directamente en Napari sobre la imagen MIP.</p>
</div>
""", unsafe_allow_html=True)

if groups and selected_file:
    base_roi, _ = os.path.splitext(selected_file)
    target_napari_path = mip_file_path if os.path.exists(mip_file_path) else raw_file_path
    seg_roi_path = os.path.join(SEGM_BASE_DIR, selected_group, selected_section, f"{base_roi}_masks.tif")
    if os.path.exists(seg_roi_path):
        target_napari_path = seg_roi_path

    # Check ROI status
    roi_json_path = os.path.join(METRICS_BASE_DIR, selected_group, selected_section, f"{base_roi}_rois.json")
    n_rois_total = 0
    roi_area_mm2 = 0.0
    reg_info = []
    if os.path.exists(roi_json_path):
        try:
            with open(roi_json_path, 'r') as f_roi:
                roi_data = json.load(f_roi)
                n_rois_total = roi_data.get("n_rois_total", 0)
                if n_rois_total == 0:
                    n_rois_total = roi_data.get("n_rois", 0)
                roi_area_mm2 = roi_data.get("total_roi_area_mm2", 0.0)
                meta = roi_data.get("regions_metadata", {})
                for r in ["A", "B", "C"]:
                    m_r = meta.get(r, {})
                    nr = m_r.get("n_rois", 0)
                    if nr > 0:
                        reg_info.append(f"Región {r}: {nr}")
        except Exception:
            pass

    if n_rois_total > 0:
        detail = " | ".join(reg_info) if reg_info else f"{n_rois_total} polígonos"
        st.success(f"🎯 **ROIs Guardadas ({detail}):** Área total ROI: {roi_area_mm2:.4f} mm²")
    else:
        st.info("ℹ️ **Sin ROIs trazadas aún.** Presiona el botón inferior para abrir Napari y dibujar polígonos en la Región A, B o C.")

    btn_r1, btn_r2 = st.columns(2)
    with btn_r1:
        if st.button("🎯 Abrir y Dibujar ROIs en Napari (Regiones A, B, C)", type="primary", use_container_width=True, key="btn_main_edit_roi"):
            cmd = [sys.executable, "src/napari_viewer.py", "--path", target_napari_path, "--pixel_size", str(px_size), "--edit-roi"]
            try:
                env = os.environ.copy()
                env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
                subprocess.Popen(cmd, env=env)
                st.success("✅ Visor Napari abierto en modo edición de ROIs. Traza los polígonos y presiona Ctrl+S para guardar.")
            except Exception as e:
                st.error(f"Error lanzando Napari: {e}")

    with btn_r2:
        if st.button("🧪 Visualizar Canales de Imagen en Napari", type="secondary", use_container_width=True, key="btn_main_view_channels"):
            cmd = [sys.executable, "src/napari_viewer.py", "--path", target_napari_path, "--pixel_size", str(px_size)]
            try:
                env = os.environ.copy()
                env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
                subprocess.Popen(cmd, env=env)
                st.success("✅ Visor Napari de canales lanzado.")
            except Exception as e:
                st.error(f"Error lanzando Napari: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: ANÁLISIS DE LA MUESTRA SELECCIONADA
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### ⚡ 3. Procesamiento Individual de la Muestra Seleccionada")
st.write(f"Ejecuta el pipeline completo de Inteligencia Artificial (Cellpose PV + PNNloc + PNNscore + Anotación de ROIs) exclusivamente sobre `{selected_file}`.")

col_single1, col_single2 = st.columns([3, 1], vertical_alignment="center")
with col_single1:
    csv_single = os.path.join(METRICS_BASE_DIR, selected_group, selected_section, f"{base_fn}_nuclei_metrics.csv")
    if os.path.exists(csv_single):
        st.success(f"✅ **Muestra Analizada Previamente:** Métrica CSV lista en `{csv_single}`")
    else:
        st.info("ℹ️ **Muestra pendiente de análisis.** Presiona el botón para procesarla.")

with col_single2:
    btn_run_single = st.button("⚡ Procesar Esta Muestra", type="primary", use_container_width=True, key="btn_run_single_sample")

if btn_run_single:
    with st.spinner(f"Procesando muestra {selected_file}..."):
        try:
            mip_p = get_or_create_mip(raw_file_path, px_size)
            out_s = os.path.join(SEGM_BASE_DIR, selected_group, selected_section)
            out_m = os.path.join(METRICS_BASE_DIR, selected_group, selected_section)
            os.makedirs(out_s, exist_ok=True)
            os.makedirs(out_m, exist_ok=True)
            
            model_pv_obj, model_loc, model_score, device = load_models()
            
            summary_single = run_pipeline_on_file(
                tif_path=mip_p,
                out_segm_dir=out_s,
                out_metrics_dir=out_m,
                model_pv_obj=model_pv_obj if do_pv else None,
                model_loc=model_loc,
                model_score=model_score,
                device=device,
                pv_filter_type=pv_filter,
                pv_diameter=pv_diam,
                pv_flow_threshold=pv_flow,
                pv_cellprob_threshold=pv_prob,
                loc_threshold=loc_threshold,
                score_threshold=score_threshold,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                px_size=px_size,
                do_pv_segmentation=do_pv,
                calib_data=calib_data
            )
            st.success(f"🎉 ¡Muestra `{selected_file}` procesada con éxito!")
            
            # Display Metric Summary Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Interneuronas PV+ Segmentadas", f"{summary_single.get('total_pv_segmentation', 0)}")
            m2.metric("PNNs Detectadas", f"{summary_single.get('total_pnn_plus', 0)}")
            m3.metric("Coinmunomarcadas (PV+/PNN+)", f"{summary_single.get('pv_pnn_plus', 0)}")
            m4.metric("Superficie Total (mm²)", f"{summary_single.get('image_area_mm2', 0.0):.3f}")
            
        except Exception as e:
            st.error(f"Error procesando muestra {selected_file}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: EJECUCIÓN GLOBAL EN BATCH DEL EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🚀 4. Ejecución Global del Experimento en Batch")

batch_col1, batch_col2 = st.columns([3, 1], vertical_alignment="center")
with batch_col1:
    st.write("Ejecuta la segmentación y análisis global de PNNs y PV+ para todas las muestras del experimento.")
with batch_col2:
    force_reprocess = st.checkbox("🔄 Forzar reprocesamiento", value=False, key="batch_force")

if st.button("▶️ Procesar Todo el Experimento en Batch", type="primary", key="btn_batch_run"):
    calib_data.update({
        'pixel_size_um': px_size,
        'do_pv_segmentation': do_pv,
        'pv_cellpose_filter_type': pv_filter,
        'pv_cellpose_diameter': pv_diam,
        'pv_cellpose_flow_threshold': pv_flow,
        'pv_cellpose_cellprob_threshold': pv_prob,
        'lupori_loc_threshold': loc_threshold,
        'lupori_score_threshold': score_threshold,
        'lupori_min_peak_dist': min_peak_dist,
        'lupori_tile_size': tile_size,
        'lupori_tile_overlap': tile_overlap,
        'channels': ["AGR", "DAPI", "WFA", "PV"]
    })
    save_config(calib_data)

    if not all_tasks:
        st.error("No se encontraron muestras en `data/raw`.")
        st.stop()

    tasks_to_run = []
    for g, s, f, path in all_tasks:
        base, _ = os.path.splitext(f)
        seg_out = os.path.join(SEGM_BASE_DIR, g, s, f"{base}_masks.tif")
        csv_out = os.path.join(METRICS_BASE_DIR, g, s, f"{base}_nuclei_metrics.csv")
        already = os.path.exists(seg_out) and os.path.exists(csv_out)
        if not already or force_reprocess:
            tasks_to_run.append((g, s, f, path))

    if not tasks_to_run:
        st.success("✅ Todas las imágenes ya están procesadas.")
        st.stop()

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    status_text.text("Cargando modelos de Inteligencia Artificial (Cellpose PV, PNNloc, PNNscore)...")
    model_pv_obj, model_loc, model_score, device = load_models()
    success_count = 0
    
    for idx, (g, s, f, raw_p) in enumerate(tasks_to_run):
        base, _ = os.path.splitext(f)
        status_text.text(f"Procesando [{idx+1}/{len(tasks_to_run)}]: {g}/{s}/{base}")
        try:
            mip_p = get_or_create_mip(raw_p, px_size)
            out_s = os.path.join(SEGM_BASE_DIR, g, s)
            out_m = os.path.join(METRICS_BASE_DIR, g, s)
            os.makedirs(out_s, exist_ok=True)
            os.makedirs(out_m, exist_ok=True)
            
            run_pipeline_on_file(
                tif_path=mip_p,
                out_segm_dir=out_s,
                out_metrics_dir=out_m,
                model_pv_obj=model_pv_obj if do_pv else None,
                model_loc=model_loc,
                model_score=model_score,
                device=device,
                pv_filter_type=pv_filter,
                pv_diameter=pv_diam,
                pv_flow_threshold=pv_flow,
                pv_cellprob_threshold=pv_prob,
                loc_threshold=loc_threshold,
                score_threshold=score_threshold,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                px_size=px_size,
                do_pv_segmentation=do_pv,
                calib_data=calib_data
            )
            success_count += 1
        except Exception as e:
            st.error(f"Error procesando {f}: {e}")

        progress_bar.progress((idx + 1) / len(tasks_to_run))

    status_text.text(f"✅ ¡Procesamiento batch completado! {success_count}/{len(tasks_to_run)} procesadas exitosamente.")
    st.success(f"✅ Se han procesado {success_count} muestras correctamente.")
    st.rerun()
