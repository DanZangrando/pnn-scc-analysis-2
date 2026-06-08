import streamlit as st
import os
import json
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PNN SSC Analysis 🧠🔬",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMarkdown h1, h2, h3 {
        color: #4facfe;
    }
    .stButton>button {
        background-image: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-image: linear-gradient(120deg, #00f2fe 0%, #4facfe 100%);
        color: white;
    }
    .status-box {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4facfe;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

CONFIG_PATH = "experiment_config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=4)

calib_data = load_config()

# Sidebar
st.sidebar.title("⚙️ Configuración Global")

st.sidebar.caption("🖼️ Visualizador: Napari (WSL/GUI)")
st.sidebar.divider()
st.sidebar.subheader("Parámetros del Pipeline")

with st.sidebar.expander("📏 Calibración de Imagen"):
    px_size = st.number_input("Tamaño de Pixel (µm)", value=float(calib_data.get('pixel_size_um', 1.0)), step=0.001, format="%.4f")

with st.sidebar.expander("🧬 Parámetros Cellpose (DAPI)"):
    st.caption("⚠️ Canal DAPI: Sólo para referencia visual en overlays/Napari (no cuantificado).")
    filter_options = ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"]
    def_filter = calib_data.get('cellpose_filter_type', "Ninguno")
    dapi_filter = st.selectbox("Filtro previo DAPI", filter_options, index=filter_options.index(def_filter) if def_filter in filter_options else 0)
    dapi_diam = st.number_input("Diámetro Núcleo (px)", value=float(calib_data.get('cellpose_diameter', 30.0)), step=1.0)
    dapi_flow = st.slider("Flow Threshold (DAPI)", 0.0, 1.0, float(calib_data.get('cellpose_flow_threshold', 0.4)))
    dapi_prob = st.slider("Cell Prob Threshold (DAPI)", -6.0, 6.0, float(calib_data.get('cellpose_cellprob_threshold', 0.1)))

with st.sidebar.expander("🧪 Parámetros Cellpose (PV)"):
    do_pv = st.checkbox("Activar segmentación PV", value=calib_data.get('do_pv_segmentation', True))
    pv_def_filter = calib_data.get('pv_cellpose_filter_type', "Ninguno")
    pv_filter = st.selectbox("Filtro previo PV", filter_options, index=filter_options.index(pv_def_filter) if pv_def_filter in filter_options else 0)
    pv_diam = st.number_input("Diámetro PV (px)", value=float(calib_data.get('pv_cellpose_diameter', 30.0)), step=1.0)
    pv_flow = st.slider("Flow Threshold (PV)", 0.0, 1.0, float(calib_data.get('pv_cellpose_flow_threshold', 0.4)))
    pv_prob = st.slider("Cell Prob Threshold (PV)", -6.0, 6.0, float(calib_data.get('pv_cellpose_cellprob_threshold', 0.0)))

with st.sidebar.expander("🕸️ Parámetros PNN (WFA)"):
    st.markdown("La detección PV+/PNN+ se realiza por colocalización espacial directa (máxima superficie). No se requiere anillo de expansión.")
    pv_expansion_dist_um = 0.0
    pnn_thresh = 0.0
    pnn_excl = 0.0
    
    thresh_methods = ["Automático (Otsu)", "Manual"]
    def_method = calib_data.get('pnn_wfa_threshold_method', "Automático (Otsu)")
    pnn_wfa_threshold_method = st.selectbox("Método Umbralado WFA", thresh_methods, index=thresh_methods.index(def_method) if def_method in thresh_methods else 0)
    pnn_wfa_manual_threshold = st.number_input("Umbral WFA Manual", value=float(calib_data.get('pnn_wfa_manual_threshold', 10000.0)), step=500.0)
    max_pnn_distance_um = st.number_input("Distancia máx. Voronoi (µm)", value=float(calib_data.get('max_pnn_distance_um', 20.0)), step=1.0)
    pnn_gaussian_sigma = st.slider("Suavizado Gaussiano WFA (Sigma)", 0.0, 3.0, float(calib_data.get('pnn_gaussian_sigma', 1.0)), step=0.1)
    
    # Advanced skeletonization parameters
    pnn_connect_fragments = st.checkbox("Conectar fragmentos cercanos", value=bool(calib_data.get('pnn_connect_fragments', False)))
    pnn_connection_radius_um = st.number_input("Radio de conexión (µm)", value=float(calib_data.get('pnn_connection_radius_um', 1.0)), step=0.1)
    pnn_pruning_min_voxels = st.number_input("Umbral de Poda (vóxeles)", value=int(calib_data.get('pnn_pruning_min_voxels', 0)), step=1)
    pnn_filter_by_nucleus = st.checkbox("Filtrar por conectividad al soma PV", value=bool(calib_data.get('pnn_filter_by_nucleus', False)))

with st.sidebar.expander("🕸️ Parámetros Cellpose (WFA)"):
    do_wfa_cp = st.checkbox("Activar segmentación WFA (Cellpose)", value=bool(calib_data.get('do_wfa_cellpose', True)))
    wfa_cp_def_filter = calib_data.get('wfa_cellpose_filter_type', "Ninguno")
    wfa_cp_filter = st.selectbox("Filtro WFA Cellpose", filter_options, index=filter_options.index(wfa_cp_def_filter) if wfa_cp_def_filter in filter_options else 0)
    wfa_cp_diam = st.number_input("Diámetro PNN WFA (px)", value=float(calib_data.get('wfa_cellpose_diameter', 30.0)), step=1.0)
    wfa_cp_flow = st.slider("Flow Threshold WFA Cellpose", 0.0, 1.0, float(calib_data.get('wfa_cellpose_flow_threshold', 0.4)))
    wfa_cp_prob = st.slider("Cellprob Threshold WFA Cellpose", -6.0, 6.0, float(calib_data.get('wfa_cellpose_cellprob_threshold', 0.0)))

if st.sidebar.button("💾 Guardar Toda la Configuración"):
    calib_data.update({
        'pixel_size_um': px_size,
        'cellpose_filter_type': dapi_filter,
        'cellpose_diameter': dapi_diam,
        'cellpose_flow_threshold': dapi_flow,
        'cellpose_cellprob_threshold': dapi_prob,
        'do_pv_segmentation': do_pv,
        'pv_cellpose_filter_type': pv_filter,
        'pv_cellpose_diameter': pv_diam,
        'pv_cellpose_flow_threshold': pv_flow,
        'pv_cellpose_cellprob_threshold': pv_prob,
        'pv_expansion_dist_um': pv_expansion_dist_um,
        'pnn_intensity_threshold': pnn_thresh,
        'pnn_exclusion_distance_um': pnn_excl,
        'pnn_wfa_threshold_method': pnn_wfa_threshold_method,
        'pnn_wfa_manual_threshold': pnn_wfa_manual_threshold,
        'max_pnn_distance_um': max_pnn_distance_um,
        'pnn_gaussian_sigma': pnn_gaussian_sigma,
        'pnn_connect_fragments': pnn_connect_fragments,
        'pnn_connection_radius_um': pnn_connection_radius_um,
        'pnn_pruning_min_voxels': pnn_pruning_min_voxels,
        'pnn_filter_by_nucleus': pnn_filter_by_nucleus,
        'do_wfa_cellpose': do_wfa_cp,
        'wfa_cellpose_filter_type': wfa_cp_filter,
        'wfa_cellpose_diameter': wfa_cp_diam,
        'wfa_cellpose_flow_threshold': wfa_cp_flow,
        'wfa_cellpose_cellprob_threshold': wfa_cp_prob,
        'channels': ["AGR", "DAPI", "WFA", "PV"]
    })
    save_config(calib_data)
    st.sidebar.success("Configuración guardada exitosamente.")

st.sidebar.divider()
st.sidebar.title("Navegación")
st.sidebar.info("Este es el punto de inicio del análisis de PNNs y ERα en la Corteza Somatosensorial.")

# Home Page Content
st.title("PNN SSC Analysis 🧠🔬")
st.subheader("Análisis Automatizado de Redes Perineuronales e Interneuronas PV+")

st.markdown("""
<div class="status-box">
<h3>📋 Descripción del Proyecto</h3>
Este proyecto procesa imágenes de <b>microscopía de fluorescencia nativa (.TIF)</b> para cuantificar la densidad y morfología de las <b>Redes Perineuronales (PNNs)</b> y su asociación con las interneuronas <b>PV+ (Parvalbúmina)</b> y núcleos <b>DAPI</b> en la Corteza Somatosensorial (SSC). Soporta comparación entre grupos experimentales y hemisferios (IPSI vs CONTRA).
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("### 📂 Explorador de Datos Estructural")
raw_data_path = "data/raw"
if not os.path.exists(raw_data_path) or not any(os.path.isdir(os.path.join(raw_data_path, d)) for d in os.listdir(raw_data_path) if not d.startswith('.')):
    raw_data_path = "data/processed/mips"

all_tasks = []

if os.path.exists(raw_data_path):
    groups = sorted([d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))])
    
    if groups:
        st.write(f"Se han detectado **{len(groups)}** grupos experimentales.")
        
        # Build tasks list for global processing
        for g in groups:
            g_path = os.path.join(raw_data_path, g)
            sections = [d for d in os.listdir(g_path) if os.path.isdir(os.path.join(g_path, d))]
            for s in sections:
                s_path = os.path.join(g_path, s)
                files = [f for f in os.listdir(s_path) if f.lower().endswith('.tif')]
                for f in files:
                    all_tasks.append((g, s, f, os.path.join(s_path, f)))
        
        selected_group = st.selectbox("Selecciona un Grupo para explorar:", groups)
        group_dir = os.path.join(raw_data_path, selected_group)
        
        sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
        if sections:
            selected_section = st.selectbox("Selecciona una Sección (ej. IPSI/CONTRA):", sections)
            section_dir = os.path.join(group_dir, selected_section)
            tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith('.tif')])
            
            if tif_files:
                st.write(f"La sección `{selected_section}` contiene **{len(tif_files)}** imágenes.")
                selected_file = st.selectbox("Selecciona una imagen:", tif_files)
                
                # Extract individual ID from format ACF_49~1.TIF
                import re as _re
                m_id = _re.match(r'(ACF_\d+)', selected_file)
                indiv_id = m_id.group(1) if m_id else selected_file.split('~')[0]
                
                # Check if already processed
                base_sel, _ = os.path.splitext(selected_file)
                seg_check = os.path.join('data/processed/segmented', selected_group, selected_section, f"{base_sel}_masks.tif")
                csv_check  = os.path.join('data/processed/metrics', selected_group, selected_section, f"{base_sel}_nuclei_metrics.csv")
                already_done = os.path.exists(seg_check) and os.path.exists(csv_check)
                
                badge = "✅ Ya procesada" if already_done else "⏳ Pendiente de procesar"
                st.info(f"Archivo seleccionado: `{selected_file}` &nbsp;&nbsp; {badge}")
                st.markdown(f"""
                - **Grupo:** {selected_group}
                - **Sección:** {selected_section}
                - **Sujeto:** {indiv_id}
                - **Formato:** .TIF (4 Canales: AGR, DAPI, WFA, PV)
                """)
            else:
                st.warning(f"No se detectaron archivos `.TIF` en la sección `{selected_section}`.")
        else:
            st.warning(f"No se detectaron subdirectorios IPSI/CONTRA en el grupo `{selected_group}`.")
    else:
        st.warning("No se detectaron subdirectorios de grupos en `data/raw`.")
else:
    st.error(f"No se encontró el directorio `{raw_data_path}`.")

st.divider()

st.markdown("### 🚀 Ejecución Global en Batch")

# Summary of pending / done tasks
SEGM_BASE_DIR   = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"

done_count    = sum(1 for g, s, f, _ in all_tasks
                    if os.path.exists(os.path.join(SEGM_BASE_DIR, g, s, f"{os.path.splitext(f)[0]}_masks.tif"))
                    and os.path.exists(os.path.join(METRICS_BASE_DIR, g, s, f"{os.path.splitext(f)[0]}_nuclei_metrics.csv")))
pending_count = len(all_tasks) - done_count

col_cnt1, col_cnt2, col_cnt3 = st.columns(3)
col_cnt1.metric("Total de imágenes", len(all_tasks))
col_cnt2.metric("✅ Ya procesadas", done_count)
col_cnt3.metric("⏳ Pendientes", pending_count)

batch_col1, batch_col2 = st.columns([3, 1])
with batch_col1:
    st.write("Procesa todas las imágenes con los parámetros del sidebar. Por defecto omite las ya procesadas.")
with batch_col2:
    force_reprocess = st.checkbox("🔄 Forzar reprocesamiento", value=False,
                                   help="Si está activo, vuelve a procesar todas las imágenes aunque ya tengan resultados.",
                                   key="batch_force")

if st.button("▶️ Procesar Todo el Experimento en Batch", type="primary", key="btn_batch_run"):
    # Save full config before processing
    calib_data.update({
        'pixel_size_um': px_size,
        'cellpose_filter_type': dapi_filter,
        'cellpose_diameter': dapi_diam,
        'cellpose_flow_threshold': dapi_flow,
        'cellpose_cellprob_threshold': dapi_prob,
        'do_pv_segmentation': do_pv,
        'pv_cellpose_filter_type': pv_filter,
        'pv_cellpose_diameter': pv_diam,
        'pv_cellpose_flow_threshold': pv_flow,
        'pv_cellpose_cellprob_threshold': pv_prob,
        'pv_expansion_dist_um': pv_expansion_dist_um,
        'pnn_intensity_threshold': pnn_thresh,
        'pnn_exclusion_distance_um': pnn_excl,
        'pnn_wfa_threshold_method': pnn_wfa_threshold_method,
        'pnn_wfa_manual_threshold': pnn_wfa_manual_threshold,
        'max_pnn_distance_um': max_pnn_distance_um,
        'pnn_gaussian_sigma': pnn_gaussian_sigma,
        'pnn_connect_fragments': pnn_connect_fragments,
        'pnn_connection_radius_um': pnn_connection_radius_um,
        'pnn_pruning_min_voxels': pnn_pruning_min_voxels,
        'pnn_filter_by_nucleus': pnn_filter_by_nucleus,
        'do_wfa_cellpose': do_wfa_cp,
        'wfa_cellpose_filter_type': wfa_cp_filter,
        'wfa_cellpose_diameter': wfa_cp_diam,
        'wfa_cellpose_flow_threshold': wfa_cp_flow,
        'wfa_cellpose_cellprob_threshold': wfa_cp_prob,
        'channels': ["AGR", "DAPI", "WFA", "PV"]
    })
    save_config(calib_data)

    if not all_tasks:
        st.error("No se encontraron imágenes `.TIF` en la estructura de `data/raw`.")
        st.stop()

    # Determine which tasks to run
    tasks_to_run = []
    tasks_skipped = []
    for g, s, f, path in all_tasks:
        base, _ = os.path.splitext(f)
        seg_out = os.path.join(SEGM_BASE_DIR, g, s, f"{base}_masks.tif")
        csv_out = os.path.join(METRICS_BASE_DIR, g, s, f"{base}_nuclei_metrics.csv")
        already = os.path.exists(seg_out) and os.path.exists(csv_out)
        if not already or force_reprocess:
            tasks_to_run.append((g, s, f, path))
        else:
            tasks_skipped.append((g, s, f))

    if not tasks_to_run:
        st.success("✅ Todas las imágenes ya están procesadas. Activa '🔄 Forzar reprocesamiento' para volver a correr.")
        st.stop()

    st.info(f"🔬 Procesando **{len(tasks_to_run)}** imágenes · Saltando **{len(tasks_skipped)}** ya procesadas.")

    from pipeline import run_pipeline_on_file
    from cellpose import models
    import torch

    use_gpu = torch.cuda.is_available()
    gpu_label = "GPU 🚀" if use_gpu else "CPU (sin CUDA)"
    st.write(f"Dispositivo: **{gpu_label}**")

    with st.spinner("Cargando modelos Cellpose..."):
        model_dapi = models.CellposeModel(gpu=use_gpu)
        model_pv   = models.CellposeModel(gpu=use_gpu) if do_pv else None

    progress_bar = st.progress(0)
    status_text  = st.empty()
    results_log  = []  # accumulate per-image results
    result_table = st.empty()  # live-updating table placeholder

    success_count = 0
    error_count   = 0

    for i, (g, s, f, path) in enumerate(tasks_to_run):
        status_text.markdown(f"**Procesando ({i+1}/{len(tasks_to_run)}):** `{g} / {s} / {f}`")
        out_segm = os.path.join(SEGM_BASE_DIR, g, s)
        out_metr = os.path.join(METRICS_BASE_DIR, g, s)
        os.makedirs(out_segm, exist_ok=True)
        os.makedirs(out_metr, exist_ok=True)

        try:
            result = run_pipeline_on_file(
                tif_path=path,
                out_segm_dir=out_segm,
                out_metrics_dir=out_metr,
                model_dapi=model_dapi,
                model_pv_obj=model_pv,
                filter_type=dapi_filter,
                diameter=dapi_diam,
                flow_threshold=dapi_flow,
                cellprob_threshold=dapi_prob,
                pv_filter_type=pv_filter,
                pv_diameter=pv_diam,
                pv_flow_threshold=pv_flow,
                pv_cellprob_threshold=pv_prob,
                pv_expansion_dist_um=pv_expansion_dist_um,
                pnn_threshold=pnn_thresh,
                pnn_exclusion_dist_um=pnn_excl,
                px_size=px_size,
                do_pv_segmentation=do_pv,
                calib_data=calib_data
            )
            success_count += 1
            results_log.append({
                "Grupo": g, "Sección": s, "Archivo": f,
                "PV+": result.get('pnn_minus', 0) + result.get('pv_pnn_plus', 0),
                "PNN+ Total": result.get('total_pnn_plus', 0),
                "PNN+ Ocup.": result.get('pv_pnn_plus', 0),
                "PNN+ Huec.": result.get('hollow_pnn_plus', 0),
                "Estado": "✅ OK"
            })
        except Exception as e:
            error_count += 1
            results_log.append({
                "Grupo": g, "Sección": s, "Archivo": f,
                "PV+": "-", "PNN+ Total": "-",
                "PNN+ Ocup.": "-", "PNN+ Huec.": "-",
                "Estado": f"❌ {str(e)[:60]}"
            })

        progress_bar.progress((i + 1) / len(tasks_to_run))

        # Update live results table
        result_table.dataframe(
            pd.DataFrame(results_log),
            use_container_width=True
        )

    # Final summary
    status_text.empty()
    if error_count == 0:
        st.success(f"✅ Batch completado: {success_count}/{len(tasks_to_run)} imágenes procesadas correctamente. {len(tasks_skipped)} omitidas (ya procesadas).")
    else:
        st.warning(f"⚠️ Batch finalizado: {success_count} OK · {error_count} errores · {len(tasks_skipped)} omitidas.")

    # Reload metrics and show global summary
    all_csv = []
    for root, dirs, files in os.walk(METRICS_BASE_DIR):
        if 'test' in root: continue
        for fname in files:
            if fname.endswith('_nuclei_metrics.csv'):
                try:
                    df_c = pd.read_csv(os.path.join(root, fname))
                    all_csv.append(df_c)
                except Exception:
                    pass
    if all_csv:
        df_global = pd.concat(all_csv, ignore_index=True)
        st.divider()
        st.subheader("📊 Resumen Global del Experimento")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Total Células/Redes", len(df_global))
        g2.metric("Total PV+", int(df_global['is_pv_plus'].sum()) if 'is_pv_plus' in df_global else "-")
        g3.metric("Total PNN+ (Redes)", int(df_global['is_pnn_plus'].sum()) if 'is_pnn_plus' in df_global else "-")
        if 'cell_type' in df_global.columns:
            occ = int((df_global['cell_type'] == 'PV+/PNN+').sum())
            hol = int((df_global['cell_type'] == 'PV-/PNN+').sum())
            g4.metric("PNN+ Ocup. / Huecas", f"{occ} / {hol}")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado para el análisis de SSC")
