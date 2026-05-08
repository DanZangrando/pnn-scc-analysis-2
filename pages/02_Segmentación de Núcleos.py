import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, regionprops_table
from skimage import exposure, draw
import pandas as pd

st.set_page_config(page_title="Segmentación de Núcleos", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #bb86fc;
    }
    div[data-testid="stMetricLabel"] {
        color: #e0e0e0;
    }
    .stHeader {
        color: #bb86fc;
    }
    .img-caption {
        font-weight: bold;
        color: #bb86fc;
        margin-bottom: 5px;
        text-align: center;
    }
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #bb86fc, transparent);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Página 2: Segmentación de Núcleos (DAPI + PV)")

with st.expander("ℹ️ ¿Cómo funciona este pipeline de segmentación? (Documentación Técnica)", expanded=False):
    st.markdown("""
    **1. Pre-procesamiento (Filtros)**
    Antes de enviar la imagen a Cellpose, se extrae el canal DAPI y se le puede aplicar un **Filtro** (desde la barra lateral).
    - **Ninguno**: La imagen pasa cruda a Cellpose. Evita borrar núcleos tenues en esquinas oscuras.
    - **Otsu Global**: Corta el ruido de fondo fijando un único umbral para toda la imagen.
    - **CLAHE (Adaptativo Local)**: Ecualizador de contraste por sectores. Realza los núcleos borrosos.

    
    **2. Segmentación (Cellpose)**
    La imagen limpia se envía a la red neuronal Cellpose. La red identifica cada núcleo individual y le asigna una etiqueta numérica única (ID: 1 a N).
    - **Flow Threshold (Umbral de Flujo):** Define qué tan estricta es la matemática agrupando píxeles. (Sugerido: 0.4)
    - **Cell Prob Threshold (Probabilidad):** Define el límite de confianza para aceptar una célula. (Sugerido: 0.1)

    
    **3. Exportación (7-Canales + Métricas)**
    Para evitar la generación lenta de miles de polígonos, el pipeline usa una técnica de inyección:
    - **Métricas (`.csv`):** Características físicas de cada célula (área, intensidad PV, presencia de PNN).
    - **Imagen (`_segmented.tif`):** Clon de 7 canales: 4 originales + 3 Máscaras (DAPI, PV, PNN).
    
    **4. Visualización en QuPath (LUTs)**
    QuPath abrirá la imagen con los 7 canales. Asigna LUTs como **Glasbey** o **Fire** a los canales de máscara (5, 6 y 7) para visualizar los resultados.
    """)


# --- Paths ---
RAW_DIR = "data/raw"
MIPS_BASE_DIR = "data/processed/mips"
SEGM_BASE_DIR = "data/processed/segmented"
METRICS_BASE_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

if not os.path.exists(MIPS_BASE_DIR):
    st.error(f"No se encontró la carpeta `{MIPS_BASE_DIR}`. Por favor, genera los MIPs en la Página 1.")
    st.stop()

# Get groups (subdirectories)
groups = sorted([d for d in os.listdir(MIPS_BASE_DIR) if os.path.isdir(os.path.join(MIPS_BASE_DIR, d))])
if not groups:
    groups = ["."]

st.sidebar.header("📁 Selección de Grupo")
selected_group = st.sidebar.selectbox("Grupo:", groups)

MIPS_DIR = os.path.join(MIPS_BASE_DIR, selected_group)
SEGM_DIR = os.path.join(SEGM_BASE_DIR, selected_group)
METRICS_DIR = os.path.join(METRICS_BASE_DIR, selected_group)

os.makedirs(SEGM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# --- Load Config ---
calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)
    st.sidebar.success("✅ Configuración Global Cargada")
    # st.sidebar.json(calib_data) # Removed for space
else:
    st.sidebar.warning("⚠️ No se encontró la configuración global. Modifica los parámetros en la Página 1.")

# --- Data Loading ---
processed_files = sorted([f for f in os.listdir(MIPS_DIR) if f.endswith('_MIP.tif')])

if not processed_files:
    st.error(f"No hay imágenes procesadas en `{MIPS_DIR}`. Por favor, genera los MIPs en la Página 1.")
    st.stop()

st.sidebar.header("📂 Selección de Muestra")
selected_filename = st.sidebar.selectbox("Archivo Procesado:", processed_files)
selected_path = os.path.join(MIPS_DIR, selected_filename)

@st.cache_data
def load_channels(path):
    img = tiff.imread(path) # Expected (C, Y, X) - Now with 4 channels
    # 0: AGR, 1: DAPI, 2: WFA, 3: PV
    agr = img[0, :, :] if img.shape[0] >= 1 else np.zeros_like(img[0])
    dapi = img[1, :, :] if img.shape[0] >= 2 else img[0, :, :]
    wfa = img[2, :, :] if img.shape[0] >= 3 else np.zeros_like(img[0])
    pv = img[3, :, :] if img.shape[0] >= 4 else np.zeros_like(img[0])
    
    # Normalize for display
    disp_dapi = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_pv = cv2.normalize(pv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_wfa = cv2.normalize(wfa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_agr = cv2.normalize(agr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return (pv, wfa, dapi, agr), (disp_pv, disp_wfa, disp_dapi, disp_agr)

(pv_raw, wfa_raw, dapi_raw, agr_raw), (pv_disp, wfa_disp, dapi_disp, agr_disp) = load_channels(selected_path)

st.subheader(f"Muestra: {selected_filename.replace('_MIP.tif', '')}")

# --- Centralized Control Panel ---
with st.container():
    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        view_mode = st.radio("Capa a visualizar en el panel derecho:", 
                             ["Núcleos (DAPI)", "Parvalbúmina (PV+)", "Redes (PNN+)"], 
                             horizontal=True, 
                             label_visibility="visible")
    with c_col2:
        st.write("") # Spacer
        st.caption("👈 Selecciona la capa para comparar con el canal DAPI original.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="img-caption">Canal DAPI (Referencia)</p>', unsafe_allow_html=True)
    st.image(dapi_disp, use_container_width=True, clamp=True, channels="GRAY")

# --- Cellpose Setup ---
st.sidebar.header("⚙️ Parámetros de Cellpose")
use_gpu = st.sidebar.checkbox("Intentar usar GPU (PyTorch)", value=True)

filter_options = ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"]
default_filter = calib_data.get('cellpose_filter_type', "Ninguno")
if default_filter not in filter_options:
    default_filter = "Ninguno"
    
filter_type = st.sidebar.selectbox("Filtro previo en DAPI", filter_options, index=filter_options.index(default_filter))

model_type = st.sidebar.selectbox("Modelo Base", ["cyto", "nuclei", "cyto2", "cyto3"], index=1)
diameter = st.sidebar.number_input("Diámetro del Núcleo (px)", value=float(calib_data.get('cellpose_diameter', 30.0)), step=1.0)
flow_threshold = st.sidebar.slider("Flow Threshold", 0.0, 1.0, float(calib_data.get('cellpose_flow_threshold', 0.4)))
cellprob_threshold = st.sidebar.slider("Cell Prob Threshold", -6.0, 6.0, float(calib_data.get('cellpose_cellprob_threshold', 0.1)))

# Calculate physical diameter for immediate feedback
if 'pixel_size_um' in calib_data:
    px_size = calib_data['pixel_size_um']
    phys_diam = diameter * px_size
    st.sidebar.info(f"Diámetro Físico Aprox: **{phys_diam:.2f} µm** (a {px_size:.4f} µm/px)")

if st.sidebar.button("💾 Guardar Parámetros por Defecto"):
    calib_data['cellpose_filter_type'] = filter_type
    calib_data['cellpose_diameter'] = diameter
    calib_data['cellpose_flow_threshold'] = flow_threshold
    calib_data['cellpose_cellprob_threshold'] = cellprob_threshold
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
    st.sidebar.success("Parámetros guardados y definidos como defecto.")

st.sidebar.divider()
st.sidebar.header("🧪 Segmentación de PV")
do_pv_segmentation = st.sidebar.checkbox("Activar segmentación PV", value=True)

pv_filter_type = calib_data.get('pv_cellpose_filter_type', "Ninguno")
pv_filter_type = st.sidebar.selectbox("Filtro previo en PV", filter_options, index=filter_options.index(pv_filter_type) if pv_filter_type in filter_options else 0)
pv_diameter = st.sidebar.number_input("Diámetro PV (px)", value=float(calib_data.get('pv_cellpose_diameter', 30.0)), step=1.0)
pv_flow_threshold = st.sidebar.slider("Flow Threshold PV", 0.0, 1.0, float(calib_data.get('pv_cellpose_flow_threshold', 0.4)))
pv_cellprob_threshold = st.sidebar.slider("Cell Prob Threshold PV", -6.0, 6.0, float(calib_data.get('pv_cellpose_cellprob_threshold', 0.0)))

if st.sidebar.button("💾 Guardar Parámetros PV"):
    calib_data['pv_cellpose_filter_type'] = pv_filter_type
    calib_data['pv_cellpose_diameter'] = pv_diameter
    calib_data['pv_cellpose_flow_threshold'] = pv_flow_threshold
    calib_data['pv_cellpose_cellprob_threshold'] = pv_cellprob_threshold
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
    st.sidebar.success("Parámetros de PV guardados.")

st.sidebar.divider()
st.sidebar.header("🕸️ Análisis de PNN (WFA)")
pnn_radius_um = st.sidebar.number_input("Radio de búsqueda (µm)", value=float(calib_data.get('pnn_radius_um', 20.0)), step=1.0)
pnn_threshold = st.sidebar.number_input("Umbral de Intensidad WFA (Suma)", value=float(calib_data.get('pnn_intensity_threshold', 500000.0)), step=10000.0)
pnn_exclusion_dist_um = st.sidebar.number_input("Distancia de exclusión (µm)", value=float(calib_data.get('pnn_exclusion_distance_um', 15.0)), step=1.0)

if st.sidebar.button("💾 Guardar Parámetros PNN"):
    calib_data['pnn_radius_um'] = pnn_radius_um
    calib_data['pnn_intensity_threshold'] = pnn_threshold
    calib_data['pnn_exclusion_distance_um'] = pnn_exclusion_dist_um
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
    st.sidebar.success("Parámetros de PNN guardados.")

st.sidebar.divider()
st.sidebar.subheader("▶️ Ejecución Individual")
if st.sidebar.button("🔬 Segmentar Muestra Actual", type="primary", use_container_width=True):
    with st.spinner("Ejecutando Pipeline completo (DAPI + PV + PNN)..."):
        try:
            px_size = calib_data.get('pixel_size_um', 1.0)
            
            # --- 1. DAPI Segmentation ---
            input_dapi = dapi_raw.copy()
            if filter_type == "Otsu Global":
                thresh = threshold_otsu(input_dapi)
                input_dapi[input_dapi < thresh] = 0
            elif filter_type == "CLAHE (Adaptativo Local)":
                clahe = exposure.equalize_adapthist(input_dapi, clip_limit=0.03)
                input_dapi = (clahe * 65535).astype(np.uint16)

            model_dapi = models.CellposeModel(gpu=use_gpu, model_type=model_type)
            masks_dapi, _, _ = model_dapi.eval(input_dapi, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
            
            # --- 2. PV Segmentation (Optional) ---
            masks_pv = np.zeros_like(masks_dapi)
            if do_pv_segmentation:
                input_pv = pv_raw.copy()
                if pv_filter_type == "Otsu Global":
                    thresh = threshold_otsu(input_pv)
                    input_pv[input_pv < thresh] = 0
                elif pv_filter_type == "CLAHE (Adaptativo Local)":
                    clahe = exposure.equalize_adapthist(input_pv, clip_limit=0.03)
                    input_pv = (clahe * 65535).astype(np.uint16)
                
                model_pv = models.CellposeModel(gpu=use_gpu, model_type="nuclei")
                masks_pv, _, _ = model_pv.eval(input_pv, diameter=pv_diameter, flow_threshold=pv_flow_threshold, cellprob_threshold=pv_cellprob_threshold)

            # --- 3. PNN and Colocalization Analysis ---
            props = regionprops(masks_dapi, intensity_image=wfa_raw)
            results = []
            
            for p in props:
                label = p.label
                centroid = p.centroid
                
                # PNN Signal
                rr, cc = draw.disk(centroid, pnn_radius_um / px_size, shape=wfa_raw.shape)
                wfa_sum = np.sum(wfa_raw[rr, cc])
                is_pnn_plus = wfa_sum > pnn_threshold
                
                # PV Colocalization
                is_pv_plus = masks_pv[int(centroid[0]), int(centroid[1])] > 0
                
                results.append({
                    'label': label,
                    'centroid_y': centroid[0],
                    'centroid_x': centroid[1],
                    'area_um2': p.area * (px_size**2),
                    'diameter_um': p.equivalent_diameter_area * px_size,
                    'dapi_mean_intensity': p.intensity_mean,
                    'wfa_sum_intensity': wfa_sum,
                    'is_pnn_plus': is_pnn_plus,
                    'is_pv_plus': is_pv_plus
                })
            
            # --- 3b. Spatial Exclusion (NMS) for PNN ---
            pnn_candidates = [r for r in results if r['is_pnn_plus']]
            if len(pnn_candidates) > 1 and pnn_exclusion_dist_um > 0:
                pnn_candidates.sort(key=lambda x: x['wfa_sum_intensity'], reverse=True)
                kept_centroids = []
                excluded_labels = []
                
                for cand in pnn_candidates:
                    curr_c = (cand['centroid_y'], cand['centroid_x'])
                    too_close = False
                    for k_c in kept_centroids:
                        dist = np.sqrt((curr_c[0]-k_c[0])**2 + (curr_c[1]-k_c[1])**2) * px_size
                        if dist < pnn_exclusion_dist_um:
                            too_close = True
                            break
                    if not too_close:
                        kept_centroids.append(curr_c)
                    else:
                        excluded_labels.append(cand['label'])
                
                for r in results:
                    if r['label'] in excluded_labels:
                        r['is_pnn_plus'] = False
            
            df = pd.DataFrame(results)
            csv_filename = selected_filename.replace('_MIP.tif', '_nuclei_metrics.csv')
            df.to_csv(os.path.join(METRICS_DIR, csv_filename), index=False)
            
            # --- 4. Global Summary Metrics ---
            total_pv = int(np.max(masks_pv))
            pnn_plus_count = int(df['is_pnn_plus'].sum())
            
            summary_data = {
                "total_dapi": len(df),
                "total_pv_segmentation": total_pv,
                "pnn_plus": pnn_plus_count,
                "pnn_minus": int(len(df) - pnn_plus_count),
                "dapi_pv_coloc": int(df['is_pv_plus'].sum()),
                "pixel_size": px_size
            }
            summary_filename = selected_filename.replace('_MIP.tif', '_summary.json')
            with open(os.path.join(METRICS_DIR, summary_filename), 'w') as f:
                json.dump(summary_data, f, indent=4)

            # --- 5. Save Multi-channel TIFF (7 channels) ---
            orig_mip = tiff.imread(selected_path)
            # Ensure we have 4 channels from original MIP
            if orig_mip.shape[0] < 4:
                # Padding or different handling if image is missing channels
                pass
            
            mask_dapi_ext = np.expand_dims(masks_dapi.astype(np.uint16), axis=0)
            mask_pv_ext = np.expand_dims(masks_pv.astype(np.uint16), axis=0)
            
            pnn_labels = df[df['is_pnn_plus'] == True]['label'].unique()
            masks_pnn = np.zeros_like(masks_dapi, dtype=np.uint16)
            if len(pnn_labels) > 0:
                max_label = int(np.max(masks_dapi))
                lut = np.zeros(max_label + 1, dtype=np.uint16)
                for l in pnn_labels:
                    if l <= max_label: lut[int(l)] = int(l)
                masks_pnn = lut[masks_dapi.astype(int)]
                
            mask_pnn_ext = np.expand_dims(masks_pnn, axis=0)
            
            segmented_stack = np.concatenate([orig_mip, mask_dapi_ext, mask_pv_ext, mask_pnn_ext], axis=0)
            seg_filename = selected_filename.replace('_MIP.tif', '_segmented.tif')
            seg_path = os.path.join(SEGM_DIR, seg_filename)
            
            ch_names = calib_data.get('channels', ['AGR', 'DAPI', 'WFA', 'PV'])
            seg_names = ch_names + ['DAPI_Mask', 'PV_Mask', 'PNN_Mask']
            
            tiff.imwrite(seg_path, segmented_stack, imagej=True, metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX', 'Labels': seg_names})
            
            st.sidebar.success(f"Detección finalizada: {len(df)} núcleos ({pnn_plus_count} en PNN+)")
            
            st.session_state[f"masks_{selected_filename}"] = masks_dapi
            st.session_state[f"masks_pv_{selected_filename}"] = masks_pv
            st.session_state['just_segmented'] = True

        except Exception as e:
            st.error(f"Error en Pipeline: {e}")
            import traceback
            st.error(traceback.format_exc())

st.sidebar.divider()

# ─────────────────────────────────────────────
# Helper function: process a single MIP file
# ─────────────────────────────────────────────
def run_pipeline_on_file(mip_path, out_segm_dir, out_metrics_dir,
                         model_dapi, model_pv_obj,
                         filter_type, diameter, flow_threshold, cellprob_threshold,
                         pv_filter_type, pv_diameter, pv_flow_threshold, pv_cellprob_threshold,
                         pnn_radius_um, pnn_threshold, pnn_exclusion_dist_um,
                         px_size, do_pv_segmentation, calib_data):
    fname = os.path.basename(mip_path)
    (p_raw, w_raw, d_raw, a_raw), _ = load_channels(mip_path)

    # DAPI preprocessing
    in_dapi = d_raw.copy()
    if filter_type == "Otsu Global":
        t = threshold_otsu(in_dapi)
        in_dapi[in_dapi < t] = 0
    elif filter_type == "CLAHE (Adaptativo Local)":
        clahe = exposure.equalize_adapthist(in_dapi, clip_limit=0.03)
        in_dapi = (clahe * 65535).astype(np.uint16)

    m_dapi, _, _ = model_dapi.eval(in_dapi, diameter=diameter, channels=[0, 0],
                                    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

    # PV preprocessing
    m_pv = np.zeros_like(m_dapi)
    if do_pv_segmentation and model_pv_obj is not None:
        in_pv = p_raw.copy()
        if pv_filter_type == "Otsu Global":
            t = threshold_otsu(in_pv)
            in_pv[in_pv < t] = 0
        elif pv_filter_type == "CLAHE (Adaptativo Local)":
            clahe = exposure.equalize_adapthist(in_pv, clip_limit=0.03)
            in_pv = (clahe * 65535).astype(np.uint16)
        m_pv, _, _ = model_pv_obj.eval(in_pv, diameter=pv_diameter, channels=[0, 0],
                                        flow_threshold=pv_flow_threshold, cellprob_threshold=pv_cellprob_threshold)

    # PNN analysis
    p_batch = regionprops(m_dapi, intensity_image=w_raw)
    r_batch = []
    for pb in p_batch:
        cr = pb.centroid
        rd, cd = draw.disk(cr, pnn_radius_um / px_size, shape=w_raw.shape)
        wfa_s = np.sum(w_raw[rd, cd])
        r_batch.append({
            'label': pb.label,
            'centroid_y': cr[0],
            'centroid_x': cr[1],
            'area_um2': pb.area * (px_size ** 2),
            'diameter_um': pb.equivalent_diameter_area * px_size,
            'dapi_mean_intensity': pb.intensity_mean,
            'wfa_sum_intensity': wfa_s,
            'is_pnn_plus': wfa_s > pnn_threshold,
            'is_pv_plus': m_pv[int(cr[0]), int(cr[1])] > 0
        })

    # NMS
    pnn_cands = [i for i, r in enumerate(r_batch) if r['is_pnn_plus']]
    if len(pnn_cands) > 1 and pnn_exclusion_dist_um > 0:
        sorted_idx = sorted(pnn_cands, key=lambda i: r_batch[i]['wfa_sum_intensity'], reverse=True)
        kept = []
        for si in sorted_idx:
            cy, cx = r_batch[si]['centroid_y'], r_batch[si]['centroid_x']
            if not any(np.sqrt((cy - k[0])**2 + (cx - k[1])**2) * px_size < pnn_exclusion_dist_um for k in kept):
                kept.append((cy, cx))
            else:
                r_batch[si]['is_pnn_plus'] = False

    df_b = pd.DataFrame(r_batch)
    df_b.to_csv(os.path.join(out_metrics_dir, fname.replace('_MIP.tif', '_nuclei_metrics.csv')), index=False)

    # TIFF output
    orig_b = tiff.imread(mip_path)
    m_pnn_b = np.zeros_like(m_dapi, dtype=np.uint16)
    pnn_lbls = df_b[df_b['is_pnn_plus']]['label'].values
    if len(pnn_lbls) > 0:
        lut = np.zeros(int(np.max(m_dapi)) + 1, dtype=np.uint16)
        for lb in pnn_lbls:
            lut[int(lb)] = int(lb)
        m_pnn_b = lut[m_dapi.astype(int)]

    stk = np.concatenate([orig_b,
                          np.expand_dims(m_dapi.astype(np.uint16), 0),
                          np.expand_dims(m_pv.astype(np.uint16), 0),
                          np.expand_dims(m_pnn_b, 0)], axis=0)
    ch_names = calib_data.get('channels', ['AGR', 'DAPI', 'WFA', 'PV'])
    tiff.imwrite(os.path.join(out_segm_dir, fname.replace('_MIP.tif', '_segmented.tif')),
                 stk, imagej=True,
                 metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                           'Labels': ch_names + ['DAPI_Mask', 'PV_Mask', 'PNN_Mask']})

    # Summary JSON
    summary = {
        "total_dapi": len(df_b),
        "total_pv_segmentation": int(np.max(m_pv)),
        "pnn_plus": int(df_b['is_pnn_plus'].sum()),
        "pnn_minus": int((~df_b['is_pnn_plus']).sum()),
        "dapi_pv_coloc": int(df_b['is_pv_plus'].sum()),
        "pixel_size": px_size
    }
    with open(os.path.join(out_metrics_dir, fname.replace('_MIP.tif', '_summary.json')), 'w') as fs:
        json.dump(summary, fs, indent=4)

    return summary


# ─────────────────────────────────────────────
# Batch Buttons
# ─────────────────────────────────────────────
st.sidebar.subheader("🚀 Procesamiento en Lote")

if st.sidebar.button("▶️ Segmentar Grupo Actual", use_container_width=True):
    px_size = calib_data.get('pixel_size_um', 1.0)
    model_d = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    model_p = models.CellposeModel(gpu=use_gpu, model_type="nuclei") if do_pv_segmentation else None
    progress = st.sidebar.progress(0)
    success = 0
    for idx, f in enumerate(processed_files):
        try:
            run_pipeline_on_file(
                os.path.join(MIPS_DIR, f), SEGM_DIR, METRICS_DIR,
                model_d, model_p,
                filter_type, diameter, flow_threshold, cellprob_threshold,
                pv_filter_type, pv_diameter, pv_flow_threshold, pv_cellprob_threshold,
                pnn_radius_um, pnn_threshold, pnn_exclusion_dist_um,
                px_size, do_pv_segmentation, calib_data
            )
            success += 1
        except Exception as e:
            st.sidebar.error(f"Error en {f}: {e}")
        progress.progress((idx + 1) / len(processed_files))
    st.sidebar.success(f"✅ Grupo '{selected_group}': {success}/{len(processed_files)} imágenes procesadas.")

st.sidebar.divider()

if st.sidebar.button("🌍 Segmentar TODOS los Grupos", use_container_width=True, type="primary"):
    px_size = calib_data.get('pixel_size_um', 1.0)
    all_groups = sorted([d for d in os.listdir(MIPS_BASE_DIR) if os.path.isdir(os.path.join(MIPS_BASE_DIR, d))])
    
    # Count total files across all groups
    all_tasks = []
    for g in all_groups:
        g_mips_dir = os.path.join(MIPS_BASE_DIR, g)
        mip_files = [f for f in os.listdir(g_mips_dir) if f.endswith('_MIP.tif')]
        for mf in mip_files:
            all_tasks.append((g, mf))
    
    if not all_tasks:
        st.sidebar.warning("No se encontraron MIPs en ningún grupo. Genera MIPs en la Página 1 primero.")
    else:
        st.sidebar.info(f"Procesando {len(all_tasks)} imágenes en {len(all_groups)} grupos...")
        
        # Load models once for efficiency
        model_d = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        model_p = models.CellposeModel(gpu=use_gpu, model_type="nuclei") if do_pv_segmentation else None
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        total_success = 0
        
        for task_idx, (grp, fname) in enumerate(all_tasks):
            g_mips = os.path.join(MIPS_BASE_DIR, grp)
            g_segm = os.path.join(SEGM_BASE_DIR, grp)
            g_metrics = os.path.join(METRICS_BASE_DIR, grp)
            os.makedirs(g_segm, exist_ok=True)
            os.makedirs(g_metrics, exist_ok=True)
            
            status_text.caption(f"⏳ [{task_idx+1}/{len(all_tasks)}] {grp} / {fname}")
            try:
                run_pipeline_on_file(
                    os.path.join(g_mips, fname), g_segm, g_metrics,
                    model_d, model_p,
                    filter_type, diameter, flow_threshold, cellprob_threshold,
                    pv_filter_type, pv_diameter, pv_flow_threshold, pv_cellprob_threshold,
                    pnn_radius_um, pnn_threshold, pnn_exclusion_dist_um,
                    px_size, do_pv_segmentation, calib_data
                )
                total_success += 1
            except Exception as e:
                st.sidebar.error(f"❌ {grp}/{fname}: {e}")
            
            progress_bar.progress((task_idx + 1) / len(all_tasks))
        
        status_text.empty()
        st.sidebar.success(f"🎉 Batch Global completado: {total_success}/{len(all_tasks)} imágenes procesadas en {len(all_groups)} grupos.")

with col2:
    st.markdown('<p class="img-caption">Visualización de Resultados</p>', unsafe_allow_html=True)
    
    seg_file = os.path.join(SEGM_DIR, selected_filename.replace('_MIP.tif', '_segmented.tif'))
    csv_file = os.path.join(METRICS_DIR, selected_filename.replace('_MIP.tif', '_nuclei_metrics.csv'))
    summary_file = os.path.join(METRICS_DIR, selected_filename.replace('_MIP.tif', '_summary.json'))
    
    current_masks = None
    base_img = dapi_disp
    
    if os.path.exists(seg_file):
        loaded = tiff.imread(seg_file)
        if view_mode == "Núcleos (DAPI)":
            current_masks = loaded[4, :, :] # DAPI Mask is now at Index 4 (after 4 original channels)
            base_img = dapi_disp
        elif view_mode == "Parvalbúmina (PV+)":
            current_masks = loaded[5, :, :] # PV Mask at Index 5
            base_img = pv_disp
        else: # PNN+
            current_masks = loaded[6, :, :] # PNN Mask at Index 6
            base_img = wfa_disp
                
    if current_masks is not None:
        overlay = label2rgb(np.squeeze(current_masks), image=base_img, bg_label=0, alpha=0.4, image_alpha=1)
        st.image(overlay, use_container_width=True, clamp=True)
    else:
        st.info("👈 Ajusta parámetros y segmenta para visualizar resultados.")

st.divider()

# --- Analysis Summary Dashboard ---
if os.path.exists(csv_file):
    df_metrics = pd.read_csv(csv_file)
    summary_data = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f: summary_data = json.load(f)
    
    if 'is_pv_plus' not in df_metrics.columns:
        st.info("💡 Por favor, vuelve a segmentar esta imagen para ver las nuevas estadísticas de PV.")
    else:
        st.subheader("📊 Dashboard de Resultados")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1: st.metric("Total DAPI (Núcleos)", len(df_metrics))
        with m_col2: st.metric("Total PV+ (Células)", summary_data.get('total_pv_segmentation', df_metrics['is_pv_plus'].sum()))
        with m_col3: 
            pnn_count = df_metrics['is_pnn_plus'].sum()
            st.metric("PNN+ (Redes)", pnn_count, f"{100*pnn_count/len(df_metrics):.1f}%")
        with m_col4: st.metric("Coloc (PV+/PNN+)", df_metrics[df_metrics['is_pnn_plus'] == True]['is_pv_plus'].sum())
            
        st.markdown("#### 🧪 Análisis Detallado de Colocalización")
        pv_plus_df = df_metrics[df_metrics['is_pv_plus'] == True]
        coloc_data = {
            "Población": ["Total (DAPI)", "En Redes (PNN+)", "Células PV+"],
            "n": [len(df_metrics), int(df_metrics['is_pnn_plus'].sum()), len(pv_plus_df)],
            "PV+ en Grupo": [df_metrics['is_pv_plus'].sum(), df_metrics[df_metrics['is_pnn_plus'] == True]['is_pv_plus'].sum(), len(pv_plus_df)],
            "% PV+": [f"{100*df_metrics['is_pv_plus'].mean():.1f}%", f"{100*df_metrics[df_metrics['is_pnn_plus'] == True]['is_pv_plus'].mean():.1f}%" if df_metrics['is_pnn_plus'].sum() > 0 else "0%", "100%"]
        }
        st.table(pd.DataFrame(coloc_data))
    
    st.dataframe(df_metrics.head(50))
    
    st.markdown("### 🖥️ Inspección Visual en QuPath")
    st.info("**Canales:** 1: AGR, 2: DAPI, 3: WFA, 4: PV | **Máscaras:** 5: DAPI_Mask, 6: PV_Mask, 7: PNN_Mask")
    if st.button("🌌 Abrir Imagen 7-Canales en QuPath", type="primary"):
        from subprocess import Popen, check_output
        def open_q(p):
            exe = st.session_state.get('qupath_path', r"C:\Users\danie\AppData\Local\QuPath-0.7.0\QuPath-0.7.0.exe")
            win = check_output(['wslpath', '-w', p]).decode('utf-8').strip()
            Popen(['powershell.exe', '-Command', f"& '{exe}' '{win}'"])
        open_q(seg_file)
        st.success("Abriendo QuPath...")

# (End of layout rendering for Page 2)
