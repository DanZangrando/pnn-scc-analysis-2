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

st.title("🧬 Página 2: Segmentación de Núcleos (DAPI + ERα)")

with st.expander("ℹ️ ¿Cómo funciona este pipeline de segmentación? (Documentación Técnica)", expanded=False):
    st.markdown("""
    **1. Pre-procesamiento (Filtros)**
    Antes de enviar la imagen a Cellpose, se extrae el canal DAPI y se le puede aplicar un **Filtro** (desde la barra lateral).
    - **Ninguno**: La imagen pasa cruda a Cellpose. Evita borrar núcleos tenues en esquinas oscuras.
    - **Otsu Global**: Corta el ruido de fondo fijando un único umbral para toda la imagen. (Peligro: borra núcleos en parches oscuros de escaneos panorámicos).
    - **CLAHE (Adaptativo Local)**: Ecualizador de contraste por sectores. Realza los núcleos borrosos sin importar si el parche está oscuro o iluminado. Excelente para "mosaicos".

    
    **2. Segmentación (Cellpose)**
    La imagen limpia se envía a la red neuronal Cellpose acelerada por hardware (PyTorch/GPU). La red identifica cada núcleo individual y le asigna una etiqueta numérica única (ID: 1 a N).
    - **Flow Threshold (Umbral de Flujo):** (0.0 a 1.0) Define qué tan estricta es la matemática agrupando píxeles. Valores altos separan mejor células que se están tocando fuertemente, pero pueden partir una célula real en dos mitades. (Sugerido: 0.4)
    - **Cell Prob Threshold (Probabilidad):** (-6.0 a 6.0) Define el límite de confianza para aceptar una célula. Valores negativos permiten detectar núcleos tenues, borrosos o mal iluminados. Valores positivos exigen que el núcleo sea muy evidente. (Sugerido: 0.0)

    
    **3. Exportación (4-Canales + Métricas)**
    Para evitar la generación lenta de miles de polígonos, el pipeline usa una técnica de inyección ultrarrápida:
    - **Métricas (`.csv`):** Se extraen las características físicas (área, centroide, diámetro) de cada célula. Cada fila tiene una columna `label` que representa su ID.
    - **Imagen (`_segmented.tif`):** Se crea una copia uniendo los 3 canales originales + un **4º Canal de 16-bits (Nuclei_Mask)**. La intensidad del píxel gris en el tejido es exactamente igual al `label` asignado en el CSV.
    
    **4. Visualización en QuPath (LUTs)**
    QuPath abrirá la imagen nativamente con 4 canales. 
    Como la máscara es de 16-bits, lo verás inicialmente en escala de grises. Para transformar esos grises en células independientes y coloridas **sin perder la correlación matemática**:
    1. Ve a la pestaña de *Brightness & Contrast* en QuPath.
    2. Haz doble clic en el 4º canal (`Nuclei_Mask`).
    3. Cambia la opción **LUT** de `Grayscale` a `Glasbey` o `Fire`.
    ¡Al instante tus núcleos tendrán colores separados correlacionando perfectamente con su `label` en el CSV!
    """)


# --- Paths ---
RAW_DIR = "data/raw"
MIPS_DIR = "data/processed/mips"
SEGM_DIR = "data/processed/segmented"
METRICS_DIR = "data/processed/metrics"
CONFIG_PATH = "experiment_config.json"

os.makedirs(MIPS_DIR, exist_ok=True)
os.makedirs(SEGM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# --- Load Config ---
calib_data = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)
    st.sidebar.success("✅ Configuración Global Cargada")
    st.sidebar.json(calib_data)
else:
    st.sidebar.warning("⚠️ No se encontró la configuración global (`experiment_config.json`). Modifica los parámetros en la Página 1 para generarla.")

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
    img = tiff.imread(path) # Expected (C, Y, X)
    # 0: ER, 1: WFA, 2: DAPI
    er = img[0, :, :] if img.shape[0] >= 1 else np.zeros_like(img[0])
    wfa = img[1, :, :] if img.shape[0] >= 2 else np.zeros_like(img[0])
    dapi = img[2, :, :] if img.shape[0] >= 3 else img[0, :, :]
    
    # Normalize for display
    disp_dapi = cv2.normalize(dapi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_er = cv2.normalize(er, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_wfa = cv2.normalize(wfa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return (er, wfa, dapi), (disp_er, disp_wfa, disp_dapi)

(er_raw, wfa_raw, dapi_raw), (er_disp, wfa_disp, dapi_disp) = load_channels(selected_path)

st.subheader(f"Muestra: {selected_filename.replace('_MIP.tif', '')}")

# --- Centralized Control Panel ---
with st.container():
    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        view_mode = st.radio("Capa a visualizar en el panel derecho:", 
                             ["Núcleos (DAPI)", "Estrógenos (ERα)", "Redes (PNN+)"], 
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
    
filter_type = st.sidebar.selectbox("Filtro previo en DAPI", filter_options, index=filter_options.index(default_filter), help="Otsu limpia fondos limpios pero falla en parches oscuros. CLAHE realza células ocultas en fondos oscuros.")

model_type = st.sidebar.selectbox("Modelo Base", ["cyto", "nuclei", "cyto2", "cyto3"], index=1)
diameter = st.sidebar.number_input("Diámetro del Núcleo (px)", value=float(calib_data.get('cellpose_diameter', 30.0)), step=1.0)
flow_threshold = st.sidebar.slider("Flow Threshold", 0.0, 1.0, float(calib_data.get('cellpose_flow_threshold', 0.4)))
cellprob_threshold = st.sidebar.slider("Cell Prob Threshold", -6.0, 6.0, float(calib_data.get('cellpose_cellprob_threshold', 0.0)))

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
st.sidebar.header("🧪 Segmentación de ERα")
do_er_segmentation = st.sidebar.checkbox("Activar segmentación ERα", value=True)

er_filter_type = calib_data.get('er_cellpose_filter_type', "Ninguno")
er_filter_type = st.sidebar.selectbox("Filtro previo en ERα", filter_options, index=filter_options.index(er_filter_type))
er_diameter = st.sidebar.number_input("Diámetro ERα (px)", value=float(calib_data.get('er_cellpose_diameter', 30.0)), step=1.0)
er_flow_threshold = st.sidebar.slider("Flow Threshold ERα", 0.0, 1.0, float(calib_data.get('er_cellpose_flow_threshold', 0.4)))
er_cellprob_threshold = st.sidebar.slider("Cell Prob Threshold ERα", -6.0, 6.0, float(calib_data.get('er_cellpose_cellprob_threshold', 0.0)))

if st.sidebar.button("💾 Guardar Parámetros ERα"):
    calib_data['er_cellpose_filter_type'] = er_filter_type
    calib_data['er_cellpose_diameter'] = er_diameter
    calib_data['er_cellpose_flow_threshold'] = er_flow_threshold
    calib_data['er_cellpose_cellprob_threshold'] = er_cellprob_threshold
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
    st.sidebar.success("Parámetros de ERα guardados.")

st.sidebar.divider()
st.sidebar.header("🕸️ Análisis de PNN (WFA)")
pnn_radius_um = st.sidebar.number_input("Radio de búsqueda (µm)", value=float(calib_data.get('pnn_radius_um', 20.0)), step=1.0)
pnn_threshold = st.sidebar.number_input("Umbral de Intensidad WFA (Suma)", value=float(calib_data.get('pnn_intensity_threshold', 500000.0)), step=10000.0)
pnn_exclusion_dist_um = st.sidebar.number_input("Distancia de exclusión (µm)", value=float(calib_data.get('pnn_exclusion_distance_um', 15.0)), step=1.0, help="Si hay varios núcleos PNN+, solo se queda con el mejor a esta distancia.")

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
    with st.spinner("Ejecutando Pipeline completo (DAPI + ER + PNN)..."):
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
            masks_dapi, _, _ = model_dapi.eval(input_dapi, diameter=diameter, channels=[0,0], flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
            
            # --- 2. ER Segmentation (Optional) ---
            masks_er = np.zeros_like(masks_dapi)
            if do_er_segmentation:
                input_er = er_raw.copy()
                if er_filter_type == "Otsu Global":
                    thresh = threshold_otsu(input_er)
                    input_er[input_er < thresh] = 0
                elif er_filter_type == "CLAHE (Adaptativo Local)":
                    clahe = exposure.equalize_adapthist(input_er, clip_limit=0.03)
                    input_er = (clahe * 65535).astype(np.uint16)
                
                model_er = models.CellposeModel(gpu=use_gpu, model_type="nuclei") # Reuse or specific model
                masks_er, _, _ = model_er.eval(input_er, diameter=er_diameter, channels=[0,0], flow_threshold=er_flow_threshold, cellprob_threshold=er_cellprob_threshold)

            # --- 3. PNN and Colocalization Analysis ---
            props = regionprops(masks_dapi, intensity_image=wfa_raw)
            results = []
            
            # Precompute ER indices for faster overlap check
            # For simplicity, we'll check if the centroid of DAPI is in an ER mask
            for p in props:
                label = p.label
                centroid = p.centroid # (y, x)
                
                # PNN Signal
                rr, cc = draw.disk(centroid, pnn_radius_um / px_size, shape=wfa_raw.shape)
                wfa_sum = np.sum(wfa_raw[rr, cc])
                is_pnn_plus = wfa_sum > pnn_threshold
                
                # ER Colocalization
                # Check if centroid falls into an ER mask
                is_er_plus = masks_er[int(centroid[0]), int(centroid[1])] > 0
                
                results.append({
                    'label': label,
                    'centroid_y': centroid[0],
                    'centroid_x': centroid[1],
                    'area_um2': p.area * (px_size**2),
                    'diameter_um': p.equivalent_diameter * px_size,
                    'dapi_mean_intensity': p.mean_intensity,
                    'wfa_sum_intensity': wfa_sum,
                    'is_pnn_plus': is_pnn_plus,
                    'is_er_plus': is_er_plus
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
                
                # Update original results
                for r in results:
                    if r['label'] in excluded_labels:
                        r['is_pnn_plus'] = False
            
            df = pd.DataFrame(results)
            csv_filename = selected_filename.replace('_MIP.tif', '_nuclei_metrics.csv')
            df.to_csv(os.path.join(METRICS_DIR, csv_filename), index=False)
            
            # --- 4. Global Summary Metrics ---
            total_er = int(np.max(masks_er))
            pnn_plus_count = int(df['is_pnn_plus'].sum())
            
            summary_data = {
                "total_dapi": len(df),
                "total_er_segmentation": total_er,
                "pnn_plus": pnn_plus_count,
                "pnn_minus": int(len(df) - pnn_plus_count),
                "dapi_er_coloc": int(df['is_er_plus'].sum()),
                "pixel_size": px_size
            }
            summary_filename = selected_filename.replace('_MIP.tif', '_summary.json')
            with open(os.path.join(METRICS_DIR, summary_filename), 'w') as f:
                json.dump(summary_data, f, indent=4)

            # --- 5. Save Multi-channel TIFF (6 channels) ---
            orig_mip = tiff.imread(selected_path)
            if orig_mip.shape[0] > 3:
                orig_mip = orig_mip[:3, :, :]
            
            mask_dapi_ext = np.expand_dims(masks_dapi.astype(np.uint16), axis=0)
            mask_er_ext = np.expand_dims(masks_er.astype(np.uint16), axis=0)
            
            # Robust PNN-only mask generation
            pnn_labels = df[df['is_pnn_plus'] == True]['label'].unique()
            masks_pnn = np.zeros_like(masks_dapi, dtype=np.uint16)
            # Efficiently map only PNN+ labels
            if len(pnn_labels) > 0:
                # Create a lookup table for faster mapping
                max_label = int(np.max(masks_dapi))
                lut = np.zeros(max_label + 1, dtype=np.uint16)
                for l in pnn_labels:
                    if l <= max_label: lut[int(l)] = int(l)
                masks_pnn = lut[masks_dapi.astype(int)]
                
            mask_pnn_ext = np.expand_dims(masks_pnn, axis=0)
            
            segmented_stack = np.concatenate([orig_mip, mask_dapi_ext, mask_er_ext, mask_pnn_ext], axis=0)
            seg_filename = selected_filename.replace('_MIP.tif', '_segmented.tif')
            seg_path = os.path.join(SEGM_DIR, seg_filename)
            
            ch_names = calib_data.get('channels', ['ERα', 'WFA', 'DAPI'])
            seg_names = ch_names[:3] + ['DAPI_Mask', 'ER_Mask', 'PNN_Mask']
            
            tiff.imwrite(seg_path, segmented_stack, imagej=True, metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX', 'Labels': seg_names})
            
            st.sidebar.success(f"Detección finalizada: {len(df)} núcleos ({pnn_plus_count} en PNN+)")
            
            # Fix session state
            st.session_state[f"masks_{selected_filename}"] = masks_dapi
            st.session_state[f"masks_er_{selected_filename}"] = masks_er
            st.session_state['just_segmented'] = True

        except Exception as e:
            st.error(f"Error en Pipeline: {e}")
            import traceback
            st.error(traceback.format_exc())

st.sidebar.divider()

# Batch Processing from Sidebar
st.sidebar.subheader("🚀 Procesamiento en Lote")
st.sidebar.write("Aplica la configuración a **TODOS** los archivos.")
if st.sidebar.button("Segmentar Todas las Imágenes", use_container_width=True):
    with st.spinner("Procesando lote (DAPI + ER + PNN)..."):
        success = 0
        model_dapi = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        model_er = models.CellposeModel(gpu=use_gpu, model_type="nuclei") if do_er_segmentation else None
        
        progress = st.sidebar.progress(0)
        px_size = calib_data.get('pixel_size_um', 1.0)
        
        for idx, f in enumerate(processed_files):
            try:
                p = os.path.join(MIPS_DIR, f)
                (e_raw, w_raw, d_raw), _ = load_channels(p)
                
                # 1. DAPI
                in_dapi = d_raw.copy()
                if filter_type == "Otsu Global":
                    thresh = threshold_otsu(in_dapi)
                    in_dapi[in_dapi < thresh] = 0
                elif filter_type == "CLAHE (Adaptativo Local)":
                    clahe = exposure.equalize_adapthist(in_dapi, clip_limit=0.03)
                    in_dapi = (clahe * 65535).astype(np.uint16)
                
                m_dapi, _, _ = model_dapi.eval(in_dapi, diameter=diameter, channels=[0,0], flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
                
                # 2. ER
                m_er = np.zeros_like(m_dapi)
                if do_er_segmentation:
                    in_er = e_raw.copy()
                    if er_filter_type == "Otsu Global":
                        thresh = threshold_otsu(in_er)
                        in_er[in_er < thresh] = 0
                    elif er_filter_type == "CLAHE (Adaptativo Local)":
                        clahe = exposure.equalize_adapthist(in_er, clip_limit=0.03)
                        in_er = (clahe * 65535).astype(np.uint16)
                    m_er, _, _ = model_er.eval(in_er, diameter=er_diameter, channels=[0,0], flow_threshold=er_flow_threshold, cellprob_threshold=er_cellprob_threshold)
                
                # 3. PNN/Coloc
                props_batch = regionprops(m_dapi, intensity_image=w_raw)
                res_batch = []
                for p_batch in props_batch:
                    c_batch = p_batch.centroid
                    rr_b, cc_b = draw.disk(c_batch, pnn_radius_um / px_size, shape=w_raw.shape)
                    w_sum_b = np.sum(w_raw[rr_b, cc_b])
                    res_batch.append({
                        'label': p_batch.label,
                        'centroid_y': c_batch[0],
                        'centroid_x': c_batch[1],
                        'area_um2': p_batch.area * (px_size**2),
                        'diameter_um': p_batch.equivalent_diameter * px_size,
                        'dapi_mean_intensity': p_batch.mean_intensity,
                        'wfa_sum_intensity': w_sum_b,
                        'is_pnn_plus': w_sum_b > pnn_threshold,
                        'is_er_plus': m_er[int(c_batch[0]), int(c_batch[1])] > 0
                    })
                
                # --- 3b. Spatial Exclusion (Batch) ---
                pnn_c_b = [r for r in res_batch if r['is_pnn_plus']]
                if len(pnn_c_b) > 1 and pnn_exclusion_dist_um > 0:
                    pnn_c_b.sort(key=lambda x: x['wfa_sum_intensity'], reverse=True)
                    k_c_b = []
                    ex_l_b = []
                    for cb in pnn_c_b:
                        curr_cb = (cb['centroid_y'], cb['centroid_x'])
                        too_c_b = False
                        for kc in k_c_b:
                            d_b = np.sqrt((curr_cb[0]-kc[0])**2 + (curr_cb[1]-kc[1])**2) * px_size
                            if d_b < pnn_exclusion_dist_um:
                                too_c_b = True
                                break
                        if not too_c_b:
                            k_c_b.append(curr_cb)
                        else:
                            ex_l_b.append(cb['label'])
                    for rb in res_batch:
                        if rb['label'] in ex_l_b:
                            rb['is_pnn_plus'] = False
                
                pd.DataFrame(res_batch).to_csv(os.path.join(METRICS_DIR, f.replace('_MIP.tif', '_nuclei_metrics.csv')), index=False)
                
                # 4. Save Tiff
                o_mip = tiff.imread(p)
                if o_mip.shape[0] > 3: o_mip = o_mip[:3, :, :]
                
                # Robust PNN mask mapping for batch
                pnn_labels_batch = [int(x['label']) for x in res_batch if x['is_pnn_plus']]
                m_pnn = np.zeros_like(m_dapi, dtype=np.uint16)
                if len(pnn_labels_batch) > 0:
                    max_l_b = int(np.max(m_dapi))
                    lut_b = np.zeros(max_l_b + 1, dtype=np.uint16)
                    for lb in pnn_labels_batch:
                        if lb <= max_l_b: lut_b[lb] = lb
                    m_pnn = lut_b[m_dapi.astype(int)]
                
                stk = np.concatenate([
                    o_mip, 
                    np.expand_dims(m_dapi.astype(np.uint16), axis=0), 
                    np.expand_dims(m_er.astype(np.uint16), axis=0),
                    np.expand_dims(m_pnn, axis=0)
                ], axis=0)
                
                seg_p = os.path.join(SEGM_DIR, f.replace('_MIP.tif', '_segmented.tif'))
                tiff.imwrite(seg_p, stk, imagej=True, metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX', 'Labels': ['ERα', 'WFA', 'DAPI', 'DAPI_Mask', 'ER_Mask', 'PNN_Mask']})
                
                # 5. Global Summary
                sum_data = {
                    "total_dapi": len(res_batch),
                    "total_er_segmentation": int(np.max(m_er)),
                    "pnn_plus": int(sum(x['is_pnn_plus'] for x in res_batch)),
                    "pnn_minus": int(len(res_batch) - sum(x['is_pnn_plus'] for x in res_batch)),
                    "dapi_er_coloc": int(sum(x['is_er_plus'] for x in res_batch)),
                    "pixel_size": px_size
                }
                with open(os.path.join(METRICS_DIR, f.replace('_MIP.tif', '_summary.json')), 'w') as f_sum:
                    json.dump(sum_data, f_sum, indent=4)
                
                success += 1
            except Exception as e:
                st.sidebar.error(f"Error en {f}: {e}")
            progress.progress((idx + 1) / len(processed_files))
            
        st.sidebar.success(f"Lote completado: {success}/{len(processed_files)} procesadas.")

with col2:
    st.markdown('<p class="img-caption">Visualización de Resultados</p>', unsafe_allow_html=True)
    
    seg_file = os.path.join(SEGM_DIR, selected_filename.replace('_MIP.tif', '_segmented.tif'))
    csv_file = os.path.join(METRICS_DIR, selected_filename.replace('_MIP.tif', '_nuclei_metrics.csv'))
    summary_file = os.path.join(METRICS_DIR, selected_filename.replace('_MIP.tif', '_summary.json'))
    
    current_masks = None
    base_img = dapi_disp # Fallback
    
    if os.path.exists(seg_file):
        loaded = tiff.imread(seg_file)
        if view_mode == "Núcleos (DAPI)":
            current_masks = loaded[3, :, :]
            base_img = dapi_disp
        elif view_mode == "Estrógenos (ERα)":
            if loaded.shape[0] < 5:
                st.warning("⚠️ Imagen antigua. Re-segmenta para ver ERα.")
                current_masks = None
            else:
                current_masks = loaded[4, :, :]
                base_img = er_disp
        else: # PNN+
            if os.path.exists(csv_file):
                df_temp = pd.read_csv(csv_file)
                pnn_labels = df_temp[df_temp['is_pnn_plus'] == True]['label'].values
                dapi_m = loaded[3, :, :]
                current_masks = np.where(np.isin(dapi_m, pnn_labels), dapi_m, 0)
                base_img = wfa_disp
            else:
                st.warning("Debe segmentar primero.")
                current_masks = None
                
    if current_masks is not None:
        current_masks = np.squeeze(current_masks)
        overlay = label2rgb(current_masks, image=base_img, bg_label=0, alpha=0.4, image_alpha=1)
        st.image(overlay, use_container_width=True, clamp=True)
    else:
        st.info("👈 Ajusta parámetros y segmenta para visualizar resultados.")

st.divider()

# --- Analysis Summary Dashboard ---
if os.path.exists(csv_file):
    df_metrics = pd.read_csv(csv_file)
    summary_data = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
    
    if 'is_pnn_plus' not in df_metrics.columns:
        st.info("💡 Por favor, vuelve a segmentar esta imagen para ver las nuevas estadísticas.")
    else:
        st.subheader("📊 Dashboard de Resultados")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total DAPI (Núcleos)", len(df_metrics))
        with m_col2:
            er_total = summary_data.get('total_er_segmentation', df_metrics['is_er_plus'].sum())
            st.metric("Total ERα (Células)", er_total)
        with m_col3:
            pnn_count = df_metrics['is_pnn_plus'].sum()
            st.metric("PNN+ (Redes)", pnn_count, f"{100*pnn_count/len(df_metrics):.1f}% del total")
        with m_col4:
            er_in_pnn = df_metrics[df_metrics['is_pnn_plus'] == True]['is_er_plus'].sum()
            st.metric("Coloc (DAPI+/ER+/PNN+)", er_in_pnn)
            
        st.markdown("#### 🧪 Análisis Detallado de Colocalización")
        
        pnn_plus_df = df_metrics[df_metrics['is_pnn_plus'] == True]
        pnn_minus_df = df_metrics[df_metrics['is_pnn_plus'] == False]
        
        coloc_data = {
            "Población": ["Total (Global)", "En Redes (PNN+)", "Fuera de Redes (PNN-)"],
            "n (DAPI)": [len(df_metrics), len(pnn_plus_df), len(pnn_minus_df)],
            "% del Total": ["100%", f"{100*len(pnn_plus_df)/len(df_metrics):.1f}%", f"{100*len(pnn_minus_df)/len(df_metrics):.1f}%"],
            "ER+ (Solapamiento)": [df_metrics['is_er_plus'].sum(), pnn_plus_df['is_er_plus'].sum(), pnn_minus_df['is_er_plus'].sum()],
            "% ER+ en Grupo": [
                f"{100*df_metrics['is_er_plus'].mean():.1f}%",
                f"{100*pnn_plus_df['is_er_plus'].mean():.1f}%" if len(pnn_plus_df) > 0 else "0%",
                f"{100*pnn_minus_df['is_er_plus'].mean():.1f}%" if len(pnn_minus_df) > 0 else "0%"
            ]
        }
        st.table(pd.DataFrame(coloc_data))
    
    st.markdown(f"**Detalle de Métricas (primeros 50)**")
    st.dataframe(df_metrics.head(50))
    st.caption("Mostrando los primeros 50 núcleos guardados en el archivo CSV generado.")
    
    import subprocess
    def open_mip_in_qupath(wsl_path):
        qupath_exe = st.session_state.get('qupath_path', r"C:\Users\danie\AppData\Local\QuPath-0.7.0\QuPath-0.7.0.exe")
        try:
            win_path = subprocess.check_output(['wslpath', '-w', wsl_path]).decode('utf-8', errors='replace').strip()
            ps_cmd = f"& '{qupath_exe}' '{win_path}'"
            subprocess.Popen(['powershell.exe', '-Command', ps_cmd])
            return True
        except Exception as e:
            st.error(f"Error lanzando QuPath: {e}")
            return False

    st.markdown("### 🖥️ Inspección Visual")
    st.info("**Visualización:** QuPath cargará automáticamente los **6 canales** al pulsar el botón. \
                \n- **Canal 4 (DAPI_Mask)**: Todos los núcleos. \
                \n- **Canal 5 (ER_Mask)**: Máscara de estrógenos. \
                \n- **Canal 6 (PNN_Mask)**: Solo núcleos envueltos en redes. \
                \nAsigna un LUT como *Glasbey* o *Fire* en QuPath a estos canales para verlos en color.")
    if st.button("🌌 Abrir Imagen 6-Canales en QuPath", type="primary"):
        open_mip_in_qupath(seg_file)
        st.success("Abriendo QuPath... Navega a los Canales 4 y 5 para inspeccionar las máscaras.")

# (End of layout rendering for Page 2)
