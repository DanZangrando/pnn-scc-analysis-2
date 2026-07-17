import streamlit as st
import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage import exposure, draw
from skimage.measure import regionprops
import pandas as pd
import sys
import subprocess
from pipeline import load_channels_tif

# Page configuration
st.set_page_config(page_title="Paso 1: Segmentación de Núcleos (DAPI)", layout="wide")

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

st.markdown('<div class="main-header">🧬 Paso 1: Segmentación de Núcleos (DAPI)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ajuste de parámetros y segmentación interactiva de núcleos celulares sobre el canal DAPI usando Cellpose.</div>', unsafe_allow_html=True)

RAW_DIR = "data/raw"
if not os.path.exists(RAW_DIR) or not any(os.path.isdir(os.path.join(RAW_DIR, d)) for d in os.listdir(RAW_DIR) if not d.startswith('.')):
    RAW_DIR = "data/processed/mips"

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
    st.warning("No hay grupos en raw data.")
    st.stop()

selected_group = st.sidebar.selectbox("Grupo:", groups, key="p1_group")
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No hay secciones en {selected_group}.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections, key="p1_section")
section_dir = os.path.join(group_dir, selected_section)

tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith('.tif')])
if not tif_files:
    st.warning("No hay archivos .tif en la sección.")
    st.stop()

selected_filename = st.sidebar.selectbox("Imagen:", tif_files, key="p1_file")
selected_path = os.path.join(section_dir, selected_filename)

SEGM_DIR = os.path.join(SEGM_BASE_DIR, selected_group, selected_section)
METRICS_DIR = os.path.join(METRICS_BASE_DIR, selected_group, selected_section)
os.makedirs(SEGM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load channels
try:
    (pv_raw, wfa_raw, dapi_raw, agr_raw) = load_channels_tif(selected_path)
    dapi_disp = cv2.normalize(dapi_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
except Exception as e:
    st.error(f"Error al cargar la imagen: {e}")
    st.stop()

# Sidebar - Cellpose DAPI Parameters
st.sidebar.header("⚙️ Ajustes de Cellpose DAPI")
use_gpu = st.sidebar.checkbox("Usar GPU (PyTorch)", value=True, key="p1_gpu")
filter_options = ["Ninguno", "Otsu Global", "CLAHE (Adaptativo Local)"]
dapi_def_filter = calib_data.get('cellpose_filter_type', "CLAHE (Adaptativo Local)")
dapi_filter = st.selectbox("Filtro previo DAPI", filter_options, index=filter_options.index(dapi_def_filter) if dapi_def_filter in filter_options else 0, key="p1_filter")
dapi_diam = st.number_input("Diámetro de Núcleo (px)", value=float(calib_data.get('cellpose_diameter', 30.0)), key="p1_diam", step=1.0)
dapi_flow = st.slider("Flow Threshold", 0.0, 1.0, float(calib_data.get('cellpose_flow_threshold', 0.4)), key="p1_flow")
dapi_prob = st.slider("Cell Prob Threshold", -6.0, 6.0, float(calib_data.get('cellpose_cellprob_threshold', 0.1)), key="p1_prob")

px_size = float(calib_data.get('pixel_size_um', 1.0))
pnn_radius_um = float(calib_data.get('pnn_radius_um', 20.0))

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔬 Segmentar Canal DAPI", type="primary", use_container_width=True)

base_fn, _ = os.path.splitext(selected_filename)
seg_file = os.path.join(SEGM_DIR, f"{base_fn}_masks.tif")
dapi_csv_file = os.path.join(METRICS_DIR, f"{base_fn}_dapi_metrics.csv")

if run_btn:
    # Save parameters to global configuration
    calib_data.update({
        'cellpose_filter_type': dapi_filter,
        'cellpose_diameter': dapi_diam,
        'cellpose_flow_threshold': dapi_flow,
        'cellpose_cellprob_threshold': dapi_prob
    })
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    with st.spinner("Ejecutando segmentación Cellpose para DAPI..."):
        try:
            in_dapi = dapi_raw.copy()
            if dapi_filter == "Otsu Global":
                t = threshold_otsu(in_dapi)
                in_dapi[in_dapi < t] = 0
            elif dapi_filter == "CLAHE (Adaptativo Local)":
                clahe = exposure.equalize_adapthist(in_dapi, clip_limit=0.03)
                in_dapi = (clahe * 65535).astype(np.uint16)
                
            model_dapi = models.CellposeModel(gpu=use_gpu)
            m_dapi, _, _ = model_dapi.eval(in_dapi, diameter=dapi_diam, 
                                            flow_threshold=dapi_flow, cellprob_threshold=dapi_prob)
            
            # Load or create segmented stack TIFF
            if os.path.exists(seg_file):
                try:
                    loaded = tiff.imread(seg_file)
                    num_ch = loaded.shape[0] if len(loaded.shape) == 3 else 1
                    m_pv = loaded[1, :, :] if num_ch >= 2 else np.zeros_like(m_dapi)
                    if num_ch >= 5:
                        m_wfa = loaded[4, :, :]
                    elif num_ch >= 3:
                        m_wfa = loaded[2, :, :]
                    else:
                        m_wfa = np.zeros_like(m_dapi)
                    stk = np.stack([m_dapi.astype(np.uint16),
                                    m_pv.astype(np.uint16),
                                    m_wfa.astype(np.uint16),
                                    wfa_raw.astype(np.uint16)], axis=0)
                except Exception:
                    stk = np.stack([m_dapi.astype(np.uint16),
                                    np.zeros_like(m_dapi, dtype=np.uint16),
                                    np.zeros_like(m_dapi, dtype=np.uint16),
                                    wfa_raw.astype(np.uint16)], axis=0)
            else:
                stk = np.stack([m_dapi.astype(np.uint16),
                                np.zeros_like(m_dapi, dtype=np.uint16),
                                np.zeros_like(m_dapi, dtype=np.uint16),
                                wfa_raw.astype(np.uint16)], axis=0)
                                
            tiff.imwrite(seg_file, stk, imagej=True,
                         metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                                   'Labels': ['DAPI_Mask', 'PV_Mask', 'PNN_Mask', 'WFA_Raw']})
                                   
            # Compute DAPI metrics
            dapi_props = regionprops(m_dapi, intensity_image=wfa_raw)
            # Check if PV mask exists to colocalize
            pv_mask = stk[1, :, :]
            pnn_mask = stk[2, :, :]
            
            dapi_batch = []
            for db in dapi_props:
                cy, cx = db.centroid
                is_pv_coloc = bool(pv_mask[int(cy), int(cx)] > 0)
                is_pnn = bool(pnn_mask[int(cy), int(cx)] > 0)
                
                r_px = pnn_radius_um / px_size if pnn_radius_um > 0 else 20.0 / px_size
                rd, cd = draw.disk((cy, cx), r_px, shape=wfa_raw.shape)
                wfa_sum = float(np.sum(wfa_raw[rd, cd]))
                
                dapi_batch.append({
                    'label': db.label,
                    'centroid_y': cy,
                    'centroid_x': cx,
                    'area_um2': db.area * (px_size ** 2),
                    'diameter_um': db.equivalent_diameter_area * px_size,
                    'dapi_mean_intensity': float(db.intensity_mean),
                    'wfa_sum_intensity': wfa_sum,
                    'is_pnn_plus': is_pnn,
                    'is_pv_plus': is_pv_coloc
                })
                
            df_dapi = pd.DataFrame(dapi_batch)
            if df_dapi.empty:
                df_dapi = pd.DataFrame(columns=[
                    'label', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um',
                    'dapi_mean_intensity', 'wfa_sum_intensity', 'is_pnn_plus', 'is_pv_plus'
                ])
            df_dapi.to_csv(dapi_csv_file, index=False)
            st.success("🎉 ¡Segmentación completada con éxito!")
            st.rerun()
        except Exception as e:
            st.error(f"Error durante la segmentación: {e}")

# Previsualización
st.subheader(f"Muestra seleccionada: `{selected_filename}`")
col_prev1, col_prev2 = st.columns(2)

with col_prev1:
    st.markdown('<p class="img-caption">Canal DAPI Original</p>', unsafe_allow_html=True)
    st.image(dapi_disp, width="stretch", clamp=True, channels="GRAY")

with col_prev2:
    st.markdown('<p class="img-caption">Núcleos DAPI Segmentados</p>', unsafe_allow_html=True)
    has_dapi_mask = False
    if os.path.exists(seg_file):
        try:
            loaded_masks = tiff.imread(seg_file)
            m_dapi = loaded_masks[0, :, :]
            if np.max(m_dapi) > 0:
                overlay = label2rgb(m_dapi, image=dapi_disp, bg_label=0, alpha=0.4, image_alpha=1.0)
                st.image(overlay, width="stretch", clamp=True)
                has_dapi_mask = True
        except Exception as e:
            st.error(f"Error al cargar máscara segmentada: {e}")
            
    if not has_dapi_mask:
        st.info("👈 Ajusta los parámetros y presiona '🔬 Segmentar Canal DAPI' para ver los resultados.")

# Inspección en Napari (siempre disponible si existe el archivo de máscaras)
if os.path.exists(seg_file):
    st.divider()
    st.markdown("### 🖥️ Inspección Visual en Napari")
    st.write("Visualiza la imagen con los canales biológicos originales y las máscaras segmentadas acumuladas hasta el momento.")
    if st.button("🧪 Abrir en Napari", type="primary", key="p1_btn_napari"):
        cmd = [sys.executable, "napari_viewer.py", "--path", seg_file, "--pixel_size", str(px_size), "--step", "dapi"]
        try:
            env = os.environ.copy()
            env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
            subprocess.Popen(cmd, env=env)
            st.success("✅ Visor Napari lanzado con éxito.")
        except Exception as e:
            st.error(f"Error al lanzar Napari: {e}")

# Métricas
if os.path.exists(dapi_csv_file):
    try:
        df_dapi = pd.read_csv(dapi_csv_file)
        if not df_dapi.empty:
            st.divider()
            st.subheader("📊 Descriptores de Núcleos DAPI")
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Núcleos Segmentados", f"{len(df_dapi)}")
            c_m2.metric("Área Promedio (µm²)", f"{df_dapi['area_um2'].mean():.2f}")
            c_m3.metric("Diámetro Promedio (µm)", f"{df_dapi['diameter_um'].mean():.2f}")
            
            st.markdown("### Tabla de Métricas de DAPI:")
            st.dataframe(df_dapi.head(100), use_container_width=True)
    except Exception as e:
        st.warning(f"Error al cargar los descriptores: {e}")
