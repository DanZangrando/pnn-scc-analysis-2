import streamlit as st
import os
import json
import numpy as np
import subprocess
import cv2
import tifffile as tiff

# Page configuration
st.set_page_config(page_title="Visualización y Calibración", layout="wide")

st.title("📏 Página 1: Visualización y Calibración")
st.write("Explora las imágenes (crops 2D) y verifica la configuración de canales.")

# --- Utility Functions ---

# --- Data Loading ---

RAW_DIR = "data/raw"
if not os.path.exists(RAW_DIR) or not any(os.path.isdir(os.path.join(RAW_DIR, d)) for d in os.listdir(RAW_DIR) if not d.startswith('.')):
    RAW_DIR = "data/processed/mips"
CONFIG_PATH = "experiment_config.json"

if not os.path.exists(RAW_DIR):
    st.error(f"No se encontró la carpeta `{RAW_DIR}` ni `data/processed/mips`.")
    st.stop()

groups = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
if not groups:
    st.warning(f"No se detectaron grupos en `{RAW_DIR}`.")
    st.stop()

st.sidebar.header("📁 Selección de Grupo y Sección")
selected_group = st.sidebar.selectbox("Grupo:", groups)
group_dir = os.path.join(RAW_DIR, selected_group)

sections = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
if not sections:
    st.warning(f"No se detectaron secciones (IPSI/CONTRA) en `{group_dir}`.")
    st.stop()

selected_section = st.sidebar.selectbox("Sección:", sections)
section_dir = os.path.join(group_dir, selected_section)

tif_files = sorted([f for f in os.listdir(section_dir) if f.lower().endswith('.tif')])

if not tif_files:
    st.warning(f"No se detectaron archivos `.TIF` en `{section_dir}`.")
    st.stop()

st.sidebar.header("📂 Selección de Muestra")
selected_filename = st.sidebar.selectbox("Archivo TIF:", tif_files)
selected_path = os.path.abspath(os.path.join(section_dir, selected_filename))

# --- Visualization ---

st.subheader(f"Muestra: {selected_filename} (Grupo: {selected_group} - {selected_section})")

try:
    img_stack = tiff.imread(selected_path)
    
    with tiff.TiffFile(selected_path) as tif:
        axes = tif.series[0].axes
        
    if 'Z' in axes and len(img_stack.shape) >= 4:
        z_idx = axes.index('Z')
        img_stack = np.max(img_stack, axis=z_idx)
        axes = axes.replace('Z', '')
        
    if axes == 'YXC':
        img_stack = np.transpose(img_stack, (2, 0, 1))
        
    num_channels = img_stack.shape[0] if len(img_stack.shape) > 2 else 1
except Exception as e:
    st.error(f"Error cargando imagen: {e}")
    st.stop()

st.sidebar.header("🔬 Info del Archivo")
st.sidebar.text(f"Canales detectados: {num_channels}")
if len(img_stack.shape) >= 3:
    st.sidebar.text(f"Resolución: {img_stack.shape[2]} x {img_stack.shape[1]}")
else:
    st.sidebar.text(f"Resolución: {img_stack.shape[1]} x {img_stack.shape[0]}")

cols = st.columns(4)

channel_map = {
    0: {"name": "AGR (C1)", "color": "Magenta", "label": "AGR"},
    1: {"name": "DAPI (C2)", "color": "Blue", "label": "DAPI"},
    2: {"name": "WFA (C3)", "color": "Green", "label": "WFA"},
    3: {"name": "PV (C4)", "color": "White", "label": "PV"}
}

for i in range(min(4, num_channels)):
    if len(img_stack.shape) >= 3:
        plane = img_stack[i]
    else:
        plane = img_stack

    disp = cv2.normalize(plane, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    with cols[i]:
        st.markdown(f"**{channel_map[i]['name']}**")
        rgb = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
        if i == 0: # C1 - Magenta
            rgb[:,:,0] = disp
            rgb[:,:,2] = disp
        elif i == 1: # C2 - Blue
            rgb[:,:,2] = disp
        elif i == 2: # C3 - Green
            rgb[:,:,1] = disp
        elif i == 3: # C4 - White
            rgb[:,:,0] = disp
            rgb[:,:,1] = disp
            rgb[:,:,2] = disp
        st.image(rgb, width='stretch')

st.divider()

st.subheader("🖥️ Inspección Externa")
st.write("Abre la imagen original directamente en Napari.")

if st.button("🧪 Abrir en Napari", type="primary"):
    import subprocess
    import sys
    
    calib_data = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            calib_data = json.load(f)
    px_size = calib_data.get('pixel_size_um', 1.0)
    
    cmd = [sys.executable, "napari_viewer.py", "--path", selected_path, "--pixel_size", str(px_size)]
    try:
        env = os.environ.copy()
        env["NAPARI_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        subprocess.Popen(cmd, env=env)
        st.success("✅ Visor Napari lanzado con éxito.")
    except Exception as e:
        st.error(f"Error al lanzar Napari: {e}")

st.session_state['selected_file'] = selected_filename

