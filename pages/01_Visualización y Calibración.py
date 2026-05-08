import streamlit as st
import os
import json
import numpy as np
import subprocess
import sys
from aicspylibczi import CziFile
import xml.etree.ElementTree as ET
import cv2
from tifffile import imwrite

# Page configuration
st.set_page_config(page_title="Visualización y Calibración", layout="wide")

st.title("📏 Página 1: Visualización y Calibración (CZI)")
st.write("Explora las imágenes nativas, verifica la calibración y genera proyecciones (MIP) para el análisis.")

# --- Utility Functions ---

def get_czi_metadata(path):
    czi = CziFile(path)
    shape = czi.get_dims_shape()[0]
    root = czi.meta 
    if isinstance(root, str):
        root = ET.fromstring(root)
    
    # Scaling (usually in meters in CZI XML)
    pixel_size = 1.0
    z_spacing = 1.0
    scaling = root.find(".//Scaling")
    if scaling is not None:
        # Distance Id='X'
        dist_x = scaling.find(".//Distance[@Id='X']")
        if dist_x is not None:
            val = dist_x.find("Value")
            if val is not None:
                v = float(val.text)
                pixel_size = v * 1e6 if v < 0.001 else v
                    
        # Distance Id='Z'
        dist_z = scaling.find(".//Distance[@Id='Z']")
        if dist_z is not None:
            val = dist_z.find("Value")
            if val is not None:
                v = float(val.text)
                z_spacing = v * 1e6 if v < 0.001 else v
                
    # Channels
    channels = []
    ch_elements = root.findall(".//Channel")
    num_channels = shape.get('C', (0, 1))[1]
    for i in range(num_channels):
        if i < len(ch_elements):
            name = ch_elements[i].get('Name', f"Ch{i}")
        else:
            name = f"Ch{i}"
        channels.append(name)

    # Dimensions
    if 'M' in czi.dims:
        bbox = czi.get_mosaic_bounding_box()
        width = bbox.w
        height = bbox.h
    else:
        width = shape.get('X', (0, 0))[1]
        height = shape.get('Y', (0, 0))[1]

    return {
        "pixel_size": pixel_size,
        "z_spacing": z_spacing,
        "channels": channels,
        "width": width,
        "height": height,
        "shape": shape
    }

def get_mip_preview(path, channel_idx, scale=0.25):
    czi = CziFile(path)
    # Read all Z for the given channel
    # dims return is HSTCZMYX
    dims = czi.dims
    shape = czi.get_dims_shape()[0]
    z_range = shape.get('Z', (0, 1))
    
    planes = []
    for z in range(z_range[0], z_range[1]):
        # read_mosaic or read_image? 
        # If it's a mosaic, read_mosaic is better
        if 'M' in dims:
            img = czi.read_mosaic(C=channel_idx, Z=z, scale_factor=scale)
        else:
            img, _ = czi.read_image(C=channel_idx, Z=z)
        planes.append(img.squeeze())
    
    # Max projection
    mip = np.max(np.stack(planes), axis=0)
    return mip

def open_in_qupath(wsl_path, qupath_exe):
    try:
        # 1. Convert WSL path to Windows UNC path
        win_path = subprocess.check_output(['wslpath', '-w', wsl_path]).decode('utf-8', errors='replace').strip()
        
        # 2. Prepare PowerShell command
        ps_cmd = f"& '{qupath_exe}' '{win_path}'"
        
        # 3. Launch and capture error
        process = subprocess.Popen(
            ['powershell.exe', '-Command', ps_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        # Wait a bit to see if it fails immediately
        try:
            stdout, stderr = process.communicate(timeout=3)
            if process.returncode != 0:
                st.error(f"Error de PowerShell: {stderr}")
                return False
        except subprocess.TimeoutExpired:
            # If it takes more than 3s, it's likely QuPath is opening
            pass
            
        return True
    except Exception as e:
        st.error(f"Error local: {e}")
        return False

# --- Data Loading ---

# --- Data Loading ---

RAW_DIR = "data/raw"
PROCESSED_BASE_DIR = "data/processed/mips"
CONFIG_PATH = "experiment_config.json"

if not os.path.exists(RAW_DIR):
    st.error(f"No se encontró la carpeta `{RAW_DIR}`.")
    st.stop()

# Get groups (subdirectories)
groups = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
if not groups:
    # If no subdirectories, fall back to root or show error
    groups = ["."] 

st.sidebar.header("📁 Selección de Grupo")
selected_group = st.sidebar.selectbox("Grupo:", groups)
group_dir = os.path.join(RAW_DIR, selected_group)

# Update PROCESSED_DIR to include group
PROCESSED_DIR = os.path.join(PROCESSED_BASE_DIR, selected_group)
os.makedirs(PROCESSED_DIR, exist_ok=True)

czi_files = sorted([f for f in os.listdir(group_dir) if f.endswith('.czi')])

if not czi_files:
    st.warning(f"No se detectaron archivos `.czi` en `{group_dir}`.")
    st.stop()

# --- Sidebar ---

st.sidebar.header("📂 Selección de Muestra")
selected_filename = st.sidebar.selectbox("Archivo CZI:", czi_files)
selected_path = os.path.abspath(os.path.join(group_dir, selected_filename))

# Metadata
meta = get_czi_metadata(selected_path)

st.sidebar.header("📏 Calibración")
calib = st.sidebar.number_input("Píxel size (µm):", value=meta['pixel_size'], format="%.4f")
z_space = st.sidebar.number_input("Z-spacing (µm):", value=meta['z_spacing'], format="%.4f")
st.session_state['pixel_size'] = calib
st.session_state['z_spacing'] = z_space

st.sidebar.divider()
if st.sidebar.button("💾 Guardar Calibración Global"):
    # Updated to 4 channels based on new project
    standard_channels = ["AGR (AF488)", "DAPI", "WFA (AF647/FarRed)", "PV (AF546)"]
    
    config_data = {
        "pixel_size_um": calib,
        "z_spacing_um": z_space,
        "channels": standard_channels
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=4)
    st.sidebar.success("✅ Configuración guardada en `experiment_config.json` para todo el experimento.")


st.sidebar.header("🔬 Info del Archivo")
st.sidebar.text(f"Canales: {len(meta['channels'])}")
st.sidebar.text(f"Z-slices: {meta['shape'].get('Z', (0, 1))[1]}")
st.sidebar.text(f"Resolución Total: {meta['width']} x {meta['height']}")

# --- Main Page ---

st.subheader(f"Muestra: {selected_filename} (Grupo: {selected_group})")

# Create 4 columns for the 4 channels
cols = st.columns(4)

# Updated channel_map for the new configuration
# 0: AGR, 1: DAPI, 2: WFA, 3: PV
channel_map = {
    0: {"name": "AGR (C1)", "color": "Magenta", "label": "AGR"},
    1: {"name": "DAPI (C2)", "color": "Blue", "label": "DAPI"},
    2: {"name": "WFA (C3)", "color": "Green", "label": "WFA"},
    3: {"name": "PV (C4)", "color": "White", "label": "PV"}
}

previews = {}

with st.spinner("Generando Proyecciones (MIP)..."):
    for i in range(min(4, len(meta['channels']))):
        mip = get_mip_preview(selected_path, i)
        # Normalize for display
        disp = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        previews[i] = mip # Keep original for saving
        
        target_col = cols[i]
        with target_col:
            st.markdown(f"**{channel_map[i]['name']}**")
            # Create RGB for colored display
            rgb = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
            if i == 0: # C1 - Magenta (R+B)
                rgb[:,:,0] = disp
                rgb[:,:,2] = disp
            elif i == 1: # C2 - Blue
                rgb[:,:,2] = disp
            elif i == 2: # C3 - Green
                rgb[:,:,1] = disp
            elif i == 3: # C4 - White (R+G+B)
                rgb[:,:,0] = disp
                rgb[:,:,1] = disp
                rgb[:,:,2] = disp
            st.image(rgb, width='stretch')

st.divider()

# --- Actions ---
act_col1, act_col2 = st.columns(2)

with act_col1:
    st.subheader("💾 Procesamiento")
    if st.button("Procesar Solo Esta Imagen (MIP)"):
        with st.spinner("Procesando MIP a resolución completa (4 canales)..."):
            final_planes = []
            try:
                for i in range(4):
                    mip_full = get_mip_preview(selected_path, i, scale=1.0)
                    final_planes.append(mip_full)
                
                # Save multi-channel TIFF
                out_name = selected_filename.replace('.czi', '_MIP.tif')
                out_path = os.path.join(PROCESSED_DIR, out_name)
                
                # Stack: (C, Y, X)
                stack = np.stack(final_planes)
                labels = ["AGR (AF488)", "DAPI", "WFA (AF647)", "PV (AF546)"]
                imwrite(out_path, stack, imagej=True, metadata={'spacing': calib, 'unit': 'um', 'Axes': 'CYX', 'Labels': labels})

                
                st.success(f"MIP guardado con éxito en: `{out_path}`")
                st.session_state['processed_mip'] = out_path
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

    st.write("---")
    
    if st.button("🚀 Procesar TODAS las Imágenes del Grupo (Batch MIPS)"):
        with st.spinner(f"Procesando {len(czi_files)} imágenes. Esto puede tardar varios minutos..."):
            success_count = 0
            for file in czi_files:
                try:
                    p = os.path.join(group_dir, file)
                    planes = [get_mip_preview(p, i, scale=1.0) for i in range(min(4, len(meta['channels'])))]
                    stack = np.stack(planes)
                    out_name = file.replace('.czi', '_MIP.tif')
                    labels = ["AGR (AF488)", "DAPI", "WFA (AF647)", "PV (AF546)"]
                    imwrite(os.path.join(PROCESSED_DIR, out_name), stack, imagej=True, metadata={'spacing': calib, 'unit': 'um', 'Axes': 'CYX', 'Labels': labels})
                    success_count += 1
                except Exception as e:
                    st.error(f"Error procesando {file}: {e}")
            st.success(f"¡Batch completado! {success_count}/{len(czi_files)} imágenes analizadas y guardadas en {PROCESSED_DIR}.")

with act_col2:
    st.subheader("🖥️ Inspección Externa")
    st.write("Abre las imágenes en tu visor local QuPath.")
    
    qupath_exe = st.session_state.get('qupath_path', r"C:\Users\danie\AppData\Local\QuPath-0.7.0\QuPath-0.7.0.exe")
    
    if st.button("🌌 Abrir Archivo Original en QuPath"):
        if open_in_qupath(selected_path, qupath_exe):
            st.success("✅ Abriendo CZI nativo en QuPath.")
            
    mip_name = selected_filename.replace('.czi', '_MIP.tif')
    mip_path = os.path.abspath(os.path.join(PROCESSED_DIR, mip_name))
    
    st.write("") # Spacer
    if os.path.exists(mip_path):
        if st.button("🖼️ Abrir MIP Generado en QuPath", type="primary"):
            if open_in_qupath(mip_path, qupath_exe):
                st.success("✅ Abriendo Proyección TIFF en QuPath.")

# Save state
st.session_state['selected_file'] = selected_filename
