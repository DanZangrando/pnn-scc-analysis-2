import streamlit as st
import os
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

# Sidebar
st.sidebar.title("⚙️ Configuración")

default_qupath = st.session_state.get('qupath_path', r"C:\Users\danie\AppData\Local\QuPath-0.7.0\QuPath-0.7.0.exe")
qupath_path = st.sidebar.text_input("Ruta de QuPath.exe (Windows):", value=default_qupath)
st.session_state['qupath_path'] = qupath_path
st.sidebar.divider()
st.sidebar.title("Navegación")
st.sidebar.info("Este es el punto de inicio del análisis de PNNs y ERα en la Corteza Somatosensorial.")

# Home Page Content
st.title("PNN SSC Analysis 🧠🔬")
st.subheader("Análisis Automatizado de Redes Perineuronales e Interneuronas PV+")

st.markdown("""
<div class="status-box">
<h3>📋 Descripción del Proyecto</h3>
Este proyecto busca automatizar el procesamiento de imágenes de <b>microscopía de fluorescencia nativa (.czi)</b> para cuantificar la densidad y morfología de las <b>Redes Perineuronales (PNNs)</b> y su asociación con las interneuronas <b>PV+ (Parvalbúmina)</b> y núcleos <b>DAPI</b> en la Corteza Somatosensorial (SSC). El flujo de trabajo incluye el procesamiento de Z-stacks de 4 canales y la validación experta mediante <b>QuPath</b>.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧠 Contexto Teórico")
    st.write("""
    - **Redes Perineuronales (PNNs):** Estructuras de la matriz extracelular que envuelven neuronas específicas, regulando la plasticidad y protegiendo contra el estrés oxidativo.
    - **Interneuronas PV+:** Neuronas GABAérgicas de disparo rápido (fast-spiking) que expresan la proteína parvalbúmina. Son el blanco principal de las PNNs en la corteza.
    """)

with col2:
    st.markdown("### 🔬 Marcadores y Canales")
    st.write("""
    - **C1: AF488 (AGR):** Marcador adicional de referencia.
    - **C2: DAPI:** Marcador nuclear para la identificación y segmentación de somas celulares.
    - **C3: WFA (AF647):** Lectina para la visualización de las Redes Perineuronales (PNN+).
    - **C4: PV (AF546):** Anticuerpo contra Parvalbúmina para identificar interneuronas específicas.
    """)

st.divider()

st.markdown("### 🚀 Objetivos del Proyecto")
st.write("""
1. **Segmentación de Núcleos:** Identificación individual de células mediante DAPI.
2. **Detección de PV+:** Clasificación de neuronas según su expresión de Parvalbúmina.
3. **Análisis de PNNs:** Cuantificación de la intensidad y presencia de redes rodeando a las células PV+.
4. **Relación Espacial:** Evaluación de la proporción de células PV envueltas por PNN en la SSC.
""")

st.divider()

st.markdown("### 📂 Explorador de Datos")
raw_data_path = "data/raw"

if os.path.exists(raw_data_path):
    # Search for CZI files in data/raw
    czi_files = sorted([f for f in os.listdir(raw_data_path) if f.endswith('.czi')])
    
    if czi_files:
        st.write(f"Se han detectado **{len(czi_files)}** archivos `.czi` nativos.")
        selected_file = st.selectbox("Selecciona una imagen nativa:", czi_files)
        
        # Display basic file info
        st.info(f"Archivo seleccionado: `{selected_file}`")
        st.markdown(f"""
        - **Formato:** Nativo Zeiss (.czi)
        - **Configuración de Canales:** 4 Canales (AGR, DAPI, WFA, PV)
        - **Z-Stacks:** Multi-plano detectado
        """)
        
        st.warning("🔬 **Integración con QuPath:** El análisis detallado se realizará abriendo estos archivos directamente en QuPath desde la aplicación.")
    else:
        st.warning("No se detectaron archivos `.czi` en `data/raw`. Verifica la ubicación de las imágenes.")
else:
    st.error(f"No se encontró el directorio `{raw_data_path}`. Por favor, verifica la estructura del proyecto.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado para el análisis de SSC")
