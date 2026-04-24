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
Este proyecto busca automatizar el procesamiento de imágenes de <b>microscopía de fluorescencia nativa (.czi)</b> para cuantificar la densidad y morfología de las <b>Redes Perineuronales (PNNs)</b> y su asociación con la expresión de <b>ERα (Receptor de Estrógenos alfa)</b> y núcleos <b>DAPI</b> en la Corteza Somatosensorial (SSC). El flujo de trabajo incluye el procesamiento de Z-stacks de alta resolución y la validación experta mediante <b>QuPath</b>.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧠 Contexto Teórico")
    st.write("""
    - **Redes Perineuronales (PNNs):** Son estructuras especializadas de la matriz extracelular que envuelven el soma y las dendritas proximales de ciertas neuronas. Son fundamentales para la regulación de la plasticidad sináptica y la estabilización de circuitos neuronales.
    - **Interneuronas PV+:** Un subtipo de interneuronas GABAérgicas que expresan la proteína parvalbúmina. Son conocidas por su alta tasa de disparo y su papel crucial en la sincronización de ritmos corticales. En la corteza, la mayoría de las PNNs envuelven a estas células.
    """)

with col2:
    st.markdown("### 🔬 Marcadores Utilizados")
    st.write("""
    - **WFA (Wisteria Floribunda Agglutinin):** Una lectina que se une específicamente a los glicosaminoglicanos de las PNNs, permitiendo su visualización.
    - **ERα (Estrogen Receptor alpha):** Un receptor nuclear que regula la expresión génica en respuesta a estrógenos, involucrado en procesos neuroprotectores y de plasticidad.
    - **DAPI:** Colorante fluorescente que marca los núcleos celulares, permitiendo la segmentación de somas y la localización espacial de las neuronas.
    """)

st.divider()

st.markdown("### 🚀 Objetivos del Proyecto")
st.write("""
1. **Segmentación Precisa:** Utilizar *Cellpose* para identificar núcleos y somas de manera robusta.
2. **Cuantificación Morfométrica:** Medir la intensidad de fluorescencia y la integridad de las PNNs.
3. **Análisis de Asociación:** Evaluar qué proporción de células PV+ están rodeadas por PNNs y cómo varían estas poblaciones en diferentes condiciones experimentales.
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
        
        # Display basic file info (mock for now, will be real in Page 1)
        st.info(f"Archivo seleccionado: `{selected_file}`")
        st.markdown(f"""
        - **Formato:** Nativo Zeiss (.czi)
        - **Canales potenciales:** WFA, ERα, DAPI
        - **Z-Stacks:** Multi-plano detectado
        """)
        
        st.warning("🔬 **Integración con QuPath:** El análisis detallado se realizará abriendo estos archivos directamente en QuPath desde la aplicación.")
    else:
        st.warning("No se detectaron archivos `.czi` en `data/raw`. Verifica la ubicación de las imágenes.")
else:
    st.error(f"No se encontró el directorio `{raw_data_path}`. Por favor, verifica la estructura del proyecto.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado para el análisis de SSC")
