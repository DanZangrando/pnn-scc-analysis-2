import streamlit as st
import os
import sys
import json
import pandas as pd

st.set_page_config(
    page_title="Metodología: Lupori vs. Lupori-Zangrando",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght=300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    
    .hero-title {
        background: linear-gradient(120deg, #bb86fc 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 1.3rem;
        line-height: 1.5;
    }
    .method-card {
        background: rgba(30, 33, 48, 0.75);
        border: 1px solid rgba(187, 134, 252, 0.35);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }
    .card-lupori {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 2px solid #38bdf8;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .card-zangrando {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px solid #a855f7;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .qa-box {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #00f2fe;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .qa-question {
        color: #00f2fe;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .qa-answer {
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

DOC_PATH = "docs/METODOLOGIA_LUPORI_VS_ZANGRANDO.md"
PARAM_JSON_PATH = "exports/graficas_y_datos_completos/00_PARAMETROS_RUN_EXPERIMENTO.json"
if not os.path.exists(PARAM_JSON_PATH):
    PARAM_JSON_PATH = "experiment_config.json"

# Load Markdown document
doc_markdown = ""
if os.path.exists(DOC_PATH):
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        doc_markdown = f.read()
else:
    doc_markdown = "# Documento Metodológico no encontrado."

def generate_printable_html(md_content):
    # Pre-render markdown directly in Python to avoid client-side parsing issues
    from markdown_it import MarkdownIt
    import re

    svg_path = "docs/flowchart.svg"
    svg_content = ""
    if os.path.exists(svg_path):
        with open(svg_path, "r", encoding="utf-8") as sf:
            svg_content = sf.read()

    md_parser = MarkdownIt("commonmark", {"html": True, "breaks": True}).enable("table")
    body_html = md_parser.render(md_content)

    # Embed pre-rendered vector SVG into the mermaid block
    if svg_content:
        mermaid_pattern = r"<pre><code class=\"language-mermaid\">[\s\S]*?</code></pre>"
        body_html = re.sub(
            mermaid_pattern,
            f'<div class="mermaid-container">{svg_content}</div>',
            body_html
        )

    # Embed architecture diagram images as self-contained base64
    import base64
    for img_name in ["diagram_cellpose.png", "diagram_faster_rcnn.png", "diagram_pnnscore.png"]:
        img_p = os.path.join("docs/images", img_name)
        if os.path.exists(img_p):
            with open(img_p, "rb") as imf:
                b64 = base64.b64encode(imf.read()).decode("utf-8")
            body_html = body_html.replace(
                f'src="images/{img_name}"',
                f'src="data:image/png;base64,{b64}" style="max-width:100%; height:auto; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.15); margin:16px 0; border:1px solid #e2e8f0;"'
            )

    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Metodología: Cuantificación de PNN e Interneuronas PV en S1</title>

<!-- Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
    body {{
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.65;
        color: #1e293b;
        background-color: #f8fafc;
        max-width: 980px;
        margin: 40px auto;
        padding: 0 25px;
    }}
    .print-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .print-btn {{
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        color: #ffffff;
        border: none;
        padding: 9px 18px;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        font-family: inherit;
        font-size: 0.95rem;
        transition: background 0.2s ease;
    }}
    .print-btn:hover {{
        background: #3730a3;
    }}
    h1, h2, h3, h4 {{ 
        color: #0f172a; 
        font-weight: 700;
        margin-top: 1.5em;
        line-height: 1.3;
    }}
    h1 {{ 
        border-bottom: 3px solid #6366f1; 
        padding-bottom: 12px;
        color: #4338ca;
        font-size: 2.1rem;
    }}
    h2 {{ 
        border-bottom: 1px solid #cbd5e1; 
        padding-bottom: 8px;
        color: #1e1b4b;
        font-size: 1.45rem;
    }}
    h3 {{ color: #2563eb; font-size: 1.2rem; }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 25px 0;
        font-size: 0.92rem;
        background: #ffffff;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    th, td {{
        border: 1px solid #e2e8f0;
        padding: 12px 14px;
        text-align: left;
    }}
    th {{
        background-color: #ede9fe;
        color: #4338ca;
        font-weight: 700;
    }}
    tr:nth-child(even) {{ background-color: #f8fafc; }}
    blockquote {{
        border-left: 4px solid #6366f1;
        margin: 20px 0;
        padding: 14px 22px;
        background-color: #eef2ff;
        color: #312e81;
        border-radius: 0 8px 8px 0;
    }}
    code {{
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 0.88em;
    }}
    pre {{
        background-color: #0f172a;
        color: #f8fafc;
        padding: 18px;
        border-radius: 8px;
        overflow-x: auto;
    }}
    .mermaid-container {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px 16px;
        margin: 30px 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        text-align: center;
        overflow-x: auto;
    }}
    .mermaid-container svg {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }}
    .footer {{
        margin-top: 60px;
        padding-top: 25px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.85em;
        color: #64748b;
        text-align: center;
    }}
    @media print {{
        body {{ background-color: #ffffff; margin: 15px; padding: 0; }}
        .print-bar {{ display: none; }}
        .mermaid-container {{ box-shadow: none; border: 1px solid #cbd5e1; page-break-inside: avoid; }}
        table {{ page-break-inside: avoid; }}
        h1, h2, h3 {{ page-break-after: avoid; }}
    }}
</style>
</head>
<body>
<div class="print-bar">
    <span style="font-size:0.9rem; color:#475569;">📄 Documento Metodológico Oficial &bull; Defensa de Tesis (Daniel Zangrando & Rodrigo Yarzábal)</span>
    <button class="print-btn" onclick="window.print()">🖨️ Imprimir / Guardar en PDF</button>
</div>
<div id="content">
{body_html}
</div>
<div class="footer">
    Documento preparado para Defensa de Tesis / Presentación de Metodología de Investigación &bull; Daniel Zangrando & Rodrigo Yarzábal &bull; Laboratorio de Neurobiología / HistoLab &bull; 2026
</div>
</body>
</html>"""
    return html_template

# Header Section
st.markdown('<div class="hero-title">📑 Metodología Experimental: Lupori et al. vs. Metodología Adoptada</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Contraste metodológico sistemático entre el marco canónico de <b>Lupori et al. (2023)</b> y el pipeline implementado en <b>HistoLab</b> para la cuantificación sin sesgos de subpoblaciones PV+ y PV-.</div>', unsafe_allow_html=True)

# Actual Parameters KPI Cards
c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
c_kpi1.metric("📐 ROI Estandarizada", "0.3480 mm²", "348,015 µm² (Fija en S1)")
c_kpi2.metric("🧠 Segmentación PV+", "Cellpose (cpsam_v2)", "Soma Real en µm²")
c_kpi3.metric("🔬 Detección PNNs", "PNNloc + PNNscore", "Tile 256 px | Score >= -1.0")
c_kpi4.metric("⭕ Medición de Red", "Disco Circular 12 µm", "Toda la circunferencia (410.7 µm²)")

st.divider()

# Navigation Tabs
tab_summary, tab_dona_vs_disco, tab_params, tab_architectures, tab_lupori, tab_diagrams, tab_presentation, tab_defense, tab_download = st.tabs([
    "📊 Comparativa General",
    "⭕ Cuantificación de PNN: Disco Continuo",
    "⚙️ Parámetros Validados del Pipeline",
    "🧠 Arquitectura de Redes Neuronales",
    "🔬 ¿Cómo lo hizo Lupori et al.?",
    "📐 Esquema Representativo",
    "📽️ Presentación de Tesis",
    "🎓 Guía de Defensa de Tesis",
    "📥 Descargas"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: COMPARATIVA GENERAL
# ─────────────────────────────────────────────────────────────────────────────
with tab_summary:
    st.subheader("⚖️ Cuadro Comparativo del Procedimiento")
    st.write("Resumen directo de las diferencias fundamentales entre el paper original y el pipeline implementado en este laboratorio:")

    df_comp = pd.DataFrame([
        {
            "Etapa del Pipeline": "Detección de PNNs",
            "Lupori et al. (2023)": "PNNloc (Faster R-CNN) + PNNscore (Ranking ConvNet) a escala fija",
            "Metodología Adoptada (HistoLab)": "Mismas redes PNNloc + PNNscore de Lupori, pero en tiles de 256 px, escala adaptada, umbral -1.0 y NMS espacial (10 px)"
        },
        {
            "Etapa del Pipeline": "Detección de PV+",
            "Lupori et al. (2023)": "NO segmentaron somas. Faster R-CNN (PVloc) + Ranking (PVscore) arrojando un punto (X,Y) sin contorno ni área",
            "Metodología Adoptada (HistoLab)": "Cellpose (cpsam_v2) segmentando el contorno real del soma (área en µm², diámetro y máscara de instancias)"
        },
        {
            "Etapa del Pipeline": "Delimitación de la PNN",
            "Lupori et al. (2023)": "'Dona' irregular segmentada píxel a píxel mediante Random Forest y 16 filtros de Gabor en parche de 80x80 px",
            "Metodología Adoptada (HistoLab)": "Disco geométrico continuo de radio 12.0 µm (410.71 µm²) midiendo la intensidad en toda la circunferencia"
        },
        {
            "Etapa del Pipeline": "Filtro Otsu en Máscaras",
            "Lupori et al. (2023)": "No aplica (usaron Gabor)",
            "Metodología Adoptada (HistoLab)": "NO se aplicó Otsu a las máscaras finales. Se conservaron discos geométricos continuos completos"
        },
        {
            "Etapa del Pipeline": "Colocalización PV/PNN",
            "Lupori et al. (2023)": "Distancia euclidiana entre centroides <= 15 px (9.675 µm)",
            "Metodología Adoptada (HistoLab)": "Inclusión topológica celular: El centroide PNN cae dentro de la máscara del soma de Cellpose (m_pv[r,c] > 0)"
        },
        {
            "Etapa del Pipeline": "Subpoblaciones Celulares",
            "Lupori et al. (2023)": "Solo dos porcentajes globales (% PV con PNN y % PNN con PV)",
            "Metodología Adoptada (HistoLab)": "3 subpoblaciones biológicas: PV+/PNN+, PV-/PNN+ (huecas) y PV+/PNN- (desnudas)"
        },
        {
            "Etapa del Pipeline": "Estandarización Espacial",
            "Lupori et al. (2023)": "Deformación elástica de encéfalos completos al atlas 3D CCFv3",
            "Metodología Adoptada (HistoLab)": "ROI Estandarizada Invariante de Columna Cortical (Área fija = 0.348 mm²) en Napari"
        }
    ])

    st.dataframe(df_comp, hide_index=True, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: CUANTIFICACIÓN DE PNN: DISCO CONTINUO ESTANDARIZADO
# ─────────────────────────────────────────────────────────────────────────────
with tab_dona_vs_disco:
    st.subheader("⭕ Cuantificación de PNN: Parche con Random Forest vs. Disco Continuo Estandarizado")

    st.markdown("""
    Una distinción metodológica central radica en **cómo se delimita la PNN para medir su intensidad de fluorescencia**:
    """)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        <div class="card-lupori">
            <h4 style="margin-top:0; color:#38bdf8;">1. Lupori et al. (2023): Parche Cuadrado y Clasificación de Píxeles</h4>
            <p style="color:#cbd5e1; font-size:0.93rem;">
                - <b>Enfoque:</b> En un parche cuadrado de 80 × 80 px centrado en el punto detectado, aplicaron un clasificador <b>Random Forest</b> (MATLAB <i>TreeBagger</i>) con <b>16 filtros de Gabor</b> (4 frecuencias × 4 orientaciones).<br>
                - <b>Clasificación de píxeles:</b> El clasificador etiqueta cada píxel como <i>foreground</i> (célula/red) o <i>background</i> (fondo).<br>
                - <b>Medición:</b> Promediaban <i>únicamente los píxeles clasificados como célula</i>.<br>
                - <b>Consideración:</b> La superficie final de píxeles medidos depende del umbral del clasificador y del contraste local de cada parche.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="card-zangrando">
            <h4 style="margin-top:0; color:#c084fc;">2. Metodología Adoptada: Disco Circular Continuo Estandarizado</h4>
            <p style="color:#e2e8f0; font-size:0.93rem;">
                - <b>Enfoque:</b> Se establece un <b>disco geométrico circular continuo</b> (<code>draw.disk</code>) de radio físico fijo <i>R</i><sub>pnn</sub> = <b>12.0 µm</b> (13.46 px a 0.8913 µm/px), centrado en el centroide de la PNN (canal WFA), generando un área idéntica de <b>410.71 µm²</b>.<br>
                - <b>Medición en toda la circunferencia:</b> Se calcula la intensidad integrada (<code>wfa_sum</code>) y la intensidad media (<code>wfa_mean</code>) <b>en la totalidad del disco circular</b> en su rango nativo de 8 bits (0 - 255) sin Otsu.<br>
                - <b>Fundamento metodológico:</b><br>
                  1. <b>Determinístico y reproducible:</b> No depende de clasificadores de píxeles fluctuantes ni de segmentaciones locales inestables.<br>
                  2. <b>Acumulación total en el microentorno:</b> Captura la condensación pericelular y ramificaciones proximales de matriz extracelular (CSPGs / WFA).<br>
                  3. <b>Misma área para todas las células:</b> Al ser la superficie exactamente constante (410.71 µm²), las sumas e intensidades medias son directamente comparables entre condiciones experimentales.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info("""
    💡 **Fundamento Biológico y Comparación Libre de Sesgos (PV+ vs. PV-):**
    En la corteza somatosensorial, entre el 15% y el 40% de las PNNs envuelven neuronas que **no expresan parvalbúmina detectable** ("redes huecas" / PV-). Si la cuantificación dependiera del contorno de una célula o de una máscara previa de PV, las PNNs PV- no tendrían referencia somática para definirse y quedarían excluidas o medidas con una regla dispar. Al trazar un disco circular invariante centrado en el centroide de la red en el canal WFA, **tanto las redes PV+ como las PV- se cuantifican exactamente bajo la misma superficie física (410.71 µm²)**, permitiendo una comparación directa y libre de sesgos metodológicos entre ambas subpoblaciones.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PARÁMETROS VALIDADOS DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_params:
    st.subheader("⚙️ Parámetros Validados del Pipeline")
    st.write("""
    Especificaciones operativas validadas para adquisición confocal, inferencia neuronal y análisis cuantitativo:
    """)

    col_opt, col_inf = st.columns(2)

    with col_opt:
        st.markdown("#### 📐 Calibración Espacial y Óptica")
        st.markdown("""
        - `pixel_size_um = 0.8913`: Calibración confocal CZI a 10x.
        - `intensity_depth = 8 bits (contenedor TIFF 16-bit)`: Rango dinámico nativo (0 - 255) procesado sin reescalado ni compresión.
        - `area_roi_estandar_corteza_mm2 = 0.348015`: Superficie constante de columna translaminar en Napari (661 × 661 px).
        - `pnn_radius_um = 12.0`: Radio del disco circular geométrico continuo (13.46 px).
        - `pnn_disk_area = 410.71 µm²`: Superficie circular constante sobre la cual se calculan `wfa_mean` y `wfa_sum`.
        - `wfa_norm_method = "Ninguno (Raw)"`: Canal WFA procesado en su rango dinámico nativo de 8 bits (0 - 255) sin Otsu.
        """)

    with col_inf:
        st.markdown("#### 🧠 Modelos de Inferencia y Filtrado")
        st.markdown("""
        - `pv_cellpose_model = "cpsam_v2"`: Modelo híbrido U-Net + SAM ViT para interneuronas PV.
        - `pv_cellpose_diameter = 20.0 px`: Diámetro nominal de soma (≈ 17.83 µm).
        - `pv_cellpose_flow_threshold = 0.80`: Tolerancia de gradiente para reconstruir somas multipolares.
        - `pv_cellpose_cellprob_threshold = -1.50`: Umbral permisivo para captar somas tenues en 8 bits.
        - `pv_cellpose_min_size = 15 px`: Tamaño somático mínimo (≈ 11.9 µm²).
        - `lupori_tile_size = 256 px` y `lupori_tile_overlap = 64 px`: Mosaicos de inferencia para Faster R-CNN.
        - `lupori_loc_threshold = 0.20`: Sensibilidad de detección de Faster R-CNN (PNNloc).
        - `lupori_score_threshold = -1.00`: Umbral permisivo de la ConvNet de ranking (PNNscore).
        - `use_min_pnn_dist = True` y `min_pnn_dist_px = 10 px`: Spatial NMS por proximidad (≈ 8.91 µm).
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ARQUITECTURA DE REDES NEURONALES
# ─────────────────────────────────────────────────────────────────────────────
with tab_architectures:
    st.subheader("🧠 Arquitectura Técnica de las Redes Neuronales Empleadas")
    st.markdown("""
    En el pipeline coexisten dos familias de redes neuronales profundas con roles biológicos complementarios:
    la segmentación de instancias somáticas para **PV+** con Cellpose SAM v2 y el binomio de detección y puntuación de **PNNs** (Lupori et al.).
    """)

    col_nn1, col_nn2 = st.columns(2)

    with col_nn1:
        st.markdown("""
        <div class="card-zangrando">
            <h4 style="margin-top:0; color:#c084fc;">1. Cellpose SAM v2 (cpsam_v2) • Rama PV</h4>
            <p style="color:#e2e8f0; font-size:0.92rem;">
                <b>Arquitectura:</b> U-Net profunda acoplada a un encoder <i>Vision Transformer (ViT)</i> derivado del modelo fundacional <i>Segment Anything Model (SAM)</i>.<br><br>
                <b>Mecanismo Matemático:</b>
                En lugar de predecir una máscara de segmentación estática, la red predice simultáneamente tres canales de salida para cada píxel:
            </p>
            <div style="background:#090d16; border:1px solid #3b4252; border-radius:6px; padding:10px; font-family:monospace; font-size:0.85rem; color:#6ee7b7; margin-bottom:10px;">
                &Phi;(x, y) = [ V_x(x, y), &ensp; V_y(x, y), &ensp; P_cell(x, y) ]
            </div>
            <p style="color:#cbd5e1; font-size:0.90rem;">
                - <b>Campos de flujo [V<sub>x</sub>, V<sub>y</sub>]:</b> Vectores de calor que apuntan gradualmente hacia el centroide topológico del soma, resolviendo naturalmente células contiguas o en contacto.<br>
                - <b>Mapa de probabilidad P<sub>cell</sub>:</b> Logit continuo de pertenencia celular.<br>
                - <b>Post-procesamiento dinámico:</b> Integración numérica de Euler rastreando los vectores de flujo hacia los puntos de atracción convergente (somas individuales).<br>
                - <b>Calibración en S1:</b> <code>diameter = 20.0 px</code> (~17.8 µm), <code>flow_threshold = 0.80</code>, <code>cellprob_threshold = -1.50</code> sobre imágenes crudas de 8 bits (0 - 255).<br>
                - <b>Ventaja Fisiológica:</b> Delinea contornos celulares reales célula a célula, generando métricas cuantitativas de área somática en µm² y diámetro.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if os.path.exists("docs/images/diagram_cellpose.png"):
            st.image("docs/images/diagram_cellpose.png", caption="Diagrama 1: Arquitectura Cellpose SAM v2 (cpsam_v2) para Segmentación de Somas PV+", use_container_width=True)

    with col_nn2:
        st.markdown("""
        <div class="card-lupori">
            <h4 style="margin-top:0; color:#38bdf8;">2. Faster R-CNN (PNNloc) & ConvNet Ranking (PNNscore) • Rama PNN</h4>
            <p style="color:#cbd5e1; font-size:0.92rem;">
                <b>A. PNNloc (Faster R-CNN de Lupori):</b><br>
                - <b>Backbone:</b> ResNet-50 con <i>Feature Pyramid Network (FPN)</i> multinivel (P2 a P6).<br>
                - <b>Propuestas de Región:</b> RPN evalúa anclas de múltiples escalas y relaciones de aspecto sobre el canal WFA.<br>
                - <b>RoIAlign:</b> Extrae mapas de características de resolución uniforme conservando precisión espacial submétrica.<br>
                - <b>Tiling Adaptado:</b> Inferencia en tiles de 256 &times; 256 px con 64 px de overlap. Umbral de confianza = 0.20.<br><br>
                <b>B. PNNscore (ConvNet de Ranking):</b><br>
                - <b>Entrada:</b> Parches normalizados de 64 &times; 64 px centrados en cada candidato de PNNloc.<br>
                - <b>Capas:</b> 5 bloques convolucionales (Conv2D 3&times;3 &rarr; BatchNorm &rarr; ReLU &rarr; MaxPool 2&times;2).<br>
                - <b>Cabezal:</b> Global Average Pooling &rarr; Capa lineal con pérdida <i>Margin Ranking Loss</i>.<br>
                - <b>Umbral Activo:</b> Score &ge; -1.00.<br><br>
                <b>C. Spatial NMS por Distancia (< 10 px):</b><br>
                Filtro euclidiano diseñado por Daniel Zangrando para suprimir duplicados generados en el solapamiento de tiles, conservando la detección de mayor score.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if os.path.exists("docs/images/diagram_faster_rcnn.png"):
            st.image("docs/images/diagram_faster_rcnn.png", caption="Diagrama 2: Faster R-CNN (PNNloc de Lupori et al.)", use_container_width=True)
        if os.path.exists("docs/images/diagram_pnnscore.png"):
            st.image("docs/images/diagram_pnnscore.png", caption="Diagrama 3: ConvNet de Ranking (PNNscore) y Spatial NMS (Zangrando)", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: ¿CÓMO LO HIZO LUPORI ET AL.?
# ─────────────────────────────────────────────────────────────────────────────
with tab_lupori:
    st.markdown("""
    <div class="card-lupori">
        <h3 style="margin-top:0; color:#38bdf8;">🔬 Paper Canónico: Lupori et al. (Cell Reports 2023, 42:112788)</h3>
        <p style="color:#cbd5e1; font-size:0.95rem;">
            Lo que presentará <b>Rodrigo Yarzábal</b>: metodología publicada para el atlas cerebral macroscópico CCFv3.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_l1, c_l2 = st.columns(2)
    with c_l1:
        st.markdown("#### ¿Cómo detectaron las células PV? (Detalle Clave)")
        st.warning("""
        **Lupori et al. NUNCA segmentaron los somas de las células PV.**
        1. Emplearon una red Faster R-CNN (`PVloc`) entrenada con anotaciones de un solo evaluador con puntos discretos (X, Y) de un solo píxel (101,348 células PV).
        2. Luego, una red de ranking (`PVscore`) evaluó parches de 64 × 64 px y descartó detecciones con score < 0.55.
        3. Para medir intensidad celular, extrajeron recortes de 80 × 80 px y aplicaron un clasificador de píxeles Random Forest en MATLAB (*TreeBagger*) con **16 filtros de Gabor** para clasificar píxel por píxel como célula o fondo.
        4. **Limitación de Lupori:** Nunca obtuvieron una máscara poligonal de la célula. No tenían área somática (µm²), ni diámetro somático, ni forma celular.
        """)

    with c_l2:
        st.markdown("#### ¿Cómo detectaron las PNNs y la Colocalización?")
        st.info("""
        - Entrenaron una Faster R-CNN con ResNet-50 FPN (`PNNloc`) sobre 678,556 PNNs dot-annotadas.
        - Luego evaluaron con una red de ranking convolucional (`PNNscore`) sobre parches de 64 × 64 px (score ≥ 0.40).
        - **Colocalización:** Como ambas detecciones eran puntos (X, Y), definieron colocalización si la distancia euclidiana entre el punto PV y el punto PNN era ≤ 15 px (9.675 µm).
        - **Fluorescencia Difusa:** Promedio de todos los píxeles de una región del atlas dividida por la media de todo el encéfalo.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: ESQUEMA REPRESENTATIVO
# ─────────────────────────────────────────────────────────────────────────────
with tab_diagrams:
    st.subheader("📐 Flujograma del Procedimiento Final ('Lupori, Zangrando')")
    
    st.markdown("""
    ```mermaid
    flowchart TD
        subgraph Entrada["1. Entrada y Calibración"]
            A["Z-stack Confocal 8-bit\n(DAPI, PV546, WFA647)"] --> B["MIP Automatizado"]
            B --> C["Calibración CZI:\n0.8913 µm/px a 10x"]
        end

        subgraph ROI["2. Estandarización Espacial"]
            C --> D["Alineación en Napari:\nROI Estandarizada Invariante = 0.348015 mm²"]
        end

        subgraph RamaPV["3. Rama PV: Cellpose (Daniel Zangrando)"]
            B --> E["Canal PV546 Raw"]
            E --> F["Cellpose (cpsam_v2)\nRaw 8-bit | Diam: 20 px | Flow: 0.80 | Prob: -1.5"]
            F --> G["Máscara de Somas (m_pv)\nÁrea Somática (µm²), Diámetro"]
        end

        subgraph RamaPNN["4. Rama PNN: Redes de Lupori"]
            B --> H["Canal WFA647 Raw"]
            H --> I["PNNloc de Lupori (Faster R-CNN)\nTiles 256 px | Overlap 64 px | Umbral 0.20"]
            I --> J["PNNscore de Lupori (ConvNet Ranking)\nParches 64x64 px | Umbral Score >= -1.0"]
            J --> K["Spatial NMS por Distancia (< 10 px)\n(Supresión de Duplicados Zangrando)"]
        end

        subgraph Biomodelo["5. Cuantificación de Red (Daniel Zangrando)"]
            G & K --> L["Disco Geométrico Continuo (Radio 12 µm = 410.7 µm²)\nCentrado en Centroide PNN (Canal WFA)\nMedición en toda la circunferencia (wfa_mean, wfa_sum)\n(SIN filtro Otsu ni anillos de erosión)"]
            L --> M["Colocalización Topológica:\n¿Centroide PNN dentro de Máscara Soma PV?"]
            M --> N["3 Subpoblaciones Biológicas:\n1. PV+/PNN+ (Consolidada)\n2. PV-/PNN+ (Red Hueca)\n3. PV+/PNN- (Interneurona Desnuda)"]
            L --> O["Fondo Difuso de Neuropilo (M_wfa == 0)"]
        end

        subgraph Salidas["6. Salidas y Validación"]
            N & O --> P["Auditoría Visual en Napari\n(TIFF de 6 Canales con PV_in_PNN_Mask)"]
            N & O --> Q["Conteo Manual por Tiles (Página 3)\n(MARE y Bland-Altman vs Humano)"]
            N & O --> R["Bioestadística Experimental (Página 4):\nShapiro-Wilk, Mann-Whitney U, Two-Way ANOVA\ny Matrices GraphPad Prism"]
        end
    ```
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: PRESENTACIÓN DE TESIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_presentation:
    st.subheader("📽️ Presentación Interactiva de Metodología para Defensa de Tesis")
    st.markdown("""
    Presentación interactiva completa diapositiva a diapositiva explicando el marco experimental, las arquitecturas neuronales,
    la delimitación por disco continuo de 12 µm y las respuestas clave para el jurado.
    """)

    col_pbtn1, col_pbtn2 = st.columns([1, 4])
    with col_pbtn1:
        if os.path.exists("docs/PRESENTACION_METODOLOGIA.html"):
            with open("docs/PRESENTACION_METODOLOGIA.html", "r", encoding="utf-8") as pres_f:
                pres_html_data = pres_f.read()
            st.download_button(
                label="⬇️ Descargar Presentación (.html)",
                data=pres_html_data.encode("utf-8"),
                file_name="PRESENTACION_METODOLOGIA_TESIS.html",
                mime="text/html",
                use_container_width=True
            )
    with col_pbtn2:
        st.caption("Tip: Abre el archivo descargado en cualquier navegador para ver las diapositivas en pantalla completa con atajos de teclado (← / →, F para pantalla completa, P para PDF).")

    # Render presentation in an iframe
    if os.path.exists("docs/PRESENTACION_METODOLOGIA.html"):
        import streamlit.components.v1 as components
        with open("docs/PRESENTACION_METODOLOGIA.html", "r", encoding="utf-8") as pres_f:
            pres_content = pres_f.read()
        components.html(pres_content, height=680, scrolling=True)
    else:
        st.warning("El archivo `docs/PRESENTACION_METODOLOGIA.html` no fue encontrado.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8: GUÍA DE DEFENSA DE TESIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_defense:
    st.subheader("🎓 Preguntas Probables del Jurado y Respuestas Clave")

    st.markdown("""
    <div class="qa-box">
        <div class="qa-question">P1: ¿Cuál es la diferencia entre cómo midió la intensidad Lupori et al. y cómo lo midieron ustedes?</div>
        <div class="qa-answer">
            <b>Respuesta Clave:</b> Lupori et al. aplicaron en cada parche de 80 × 80 px un clasificador de píxeles Random Forest con 16 filtros de Gabor para clasificar píxeles como célula o fondo, promediando únicamente los píxeles etiquetados como célula.<br>
            Nosotros optamos por un <b>disco geométrico continuo de radio fijo 12.0 µm (410.71 µm²)</b> centrado en el centroide de la red detectada en el canal WFA, midiendo la intensidad integrada y media en toda la circunferencia. Esta decisión metodológica garantiza una <b>comparación libre de sesgos entre subpoblaciones PV+ y PV- (redes huecas)</b>: dado que las PNNs que rodean neuronas PV- no cuentan con una máscara somática previa, el disco continuo permite cuantificar ambos grupos exactamente bajo la misma superficie física y sin recurrir a filtros de umbralización inestables.
        </div>
    </div>
    
    <div class="qa-box">
        <div class="qa-question">P2: ¿Ustedes utilizaron las redes neuronales de Lupori et al. o entrenaron sus propias redes?</div>
        <div class="qa-answer">
            <b>Respuesta Clave:</b> Para la detección de las redes perineuronales en el canal WFA utilizamos exactamente las redes convolucionales preentrenadas por Lupori et al. (<code>PNNloc</code> Faster R-CNN y <code>PNNscore</code> ConvNet de ranking), aprovechando los pesos entrenados sobre más de 670,000 redes y validados por 7 evaluadores expertos.<br>
            Adaptamos su inferencia dividiendo la imagen en mosaicos de 256 px con solapamiento de 64 px, establecimos un umbral permisivo de score en -1.00 para capturar redes intermedias, e implementamos un algoritmo de supresión espacial no máxima por proximidad euclidiana (10 px) para evitar contar duplicados en los bordes de los tiles.
        </div>
    </div>

    <div class="qa-box">
        <div class="qa-question">P3: ¿Cómo hubiera detectado Lupori las células PV y por qué usaron Cellpose?</div>
        <div class="qa-answer">
            <b>Respuesta Clave:</b> Lupori no segmentaba los somas de las células PV; usaron un detector Faster R-CNN (<code>PVloc</code>) que arrojaba únicamente una coordenada puntual (X, Y) de un solo píxel y una red de ranking (<code>PVscore</code>). Luego, para la intensidad, aplicaban el clasificador Random Forest en un parche de 80 × 80 px.<br>
            Ese método no produce contornos celulares ni áreas somáticas. Por eso, nosotros utilizamos <b>Cellpose</b> (<code>cpsam_v2</code>), que delinea con campos de flujo vectorial la membrana somática de cada interneurona, permitiéndonos reportar el área somática real (µm²) para evaluar cambios tróficos somáticos y determinar la colocalización por inclusión topológica del centroide de la red dentro del perímetro somático real.
        </div>
    </div>

    <div class="qa-box">
        <div class="qa-question">P4: ¿Por qué usaron una ROI fija de 0.348 mm² en vez de registrar al atlas CCFv3?</div>
        <div class="qa-answer">
            <b>Respuesta Clave:</b> El registro elástico al atlas CCFv3 requiere cortes coronales enteros con referencias anatómicas de todo el encéfalo. En microscopía confocal a 10x enfocada en la corteza somatosensorial local, deformar elásticamente el tejido introduce distorsiones en la escala física de las células.<br>
            Diseñamos una ROI de columna cortical con un área estrictamente constante de 0.348015 mm² que abarca todas las capas de S1. Al fijar rígidamente esta superficie en Napari en todas las muestras, cualquier variación en el conteo celular representa una diferencia biológica real de densidad, libre de artefactos por área de corte o retracción tisular.
        </div>
    </div>
    """, unsafe_allow_html=True)
