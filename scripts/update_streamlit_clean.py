#!/usr/bin/env python3
"""
Clean Tab 2, Tab 3, Tab 4, and Tab 8 in pages/06_Metodología_Lupori_vs_Zangrando.py.
"""
import re

with open("pages/06_Metodología_Lupori_vs_Zangrando.py", "r", encoding="utf-8") as f:
    code = f.read()

# ─────────────────────────────────────────────────────────────────────────────
# 1. TAB 2 REPLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
tab2_old = re.search(r'(# ─+\s+# TAB 2:.*?)(?=# ─+\s+# TAB 3:)', code, re.DOTALL).group(1)

tab2_new = """# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: CUANTIFICACIÓN DE PNN: DISCO CONTINUO ESTANDARIZADO
# ─────────────────────────────────────────────────────────────────────────────
with tab_dona_vs_disco:
    st.subheader("⭕ Cuantificación de PNN: Parche con Random Forest vs. Disco Continuo Estandarizado")

    st.markdown(\"\"\"
    Una distinción metodológica central radica en **cómo se delimita la PNN para medir su intensidad de fluorescencia**:
    \"\"\")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(\"\"\"
        <div class="card-lupori">
            <h4 style="margin-top:0; color:#38bdf8;">1. Lupori et al. (2023): Parche Cuadrado y Clasificación de Píxeles</h4>
            <p style="color:#cbd5e1; font-size:0.93rem;">
                - <b>Enfoque:</b> En un parche cuadrado de 80 × 80 px centrado en el punto detectado, aplicaron un clasificador <b>Random Forest</b> (MATLAB <i>TreeBagger</i>) con <b>16 filtros de Gabor</b> (4 frecuencias × 4 orientaciones).<br>
                - <b>Clasificación de píxeles:</b> El clasificador etiqueta cada píxel como <i>foreground</i> (célula/red) o <i>background</i> (fondo).<br>
                - <b>Medición:</b> Promediaban <i>únicamente los píxeles clasificados como célula</i>.<br>
                - <b>Consideración:</b> La superficie final de píxeles medidos depende del umbral del clasificador y del contraste local de cada parche.
            </p>
        </div>
        \"\"\", unsafe_allow_html=True)

    with col_m2:
        st.markdown(\"\"\"
        <div class="card-zangrando">
            <h4 style="margin-top:0; color:#c084fc;">2. Metodología Adoptada: Disco Circular Continuo Estandarizado</h4>
            <p style="color:#e2e8f0; font-size:0.93rem;">
                - <b>Enfoque:</b> Se establece un <b>disco geométrico circular continuo</b> (<code>draw.disk</code>) de radio físico fijo <i>R</i><sub>pnn</sub> = <b>12.0 µm</b> (13.46 px a 0.8913 µm/px), centrado en el centroide de la PNN (canal WFA), generando un área idéntica de <b>410.71 µm²</b>.<br>
                - <b>Medición en toda la circunferencia:</b> Se calcula la intensidad integrada (<code>wfa_sum</code>) y la intensidad media (<code>wfa_mean</code>) <b>en la totalidad del disco circular</b> en 16 bits crudos sin Otsu.<br>
                - <b>Fundamento metodológico:</b><br>
                  1. <b>Determinístico y reproducible:</b> No depende de clasificadores de píxeles fluctuantes ni de segmentaciones locales inestables.<br>
                  2. <b>Acumulación total en el microentorno:</b> Captura la condensación pericelular y ramificaciones proximales de matriz extracelular (CSPGs / WFA).<br>
                  3. <b>Misma área para todas las células:</b> Al ser la superficie exactamente constante (410.71 µm²), las sumas e intensidades medias son directamente comparables entre condiciones experimentales.
            </p>
        </div>
        \"\"\", unsafe_allow_html=True)

    st.info(\"\"\"
    💡 **Fundamento Biológico y Comparación Libre de Sesgos (PV+ vs. PV-):**
    En la corteza somatosensorial, entre el 15% y el 40% de las PNNs envuelven neuronas que **no expresan parvalbúmina detectable** ("redes huecas" / PV-). Si la cuantificación dependiera del contorno de una célula o de una máscara previa de PV, las PNNs PV- no tendrían referencia somática para definirse y quedarían excluidas o medidas con una regla dispar. Al trazar un disco circular invariante centrado en el centroide de la red en el canal WFA, **tanto las redes PV+ como las PV- se cuantifican exactamente bajo la misma superficie física (410.71 µm²)**, permitiendo una comparación directa y libre de sesgos metodológicos entre ambas subpoblaciones.
    \"\"\")

"""

code = code.replace(tab2_old, tab2_new)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TAB 3 REPLACEMENT (ONLY ACTIVE PARAMETERS)
# ─────────────────────────────────────────────────────────────────────────────
tab3_old = re.search(r'(# ─+\s+# TAB 3:.*?)(?=# ─+\s+# TAB 4:)', code, re.DOTALL).group(1)

tab3_new = """# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PARÁMETROS VALIDADOS DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_params:
    st.subheader("⚙️ Parámetros Validados del Pipeline")
    st.write(\"\"\"
    Especificaciones operativas validadas para adquisición confocal, inferencia neuronal y análisis cuantitativo:
    \"\"\")

    col_opt, col_inf = st.columns(2)

    with col_opt:
        st.markdown("#### 📐 Calibración Espacial y Óptica")
        st.markdown(\"\"\"
        - `pixel_size_um = 0.8913`: Calibración confocal CZI a 10x.
        - `intensity_depth = 16 bits`: Rango dinámico nativo (0 - 65535) procesado sin compresión.
        - `area_roi_estandar_corteza_mm2 = 0.348015`: Superficie constante de columna translaminar en Napari (661 × 661 px).
        - `pnn_radius_um = 12.0`: Radio del disco circular geométrico continuo (13.46 px).
        - `pnn_disk_area = 410.71 µm²`: Superficie circular constante sobre la cual se calculan `wfa_mean` y `wfa_sum`.
        - `wfa_norm_method = "Ninguno (Raw)"`: Canal WFA procesado en su rango dinámico nativo de 16 bits sin Otsu.
        \"\"\")

    with col_inf:
        st.markdown("#### 🧠 Modelos de Inferencia y Filtrado")
        st.markdown(\"\"\"
        - `pv_cellpose_model = "cpsam_v2"`: Modelo híbrido U-Net + SAM ViT para interneuronas PV.
        - `pv_cellpose_diameter = 20.0 px`: Diámetro nominal de soma (≈ 17.83 µm).
        - `pv_cellpose_flow_threshold = 0.80`: Tolerancia de gradiente para reconstruir somas multipolares.
        - `pv_cellpose_cellprob_threshold = -1.50`: Umbral permisivo para captar somas tenues en 16 bits.
        - `pv_cellpose_min_size = 15 px`: Tamaño somático mínimo (≈ 11.9 µm²).
        - `lupori_tile_size = 256 px` y `lupori_tile_overlap = 64 px`: Mosaicos de inferencia para Faster R-CNN.
        - `lupori_loc_threshold = 0.20`: Sensibilidad de detección de Faster R-CNN (PNNloc).
        - `lupori_score_threshold = -1.00`: Umbral permisivo de la ConvNet de ranking (PNNscore).
        - `use_min_pnn_dist = True` y `min_pnn_dist_px = 10 px`: Spatial NMS por proximidad (≈ 8.91 µm).
        \"\"\")

"""

code = code.replace(tab3_old, tab3_new)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TAB 4 ARCHITECTURE CLEANUP
# ─────────────────────────────────────────────────────────────────────────────
code = code.replace(
    "la segmentación de instancias somáticas para **PV+** (Daniel Zangrando) y el binomio de detección y puntuación de **PNNs** (Lupori et al.).",
    "la segmentación de instancias somáticas para **PV+** con Cellpose SAM v2 y el binomio de detección y puntuación de **PNNs** (Lupori et al.)."
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TAB 8 DEFENSE QA UPDATE
# ─────────────────────────────────────────────────────────────────────────────
tab8_old = re.search(r'(# ─+\s+# TAB 8:.*?)(?=\Z)', code, re.DOTALL).group(1)

tab8_new = """# ─────────────────────────────────────────────────────────────────────────────
# TAB 8: GUÍA DE DEFENSA DE TESIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_defense:
    st.subheader("🎓 Preguntas Probables del Jurado y Respuestas Clave")

    st.markdown(\"\"\"
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
    \"\"\", unsafe_allow_html=True)
"""

code = code.replace(tab8_old, tab8_new)

with open("pages/06_Metodología_Lupori_vs_Zangrando.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Updated pages/06_Metodología_Lupori_vs_Zangrando.py successfully!")
