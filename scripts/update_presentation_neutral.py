#!/usr/bin/env python3
"""
Update PRESENTACION_METODOLOGIA.html with neutral scientific phrasing,
clean parameters (no deprecated parameters mentioned),
accurate description of Lupori's PVloc/PVscore and Random Forest Gabor,
and clear rationale for the circular disk (unbiased comparison between PV+ and PV-).
"""
import re

with open("docs/images/diagram_cellpose.svg", "r", encoding="utf-8") as f:
    svg_cellpose = f.read().strip()

with open("docs/images/diagram_faster_rcnn.svg", "r", encoding="utf-8") as f:
    svg_faster_rcnn = f.read().strip()

with open("docs/images/diagram_pnnscore.svg", "r", encoding="utf-8") as f:
    svg_pnnscore = f.read().strip()

with open("docs/PRESENTACION_METODOLOGIA.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update Title and Topbar
html = html.replace(
    "<title>Presentación Metodológica: Lupori et al. vs. Lupori, Zangrando</title>",
    "<title>Presentación Metodológica: Cuantificación PNN y PV en S1</title>"
)

# 2. Slide 1 (Portada)
old_slide_1 = re.search(r'(<!-- SLIDE 1: Portada -->.*?)(?=<!-- SLIDE 2:)', html, re.DOTALL).group(1)
new_slide_1 = """<!-- SLIDE 1: Portada -->
    <section class="slide active" id="slide-1">
      <div class="slide-body" style="justify-content: center; align-items: center; text-align: center; max-width: 900px; margin: 0 auto;">
        <span class="pill pill-blue" style="margin-bottom: 16px; font-size: 12px; padding: 6px 14px;">Defensa Metodológica Experimental</span>
        <h1 style="font-size: 40px; font-weight: 800; line-height: 1.15; margin-bottom: 20px; background: linear-gradient(135deg, #60A5FA, #A78BFA, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
          Cuantificación y Morfometría de Redes Perineuronales (PNN) e Interneuronas PV+ en Corteza Somatosensorial (S1)
        </h1>
        <p style="font-size: 17px; color: #9CA3AF; max-width: 780px; line-height: 1.6; margin-bottom: 28px;">
          Comparación metodológica entre el marco de <strong>Lupori et al. (2023)</strong> y el pipeline adaptado para el análisis celular sin sesgos en <strong>HistoLab</strong>.
        </p>

        <div class="grid-2" style="width: 100%; max-width: 740px; text-align: left; height: auto;">
          <div class="card highlight-blue">
            <span class="pill pill-blue" style="width: fit-content; margin-bottom: 8px;">Marco Canónico de Referencia</span>
            <h4 style="font-size: 16px; color: #FFFFFF;">Lupori et al. (2023)</h4>
            <p style="font-size: 13px; color: #9CA3AF; margin-top: 4px;">Detección puntual en canal WFA (PNNloc/PNNscore) y canal PV (PVloc/PVscore) con cuantificación por Random Forest y filtros de Gabor.</p>
          </div>
          <div class="card highlight-purple">
            <span class="pill pill-purple" style="width: fit-content; margin-bottom: 8px;">Metodología del Estudio Actual</span>
            <h4 style="font-size: 16px; color: #FFFFFF;">HistoLab (2026)</h4>
            <p style="font-size: 13px; color: #9CA3AF; margin-top: 4px;">Segmentación somática de PV con Cellpose SAM, cuantificación isotrópica por disco continuo de 12 µm y estandarización espacial en S1.</p>
          </div>
        </div>

        <div style="margin-top: 26px;">
          <p style="font-size: 14px; color: #E2E8F0; font-weight: 600;">Autores: Daniel Zangrando & Rodrigo Yarzábal</p>
          <p style="font-size: 12px; color: #64748B; margin-top: 4px;">HistoLab • Instituto de Investigaciones Biológicas • 2026</p>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_1, new_slide_1)

# 3. Slide 3 (Lupori Canónico)
old_slide_3 = re.search(r'(<!-- SLIDE 3: Paper Canónico Lupori 2023 -->.*?)(?=<!-- SLIDE 4:)', html, re.DOTALL).group(1)
new_slide_3 = """<!-- SLIDE 3: Paper Canónico Lupori 2023 -->
    <section class="slide" id="slide-3">
      <div class="slide-header">
        <div class="slide-category">02. Marco Metodológico Canónico</div>
        <h2 class="slide-title">El Pipeline Canónico de Lupori et al. (2023)</h2>
        <div class="slide-subtitle">Detección puntual de objetos y cuantificación en secciones de cerebro completo</div>
      </div>
      <div class="slide-body">
        <div class="grid-2">
          <div class="card">
            <div class="card-title">Estructura del Pipeline en Lupori et al.</div>
            <p>Lupori et al. desarrollaron un flujo automatizado para mapear PNNs y células PV registradas al atlas Allen CCFv3:</p>
            <ul>
              <li><strong>Canal WFA (PNNs):</strong> Localización con Faster R-CNN (<code>PNNloc</code>) y descarte de falsos positivos mediante una ConvNet de ranking (<code>PNNscore</code>, umbral ≥ 0.40).</li>
              <li><strong>Canal PV:</strong> Detección puntual con <code>PVloc</code> (Faster R-CNN) y <code>PVscore</code> (ConvNet de ranking, umbral ≥ 0.55). No genera máscaras de contorno somático.</li>
              <li><strong>Cuantificación de Tinción:</strong> Parches de 80 × 80 px centrados en cada objeto detectado; clasificación de píxeles (célula vs. fondo) mediante <strong>Random Forest</strong> (TreeBagger) con 16 filtros de Gabor.</li>
              <li><strong>Colocalización:</strong> Distancia euclidiana entre centroides de PNN y PV ≤ 15 píxeles (9.675 µm).</li>
            </ul>
          </div>

          <div class="card highlight-blue">
            <div class="card-title">Adaptación a los Objetivos de Nuestro Estudio</div>
            <p>Al aplicar este marco a los experimentos confocales en el laboratorio de histología, definimos adaptaciones metodológicas específicas:</p>
            <ul>
              <li><strong>Morfometría Somática Real:</strong> Lupori detecta la posición puntual de las células PV, pero no su superficie. Para estudiar variaciones morfométricas incorporamos la segmentación de contornos somáticos (área en µm²).</li>
              <li><strong>Comparación Libre de Sesgos (PV+ vs. PV-):</strong> En Lupori la señal depende de los píxeles de Gabor en 80 × 80 px. Para comparar objetivamente las PNNs con soma PV frente a las PNNs sin soma (redes huecas), adoptamos un <strong>disco continuo invariante de 12 µm</strong> que evalúa ambas poblaciones bajo la misma área geométrica.</li>
              <li><strong>Estandarización Espacial en S1:</strong> Frente a distorsiones elásticas del registro atlas CCFv3 en cortes coronales con variación angular, adoptamos un marco muestral constante de 0.348 mm².</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_3, new_slide_3)

# 4. Slide 4 (Cuadro Comparativo)
old_slide_4 = re.search(r'(<!-- SLIDE 4: Cuadro Comparativo Global -->.*?)(?=<!-- SLIDE 5:)', html, re.DOTALL).group(1)
new_slide_4 = """<!-- SLIDE 4: Cuadro Comparativo Global -->
    <section class="slide" id="slide-4">
      <div class="slide-header">
        <div class="slide-category">03. Metodología Comparada</div>
        <h2 class="slide-title">Cuadro Comparativo: Lupori et al. vs. Metodología del Estudio</h2>
        <div class="slide-subtitle">Comparación sistemática de las dimensiones metodológicas entre ambos enfoques</div>
      </div>
      <div class="slide-body">
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Dimensión Metodológica</th>
                <th>Lupori et al. (2023) [Canónico]</th>
                <th>Metodología Adoptada [HistoLab]</th>
                <th>Fundamento Metodológico</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>1. Normalización Espacial</strong></td>
                <td>Registro no rígido al atlas Allen CCFv3</td>
                <td><span class="pill pill-emerald">ROI Estandarizada Fija</span> (0.348015 mm² en Napari)</td>
                <td>Invariante ante deformaciones elásticas del corte; densidades comparables directamente.</td>
              </tr>
              <tr>
                <td><strong>2. Detección de PV+</strong></td>
                <td>Detección puntual (PVloc + PVscore)</td>
                <td><span class="pill pill-purple">Cellpose cpsam_v2</span> (Campos de Flujo + SAM ViT)</td>
                <td>Segmentación de contornos somáticos reales célula a célula; cálculo de área somática en µm².</td>
              </tr>
              <tr>
                <td><strong>3. Detección de PNN (WFA)</strong></td>
                <td>Faster R-CNN (PNNloc) + ConvNet (PNNscore)</td>
                <td><span class="pill pill-blue">PNNloc + PNNscore</span> + <strong>Spatial NMS por Distancia (&lt; 10 px ≈ 8.91 µm)</strong></td>
                <td>Pesos originales de Lupori et al.; se añade desduplicación espacial por solapamiento de tiles.</td>
              </tr>
              <tr>
                <td><strong>4. Cuantificación de PNN</strong></td>
                <td>Parches 80 × 80 px + Random Forest (Gabor)</td>
                <td><span class="pill pill-cyan" style="color:#06B6D4;">Disco Circular Continuo</span> (Radio 12 µm = 410.7 µm²)</td>
                <td>Superficie idéntica para comparar subpoblaciones PV+ y PV- sin sesgo metodológico.</td>
              </tr>
              <tr>
                <td><strong>5. Colocalización</strong></td>
                <td>Distancia entre centroides ≤ 15 px (9.675 µm)</td>
                <td><span class="pill pill-emerald">Inclusión Topológica</span> en máscara somática PV</td>
                <td>Clasificación celular exacta: PV+/PNN+, PV-/PNN+ (red hueca) y PV+/PNN- (interneurona desnuda).</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_4, new_slide_4)

# 5. Slide 5 (Cellpose)
old_slide_5 = re.search(r'(<!-- SLIDE 5:.*?-->.*?)(?=<!-- SLIDE 6:)', html, re.DOTALL).group(1)
new_slide_5 = f"""<!-- SLIDE 5: Arquitectura Cellpose SAM v2 -->
    <section class="slide" id="slide-5">
      <div class="slide-header">
        <div class="slide-category">04. Redes Neuronales • Rama PV</div>
        <h2 class="slide-title">Arquitectura 1: Cellpose SAM v2 (cpsam_v2)</h2>
        <div class="slide-subtitle">Segmentación de contornos somáticos de interneuronas PV+ mediante campos vectoriales</div>
      </div>
      <div class="slide-body">
        <div class="grid-architecture">
          <div style="display:flex; flex-direction:column; gap:12px; overflow-y:auto;">
            <div class="card highlight-purple" style="padding:16px;">
              <div class="card-title" style="font-size:16px;">Estructura Cellpose SAM v2</div>
              <p style="font-size:12.5px; line-height:1.5;">Backbone híbrido U-Net + Segment Anything Model (SAM ViT):</p>
              <ul style="font-size:12px; line-height:1.45; margin-top:4px; padding-left:16px;">
                <li><b>Entrada:</b> Canal PV546 crudo en 8 bits (0 - 255) sin saturación.</li>
                <li><b>Rama de Flujo:</b> Predice campos vectoriales de difusión de calor [<i>V</i><sub>x</sub>, <i>V</i><sub>y</sub>].</li>
                <li><b>Rama de Probabilidad:</b> Mapa de probabilidad de presencia celular <i>P</i><sub>cell</sub>.</li>
                <li><b>Integración de Euler:</b> Desplaza vóxeles hacia los atractores somáticos centrales.</li>
              </ul>
              <div class="formula-box" style="padding:10px; font-size:11.5px; margin:8px 0;">
                x<sub>t+1</sub> = x<sub>t</sub> + η · V(x<sub>t</sub>) &emsp; [Convergencia hacia centroide somático]
              </div>
            </div>

            <div class="card" style="padding:14px;">
              <div class="card-title" style="font-size:14px; margin-bottom:6px;">Calibración Somática</div>
              <ul style="font-size:11.5px; line-height:1.45; margin:0; padding-left:16px;">
                <li><b>Diámetro Nominal:</b> <code>diameter = 20.0 px</code> (≈ 17.8 µm a 0.8913 µm/px).</li>
                <li><b>Umbrales:</b> <code>flow_threshold = 0.80</code>, <code>cellprob_threshold = -1.50</code>.</li>
                <li><b>Salida:</b> Máscara somática con cálculo de área real en µm² e inclusión topológica.</li>
              </ul>
            </div>
          </div>

          <div class="diagram-card">
            <div class="diagram-header">
              <span style="color:#c084fc;">📐 Diagrama Arquitectural del Modelo</span>
              <span class="pill pill-purple">Dual U-Net + SAM ViT</span>
            </div>
            {svg_cellpose}
          </div>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_5, new_slide_5)

# 6. Slide 6 (Faster R-CNN)
old_slide_6 = re.search(r'(<!-- SLIDE 6: Arquitectura Faster R-CNN \(PNNloc\) -->.*?)(?=<!-- SLIDE 7:)', html, re.DOTALL).group(1)
new_slide_6 = f"""<!-- SLIDE 6: Arquitectura Faster R-CNN (PNNloc) -->
    <section class="slide" id="slide-6">
      <div class="slide-header">
        <div class="slide-category">05. Redes Neuronales • Rama PNN (1/2)</div>
        <h2 class="slide-title">Arquitectura 2: Faster R-CNN (PNNloc de Lupori et al.)</h2>
        <div class="slide-subtitle">Detección de propuestas de región para mallas perineuronales en canal WFA</div>
      </div>
      <div class="slide-body">
        <div class="grid-architecture">
          <div style="display:flex; flex-direction:column; gap:12px; overflow-y:auto;">
            <div class="card highlight-blue" style="padding:16px;">
              <div class="card-title" style="font-size:16px;">Estructura Faster R-CNN</div>
              <p style="font-size:12.5px; line-height:1.5;">Detección en dos etapas adaptada a fluorescencia de redes:</p>
              <ul style="font-size:12px; line-height:1.45; margin-top:4px; padding-left:16px;">
                <li><b>ResNet-50 + FPN:</b> Pirámide multiescala (P2 a P6) para detectar PNNs de cualquier tamaño.</li>
                <li><b>RPN:</b> 3 anclas (1:1, 1:2, 2:1) evaluando acumulación de WFA.</li>
                <li><b>RoIAlign:</b> Muestreo bilineal sin cuantización para precisión submétrica.</li>
              </ul>
              <div class="formula-box" style="padding:10px; font-size:11.5px; margin:8px 0;">
                Centroide: <i>c</i><sub>x</sub> = (<i>x</i>₁ + <i>x</i>₂) / 2, &emsp; <i>c</i><sub>y</sub> = (<i>y</i>₁ + <i>y</i>₂) / 2
              </div>
            </div>

            <div class="card" style="padding:14px;">
              <div class="card-title" style="font-size:14px; margin-bottom:6px;">Inferencia por Mosaicos (Tiles)</div>
              <ul style="font-size:11.5px; line-height:1.45; margin:0; padding-left:16px;">
                <li><b>Tamaño de Tile:</b> 256 × 256 px en canal WFA.</li>
                <li><b>Solapamiento (Overlap):</b> 64 px entre tiles para no cortar PNNs en bordes.</li>
                <li><b>Umbral de Confianza:</b> <code>loc_threshold = 0.20</code>.</li>
              </ul>
            </div>
          </div>

          <div class="diagram-card">
            <div class="diagram-header">
              <span style="color:#38bdf8;">📐 Diagrama Arquitectural del Modelo</span>
              <span class="pill pill-blue">ResNet-50 + FPN + RoIAlign</span>
            </div>
            {svg_faster_rcnn}
          </div>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_6, new_slide_6)

# 7. Slide 7 (ConvNet Ranking & Spatial NMS)
old_slide_7 = re.search(r'(<!-- SLIDE 7: ConvNet Ranking \(PNNscore\) & Spatial NMS -->.*?)(?=<!-- SLIDE 8:)', html, re.DOTALL).group(1)
new_slide_7 = f"""<!-- SLIDE 7: ConvNet Ranking (PNNscore) & Spatial NMS -->
    <section class="slide" id="slide-7">
      <div class="slide-header">
        <div class="slide-category">06. Redes Neuronales • Rama PNN (2/2)</div>
        <h2 class="slide-title">Arquitectura 3: ConvNet de Ranking &amp; Spatial NMS</h2>
        <div class="slide-subtitle">Filtrado cualitativo y desduplicación euclidiana de detecciones</div>
      </div>
      <div class="slide-body">
        <div class="grid-architecture">
          <div style="display:flex; flex-direction:column; gap:12px; overflow-y:auto;">
            <div class="card highlight-emerald" style="padding:16px;">
              <div class="card-title" style="font-size:16px;">PNNscore (ConvNet 5 Bloques)</div>
              <ul style="font-size:12px; line-height:1.45; margin-top:4px; padding-left:16px;">
                <li><b>Entrada:</b> Parches de 64 × 64 px centrados en (<i>c</i><sub>x</sub>, <i>c</i><sub>y</sub>).</li>
                <li><b>Backbone:</b> 5 bloques (Conv2D 3×3 → BN → ReLU → Pool 2×2).</li>
                <li><b>Ranking Head:</b> Global Average Pooling → Capa lineal de 512-d con <i>Margin Ranking Loss</i>.</li>
                <li><b>Corte de Aceptación:</b> <code>PNNscore ≥ -1.00</code>.</li>
              </ul>
            </div>

            <div class="card highlight-purple" style="padding:16px;">
              <div class="card-title" style="font-size:16px;">Spatial NMS por Proximidad Euclidiana</div>
              <p style="font-size:12px; line-height:1.45;">Supresión no máxima por distancia física en el plano:</p>
              <div class="formula-box" style="padding:10px; font-size:11.5px; margin:8px 0;">
                Si <i>d</i>((<i>c</i><sub>x</sub><sup>i</sup>, <i>c</i><sub>y</sub><sup>i</sup>), (<i>c</i><sub>x</sub><sup>j</sup>, <i>c</i><sub>y</sub><sup>j</sup>)) &lt; 10 px (≈ 8.91 µm):<br>
                → Conservar detección con mayor score; descartar duplicado.
              </div>
              <p style="font-size:11.5px; color:#cbd5e1;">Elimina conteos dobles generados por el solapamiento de tiles de 256 px.</p>
            </div>
          </div>

          <div class="diagram-card">
            <div class="diagram-header">
              <span style="color:#10b981;">📐 Diagrama Arquitectural del Modelo</span>
              <span class="pill pill-emerald">5-Stage ConvNet + Spatial NMS</span>
            </div>
            {svg_pnnscore}
          </div>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_7, new_slide_7)

# 8. Slide 8 (Delimitación de PNN: Disco Continuo Estandarizado)
old_slide_8 = re.search(r'(<!-- SLIDE 8:.*?-->.*?)(?=<!-- SLIDE 9:)', html, re.DOTALL).group(1)
new_slide_8 = """<!-- SLIDE 8: Cuantificación de PNN: Disco Continuo Estandarizado -->
    <section class="slide" id="slide-8">
      <div class="slide-header">
        <div class="slide-category">07. Cuantificación Pericelular</div>
        <h2 class="slide-title">Cuantificación de PNN: Disco Continuo Estandarizado</h2>
        <div class="slide-subtitle">Medición geométrica uniforme para comparar subpoblaciones PV+ y PV- sin sesgo metodológico</div>
      </div>
      <div class="slide-body">
        <div class="grid-2">
          <div class="card highlight-cyan" style="border-color: rgba(6, 182, 212, 0.4);">
            <div class="card-title" style="color: var(--accent-cyan);">🔵 Disco Continuo de Radio Fijo (12.0 µm)</div>
            <p>Se construye un disco geométrico continuo centrado en el centroide de la PNN (canal WFA):</p>
            <div class="formula-box">
              Radio = 12.0 µm &emsp; (≈ 13.46 px a 0.8913 µm/px)<br>
              Área = π · <i>R</i>² ≈ 452.39 px² ≈ <b>410.71 µm²</b><br>
              Métricas: <code>wfa_mean</code> (intensidad media), <code>wfa_sum</code> (señal integrada)
            </div>
            <ul>
              <li><strong>Muestreo integral:</strong> Mide la fluorescencia en toda la circunferencia, capturando condensación pericelular y ramificaciones proximales.</li>
              <li><strong>Medición cruda de 8 bits (0 - 255):</strong> Sin binarizaciones de Otsu artificiales que censuren intensidades intermedias.</li>
              <li><strong>Invariancia geométrica:</strong> Proporciona una superficie estandarizada idéntica de 410.71 µm² para todas las detecciones.</li>
            </ul>
          </div>

          <div class="card highlight-blue">
            <div class="card-title" style="color: #60A5FA;">⚖️ Justificación Metodológica: Comparabilidad Libre de Sesgos</div>
            <p>La adopción del disco continuo responde a una necesidad biológica y estadística fundamental:</p>
            <ul>
              <li><strong>Heterogeneidad de Subpoblaciones:</strong> En la corteza, entre el 15% y el 40% de las PNNs envuelven neuronas que no expresan PV detectable ("redes huecas").</li>
              <li><strong>Ausencia de Máscara Somática en PV-:</strong> Si la cuantificación dependiera del contorno de una célula o de una máscara celular previa, las PNNs PV- no tendrían referencia somática y se medirían con un criterio dispar o quedarían excluidas.</li>
              <li><strong>Mismo Marco para Ambas Poblaciones:</strong> Al evaluar un disco circular invariante centrado en el centroide de la PNN, <strong>tanto las PNNs PV+ como las PV- se miden bajo exactamente la misma superficie física</strong>, permitiendo compararlas de manera directa, robusta y libre de sesgos metodológicos.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_8, new_slide_8)

# 9. Slide 11 (Parámetros Validados del Pipeline - NO DEPRECATED)
old_slide_11 = re.search(r'(<!-- SLIDE 11:.*?-->.*?)(?=<!-- SLIDE 12:)', html, re.DOTALL).group(1)
new_slide_11 = """<!-- SLIDE 11: Parámetros y Calibración del Pipeline -->
    <section class="slide" id="slide-11">
      <div class="slide-header">
        <div class="slide-category">10. Parámetros y Calibración</div>
        <h2 class="slide-title">Parámetros de Calibración e Inferencia del Pipeline</h2>
        <div class="slide-subtitle">Valores validados para adquisición confocal, inferencia neuronal y análisis cuantitativo</div>
      </div>
      <div class="slide-body">
        <div class="grid-2">
          <div class="card highlight-emerald">
            <div class="card-title" style="color: var(--accent-emerald);">📐 Calibración Espacial y Óptica</div>
            <p style="font-size: 13px; color: #94A3B8; margin-bottom: 10px;">Parámetros del sistema confocal y estandarización espacial en corteza S1:</p>
            <div class="table-container">
              <table>
                <thead>
                  <tr><th>Parámetro</th><th>Valor</th><th>Función Metodológica</th></tr>
                </thead>
                <tbody>
                  <tr><td><code>pixel_size_um</code></td><td><code>0.8913</code></td><td>Factor de escala confocal a 10x (CZI Zeiss)</td></tr>
                  <tr><td><code>intensity_depth</code></td><td><code>8 bits (TIFF 16-bit)</code></td><td>Rango dinámico nativo (0 - 255) sin reescalado</td></tr>
                  <tr><td><code>roi_dimensions</code></td><td><code>661 × 661 px</code></td><td>Área estandarizada constante (0.348015 mm²)</td></tr>
                  <tr><td><code>pnn_radius_um</code></td><td><code>12.0 µm</code></td><td>Radio del disco de cuantificación pericelular</td></tr>
                  <tr><td><code>pnn_disk_area</code></td><td><code>410.71 µm²</code></td><td>Superficie circular uniforme para medir WFA</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="card highlight-blue">
            <div class="card-title" style="color: #60A5FA;">🧠 Modelos de Inferencia y Filtrado</div>
            <p style="font-size: 13px; color: #94A3B8; margin-bottom: 10px;">Parámetros de segmentación somática y detección de redes:</p>
            <div class="table-container">
              <table>
                <thead>
                  <tr><th>Parámetro</th><th>Valor</th><th>Función Metodológica</th></tr>
                </thead>
                <tbody>
                  <tr><td><code>cellpose_diameter</code></td><td><code>20.0 px</code></td><td>Diámetro nominal para segmentación somática PV</td></tr>
                  <tr><td><code>flow_threshold</code></td><td><code>0.80</code></td><td>Sensibilidad del campo vectorial de flujo</td></tr>
                  <tr><td><code>cellprob_threshold</code></td><td><code>-1.50</code></td><td>Corte permisivo para detección somática en 8-bit</td></tr>
                  <tr><td><code>PNNloc_threshold</code></td><td><code>0.20</code></td><td>Umbral de propuesta de región Faster R-CNN</td></tr>
                  <tr><td><code>PNNscore_threshold</code></td><td><code>-1.00</code></td><td>Corte permisivo de la ConvNet de ranking</td></tr>
                  <tr><td><code>nms_distance_px</code></td><td><code>10 px</code></td><td>Spatial NMS por proximidad (≈ 8.91 µm)</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>

    """

html = html.replace(old_slide_11, new_slide_11)

# 10. Slide 12 (Flowchart SVG update labels inside Slide 12)
html = html.replace("3. Rama PV: Cellpose (Zangrando)", "3. Rama PV: Cellpose SAM v2")
html = html.replace("(Supresión de Duplicados Zangrando)", "(Supresión de Duplicados por Proximidad)")
html = html.replace("5. Cuantificación Pericelular (Zangrando)", "5. Cuantificación Pericelular Estandarizada")
html = html.replace("(SIN filtro Otsu ni anillos de erosión)", "(Medición en Toda el Área Circular)")

# 11. Slide 13 (FAQ)
old_slide_13 = re.search(r'(<!-- SLIDE 13: Defensa de Tesis / FAQ -->.*?)(?=</main>)', html, re.DOTALL).group(1)
new_slide_13 = """<!-- SLIDE 13: Defensa de Tesis / FAQ -->
    <section class="slide" id="slide-13">
      <div class="slide-header">
        <div class="slide-category">12. Preparación para el Jurado</div>
        <h2 class="slide-title">Preguntas Clave y Respuestas de Defensa</h2>
        <div class="slide-subtitle">Fundamentación técnica de las decisiones metodológicas adoptadas</div>
      </div>
      <div class="slide-body">
        <div class="grid-2">
          <div class="card">
            <h4 style="font-size: 15px; color: var(--accent-cyan); margin-bottom: 6px;">P1: ¿Por qué no registrar contra el atlas CCFv3 como Lupori?</h4>
            <p style="font-size: 13px;">El CCFv3 exige hemisferios intactos con orientación estereotáxica perfecta. En rebanadas de corte manual o parciales de S1, el registro no rígido genera distorsiones geométricas inaceptables. La <strong>ROI normalizada de 0.348 mm²</strong> elimina el sesgo de deformación y proporciona densidades volumétricas rigurosas e invariantes.</p>
          </div>

          <div class="card">
            <h4 style="font-size: 15px; color: var(--accent-cyan); margin-bottom: 6px;">P2: ¿Por qué cuantificar las PNNs con un disco fijo de 12 µm en vez de una máscara celular irregular?</h4>
            <p style="font-size: 13px;">Para garantizar una comparación metodológica sin sesgos entre PNNs PV+ y PNNs PV- (redes huecas). Si la cuantificación dependiera del contorno de una célula o de una máscara celular previa, las PNNs que envuelven neuronas PV- no tendrían máscara somática de referencia y se medirían con criterios dispares. El <strong>disco circular estandarizado de 12 µm (410.71 µm²)</strong> evalúa ambas poblaciones bajo exactamente la misma superficie y regla geométrica.</p>
          </div>

          <div class="card">
            <h4 style="font-size: 15px; color: var(--accent-cyan); margin-bottom: 6px;">P3: ¿Por qué usar Cellpose SAM v2 en lugar del detector puntual de Lupori para PV?</h4>
            <p style="font-size: 13px;">Lupori et al. detectaban PV como puntos (x, y) con Faster R-CNN (PVloc). En nuestro estudio necesitábamos caracterizar la <strong>morfometría real del soma (área en µm² y diámetro)</strong> para evaluar posibles cambios tróficos celulares, además de permitir una colocalización inequívoca mediante inclusión topológica en la máscara somática real.</p>
          </div>

          <div class="card">
            <h4 style="font-size: 15px; color: var(--accent-cyan); margin-bottom: 6px;">P4: ¿Cómo se validó la precisión frente al evaluador humano?</h4>
            <p style="font-size: 13px;">Se implementó un módulo de conteo manual por tiles en Streamlit (Página 3), validando las predicciones automáticas mediante análisis de <strong>Bland-Altman</strong> y cálculo de error medio absoluto relativo (MARE ≤ 10%), confirmando alta concordancia con el evaluador biológico.</p>
          </div>
        </div>
      </div>
    </section>

  """

html = html.replace(old_slide_13, new_slide_13)

with open("docs/PRESENTACION_METODOLOGIA.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Updated docs/PRESENTACION_METODOLOGIA.html successfully!")
