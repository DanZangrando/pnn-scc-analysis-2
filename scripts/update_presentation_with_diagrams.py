#!/usr/bin/env python3
"""
Update PRESENTACION_METODOLOGIA.html:
1. Replaces all raw LaTeX math ($...$, \times, \to, etc.) with clean Unicode / HTML.
2. Inlines the 3 architecture diagram SVGs into Slides 5, 6, and 7.
3. Formats all formulas with elegant sub/sup tags and symbols.
"""
import os
import re

PRES_FILE = "docs/PRESENTACION_METODOLOGIA.html"
with open(PRES_FILE, "r", encoding="utf-8") as f:
    html = f.read()

with open("docs/images/diagram_cellpose.svg", "r", encoding="utf-8") as f:
    svg_cellpose = f.read()

with open("docs/images/diagram_faster_rcnn.svg", "r", encoding="utf-8") as f:
    svg_faster_rcnn = f.read()

with open("docs/images/diagram_pnnscore.svg", "r", encoding="utf-8") as f:
    svg_pnnscore = f.read()

# 1. Clean math and LaTeX across all slides
replacements = [
    # LaTeX to clean HTML
    (r"\$64 \\times 64\$", "64 × 64"),
    (r"\\times", "×"),
    (r"\\to", "→"),
    (r"\\ge", "≥"),
    (r"\\le", "≤"),
    (r"\\approx", "≈"),
    (r"\\mu\\text\{m\}", "µm"),
    (r"\\mu", "µ"),
    (r"\$A_{\\text\{soma\}}\$", "<i>A</i><sub>soma</sub>"),
    (r"\$\\mu\\text\{m\}\^2\$", "µm²"),
    (r"\$\\mu\\text\{m\}\$", "µm"),
    (r"\$\(x_1, y_1, x_2, y_2\)\$", "[<i>x</i>₁, <i>y</i>₁, <i>x</i>₂, <i>y</i>₂]"),
    (r"\$\(> 1024 \\times 1024 px\)\$", "(> 1024 × 1024 px)"),
    (r"\$256 \\times 256\$", "256 × 256"),
    (r"\$64\$ píxeles", "64 píxeles"),
    (r"\$256\$ px", "256 px"),
    (r"Conv2D \$3\\times 3 \\to\$ BatchNorm \\to\$ ReLU \\to\$ MaxPool \$2\\times 2\$", "Conv2D 3×3 → BatchNorm → ReLU → MaxPool 2×2"),
    (r"Conv2D \$3\\times 3", "Conv2D 3×3"),
    (r"MaxPool \$2\\times 2\$", "MaxPool 2×2"),
    (r"Pooling \$\\to\$ Capa", "Pooling → Capa"),
    (r"c_x = \(x_1 \+ x_2\)/2, &emsp; c_y = \(y_1 \+ y_2\)/2", "<i>c</i><sub>x</sub> = (<i>x</i>₁ + <i>x</i>₂) / 2, &emsp; <i>c</i><sub>y</sub> = (<i>y</i>₁ + <i>y</i>₂) / 2"),
    (r"Si d\(\(c_x\^i, c_y\^i\), \(c_x\^j, c_y\^j\)\) &lt; 10 px \(&approx; 8\.91 &mu;m\):", "Si <i>d</i>((<i>c</i><sub>x</sub><sup>i</sup>, <i>c</i><sub>y</sub><sup>i</sup>), (<i>c</i><sub>x</sub><sup>j</sup>, <i>c</i><sub>y</sub><sup>j</sup>)) &lt; 10 px (≈ 8.91 µm):"),
    (r"&rArr; Conservar detecci&oacute;n con mayor PNNscore; descartar duplicado\.", "→ Conservar detección con mayor PNNscore; descartar duplicado de overlap."),
    (r"PNNscore &ge; -1\.00", "PNNscore ≥ -1.00"),
    (r"Score &ge; -1\.0", "Score ≥ -1.0"),
    (r"Score &ge; 0\.40", "Score ≥ 0.40"),
    (r"Score &lt; 0\.55", "Score &lt; 0.55"),
    (r"distancia euclidiana entre el punto PV y el punto PNN era &le; 15 px", "distancia euclidiana entre el punto PV y el punto PNN era ≤ 15 px (9.68 µm)"),
    (r"Distancia euclidiana entre centroides &lt;= 15 px", "Distancia euclidiana entre centroides ≤ 15 px (9.68 µm)"),
    (r"Spatial NMS por Distancia \(&lt; 10 px\)", "Spatial NMS por Distancia (&lt; 10 px ≈ 8.91 µm)"),
    (r"Radio = 12\.0 &mu;m &emsp; \(&approx; 13\.46 px\)<br>\s*&Aacute;rea = &pi; &middot; R\^2 &approx; 452\.39 px\^2 &approx; 410\.71 &mu;m\^2", "Radio = 12.0 µm &emsp; (≈ 13.46 px a 0.8913 µm/px)<br>Área = π · <i>R</i>² ≈ 452.39 px² ≈ <b>410.71 µm²</b>"),
    (r"Una PNN colocaliza con PV &hArr; \(c_x, c_y\)_\{PNN\} &isin; M&aacute;scara_\{Soma_PV\}", "Una PNN colocaliza con PV &emsp;⇔&emsp; (<i>c</i><sub>x</sub>, <i>c</i><sub>y</sub>)<sub>PNN</sub> ∈ Máscara<sub>Soma_PV</sub>"),
    (r"Dimensiones: 661 &times; 661 p&iacute;xeles<br>\s*Calibraci&oacute;n: 0\.8913 &mu;m / px \(Objetivo 10x Confocal\)<br>\s*Lado real: 589\.92 &mu;m<br>\s*&Aacute;rea total: 348,015 &mu;m\^2 = 0\.348015 mm\^2", "Dimensiones: 661 × 661 píxeles (Objetivo 10x Confocal)<br>Calibración: 0.8913 µm/px &emsp;→&emsp; Lado real: 589.92 µm<br>Área total = 348,015 µm² = <b>0.348015 mm²</b>"),
    (r"MARE &le; 10%", "MARE ≤ 10%"),
    (r"Salida Red: &Phi;\(x, y\) = \[ V_x\(x, y\), V_y\(x, y\), P_cell\(x, y\) \]<br>\s*Evolución: X_\{t\+1\} = X_t \+ &eta; &middot; V\(X_t\) &emsp; \[Integración de Euler\]", "Salida: Φ(x, y) = [ <i>V</i><sub>x</sub>(x, y),  <i>V</i><sub>y</sub>(x, y),  <i>P</i><sub>cell</sub>(x, y) ]<br>Evolución: <i>X</i><sub>t+1</sub> = <i>X</i><sub>t</sub> + η · <i>V</i>(<i>X</i><sub>t</sub>) &emsp; [Integración de Euler]"),
    (r"\$([A-Za-z0-9_\\\+-\/]+)\$", r"\1")
]

for pat, rep in replacements:
    html = re.sub(pat, rep, html)

# 2. Add CSS rules for architecture diagrams
arch_css = """
    /* Architecture Grid & Diagram Styling */
    .grid-architecture {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
      height: 100%;
      min-height: 0;
    }
    .diagram-card {
      background: #070b14;
      border: 1px solid #1f293d;
      border-radius: 12px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
      overflow: hidden;
      min-height: 0;
    }
    .diagram-card svg {
      width: 100%;
      height: auto;
      max-height: 440px;
      border-radius: 8px;
      display: block;
    }
    .diagram-header {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
"""

if ".grid-architecture" not in html:
    html = html.replace("/* Cards */", arch_css + "\n    /* Cards */")

# 3. New Slide 5 (Cellpose)
new_slide_5 = f"""    <!-- SLIDE 5: Arquitectura Cellpose -->
    <section class="slide" id="slide-5">
      <div class="slide-header">
        <div class="slide-category">04. Redes Neuronales • Rama PV</div>
        <h2 class="slide-title">Arquitectura 1: Cellpose SAM v2 (cpsam_v2)</h2>
        <div class="slide-subtitle">Segmentación de instancias de somas PV+ mediante campos vectoriales de difusión de calor</div>
      </div>
      <div class="slide-body">
        <div class="grid-architecture">
          <div style="display:flex; flex-direction:column; gap:12px; overflow-y:auto;">
            <div class="card highlight-purple" style="padding:16px;">
              <div class="card-title" style="font-size:16px;">Mecanismo Matemático</div>
              <p style="font-size:12.5px; line-height:1.5;">La red no predice máscaras binarias estáticas, sino un sistema dinámico reversible:</p>
              <div class="formula-box" style="padding:10px; font-size:11.5px; margin:8px 0;">
                Salida: Φ(x, y) = [ <i>V</i><sub>x</sub>, <i>V</i><sub>y</sub>, <i>P</i><sub>cell</sub> ]<br>
                Evolución: <i>X</i><sub>t+1</sub> = <i>X</i><sub>t</sub> + η · <i>V</i>(<i>X</i><sub>t</sub>)
              </div>
              <ul style="font-size:12px; line-height:1.45; padding-left:16px;">
                <li><b>Campos de Flujo:</b> Los vectores convergen al centro geométrico del soma.</li>
                <li><b>Tracking Euleriano:</b> Agrupa vóxeles en cuencas de atracción individuales.</li>
                <li><b>Morfometría:</b> Permite medir área somática real (<i>A</i><sub>soma</sub> en µm²) y diámetro.</li>
              </ul>
            </div>

            <div class="card" style="padding:14px;">
              <div class="card-title" style="font-size:14px; margin-bottom:6px;">Parámetros Calibrados (S1)</div>
              <table style="font-size:11.5px; margin:0;">
                <tbody>
                  <tr><td><code>model_type</code></td><td><code>cpsam_v2</code> (SAM ViT)</td></tr>
                  <tr><td><code>diameter</code></td><td><code>20.0 px</code> (~17.8 µm)</td></tr>
                  <tr><td><code>flow_thresh</code></td><td><code>0.80</code> (Permisivo)</td></tr>
                  <tr><td><code>cellprob</code></td><td><code>-1.50</code> (Rescata tenues)</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="diagram-card">
            <div class="diagram-header">
              <span style="color:#c084fc;">📐 Diagrama Arquitectural del Modelo</span>
              <span class="pill pill-purple">Dual-Head ViT + U-Net</span>
            </div>
            {svg_cellpose}
          </div>
        </div>
      </div>
    </section>"""

# 4. New Slide 6 (Faster R-CNN)
new_slide_6 = f"""    <!-- SLIDE 6: Arquitectura Faster R-CNN (PNNloc) -->
    <section class="slide" id="slide-6">
      <div class="slide-header">
        <div class="slide-category">05. Redes Neuronales • Rama PNN (1/2)</div>
        <h2 class="slide-title">Arquitectura 2: Faster R-CNN (PNNloc de Lupori)</h2>
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
              <div class="card-title" style="font-size:14px; margin-bottom:6px;">Inferencia por Mosaicos (Zangrando)</div>
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
    </section>"""

# 5. New Slide 7 (ConvNet Ranking & Spatial NMS)
new_slide_7 = f"""    <!-- SLIDE 7: ConvNet Ranking (PNNscore) & Spatial NMS -->
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
              <div class="card-title" style="font-size:16px;">Spatial NMS por Distancia (Zangrando)</div>
              <p style="font-size:12px; line-height:1.45;">Supresión no máxima por proximidad euclidiana:</p>
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
    </section>"""

# Replace in html
s5_start = html.find("<!-- SLIDE 5: Arquitectura Cellpose -->")
s6_start = html.find("<!-- SLIDE 6: Arquitectura Faster R-CNN (PNNloc) -->")
s7_start = html.find("<!-- SLIDE 7: ConvNet Ranking (PNNscore) & Spatial NMS -->")
s8_start = html.find("<!-- SLIDE 8: Delimitación de PNN: Disco Continuo vs Dona -->")

assert s5_start != -1 and s6_start != -1 and s7_start != -1 and s8_start != -1, "Slide markers not found!"

html = html[:s5_start] + new_slide_5 + "\n\n" + new_slide_6 + "\n\n" + new_slide_7 + "\n\n" + html[s8_start:]

with open(PRES_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print("Saved updated PRESENTACION_METODOLOGIA.html successfully!")
