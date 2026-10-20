#!/usr/bin/env python3
"""
Generate professional architectural diagrams for:
1. Cellpose SAM v2 (cpsam_v2)
2. Faster R-CNN (PNNloc)
3. ConvNet Ranking (PNNscore) & Spatial NMS
Using pure UTF-8 XML-safe Unicode characters.
"""
import os
import subprocess

OUT_DIR = "docs/images"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CELLPOSE SAM v2 DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
svg_cellpose = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 520" width="1000" height="520" style="background:#090d16; font-family:'Inter',system-ui,-apple-system,sans-serif;">
  <defs>
    <linearGradient id="bgGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0b1120"/>
      <stop offset="100%" stop-color="#050811"/>
    </linearGradient>
    <linearGradient id="pvInput" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#db2777"/>
      <stop offset="100%" stop-color="#831843"/>
    </linearGradient>
    <linearGradient id="vitGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#4c1d95"/>
    </linearGradient>
    <linearGradient id="unetGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#312e81"/>
    </linearGradient>
    <linearGradient id="flowGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#0e7490"/>
    </linearGradient>
    <linearGradient id="probGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#10b981"/>
      <stop offset="100%" stop-color="#065f46"/>
    </linearGradient>
    <linearGradient id="outGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#b45309"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94a3b8"/>
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#c084fc"/>
    </marker>
  </defs>

  <!-- Canvas Background -->
  <rect width="1000" height="520" fill="url(#bgGrad)" rx="12" stroke="#1e293b" stroke-width="2"/>

  <!-- Title Badge -->
  <g transform="translate(30, 25)">
    <rect width="940" height="42" rx="8" fill="#111827" stroke="#334155" stroke-width="1"/>
    <text x="20" y="26" fill="#c084fc" font-size="16" font-weight="700">ARQUITECTURA CELLPOSE SAM v2 (cpsam_v2) • RAMA PV</text>
    <text x="680" y="26" fill="#94a3b8" font-size="12" font-weight="500">Segmentación de Instancias por Flujo Vectorial</text>
  </g>

  <!-- 1. INPUT -->
  <g transform="translate(35, 95)">
    <rect width="125" height="295" rx="10" fill="#111827" stroke="#db2777" stroke-width="1.5"/>
    <rect x="15" y="20" width="95" height="95" rx="8" fill="url(#pvInput)" opacity="0.85"/>
    <circle cx="62" cy="67" r="22" fill="#f43f5e" opacity="0.9"/>
    <circle cx="62" cy="67" r="12" fill="#ffe4e6"/>
    <text x="62" y="145" fill="#fbcfe8" font-size="13" font-weight="700" text-anchor="middle">Canal PV546</text>
    <text x="62" y="168" fill="#94a3b8" font-size="11" text-anchor="middle">Confocal 8-bit</text>
    <text x="62" y="188" fill="#94a3b8" font-size="11" text-anchor="middle">[H × W × 1]</text>
    <rect x="15" y="215" width="95" height="55" rx="6" fill="#1e1b4b" stroke="#4338ca" stroke-width="1"/>
    <text x="62" y="234" fill="#a5b4fc" font-size="10" font-weight="600" text-anchor="middle">Calibración S1</text>
    <text x="62" y="250" fill="#e0e7ff" font-size="11" font-weight="700" text-anchor="middle">0.8913 µm/px</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <path d="M 160 240 L 195 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- 2. ENCODER BACKBONE (SAM ViT + U-Net) -->
  <g transform="translate(200, 90)">
    <rect width="210" height="305" rx="10" fill="#0f172a" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="105" y="28" fill="#c084fc" font-size="13" font-weight="700" text-anchor="middle">ENCODER HÍBRIDO</text>
    
    <!-- SAM ViT Block -->
    <rect x="15" y="42" width="180" height="110" rx="8" fill="url(#vitGrad)" opacity="0.9"/>
    <text x="105" y="66" fill="#ffffff" font-size="12" font-weight="700" text-anchor="middle">SAM Vision Transformer</text>
    <text x="105" y="84" fill="#ddd6fe" font-size="10" text-anchor="middle">Patch Embedding (16×16 px)</text>
    <text x="105" y="102" fill="#ddd6fe" font-size="10" text-anchor="middle">Multi-Head Self-Attention</text>
    <text x="105" y="120" fill="#a7f3d0" font-size="10" font-weight="600" text-anchor="middle">Pesos Fundacionales SAM</text>
    <rect x="30" y="128" width="150" height="16" rx="4" fill="#1e1b4b"/>
    <text x="105" y="140" fill="#c7d2fe" font-size="9" text-anchor="middle">Invariante a fondo y escala</text>

    <!-- Downsampling U-Net -->
    <rect x="15" y="165" width="180" height="120" rx="8" fill="url(#unetGrad)" opacity="0.85"/>
    <text x="105" y="188" fill="#ffffff" font-size="12" font-weight="700" text-anchor="middle">Downsampling U-Net</text>
    <text x="105" y="206" fill="#c7d2fe" font-size="10" text-anchor="middle">4 Niveles Residuales</text>
    <text x="105" y="224" fill="#c7d2fe" font-size="10" text-anchor="middle">Conv2D 3×3 + BN + ReLU</text>
    <text x="105" y="244" fill="#fef08a" font-size="11" font-weight="600" text-anchor="middle">diameter = 20 px (~17.8 µm)</text>
    <text x="105" y="264" fill="#93c5fd" font-size="10" text-anchor="middle">Cuello de Botella Latente</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <path d="M 410 240 L 445 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
  <!-- Dotted Skip Connection -->
  <path d="M 395 190 C 420 170, 435 170, 455 190" stroke="#818cf8" stroke-width="1.5" stroke-dasharray="3,3" fill="none" marker-end="url(#arrow-purple)"/>

  <!-- 3. DECODER & DUAL PREDICTION HEADS -->
  <g transform="translate(450, 90)">
    <rect width="230" height="305" rx="10" fill="#0f172a" stroke="#06b6d4" stroke-width="1.5"/>
    <text x="115" y="28" fill="#22d3ee" font-size="13" font-weight="700" text-anchor="middle">DECODER &amp; CABEZALES DUALES</text>

    <!-- Upsampling block -->
    <rect x="15" y="42" width="200" height="50" rx="6" fill="#164e63" stroke="#0891b2" stroke-width="1"/>
    <text x="115" y="62" fill="#cffafe" font-size="11" font-weight="700" text-anchor="middle">Up-Sampling Convolucional</text>
    <text x="115" y="78" fill="#a5f3fc" font-size="9" text-anchor="middle">Concatenación de Skip Connections</text>

    <!-- Branch A: Flows -->
    <rect x="15" y="102" width="200" height="88" rx="8" fill="url(#flowGrad)" opacity="0.85"/>
    <text x="115" y="124" fill="#ffffff" font-size="12" font-weight="700" text-anchor="middle">Rama 1: Campos de Flujo</text>
    <text x="115" y="142" fill="#e0f2fe" font-size="12" font-weight="600" text-anchor="middle">[ V_x(x,y),  V_y(x,y) ]</text>
    <text x="115" y="158" fill="#bae6fd" font-size="10" text-anchor="middle">Gradiente al centro somático</text>
    <text x="115" y="174" fill="#fef08a" font-size="10" font-weight="600" text-anchor="middle">flow_threshold = 0.80</text>

    <!-- Branch B: Cell Probability -->
    <rect x="15" y="200" width="200" height="88" rx="8" fill="url(#probGrad)" opacity="0.85"/>
    <text x="115" y="222" fill="#ffffff" font-size="12" font-weight="700" text-anchor="middle">Rama 2: Probabilidad Celular</text>
    <text x="115" y="240" fill="#ecfdf5" font-size="12" font-weight="600" text-anchor="middle">P_cell(x, y) ∈ [-∞, +∞]</text>
    <text x="115" y="256" fill="#a7f3d0" font-size="10" text-anchor="middle">Mapa de pertenencia intracelular</text>
    <text x="115" y="272" fill="#fef08a" font-size="10" font-weight="600" text-anchor="middle">cellprob_threshold = -1.50</text>
  </g>

  <!-- Arrow 3 -> 4 -->
  <path d="M 680 240 L 715 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- 4. DYNAMIC EULER INTEGRATION & INSTANCES -->
  <g transform="translate(720, 90)">
    <rect width="245" height="305" rx="10" fill="#0f172a" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="122" y="28" fill="#fbbf24" font-size="13" font-weight="700" text-anchor="middle">POST-PROCESO DINÁMICO</text>

    <!-- Euler box -->
    <rect x="15" y="42" width="215" height="80" rx="6" fill="#1e1b4b" stroke="#6366f1" stroke-width="1"/>
    <text x="122" y="62" fill="#c7d2fe" font-size="11" font-weight="700" text-anchor="middle">Integración de Euler (Gradiente)</text>
    <text x="122" y="82" fill="#6ee7b7" font-size="11" font-family="monospace" text-anchor="middle">X_{t+1} = X_t + η · V(X_t)</text>
    <text x="122" y="104" fill="#cbd5e1" font-size="9.5" text-anchor="middle">Convergencia en cuencas de atracción</text>

    <!-- Output mask -->
    <rect x="15" y="132" width="215" height="155" rx="8" fill="url(#outGrad)" opacity="0.9"/>
    <text x="122" y="154" fill="#ffffff" font-size="12" font-weight="700" text-anchor="middle">MÁSCARA DE INSTANCIAS (m_pv)</text>
    
    <!-- Mini cells representation -->
    <circle cx="70" cy="188" r="17" fill="#1e293b" stroke="#fde68a" stroke-width="2"/>
    <text x="70" y="192" fill="#fde68a" font-size="10" font-weight="700" text-anchor="middle">ID:1</text>
    <circle cx="125" cy="180" r="14" fill="#1e293b" stroke="#fde68a" stroke-width="2"/>
    <text x="125" y="184" fill="#fde68a" font-size="9" font-weight="700" text-anchor="middle">ID:2</text>
    <circle cx="175" cy="192" r="15" fill="#1e293b" stroke="#fde68a" stroke-width="2"/>
    <text x="175" y="196" fill="#fde68a" font-size="10" font-weight="700" text-anchor="middle">ID:3</text>

    <rect x="25" y="222" width="195" height="55" rx="6" fill="#18181b" opacity="0.85"/>
    <text x="122" y="238" fill="#fef08a" font-size="10" font-weight="700" text-anchor="middle">• Área Somática Real (A_soma en µm²)</text>
    <text x="122" y="252" fill="#e4e4e7" font-size="9.5" text-anchor="middle">• Diámetro y Morfometría Real</text>
    <text x="122" y="267" fill="#a7f3d0" font-size="9.5" text-anchor="middle">• Inclusión Topológica: (c_x, c_y) ∈ m_pv</text>
  </g>

  <!-- Footer Banner -->
  <g transform="translate(30, 455)">
    <rect width="940" height="42" rx="8" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <text x="30" y="26" fill="#38bdf8" font-size="12" font-weight="600">SEGMENTACIÓN SOMÁTICA:</text>
    <text x="230" y="26" fill="#cbd5e1" font-size="11.5">Genera contornos celulares somáticos reales célula a célula, permitiendo calcular áreas en µm² y evaluar colocalización topológica.</text>
  </g>
</svg>"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. FASTER R-CNN (PNNloc) DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
svg_faster_rcnn = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 520" width="1000" height="520" style="background:#090d16; font-family:'Inter',system-ui,-apple-system,sans-serif;">
  <defs>
    <linearGradient id="bgGrad2" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0b1120"/>
      <stop offset="100%" stop-color="#050811"/>
    </linearGradient>
    <linearGradient id="wfaGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#1e3a8a"/>
    </linearGradient>
    <linearGradient id="resnetGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="fpnGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0369a1"/>
    </linearGradient>
    <linearGradient id="rpnGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#d97706"/>
      <stop offset="100%" stop-color="#92400e"/>
    </linearGradient>
    <linearGradient id="headGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#064e3b"/>
    </linearGradient>
    <marker id="arrow2" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94a3b8"/>
    </marker>
  </defs>

  <rect width="1000" height="520" fill="url(#bgGrad2)" rx="12" stroke="#1e293b" stroke-width="2"/>

  <!-- Title Badge -->
  <g transform="translate(30, 25)">
    <rect width="940" height="42" rx="8" fill="#111827" stroke="#334155" stroke-width="1"/>
    <text x="20" y="26" fill="#38bdf8" font-size="16" font-weight="700">ARQUITECTURA FASTER R-CNN (PNNloc de Lupori et al.) • RAMA PNN (1/2)</text>
    <text x="690" y="26" fill="#94a3b8" font-size="12" font-weight="500">Detección de Propuestas de Bounding Box</text>
  </g>

  <!-- 1. INPUT & TILING -->
  <g transform="translate(35, 95)">
    <rect width="135" height="295" rx="10" fill="#111827" stroke="#38bdf8" stroke-width="1.5"/>
    <rect x="18" y="20" width="100" height="100" rx="6" fill="url(#wfaGrad)" opacity="0.85"/>
    <rect x="25" y="25" width="40" height="40" fill="none" stroke="#fde047" stroke-width="1.5" stroke-dasharray="2,2"/>
    <rect x="45" y="45" width="40" height="40" fill="none" stroke="#38bdf8" stroke-width="1.5"/>
    <text x="67" y="145" fill="#bae6fd" font-size="13" font-weight="700" text-anchor="middle">Canal WFA647</text>
    <text x="67" y="166" fill="#94a3b8" font-size="11" text-anchor="middle">Tiles 256 × 256 px</text>
    <text x="67" y="185" fill="#fde047" font-size="10.5" font-weight="600" text-anchor="middle">Overlap = 64 px</text>
    
    <rect x="15" y="210" width="105" height="60" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <text x="67" y="230" fill="#93c5fd" font-size="10" font-weight="600" text-anchor="middle">Ventana Deslizante</text>
    <text x="67" y="246" fill="#e2e8f0" font-size="9.5" text-anchor="middle">Garantiza cobertura</text>
    <text x="67" y="258" fill="#e2e8f0" font-size="9.5" text-anchor="middle">en bordes de tile</text>
  </g>

  <path d="M 170 240 L 200 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow2)"/>

  <!-- 2. RESNET-50 BACKBONE -->
  <g transform="translate(205, 90)">
    <rect width="180" height="305" rx="10" fill="#0f172a" stroke="#2563eb" stroke-width="1.5"/>
    <text x="90" y="28" fill="#60a5fa" font-size="13" font-weight="700" text-anchor="middle">RESNET-50 BACKBONE</text>
    
    <!-- Conv Stages -->
    <rect x="15" y="42" width="150" height="40" rx="6" fill="url(#resnetGrad)" opacity="0.8"/>
    <text x="90" y="66" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Stage 1: 7×7 Conv, /2</text>

    <rect x="15" y="90" width="150" height="40" rx="6" fill="url(#resnetGrad)" opacity="0.85"/>
    <text x="90" y="114" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Stage 2: 3x Bottleneck (C2)</text>

    <rect x="15" y="138" width="150" height="40" rx="6" fill="url(#resnetGrad)" opacity="0.9"/>
    <text x="90" y="162" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Stage 3: 4x Bottleneck (C3)</text>

    <rect x="15" y="186" width="150" height="40" rx="6" fill="url(#resnetGrad)" opacity="0.95"/>
    <text x="90" y="210" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Stage 4: 6x Bottleneck (C4)</text>

    <rect x="15" y="234" width="150" height="40" rx="6" fill="#1e3a8a" stroke="#3b82f6" stroke-width="1"/>
    <text x="90" y="258" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Stage 5: 3x Bottleneck (C5)</text>
  </g>

  <path d="M 385 240 L 415 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow2)"/>

  <!-- 3. FPN (Feature Pyramid Network) -->
  <g transform="translate(420, 90)">
    <rect width="165" height="305" rx="10" fill="#0f172a" stroke="#0284c7" stroke-width="1.5"/>
    <text x="82" y="28" fill="#38bdf8" font-size="13" font-weight="700" text-anchor="middle">FEATURE PYRAMID (FPN)</text>
    
    <!-- Levels -->
    <rect x="15" y="45" width="135" height="35" rx="6" fill="url(#fpnGrad)"/>
    <text x="82" y="67" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">P6 (Stride 64)</text>

    <rect x="15" y="90" width="135" height="35" rx="6" fill="url(#fpnGrad)"/>
    <text x="82" y="112" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">P5 (Stride 32)</text>

    <rect x="15" y="135" width="135" height="35" rx="6" fill="url(#fpnGrad)"/>
    <text x="82" y="157" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">P4 (Stride 16)</text>

    <rect x="15" y="180" width="135" height="35" rx="6" fill="url(#fpnGrad)"/>
    <text x="82" y="202" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">P3 (Stride 8)</text>

    <rect x="15" y="225" width="135" height="35" rx="6" fill="url(#fpnGrad)"/>
    <text x="82" y="247" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">P2 (Stride 4)</text>

    <text x="82" y="282" fill="#93c5fd" font-size="9.5" text-anchor="middle">Detección Multiescala</text>
  </g>

  <path d="M 585 240 L 615 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow2)"/>

  <!-- 4. RPN & ROI-ALIGN -->
  <g transform="translate(620, 90)">
    <rect width="165" height="305" rx="10" fill="#0f172a" stroke="#d97706" stroke-width="1.5"/>
    <text x="82" y="28" fill="#f59e0b" font-size="13" font-weight="700" text-anchor="middle">RPN &amp; RoIAlign</text>
    
    <rect x="12" y="45" width="140" height="95" rx="8" fill="url(#rpnGrad)" opacity="0.85"/>
    <text x="82" y="68" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Region Proposal (RPN)</text>
    <text x="82" y="86" fill="#fde68a" font-size="10" text-anchor="middle">3 Anclas × Escala</text>
    <text x="82" y="102" fill="#fde68a" font-size="9.5" text-anchor="middle">Ratios: 1:1, 1:2, 2:1</text>
    <text x="82" y="122" fill="#ffffff" font-size="10" font-weight="600" text-anchor="middle">Objectness Score</text>

    <rect x="12" y="155" width="140" height="120" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="1"/>
    <text x="82" y="180" fill="#fbbf24" font-size="12" font-weight="700" text-anchor="middle">RoIAlign</text>
    <text x="82" y="200" fill="#e2e8f0" font-size="10" text-anchor="middle">Muestreo Bilineal</text>
    <text x="82" y="218" fill="#cbd5e1" font-size="9" text-anchor="middle">Sin cuantización entera</text>
    <text x="82" y="238" fill="#a7f3d0" font-size="10" font-weight="600" text-anchor="middle">Preserva resolución</text>
    <text x="82" y="254" fill="#a7f3d0" font-size="9.5" text-anchor="middle">submétrica</text>
  </g>

  <path d="M 785 240 L 815 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow2)"/>

  <!-- 5. FAST R-CNN HEAD & CENTROIDS -->
  <g transform="translate(820, 90)">
    <rect width="150" height="305" rx="10" fill="#0f172a" stroke="#059669" stroke-width="1.5"/>
    <text x="75" y="28" fill="#34d399" font-size="13" font-weight="700" text-anchor="middle">DETECCIÓN PNN</text>
    
    <rect x="12" y="45" width="126" height="65" rx="6" fill="url(#headGrad)" opacity="0.9"/>
    <text x="75" y="68" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">FC 1024 Layers</text>
    <text x="75" y="86" fill="#a7f3d0" font-size="10" text-anchor="middle">Score ≥ 0.20</text>
    <text x="75" y="100" fill="#e2e8f0" font-size="9" text-anchor="middle">Filtro inicial RPN</text>

    <rect x="12" y="125" width="126" height="155" rx="6" fill="#134e4a" stroke="#10b981" stroke-width="1"/>
    <text x="75" y="148" fill="#6ee7b7" font-size="11" font-weight="700" text-anchor="middle">Bounding Box</text>
    <text x="75" y="168" fill="#ffffff" font-size="10" font-family="monospace" text-anchor="middle">[x_1, y_1, x_2, y_2]</text>
    
    <line x1="20" y1="185" x2="130" y2="185" stroke="#0d9488" stroke-width="1"/>
    
    <text x="75" y="206" fill="#fde047" font-size="11" font-weight="700" text-anchor="middle">CENTROIDE PNN</text>
    <text x="75" y="226" fill="#ffffff" font-size="10" font-family="monospace" text-anchor="middle">c_x = (x_1+x_2)/2</text>
    <text x="75" y="244" fill="#ffffff" font-size="10" font-family="monospace" text-anchor="middle">c_y = (y_1+y_2)/2</text>
    <text x="75" y="265" fill="#a7f3d0" font-size="9" text-anchor="middle">→ Pasa a PNNscore</text>
  </g>

  <!-- Footer Banner -->
  <g transform="translate(30, 455)">
    <rect width="940" height="42" rx="8" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <text x="30" y="26" fill="#38bdf8" font-size="12" font-weight="600">INFERENCIA POR MOSAICOS:</text>
    <text x="240" y="26" fill="#cbd5e1" font-size="11.5">Tiles de 256 px con overlap de 64 px y umbral de 0.20 para alta sensibilidad en redes de corteza somatosensorial.</text>
  </g>
</svg>"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. PNNscore & SPATIAL NMS DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
svg_pnnscore = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 520" width="1000" height="520" style="background:#090d16; font-family:'Inter',system-ui,-apple-system,sans-serif;">
  <defs>
    <linearGradient id="bgGrad3" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0b1120"/>
      <stop offset="100%" stop-color="#050811"/>
    </linearGradient>
    <linearGradient id="patchGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="cblockGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#047857"/>
    </linearGradient>
    <linearGradient id="rankGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#d97706"/>
      <stop offset="100%" stop-color="#b45309"/>
    </linearGradient>
    <linearGradient id="nmsGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#6d28d9"/>
    </linearGradient>
    <marker id="arrow3" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94a3b8"/>
    </marker>
  </defs>

  <rect width="1000" height="520" fill="url(#bgGrad3)" rx="12" stroke="#1e293b" stroke-width="2"/>

  <!-- Title Badge -->
  <g transform="translate(30, 25)">
    <rect width="940" height="42" rx="8" fill="#111827" stroke="#334155" stroke-width="1"/>
    <text x="20" y="26" fill="#10b981" font-size="16" font-weight="700">ARQUITECTURA ConvNet DE RANKING (PNNscore) &amp; SPATIAL NMS • RAMA PNN (2/2)</text>
    <text x="690" y="26" fill="#94a3b8" font-size="12" font-weight="500">Filtrado Cualitativo y Desduplicación Espacial</text>
  </g>

  <!-- 1. INPUT PATCH -->
  <g transform="translate(35, 95)">
    <rect width="125" height="295" rx="10" fill="#111827" stroke="#3b82f6" stroke-width="1.5"/>
    <rect x="22" y="25" width="80" height="80" rx="8" fill="url(#patchGrad)" opacity="0.85"/>
    <circle cx="62" cy="65" r="24" fill="none" stroke="#93c5fd" stroke-width="2" stroke-dasharray="3,2"/>
    <circle cx="62" cy="65" r="3" fill="#fde047"/>
    <text x="62" y="135" fill="#bae6fd" font-size="12.5" font-weight="700" text-anchor="middle">Parche 64 × 64 px</text>
    <text x="62" y="155" fill="#94a3b8" font-size="11" text-anchor="middle">Centrado en (c_x, c_y)</text>
    <text x="62" y="175" fill="#94a3b8" font-size="11" text-anchor="middle">Normalizado [0, 1]</text>
    <rect x="12" y="205" width="100" height="65" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <text x="62" y="224" fill="#60a5fa" font-size="10" font-weight="600" text-anchor="middle">Candidato de</text>
    <text x="62" y="240" fill="#93c5fd" font-size="11" font-weight="700" text-anchor="middle">Faster R-CNN</text>
    <text x="62" y="255" fill="#cbd5e1" font-size="9" text-anchor="middle">Canal WFA Raw</text>
  </g>

  <path d="M 160 240 L 185 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow3)"/>

  <!-- 2. FIVE CONVOLUTIONAL BLOCKS -->
  <g transform="translate(190, 85)">
    <rect width="270" height="315" rx="10" fill="#0f172a" stroke="#10b981" stroke-width="1.5"/>
    <text x="135" y="26" fill="#34d399" font-size="13" font-weight="700" text-anchor="middle">5 BLOQUES CONVOLUCIONALES</text>

    <!-- Block 1 -->
    <rect x="15" y="38" width="240" height="42" rx="6" fill="url(#cblockGrad)" opacity="0.85"/>
    <text x="25" y="56" fill="#ffffff" font-size="11" font-weight="700">Bloque 1:</text>
    <text x="85" y="56" fill="#e2e8f0" font-size="10.5">Conv2D 3×3 (32) + BN + ReLU + Pool /2</text>
    <text x="245" y="72" fill="#a7f3d0" font-size="10" font-weight="700" text-anchor="end">[32×32×32]</text>

    <!-- Block 2 -->
    <rect x="15" y="88" width="240" height="42" rx="6" fill="url(#cblockGrad)" opacity="0.88"/>
    <text x="25" y="106" fill="#ffffff" font-size="11" font-weight="700">Bloque 2:</text>
    <text x="85" y="106" fill="#e2e8f0" font-size="10.5">Conv2D 3×3 (64) + BN + ReLU + Pool /2</text>
    <text x="245" y="122" fill="#a7f3d0" font-size="10" font-weight="700" text-anchor="end">[16×16×64]</text>

    <!-- Block 3 -->
    <rect x="15" y="138" width="240" height="42" rx="6" fill="url(#cblockGrad)" opacity="0.91"/>
    <text x="25" y="156" fill="#ffffff" font-size="11" font-weight="700">Bloque 3:</text>
    <text x="85" y="156" fill="#e2e8f0" font-size="10.5">Conv2D 3×3 (128) + BN + ReLU + Pool /2</text>
    <text x="245" y="172" fill="#a7f3d0" font-size="10" font-weight="700" text-anchor="end">[8×8×128]</text>

    <!-- Block 4 -->
    <rect x="15" y="188" width="240" height="42" rx="6" fill="url(#cblockGrad)" opacity="0.94"/>
    <text x="25" y="206" fill="#ffffff" font-size="11" font-weight="700">Bloque 4:</text>
    <text x="85" y="206" fill="#e2e8f0" font-size="10.5">Conv2D 3×3 (256) + BN + ReLU + Pool /2</text>
    <text x="245" y="222" fill="#a7f3d0" font-size="10" font-weight="700" text-anchor="end">[4×4×256]</text>

    <!-- Block 5 -->
    <rect x="15" y="238" width="240" height="42" rx="6" fill="#065f46" stroke="#10b981" stroke-width="1"/>
    <text x="25" y="256" fill="#ffffff" font-size="11" font-weight="700">Bloque 5:</text>
    <text x="85" y="256" fill="#e2e8f0" font-size="10.5">Conv2D 3×3 (512) + BN + ReLU + Pool /2</text>
    <text x="245" y="272" fill="#fde047" font-size="10" font-weight="700" text-anchor="end">[2×2×512]</text>

    <text x="135" y="302" fill="#6ee7b7" font-size="10" text-anchor="middle">Campos Receptivos Progresivos</text>
  </g>

  <path d="M 460 240 L 485 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow3)"/>

  <!-- 3. GLOBAL POOLING & RANKING HEAD -->
  <g transform="translate(490, 85)">
    <rect width="190" height="315" rx="10" fill="#0f172a" stroke="#d97706" stroke-width="1.5"/>
    <text x="95" y="26" fill="#fbbf24" font-size="13" font-weight="700" text-anchor="middle">RANKING LOSS HEAD</text>

    <rect x="15" y="42" width="160" height="60" rx="6" fill="#1e293b" stroke="#f59e0b" stroke-width="1"/>
    <text x="95" y="64" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Global Avg Pooling</text>
    <text x="95" y="82" fill="#fef08a" font-size="11" font-family="monospace" text-anchor="middle">[512-dim Vector]</text>

    <rect x="15" y="115" width="160" height="60" rx="6" fill="#1e293b" stroke="#f59e0b" stroke-width="1"/>
    <text x="95" y="137" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">Dense Layer (512 → 1)</text>
    <text x="95" y="155" fill="#fde68a" font-size="10" text-anchor="middle">Margin Ranking Loss</text>

    <!-- Score decision box -->
    <rect x="15" y="190" width="160" height="100" rx="8" fill="url(#rankGrad)" opacity="0.9"/>
    <text x="95" y="215" fill="#ffffff" font-size="13" font-weight="800" text-anchor="middle">PNNscore ∈ ℝ</text>
    <rect x="25" y="228" width="140" height="50" rx="6" fill="#1e1b4b" stroke="#a78bfa" stroke-width="1"/>
    <text x="95" y="246" fill="#c7d2fe" font-size="10" font-weight="600" text-anchor="middle">Corte de Aceptación</text>
    <text x="95" y="264" fill="#34d399" font-size="13" font-weight="800" text-anchor="middle">Score ≥ -1.00</text>
  </g>

  <path d="M 680 240 L 705 240" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow3)"/>

  <!-- 4. SPATIAL NMS MODULE (DESDUPLICACIÓN) -->
  <g transform="translate(710, 85)">
    <rect width="255" height="315" rx="10" fill="#0f172a" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="127" y="26" fill="#c084fc" font-size="13" font-weight="700" text-anchor="middle">SPATIAL NMS POR DISTANCIA</text>

    <!-- Formula Box -->
    <rect x="15" y="42" width="225" height="110" rx="8" fill="#1e1b4b" stroke="#7c3aed" stroke-width="1"/>
    <text x="127" y="66" fill="#c4b5fd" font-size="11" font-weight="700" text-anchor="middle">Criterio de Distancia Euclidiana</text>
    <text x="127" y="92" fill="#6ee7b7" font-size="11" font-family="monospace" text-anchor="middle">d(p_i, p_j) &lt; 10 px</text>
    <text x="127" y="112" fill="#a7f3d0" font-size="10.5" text-anchor="middle">Distancia Física ≈ 8.91 µm</text>
    <text x="127" y="134" fill="#fde047" font-size="10" font-weight="600" text-anchor="middle">Elimina Duplicados por Overlap</text>

    <!-- Final result box -->
    <rect x="15" y="165" width="225" height="125" rx="8" fill="url(#nmsGrad)" opacity="0.9"/>
    <text x="127" y="190" fill="#ffffff" font-size="12" font-weight="700" text-anchor="middle">DETECCIONES FINALES PNN</text>
    <circle cx="85" cy="225" r="14" fill="#1e293b" stroke="#34d399" stroke-width="2"/>
    <text x="85" y="229" fill="#34d399" font-size="9" font-weight="700" text-anchor="middle">PNN_1</text>
    <circle cx="165" cy="225" r="14" fill="#1e293b" stroke="#34d399" stroke-width="2"/>
    <text x="165" y="229" fill="#34d399" font-size="9" font-weight="700" text-anchor="middle">PNN_2</text>
    
    <text x="127" y="258" fill="#e9d5ff" font-size="10" text-anchor="middle">• Centroides Únicos Invariantes</text>
    <text x="127" y="274" fill="#a7f3d0" font-size="10" font-weight="600" text-anchor="middle">• Pasa al Disco Continuo (12 µm)</text>
  </g>

  <!-- Footer Banner -->
  <g transform="translate(30, 455)">
    <rect width="940" height="42" rx="8" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <text x="30" y="26" fill="#10b981" font-size="12" font-weight="600">FILTRADO CIENTÍFICO:</text>
    <text x="210" y="26" fill="#cbd5e1" font-size="11.5">Score continuo permisivo (-1.00) rescata mallas tenues; Spatial NMS (&lt;10 px) garantiza que ninguna PNN se cuente dos veces.</text>
  </g>
</svg>"""

with open(f"{OUT_DIR}/diagram_cellpose.svg", "w", encoding="utf-8") as f:
    f.write(svg_cellpose)
print("Saved docs/images/diagram_cellpose.svg")

with open(f"{OUT_DIR}/diagram_faster_rcnn.svg", "w", encoding="utf-8") as f:
    f.write(svg_faster_rcnn)
print("Saved docs/images/diagram_faster_rcnn.svg")

with open(f"{OUT_DIR}/diagram_pnnscore.svg", "w", encoding="utf-8") as f:
    f.write(svg_pnnscore)
print("Saved docs/images/diagram_pnnscore.svg")

# Render PNGs using headless Chrome
for name in ["diagram_cellpose", "diagram_faster_rcnn", "diagram_pnnscore"]:
    svg_file = f"file://{os.path.abspath(OUT_DIR)}/{name}.svg"
    png_file = f"{OUT_DIR}/{name}.png"
    cmd = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        f"--screenshot={png_file}",
        "--window-size=1000,520",
        "--default-background-color=00000000",
        svg_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Rendered {png_file}")
