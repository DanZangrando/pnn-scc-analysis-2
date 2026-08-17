import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2_contingency

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys
sys.path.append(os.path.abspath("src"))
from roi import load_rois, get_roi_json_path, points_in_rois, get_point_region_assignment
from image_io import extract_animal_id

st.set_page_config(page_title="Paso 5: Espacio Latente & Modelos Mixtos Lineales", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght=300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main-header {
        background: linear-gradient(120deg, #bb86fc 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header { color: #a0aec0; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .stats-card {
        background: rgba(30, 33, 48, 0.75);
        border: 1px solid rgba(187, 134, 252, 0.4);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .level-badge {
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 15px;
    }
    .table-caption {
        font-weight: 700;
        color: #00f2fe;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 0.95rem;
    }
    hr { border: 0; height: 1px;
         background: linear-gradient(to right, transparent, #bb86fc, transparent);
         margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Paso 5: Espacio Latente (Autoencoders) y Modelos Mixtos Lineales (LMM)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Descubrimiento de fenotipos celulares mediante aprendizaje profundo no supervisado (Autoencoders PyTorch) y análisis estadístico masivo single-cell con modelos mixtos lineales.</div>', unsafe_allow_html=True)

METRICS_BASE_DIR = "data/processed/metrics"

def extract_animal_id(filename):
    clean = os.path.basename(str(filename))
    for suf in ['_nuclei_metrics.csv', '_masks.tif', '_prob_map.tif', '_segmented.tif', '.TIF', '.tif', '.czi']:
        clean = clean.replace(suf, '')
    m = re.match(r'^(ACF_[A-Za-z0-9]+)', clean)
    if m:
        return m.group(1)
    return clean.split('~')[0].split('_')[0]

def parse_group(name):
    import unicodedata
    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    name_norm = strip_accents(name).upper().replace('_', ' ')

    sex = 'HEMBRA' if 'HEMBRA' in name_norm else 'MACHO' if 'MACHO' in name_norm else None
    if 'NONE' in name_norm: cond, order = 'NONE', 0
    elif '3' in name_norm and 'DIA' in name_norm: cond, order = '3 DÍAS', 1
    elif '14' in name_norm and 'DIA' in name_norm: cond, order = '14 DÍAS', 2
    else: cond, order = name, 99

    return sex, cond, order

COND_ORDER  = ['NONE', '3 DÍAS', '14 DÍAS']
COND_COLORS = {'NONE': '#4facfe', '3 DÍAS': '#bb86fc', '14 DÍAS': '#00ffcc'}
SEX_COLORS  = {'MACHO': '#5bc0de', 'HEMBRA': '#e83e8c'}

def get_dataset_mtime_hash():
    max_mtime = 0.0
    file_count = 0
    if os.path.exists(METRICS_BASE_DIR):
        for root, dirs, files in os.walk(METRICS_BASE_DIR):
            for f in files:
                if f.endswith('_rois.json') or f.endswith('_nuclei_metrics.csv'):
                    p = os.path.join(root, f)
                    try:
                        m = os.path.getmtime(p)
                        if m > max_mtime:
                            max_mtime = m
                        file_count += 1
                    except Exception:
                        pass
    return f"{file_count}_{max_mtime}"

@st.cache_data
def load_all_dataset(_hash=None):
    all_cells = []
    image_catalog = []

    RAW_BASE = "data/raw"
    if not os.path.exists(RAW_BASE) or not any(os.path.isdir(os.path.join(RAW_BASE, d)) for d in os.listdir(RAW_BASE) if not d.startswith('.')):
        RAW_BASE = "data/processed/mips"

    if not os.path.exists(RAW_BASE):
        return pd.DataFrame(), pd.DataFrame()

    raw_items = []
    for group in sorted(os.listdir(RAW_BASE)):
        g_path = os.path.join(RAW_BASE, group)
        if not os.path.isdir(g_path) or group.startswith('.'):
            continue
        for sec in sorted(os.listdir(g_path)):
            s_path = os.path.join(g_path, sec)
            if not os.path.isdir(s_path) or sec.startswith('.'):
                continue
            for f in sorted(os.listdir(s_path)):
                if f.lower().endswith(('.tif', '.czi')):
                    base_name = os.path.splitext(f)[0]
                    raw_items.append((group, sec, f, base_name))

    for group, sec, fname, base_name in raw_items:
        metrics_dir = os.path.join(METRICS_BASE_DIR, group, sec)
        csv_file = os.path.join(metrics_dir, f"{base_name}_nuclei_metrics.csv")
        roi_file = os.path.join(metrics_dir, f"{base_name}_rois.json")
        if not os.path.exists(roi_file):
            roi_file = get_roi_json_path(metrics_dir, fname)

        regions_dict = load_rois(roi_file)
        has_a = len(regions_dict.get('A', [])) > 0
        has_b = len(regions_dict.get('B', [])) > 0
        has_c = len(regions_dict.get('C', [])) > 0
        has_any = (has_a or has_b or has_c)

        animal_id = extract_animal_id(fname)
        m2 = re.search(r'~(\d+)$', fname)
        corte_num = int(m2.group(1)) if m2 else 1

        image_catalog.append({
            'group': group, 'section': sec, 'image_name': fname, 'base_name': base_name,
            'animal_id': animal_id, 'corte_num': corte_num,
            'has_roi_a': has_a, 'has_roi_b': has_b, 'has_roi_c': has_c,
            'has_roi_any': has_any
        })

        if os.path.exists(csv_file):
            try:
                df_c = pd.read_csv(csv_file)
                if not df_c.empty:
                    df_c['group']      = group
                    df_c['section']    = sec
                    df_c['image_name'] = fname
                    df_c['animal_id']  = animal_id
                    df_c['corte_num']  = corte_num
                    
                    coords = df_c[['centroid_y', 'centroid_x']].values
                    df_c['is_in_roi']  = points_in_rois(coords, regions_dict, target_region='ALL') if has_any else False
                    df_c['roi_region'] = get_point_region_assignment(coords, regions_dict) if has_any else 'NONE'
                    all_cells.append(df_c)
            except Exception:
                pass

    df_cells = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    df_imgs = pd.DataFrame(image_catalog)
    return df_cells, df_imgs

df_raw_cells, df_all_images = load_all_dataset(_hash=get_dataset_mtime_hash())

if df_raw_cells.empty or df_all_images.empty:
    st.info("👋 No se encontraron métricas. Ejecuta el pipeline primero.")
    st.stop()

# Ensure backwards compatibility columns
if 'pv_area_um2' not in df_raw_cells.columns:
    df_raw_cells['pv_area_um2'] = df_raw_cells.apply(lambda r: r['area_um2'] if r['is_pv_plus'] else 0.0, axis=1)
    df_raw_cells['pv_diameter_um'] = df_raw_cells.apply(lambda r: r['diameter_um'] if r['is_pv_plus'] else 0.0, axis=1)
    df_raw_cells['pnn_area_um2'] = df_raw_cells.apply(lambda r: r['area_um2'] if r['is_pnn_plus'] else 0.0, axis=1)
    df_raw_cells['pnn_diameter_um'] = df_raw_cells.apply(lambda r: r['diameter_um'] if r['is_pnn_plus'] else 0.0, axis=1)
if 'score' not in df_raw_cells.columns:
    df_raw_cells['score'] = 0.0
if 'wfa_pericellular_norm' not in df_raw_cells.columns:
    df_raw_cells['wfa_pericellular_norm'] = df_raw_cells['wfa_mean_intensity'] if 'wfa_mean_intensity' in df_raw_cells.columns else 0.0

# ─── SIDEBAR CONTROLS ───
st.sidebar.header("⚙️ Configuración del Análisis")
if st.sidebar.button("🔄 Recargar Datos del Disco", key="ae_reload_btn"):
    st.cache_data.clear()
    st.rerun()

area_scope = st.sidebar.radio(
    "Área de Cuantificación:",
    [
        "🌐 Toda la Imagen (Global)",
        "🎯 Todas las ROIs (A, B y C combinadas)",
        "🅰️ Solo Región A",
        "🅱️ Solo Región B",
        "🅒 Solo Región C"
    ],
    index=0,
    key="ae_area_scope"
)

use_bonferroni = st.sidebar.checkbox(
    "Corrección Bonferroni (para comparaciones múltiples)",
    value=True, key="ae_bonferroni"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Configuración del Autoencoder")
latent_dim = st.sidebar.slider("Dimensiones Espacio Latente", 2, 5, 3, step=1)
epochs = st.sidebar.select_slider("Épocas de Entrenamiento PyTorch", options=[10, 20, 30, 50, 100], value=30)
n_clusters = st.sidebar.slider("Número de Clusters Fenotípicos (K-Means)", 2, 8, 3, step=1)

# Apply ROI Filter if selected
if area_scope == "🎯 Todas las ROIs (A, B y C combinadas)":
    valid_images_df = df_all_images[df_all_images['has_roi_any'] == True].copy()
    active_cells_df = df_raw_cells[(df_raw_cells['is_in_roi'] == True) & (df_raw_cells['image_name'].isin(valid_images_df['image_name']))].copy()
elif area_scope == "🅰️ Solo Región A":
    valid_images_df = df_all_images[df_all_images['has_roi_a'] == True].copy()
    active_cells_df = df_raw_cells[(df_raw_cells['roi_region'] == 'A') & (df_raw_cells['image_name'].isin(valid_images_df['image_name']))].copy()
elif area_scope == "🅱️ Solo Región B":
    valid_images_df = df_all_images[df_all_images['has_roi_b'] == True].copy()
    active_cells_df = df_raw_cells[(df_raw_cells['roi_region'] == 'B') & (df_raw_cells['image_name'].isin(valid_images_df['image_name']))].copy()
elif area_scope == "🅒 Solo Región C":
    valid_images_df = df_all_images[df_all_images['has_roi_c'] == True].copy()
    active_cells_df = df_raw_cells[(df_raw_cells['roi_region'] == 'C') & (df_raw_cells['image_name'].isin(valid_images_df['image_name']))].copy()
else:
    valid_images_df = df_all_images.copy()
    active_cells_df = df_raw_cells.copy()

# Add group metadata annotations
group_meta = {}
for g in active_cells_df['group'].dropna().unique():
    sex, cond, order = parse_group(g)
    if sex is not None:
        group_meta[g] = {'sex': sex, 'condition': cond, 'order': order}

active_cells_df['sex']       = active_cells_df['group'].map(lambda g: group_meta.get(g, {}).get('sex'))
active_cells_df['condition'] = active_cells_df['group'].map(lambda g: group_meta.get(g, {}).get('condition'))
active_cells_df = active_cells_df[active_cells_df['sex'].notna() & active_cells_df['condition'].notna() & active_cells_df['section'].isin(['IPSI', 'CONTRA'])].reset_index(drop=True)

# Summary Card
total_imgs_all = len(df_all_images)
valid_imgs_cnt = len(valid_images_df)
excluded_imgs_cnt = total_imgs_all - valid_imgs_cnt
total_cells_cnt = len(active_cells_df)
total_subjs_cnt = active_cells_df['animal_id'].nunique() if not active_cells_df.empty else 0

st.markdown('<div class="stats-card">', unsafe_allow_html=True)
c_b1, c_b2, c_b3, c_b4 = st.columns(4)
c_b1.metric("📐 Área Seleccionada", area_scope.split(" ")[1] if " " in area_scope else area_scope)
c_b2.metric("🐀 Sujetos Activos (N_suj)", f"{total_subjs_cnt}")
c_b3.metric("🔬 Preparados Validados (N_prep)", f"{valid_imgs_cnt} / {total_imgs_all}")
c_b4.metric("🧫 Células en Muestra (N_células)", f"{total_cells_cnt:,}")

if excluded_imgs_cnt > 0:
    st.warning(f"⚠️ **Atención sobre Tamaño Muestral ($N$):** Se excluyeron **{excluded_imgs_cnt}** preparados del análisis porque no tenían la ROI seleccionada (`{area_scope}`) trazada. Se analizan **{valid_imgs_cnt}** preparados para asegurar consistencia.")
else:
    st.success(f"✅ **Consistencia del Muestreo ($N$):** Se están considerando los **{valid_imgs_cnt}** preparados completos del dataset.")

st.markdown('</div>', unsafe_allow_html=True)

tab_ae, tab_lmm, tab_clusters = st.tabs([
    "🧠 Autoencoder & Espacio Latente (PyTorch)",
    "📉 Modelos Mixtos Lineales (LMM - Gran N)",
    "📊 Fenotipos & Proporciones por Grupo"
])

# ─── AUTOENCODER MODEL CLASS ───
class SingleCellAutoencoder(nn.Module):
    def __init__(self, in_features, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, in_features)
        )
    def forward(self, x):
        lat = self.encoder(x)
        rec = self.decoder(lat)
        return rec, lat

# Prepare Features
FEATURE_COLS = [
    'area_um2', 'diameter_um', 'wfa_sum_intensity',
    'wfa_mean_intensity', 'wfa_pericellular_norm',
    'pv_area_um2', 'score'
]
avail_features = [c for c in FEATURE_COLS if c in active_cells_df.columns]

@st.cache_data
def fit_autoencoder_and_clusters(df_features, feature_names, lat_dim, ep_cnt, k_cnt):
    X = df_features[feature_names].fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(512, len(X)), shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleCellAutoencoder(in_features=len(feature_names), latent_dim=lat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(ep_cnt):
        for (b_x,) in loader:
            b_x = b_x.to(device)
            optimizer.zero_grad()
            rec, _ = model(b_x)
            loss = criterion(rec, b_x)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        all_tensor_x = tensor_X.to(device)
        _, latent_repr = model(all_tensor_x)
        latent_np = latent_repr.cpu().numpy()

    kmeans = KMeans(n_clusters=k_cnt, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(latent_np)

    return latent_np, clusters

# ─── TAB 1: AUTOENCODER & ESPACIO LATENTE ───
with tab_ae:
    st.header("🧠 Red Neuronal Autoencoder & Reducción Espacial")
    st.markdown("""
    El Autoencoder comprime las múltiples variables morfológicas e intensitarias de cada célula en un **espacio latente continuo de baja dimensión ($Z$)**, capturando patrones biológicos no lineales sin supervisión.
    """)

    if len(active_cells_df) < 10:
        st.warning("No hay suficientes células en la muestra seleccionada para entrenar el Autoencoder.")
    else:
        with st.spinner("Entrenando Autoencoder en PyTorch y proyectando espacio latente..."):
            latent_coords, cluster_labels = fit_autoencoder_and_clusters(
                active_cells_df, avail_features, latent_dim, epochs, n_clusters
            )

        df_ae = active_cells_df.copy()
        for i in range(latent_dim):
            df_ae[f'Z_{i+1}'] = latent_coords[:, i]
        df_ae['Cluster'] = [f"Fenotipo {c+1}" for c in cluster_labels]

        col_ae1, col_ae2 = st.columns([3, 1])
        with col_ae2:
            st.markdown("### 🎨 Parámetros Visuales")
            color_by = st.selectbox(
                "Colorear Puntos por:",
                ["Cluster", "condition", "section", "sex", "cell_type"],
                format_func=lambda x: {
                    "Cluster": "Fenotipo Celular (Cluster)",
                    "condition": "Condición Experimental",
                    "section": "Hemisferio (IPSI / CONTRA)",
                    "sex": "Sexo (MACHO / HEMBRA)",
                    "cell_type": "Tipo Celular (PV+/PNN+)"
                }.get(x, x)
            )

        with col_ae1:
            if latent_dim >= 3:
                fig_ae = px.scatter_3d(
                    df_ae.head(5000), x='Z_1', y='Z_2', z='Z_3',
                    color=color_by,
                    hover_data=['animal_id', 'group', 'cell_type'],
                    title=f"Espacio Latente 3D (Autoencoder PyTorch) — N_células={len(df_ae):,}",
                    opacity=0.7,
                    template='plotly_dark'
                )
                fig_ae.update_traces(marker=dict(size=3))
                fig_ae.update_layout(height=650)
                st.plotly_chart(fig_ae, use_container_width=True)
            else:
                fig_ae = px.scatter(
                    df_ae.head(10000), x='Z_1', y='Z_2',
                    color=color_by,
                    hover_data=['animal_id', 'group', 'cell_type'],
                    title=f"Espacio Latente 2D (Autoencoder PyTorch) — N_células={len(df_ae):,}",
                    opacity=0.7,
                    template='plotly_dark'
                )
                fig_ae.update_layout(height=550)
                st.plotly_chart(fig_ae, use_container_width=True)

        st.divider()
        st.subheader("📋 Perfil de Fenotipos Celulares (Promedio por Cluster)")
        cluster_profile = df_ae.groupby('Cluster')[avail_features].mean().reset_index()
        st.dataframe(cluster_profile, use_container_width=True)

# ─── TAB 2: MODELOS MIXTOS LINEALES (LMM / LME) ───
with tab_lmm:
    st.header("📉 Modelos Mixtos Lineales (LMM / LME)")
    st.markdown("""
    **Ventaja del Modelo Mixto Lineal:**
    A diferencia de promediar las células por sujeto (lo que descarta la variabilidad intrínseca) o de tratar todas las células como independientes (lo que pseudoreplica y genera p-valores falsamente bajos), los **LMM** evalúan **las 160.000+ células individualmente** incluyendo un **efecto aleatorio por sujeto ($1 | \text{animal\_id}$)**.
    """)

    LMM_VARS = {
        "wfa_pericellular_norm": "Intensidad Circundante WFA Normalizada (Norm 0-1)",
        "wfa_mean_intensity": "Intensidad Media WFA Raw",
        "pnn_area_um2": "Área de PNN (µm²)",
        "pv_area_um2": "Área Soma PV+ (µm²)",
        "score": "Confianza PNNscore (IA)"
    }

    LMM_FORMULAS = {
        "Condición * Hemisferio + Sexo": "{y} ~ C(condition, Treatment(reference='NONE')) * C(section, Treatment(reference='CONTRA')) + C(sex)",
        "Condición * Hemisferio": "{y} ~ C(condition, Treatment(reference='NONE')) * C(section, Treatment(reference='CONTRA'))",
        "Condición + Hemisferio + Sexo (Aditivo)": "{y} ~ C(condition, Treatment(reference='NONE')) + C(section, Treatment(reference='CONTRA')) + C(sex)",
        "Condición * Hemisferio * Sexo (Interacción Completa)": "{y} ~ C(condition, Treatment(reference='NONE')) * C(section, Treatment(reference='CONTRA')) * C(sex)"
    }

    col_form1, col_form2 = st.columns([2, 2])
    with col_form1:
        sel_lmm_var = st.selectbox(
            "Seleccionar Variable Respuesta Single-Cell ($Y$):",
            list(LMM_VARS.keys()),
            format_func=lambda x: LMM_VARS[x],
            key="lmm_var"
        )
    with col_form2:
        sel_formula_name = st.selectbox(
            "Estructura del Modelo LMM (Fórmula):",
            list(LMM_FORMULAS.keys()),
            index=0,
            key="lmm_formula_name"
        )

    if active_cells_df.empty or sel_lmm_var not in active_cells_df.columns:
        st.warning("No hay suficientes datos celulares para ajustar el Modelo Mixto Lineal.")
    else:
        df_lmm_full = active_cells_df[[sel_lmm_var, 'condition', 'section', 'sex', 'animal_id', 'group']].dropna()
        df_lmm_full['condition'] = pd.Categorical(df_lmm_full['condition'], categories=COND_ORDER, ordered=True)
        df_lmm_full = df_lmm_full.dropna().reset_index(drop=True)

        # Optimize sample size for instant REML convergence if dataset > 30,000 cells
        if len(df_lmm_full) > 30000:
            df_lmm = df_lmm_full.sample(n=30000, random_state=42).reset_index(drop=True)
            st.info(f"⚡ **Optimización Muestral Activa:** Se seleccionó un submuestreo representativo de **30.000 células** (de {len(df_lmm_full):,}) para un ajuste REML instantáneo e hiperpreciso del Modelo Mixto.")
        else:
            df_lmm = df_lmm_full
        
        with st.spinner("Ajustando Modelo Mixto Lineal (statsmodels MixedLM)..."):
            result_lmm = None
            formula_str = LMM_FORMULAS[sel_formula_name].format(y=sel_lmm_var)
            
            try:
                model_lmm = smf.mixedlm(formula_str, df_lmm, groups="animal_id")
                result_lmm = model_lmm.fit(reml=True)
            except Exception as e_primary:
                # Robust Fallback Ladder
                st.warning(f"⚠️ La fórmula de interacción seleccionada generó una matriz singular o colineal ({e_primary}). Probando estructura alternativa...")
                fallback_formulas = [
                    LMM_FORMULAS["Condición * Hemisferio"].format(y=sel_lmm_var),
                    LMM_FORMULAS["Condición + Hemisferio + Sexo (Aditivo)"].format(y=sel_lmm_var)
                ]
                for f_alt in fallback_formulas:
                    try:
                        model_alt = smf.mixedlm(f_alt, df_lmm, groups="animal_id")
                        result_lmm = model_alt.fit(reml=True)
                        formula_str = f_alt
                        st.info(f"💡 Se utilizó con éxito la fórmula alternativa: `{formula_str}`")
                        break
                    except Exception:
                        pass

            if result_lmm is not None:
                st.success(f"✅ Modelo Mixto Lineal ajustado correctamente con optimización REML. (Fórmula: `{formula_str}`)")

                # Fixed Effects Table
                fe_summary = pd.DataFrame({
                    "Coeficiente (β)": result_lmm.params,
                    "Error Estándar": result_lmm.bse,
                    "z-score": result_lmm.tvalues,
                    "p-valor": result_lmm.pvalues,
                    "IC 95% Inferior": result_lmm.conf_int()[0],
                    "IC 95% Superior": result_lmm.conf_int()[1]
                })

                if use_bonferroni:
                    n_tests = len(fe_summary)
                    fe_summary["p-valor (Bonferroni)"] = np.minimum(fe_summary["p-valor"] * n_tests, 1.0)

                fe_summary["Significancia"] = fe_summary["p-valor"].apply(lambda p: "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns")))

                col_lmm1, col_lmm2 = st.columns([2, 1])

                with col_lmm1:
                    st.markdown(f'<div class="table-caption">📋 Efectos Fijos del Modelo (Variable: {LMM_VARS[sel_lmm_var]})</div>', unsafe_allow_html=True)
                    num_cols = [c for c in fe_summary.columns if c != "Significancia"]
                    st.dataframe(fe_summary.style.format("{:.4f}", subset=num_cols), width="stretch")

                with col_lmm2:
                    st.markdown('<div class="table-caption">🔍 Componentes de Varianza y Ajuste</div>', unsafe_allow_html=True)
                    var_random = float(result_lmm.cov_re.iloc[0, 0]) if hasattr(result_lmm, 'cov_re') else 0.0
                    var_resid = float(result_lmm.scale)
                    icc = var_random / (var_random + var_resid + 1e-8)

                    st.markdown(
                        f"* **Varianza Entre Animales (Sujetos):** `{var_random:.4f}`\n"
                        f"* **Varianza Residual:** `{var_resid:.4f}`\n"
                        f"* **Coeficiente de Correlación Intraclase (ICC):** `{icc:.4f}` ({icc*100:.1f}% de la varianza explicada por diferencias individuales).\n"
                        f"* **Log-Likelihood:** `{result_lmm.llf:.2f}`\n"
                        f"* **$N$ total células:** `{len(df_lmm):,}`\n"
                        f"* **$N$ grupos (animales):** `{df_lmm['animal_id'].nunique()}`"
                    )

                # Plot Marginal Means
                st.divider()
                st.subheader("📈 Promedios Marginales Estimados (Efectos Fijos LMM)")
                marginal_means = df_lmm.groupby(['condition', 'section', 'sex'])[sel_lmm_var].agg(['mean', 'std', 'count']).reset_index()

                fig_lmm = px.bar(
                    marginal_means, x='condition', y='mean', color='section', barmode='group',
                    facet_col='sex', error_y='std',
                    title=f"Estimaciones por Condición, Hemisferio y Sexo — {LMM_VARS[sel_lmm_var]}",
                    color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                    template='plotly_dark'
                )
                fig_lmm.update_layout(height=450)
                st.plotly_chart(fig_lmm, use_container_width=True)
            else:
                st.error("No se pudo ajustar el Modelo Mixto Lineal con las fórmulas disponibles. Verifique los datos.")

# ─── TAB 3: FENOTIPOS & PROPORCIONES POR GRUPO ───
with tab_clusters:
    st.header("📊 Distribución de Fenotipos por Condición y Hemisferio")
    st.markdown("""
    Evalúa la reorganización de subpoblaciones celulares a lo largo del tiempo (`NONE` $\\rightarrow$ `3 DÍAS` $\\rightarrow$ `14 DÍAS`) para determinar si la neuroplasticidad altera las proporciones relativas de cada fenotipo.
    """)

    if len(active_cells_df) >= 10:
        latent_coords, cluster_labels = fit_autoencoder_and_clusters(
            active_cells_df, avail_features, latent_dim, epochs, n_clusters
        )
        df_prop = active_cells_df.copy()
        df_prop['Cluster'] = [f"Fenotipo {c+1}" for c in cluster_labels]

        ct = pd.crosstab(index=df_prop['condition'], columns=df_prop['Cluster'], normalize='index') * 100

        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            fig_prop = px.bar(
                ct, x=ct.index, y=ct.columns,
                title="Proporción de Fenotipos Celulares (%) por Condición Experimental",
                labels={'value': 'Porcentaje (%)', 'condition': 'Condición'},
                template='plotly_dark'
            )
            fig_prop.update_layout(height=450)
            st.plotly_chart(fig_prop, use_container_width=True)

        with col_p2:
            st.markdown('<div class="table-caption">🧪 Prueba de Chi-Cuadrada (Independencia)</div>', unsafe_allow_html=True)
            ct_raw = pd.crosstab(index=df_prop['condition'], columns=df_prop['Cluster'])
            chi2, p_val, dof, _ = chi2_contingency(ct_raw)

            st.markdown(f"""
            * **Estadístico $\chi^2$:** `{chi2:.2f}`
            * **Grados de Libertad:** `{dof}`
            * **p-valor:** `{p_val:.4e}`
            * **Resultado:** {"✅ Cambio significativo en proporciones fenotípicas" if p_val < 0.05 else "ns (Sin cambio significativo)"}
            """)

            st.dataframe(ct_raw, use_container_width=True)
