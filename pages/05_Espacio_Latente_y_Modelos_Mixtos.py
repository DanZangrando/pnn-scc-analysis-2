import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score

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
    elif '7' in name_norm and 'DIA' in name_norm: cond, order = '7 DÍAS', 2
    elif '14' in name_norm and 'DIA' in name_norm: cond, order = '14 DÍAS', 3
    else: cond, order = name, 99

    return sex, cond, order

COND_ORDER  = ['NONE', '3 DÍAS', '7 DÍAS', '14 DÍAS']
COND_COLORS = {'NONE': '#4facfe', '3 DÍAS': '#bb86fc', '7 DÍAS': '#f59e0b', '14 DÍAS': '#00ffcc'}
SEX_COLORS  = {'MACHO': '#5bc0de', 'HEMBRA': '#e83e8c'}

def get_color_discrete_map(color_by, unique_values):
    if color_by == 'condition':
        return COND_COLORS
    elif color_by == 'section':
        return {'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'}
    elif color_by == 'sex':
        return SEX_COLORS
    elif color_by == 'cell_type':
        return {'PV+/PNN+': '#00ff7f', 'PV-/PNN+': '#ff4757', 'PV+/PNN-': '#ffa502'}
    elif color_by == 'roi_region':
        return {'A': '#00ffff', 'B': '#ff00ff', 'C': '#ffff00', 'NONE': '#64748b'}
    elif color_by == 'cond_sec':
        cond_sec_palette = {
            'NONE (CONTRA)': '#38bdf8', 'NONE (IPSI)': '#00f2fe',
            '3 DÍAS (CONTRA)': '#c084fc', '3 DÍAS (IPSI)': '#a855f7',
            '7 DÍAS (CONTRA)': '#fbbf24', '7 DÍAS (IPSI)': '#f59e0b',
            '14 DÍAS (CONTRA)': '#34d399', '14 DÍAS (IPSI)': '#10b981'
        }
        res = {str(v): cond_sec_palette[str(v)] for v in unique_values if str(v) in cond_sec_palette}
        if len(res) == len(unique_values):
            return res
    elif color_by == 'sex_cond':
        sex_cond_palette = {
            'MACHO - NONE': '#38bdf8', 'MACHO - 3 DÍAS': '#818cf8', 'MACHO - 7 DÍAS': '#fbbf24', 'MACHO - 14 DÍAS': '#34d399',
            'HEMBRA - NONE': '#f472b6', 'HEMBRA - 3 DÍAS': '#e879f9', 'HEMBRA - 7 DÍAS': '#fb923c', 'HEMBRA - 14 DÍAS': '#2dd4bf'
        }
        res = {str(v): sex_cond_palette[str(v)] for v in unique_values if str(v) in sex_cond_palette}
        if len(res) == len(unique_values):
            return res
            
    qual_palette = [
        '#00f2fe', '#bb86fc', '#ff7b00', '#00ff7f', '#e83e8c', '#ffff00',
        '#f59e0b', '#38bdf8', '#c084fc', '#4ade80', '#fb7185', '#a3e635',
        '#f43f5e', '#8b5cf6', '#06b6d4', '#f97316', '#ec4899', '#14b8a6'
    ]
    return {str(val): qual_palette[idx % len(qual_palette)] for idx, val in enumerate(unique_values)}

def compute_cramers_v(contingency_table):
    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    if n == 0:
        return 0.0
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))

def compute_standardized_residuals(contingency_table):
    O = contingency_table.values.astype(float)
    N = O.sum()
    if N == 0 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return pd.DataFrame(0.0, index=contingency_table.index, columns=contingency_table.columns)
    
    R = O.sum(axis=1, keepdims=True)
    C = O.sum(axis=0, keepdims=True)
    E = (R @ C) / N
    
    row_prop = R / N
    col_prop = C / N
    
    denom = np.sqrt(E * (1.0 - row_prop) * (1.0 - col_prop) + 1e-8)
    adj_residuals = (O - E) / denom
    return pd.DataFrame(adj_residuals, index=contingency_table.index, columns=contingency_table.columns)

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

st.sidebar.subheader("🎯 Filtros Poblacionales de Muestra")

area_scope = st.sidebar.radio(
    "Área de Cuantificación / ROIs:",
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

# 1. Apply ROI Filter
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

# Dynamic Filters for Hemisphere, Sex, Condition and Cell Type
section_scope = st.sidebar.radio(
    "Hemisferio / Sección:",
    ["🌐 Ambos (IPSI + CONTRA)", "⚡ Solo IPSI (Lesión / Exp)", "🛡️ Solo CONTRA (Control)"],
    index=0,
    key="ae_section_scope"
)
if section_scope == "⚡ Solo IPSI (Lesión / Exp)":
    active_cells_df = active_cells_df[active_cells_df['section'] == 'IPSI']
elif section_scope == "🛡️ Solo CONTRA (Control)":
    active_cells_df = active_cells_df[active_cells_df['section'] == 'CONTRA']

sex_scope = st.sidebar.radio(
    "Sexo:",
    ["👫 Ambos (Macho + Hembra)", "♂️ Solo MACHOS", "♀️ Solo HEMBRAS"],
    index=0,
    key="ae_sex_scope"
)
if sex_scope == "♂️ Solo MACHOS":
    active_cells_df = active_cells_df[active_cells_df['sex'] == 'MACHO']
elif sex_scope == "♀️ Solo HEMBRAS":
    active_cells_df = active_cells_df[active_cells_df['sex'] == 'HEMBRA']

# Dynamic Conditions Filter
all_available_conds = [c for c in COND_ORDER if c in active_cells_df['condition'].unique()]
if not all_available_conds:
    all_available_conds = sorted(active_cells_df['condition'].dropna().unique().tolist())

selected_conds = st.sidebar.multiselect(
    "Condiciones Experimentales a incluir:",
    options=all_available_conds,
    default=all_available_conds,
    key="ae_selected_conds"
)
if selected_conds:
    active_cells_df = active_cells_df[active_cells_df['condition'].isin(selected_conds)]

# Cell Type Filter
avail_cell_types = ["Todos los tipos"] + sorted(active_cells_df['cell_type'].dropna().unique().tolist())
cell_type_scope = st.sidebar.selectbox(
    "Tipo Celular:",
    avail_cell_types,
    index=0,
    key="ae_cell_type_scope"
)
if cell_type_scope != "Todos los tipos":
    active_cells_df = active_cells_df[active_cells_df['cell_type'] == cell_type_scope]

active_cells_df = active_cells_df.reset_index(drop=True)

# Generate composite grouping columns
if not active_cells_df.empty:
    active_cells_df['cond_sec'] = active_cells_df['condition'] + " (" + active_cells_df['section'] + ")"
    active_cells_df['sex_cond'] = active_cells_df['sex'] + " - " + active_cells_df['condition']
    active_cells_df['group_sec'] = active_cells_df['group'] + " - " + active_cells_df['section']

use_bonferroni = st.sidebar.checkbox(
    "Corrección Bonferroni (para comparaciones múltiples)",
    value=True, key="ae_bonferroni"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Configuración del Autoencoder")
latent_dim = st.sidebar.slider("Dimensiones Espacio Latente", 2, 5, 3, step=1)
epochs = st.sidebar.select_slider("Épocas de Entrenamiento PyTorch", options=[10, 20, 30, 50, 100], value=30)
n_clusters = st.sidebar.slider("Número de Clusters Fenotípicos (K-Means)", 2, 8, 3, step=1)

# Summary Card
total_imgs_all = len(df_all_images)
valid_imgs_cnt = active_cells_df['image_name'].nunique() if not active_cells_df.empty else 0
excluded_imgs_cnt = total_imgs_all - valid_imgs_cnt
total_cells_cnt = len(active_cells_df)
total_subjs_cnt = active_cells_df['animal_id'].nunique() if not active_cells_df.empty else 0

st.markdown('<div class="stats-card">', unsafe_allow_html=True)
c_b1, c_b2, c_b3, c_b4 = st.columns(4)
c_b1.metric("📐 Filtro / Población Activa", f"{len(active_cells_df['condition'].unique()) if not active_cells_df.empty else 0} Conds")
c_b2.metric("🐀 Sujetos Activos (N_suj)", f"{total_subjs_cnt}")
c_b3.metric("🔬 Preparados Validados (N_prep)", f"{valid_imgs_cnt} / {total_imgs_all}")
c_b4.metric("🧫 Células en Muestra (N_células)", f"{total_cells_cnt:,}")

if total_cells_cnt == 0:
    st.error("⚠️ **No hay células que coincidan con la combinación de filtros seleccionada.** Por favor ajusta los filtros en la barra lateral.")
    st.stop()
elif excluded_imgs_cnt > 0:
    st.warning(f"ℹ️ **Filtros Activos:** Se analizan **{total_cells_cnt:,} células** de **{valid_imgs_cnt}** preparados correspondientes al subconjunto filtrado.")
else:
    st.success(f"✅ **Muestreo Completo:** Se están considerando todas las **{total_cells_cnt:,} células** de los **{valid_imgs_cnt}** preparados del dataset.")

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

def get_descriptive_cluster_names(df_temp, cluster_col, cond_col='condition', wfa_col='wfa_pericellular_norm'):
    cluster_names = {}
    mean_wfa_global = df_temp[wfa_col].mean() if wfa_col in df_temp.columns else 1.0
    for cl in sorted(df_temp[cluster_col].unique()):
        sub = df_temp[df_temp[cluster_col] == cl]
        if sub.empty:
            cluster_names[cl] = f"Cluster {cl+1}"
            continue
        
        # Dominant condition
        dom_cond = sub[cond_col].mode().iloc[0] if cond_col in sub.columns and not sub[cond_col].empty else f"C{cl+1}"
        dom_pct = int(round((sub[cond_col] == dom_cond).mean() * 100)) if cond_col in sub.columns else 0
        
        # Relative WFA intensity
        mean_wfa_cl = sub[wfa_col].mean() if wfa_col in sub.columns else 1.0
        intensity_tag = "WFA-Alto" if mean_wfa_cl >= 1.15 * mean_wfa_global else ("WFA-Bajo" if mean_wfa_cl <= 0.85 * mean_wfa_global else "WFA-Medio")
        
        cluster_names[cl] = f"Cluster {cl+1} [{intensity_tag} | {dom_cond} {dom_pct}%]"
    return cluster_names

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
            
        df_ae['Cluster_ID'] = cluster_labels
        descriptive_map = get_descriptive_cluster_names(df_ae, 'Cluster_ID', cond_col='cond_sec')
        df_ae['Cluster_Descriptivo'] = df_ae['Cluster_ID'].map(descriptive_map)
        df_ae['Cluster_Simple'] = [f"Cluster {c+1}" for c in cluster_labels]

        col_ae1, col_ae2 = st.columns([3, 1])
        with col_ae2:
            st.markdown("### 🎨 Parámetros Visuales")
            
            cluster_naming = st.radio(
                "Nomenclatura de Clusters IA:",
                ["🏷️ Descriptiva (Biomarcador + Grupo Dominante)", "🔢 Numérica Simple (Cluster 1, 2...)"],
                index=0,
                key="ae_cluster_naming"
            )
            df_ae['Cluster'] = df_ae['Cluster_Descriptivo'] if "Descriptiva" in cluster_naming else df_ae['Cluster_Simple']

            color_options = [
                "Cluster",
                "condition",
                "cond_sec",
                "sex_cond",
                "section",
                "sex",
                "cell_type",
                "roi_region",
                "group"
            ]
            
            format_dict = {
                "Cluster": "🤖 Clusters IA (K-Means)",
                "condition": "🏷️ Condición (NONE, 3d, 7d, 14d)",
                "cond_sec": "⚡ Condición × Hemisferio (ej. 3d IPSI)",
                "sex_cond": "👫 Sexo × Condición (ej. Macho 3d)",
                "section": "🛡️ Hemisferio (IPSI / CONTRA)",
                "sex": "⚧ Sexo (MACHO / HEMBRA)",
                "cell_type": "🧪 Tipo Celular (PV+/PNN+)",
                "roi_region": "🎯 Región ROI (A, B, C)",
                "group": "🔬 Grupo Completo"
            }
            
            color_by = st.selectbox(
                "Colorear Puntos por:",
                color_options,
                format_func=lambda x: format_dict.get(x, x),
                key="ae_color_by"
            )

        with col_ae1:
            hover_cols = ['animal_id', 'group', 'section', 'cell_type', 'roi_region']
            color_map = get_color_discrete_map(color_by, sorted(df_ae[color_by].dropna().unique()))
            
            if latent_dim >= 3:
                fig_ae = px.scatter_3d(
                    df_ae.head(5000), x='Z_1', y='Z_2', z='Z_3',
                    color=color_by,
                    color_discrete_map=color_map,
                    hover_data=hover_cols,
                    title=f"Espacio Latente 3D (Autoencoder PyTorch) — N_células={len(df_ae):,}",
                    opacity=0.75,
                    template='plotly_dark'
                )
                fig_ae.update_traces(marker=dict(size=3.5))
                fig_ae.update_layout(height=650)
                st.plotly_chart(fig_ae, use_container_width=True)
            else:
                fig_ae = px.scatter(
                    df_ae.head(10000), x='Z_1', y='Z_2',
                    color=color_by,
                    color_discrete_map=color_map,
                    hover_data=hover_cols,
                    title=f"Espacio Latente 2D (Autoencoder PyTorch) — N_células={len(df_ae):,}",
                    opacity=0.75,
                    template='plotly_dark'
                )
                fig_ae.update_layout(height=550)
                st.plotly_chart(fig_ae, use_container_width=True)

        st.divider()
        st.subheader("📋 Perfil de Fenotipos Celulares (Promedio de Biomarcadores por Cluster)")
        cluster_profile = df_ae.groupby('Cluster')[avail_features].mean().reset_index()
        st.dataframe(cluster_profile, use_container_width=True)

# ─── TAB 2: MODELOS MIXTOS LINEALES (LMM / LME) ───
with tab_lmm:
    st.header("📉 Modelos Mixtos Lineales (LMM / LME)")
    st.markdown("""
    **Ventaja del Modelo Mixto Lineal:**
    A diferencia de promediar las células por sujeto (lo que descarta la variabilidad intrínseca) o de tratar todas las células como independientes (lo que pseudoreplica y genera p-valores falsamente bajos), los **LMM** evalúan **las células individualmente** incluyendo un **efecto aleatorio por sujeto ($1 | \text{animal\_id}$)**.
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
    st.header("📊 Distribución, Proporciones y Validación Estadística de Fenotipos")
    st.markdown("""
    Evalúa de forma interactiva la reorganización de subpoblaciones celulares a través de las condiciones experimentales (`NONE`, `3 DÍAS`, `7 DÍAS`, `14 DÍAS`), hemisferios (`IPSI` vs `CONTRA`) y sexos (`MACHO` vs `HEMBRA`).
    """)

    if len(active_cells_df) >= 10:
        latent_coords, cluster_labels = fit_autoencoder_and_clusters(
            active_cells_df, avail_features, latent_dim, epochs, n_clusters
        )
        df_prop = active_cells_df.copy()
        df_prop['Cluster_ID'] = cluster_labels
        descriptive_map = get_descriptive_cluster_names(df_prop, 'Cluster_ID', cond_col='cond_sec')
        df_prop['Cluster_Descriptivo'] = df_prop['Cluster_ID'].map(descriptive_map)
        df_prop['Cluster_Simple'] = [f"Cluster {c+1}" for c in cluster_labels]
        df_prop['Cluster'] = df_prop['Cluster_Descriptivo'] if "Descriptiva" in st.session_state.get('ae_cluster_naming', 'Descriptiva') else df_prop['Cluster_Simple']

        col_ctl1, col_ctl2 = st.columns(2)
        with col_ctl1:
            x_group_by = st.selectbox(
                "Agrupar Eje X por:",
                ["condition", "cond_sec", "sex_cond", "section", "sex", "group"],
                format_func=lambda x: {
                    "condition": "Condición Experimental (NONE, 3d, 7d, 14d)",
                    "cond_sec": "Condición × Hemisferio (ej. 3d IPSI vs 3d CONTRA)",
                    "sex_cond": "Sexo × Condición (ej. Macho 3d vs Hembra 3d)",
                    "section": "Hemisferio (IPSI vs CONTRA)",
                    "sex": "Sexo (MACHO vs HEMBRA)",
                    "group": "Grupo Experimental Completo"
                }.get(x, x),
                key="prop_x_group"
            )
        with col_ctl2:
            subpop_by = st.selectbox(
                "Subpoblación / Barras:",
                ["Cluster", "cell_type", "roi_region"],
                format_func=lambda x: {
                    "Cluster": "Clusters IA del Autoencoder",
                    "cell_type": "Tipo Celular (PV+/PNN+, PV-/PNN+, etc.)",
                    "roi_region": "Región ROI (A, B, C)"
                }.get(x, x),
                key="prop_subpop"
            )

        subpop_map = get_color_discrete_map(subpop_by, sorted(df_prop[subpop_by].dropna().unique()))
        ct = pd.crosstab(index=df_prop[x_group_by], columns=df_prop[subpop_by], normalize='index') * 100
        ct_raw = pd.crosstab(index=df_prop[x_group_by], columns=df_prop[subpop_by])

        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            fig_prop = px.bar(
                ct, x=ct.index, y=ct.columns,
                title=f"Proporción (%) de {format_dict.get(subpop_by, subpop_by)} por {x_group_by}",
                labels={'value': 'Porcentaje (%)', x_group_by: 'Grupo'},
                color_discrete_map=subpop_map,
                template='plotly_dark'
            )
            fig_prop.update_layout(height=480, barmode='stack')
            st.plotly_chart(fig_prop, use_container_width=True)

        with col_p2:
            st.markdown('<div class="table-caption">🧪 Métricas y Tests de Asociación Estadística</div>', unsafe_allow_html=True)
            if ct_raw.shape[0] > 1 and ct_raw.shape[1] > 1:
                chi2, p_val, dof, _ = chi2_contingency(ct_raw)
                v_cramer = compute_cramers_v(ct_raw)
                
                try:
                    nmi_score = normalized_mutual_info_score(df_prop[x_group_by].astype(str), df_prop[subpop_by].astype(str))
                    homog_score = homogeneity_score(df_prop[x_group_by].astype(str), df_prop[subpop_by].astype(str))
                except Exception:
                    nmi_score, homog_score = np.nan, np.nan
                
                v_interp = "Fuerte (Asociación marcada)" if v_cramer >= 0.30 else ("Moderada" if v_cramer >= 0.10 else "Débil")
                
                st.markdown(f"""
                * **Chi-Cuadrada ($\chi^2$):** `{chi2:.2f}` (gl: `{dof}`, $p = {p_val:.4e}$)
                * **$V$ de Cramér (Fuerza de Efecto):** `{v_cramer:.4f}` $\rightarrow$ **{v_interp}**
                * **NMI (Información Mutua Normalizada):** `{nmi_score:.4f}`
                * **Homogeneidad de Partición:** `{homog_score:.4f}`
                * **Conclusión:** {"✅ Asociación altamente significativa entre grupos y fenotipos ($p < 0.05$)" if p_val < 0.05 else "ns (Sin asociación estadísticamente significativa)"}
                """)
            else:
                st.info("Se requiere más de 1 categoría en ambos ejes para calcular $\chi^2$ y $V$ de Cramér.")

            st.markdown('<div class="table-caption">📊 Conteo Absoluto de Células ($N$)</div>', unsafe_allow_html=True)
            st.dataframe(ct_raw, use_container_width=True)

        # ─── HEATMAP DE RESIDUOS ESTANDARIZADOS (ENRIQUECIMIENTO) ───
        if ct_raw.shape[0] > 1 and ct_raw.shape[1] > 1:
            st.divider()
            st.subheader("🔥 Mapa de Calor de Enriquecimiento / Depleción Fenotípica (Residuos Ajustados Z)")
            st.markdown("""
            Los **Residuos Estandarizados Ajustados ($Z$)** cuantifican con precisión qué combinaciones de Grupo $\\times$ Fenotipo están **significativamente alteradas**:
            * 🟥 **$Z > +1.96$ ($p < 0.05$):** Fenotipo **significativamente enriquecido / sobrerrepresentado**.
            * 🟦 **$Z < -1.96$ ($p < 0.05$):** Fenotipo **significativamente depletado / disminuido**.
            * ⬜ **$-1.96 \\le Z \\le +1.96$:** Frecuencia esperada por azar (neutral).
            """)
            
            res_df = compute_standardized_residuals(ct_raw)
            max_abs_z = float(max(4.0, np.nanmax(np.abs(res_df.values))))
            fig_hm = px.imshow(
                res_df,
                text_auto=".2f",
                color_continuous_scale="balance",
                zmin=-max_abs_z,
                zmax=max_abs_z,
                title=f"Enriquecimiento Fenotípico (Z-Scores de Haberman): {x_group_by} vs {subpop_by}",
                labels=dict(x=subpop_by, y=x_group_by, color="Z-score"),
                template='plotly_dark'
            )
            fig_hm.update_layout(height=450)
            st.plotly_chart(fig_hm, use_container_width=True)

