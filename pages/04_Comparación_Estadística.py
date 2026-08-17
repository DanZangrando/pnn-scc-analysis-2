import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import mannwhitneyu, wilcoxon, f_oneway, kruskal, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import sys
import tifffile as tiff
sys.path.append(os.path.abspath("src"))
from roi import load_rois, get_roi_json_path, points_in_rois, get_point_region_assignment, create_roi_mask
from image_io import extract_animal_id

st.set_page_config(page_title="Comparación Estadística", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMarkdown h1 {
        color: #bb86fc;
        text-align: center;
        background: linear-gradient(120deg, #bb86fc 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        margin-bottom: 25px;
    }
    .stats-box {
        background: rgba(30, 33, 48, 0.7);
        border: 1px solid rgba(187, 134, 252, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .level-info {
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 16px;
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

st.title("📊 Paso 4: Comparación Estadística")

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
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    name_norm = strip_accents(name).upper().replace('_', ' ')

    sex = None
    if 'HEMBRA' in name_norm:
        sex = 'HEMBRA'
    elif 'MACHO' in name_norm:
        sex = 'MACHO'

    if 'NONE' in name_norm:
        cond, order = 'NONE', 0
    elif '3' in name_norm and 'DIA' in name_norm:
        cond, order = '3 DÍAS', 1
    elif '7' in name_norm and 'DIA' in name_norm:
        cond, order = '7 DÍAS', 2
    elif '14' in name_norm and 'DIA' in name_norm:
        cond, order = '14 DÍAS', 3
    else:
        cond, order = name, 99

    return sex, cond, order

def sig_stars(p, alpha=0.05):
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < alpha:
        return "*"
    else:
        return "ns"

def run_mwu(a, b):
    a = np.array(a)[~np.isnan(a)]
    b = np.array(b)[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    try:
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        return stat, p
    except Exception:
        return np.nan, np.nan

COND_ORDER  = ['NONE', '3 DÍAS', '7 DÍAS', '14 DÍAS']
COND_COLORS = {'NONE': '#4facfe', '3 DÍAS': '#bb86fc', '7 DÍAS': '#f59e0b', '14 DÍAS': '#00ffcc'}
SEX_COLORS  = {'MACHO': '#5bc0de', 'HEMBRA': '#e83e8c'}

def _ensure_roi_annotations(df):
    if df.empty or ('is_in_roi' in df.columns and 'roi_region' in df.columns):
        return df
    
    df = df.copy()
    is_roi_list = []
    roi_region_list = []
    roi_cache = {}
    
    for idx, row in df.iterrows():
        img_name = row.get('image_name', '')
        group = row.get('group', '')
        section = row.get('section', '')
        cy = row.get('centroid_y', 0)
        cx = row.get('centroid_x', 0)
        
        cache_key = (group, section, img_name)
        if cache_key not in roi_cache:
            roi_json_path = os.path.join(METRICS_BASE_DIR, group, section, f"{img_name}_rois.json")
            if not os.path.exists(roi_json_path):
                roi_json_path = get_roi_json_path(METRICS_BASE_DIR, img_name)
            roi_cache[cache_key] = load_rois(roi_json_path)
            
        regions_dict = roi_cache[cache_key]
        in_roi = bool(points_in_rois([[cy, cx]], regions_dict, target_region="ALL")[0]) if regions_dict else False
        reg_name = get_point_region_assignment([[cy, cx]], regions_dict)[0] if regions_dict else "NONE"
        
        is_roi_list.append(in_roi)
        roi_region_list.append(reg_name)
        
    df['is_in_roi'] = is_roi_list
    df['roi_region'] = roi_region_list
    return df

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

df_raw_nuclei, df_all_images = load_all_dataset(_hash=get_dataset_mtime_hash())

if df_raw_nuclei.empty or df_all_images.empty:
    st.info("👋 No se encontraron métricas. Ejecuta el pipeline primero.")
    st.stop()

# Ensure backwards compatibility columns for nuclei
if not df_raw_nuclei.empty:
    if 'pv_area_um2' not in df_raw_nuclei.columns:
        df_raw_nuclei['pv_area_um2']    = df_raw_nuclei.apply(lambda r: r['area_um2'] if r['is_pv_plus'] else 0.0, axis=1)
        df_raw_nuclei['pv_diameter_um'] = df_raw_nuclei.apply(lambda r: r['diameter_um'] if r['is_pv_plus'] else 0.0, axis=1)
        df_raw_nuclei['pnn_area_um2']   = df_raw_nuclei.apply(lambda r: r['area_um2'] if r['is_pnn_plus'] else 0.0, axis=1)
        df_raw_nuclei['pnn_diameter_um']= df_raw_nuclei.apply(lambda r: r['diameter_um'] if r['is_pnn_plus'] else 0.0, axis=1)
    if 'score' not in df_raw_nuclei.columns:
        df_raw_nuclei['score'] = 0.0

SKEL_COLS = ['score']

# Sidebar controls
st.sidebar.header("⚙️ Configuración del Análisis")
if st.sidebar.button("🔄 Recargar Datos del Disco", key="stats_reload_btn"):
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
    key="stats_area_scope"
)

level_type = st.sidebar.radio(
    "Nivel de Análisis:",
    ["Por Sujeto (animal, promediando cortes)",
     "Por Preparado/Corte (imagen)",
     "Por Célula (distribuciones individuales)"],
    index=0,
    key="stats_level_select"
)

use_bonferroni = st.sidebar.checkbox(
    "Corrección Bonferroni (para comparaciones múltiples)",
    value=True, key="stats_bonferroni"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filtros de Cohorte (Subpoblación)")
filter_sex = st.sidebar.selectbox(
    "Filtro de Sexo:",
    ["Ambos (MACHO y HEMBRA)", "Solo MACHO", "Solo HEMBRA"],
    index=0,
    key="stats_filter_sex"
)

filter_sec = st.sidebar.selectbox(
    "Filtro de Hemisferio:",
    ["Ambos (IPSI y CONTRA)", "Solo IPSI", "Solo CONTRA"],
    index=0,
    key="stats_filter_sec"
)

stats_mode = st.sidebar.radio(
    "Modo de Análisis Estadístico:",
    [
        "🔬 ANOVA de 1 Vía (Condiciones) + Post-Hoc",
        "📊 Desglose Tradicional (Panel por Sexo)"
    ],
    index=0,
    key="stats_mode_select"
)

# ─── ROI COVERAGE & DATA FILTERING ───
if area_scope == "🎯 Todas las ROIs (A, B y C combinadas)":
    valid_images_df = df_all_images[df_all_images['has_roi_any'] == True].copy()
    active_cells_df = df_raw_nuclei[(df_raw_nuclei['is_in_roi'] == True) & (df_raw_nuclei['image_name'].isin(valid_images_df['image_name']))].copy()
elif area_scope == "🅰️ Solo Región A":
    valid_images_df = df_all_images[df_all_images['has_roi_a'] == True].copy()
    active_cells_df = df_raw_nuclei[(df_raw_nuclei['roi_region'] == 'A') & (df_raw_nuclei['image_name'].isin(valid_images_df['image_name']))].copy()
elif area_scope == "🅱️ Solo Región B":
    valid_images_df = df_all_images[df_all_images['has_roi_b'] == True].copy()
    active_cells_df = df_raw_nuclei[(df_raw_nuclei['roi_region'] == 'B') & (df_raw_nuclei['image_name'].isin(valid_images_df['image_name']))].copy()
elif area_scope == "🅒 Solo Región C":
    valid_images_df = df_all_images[df_all_images['has_roi_c'] == True].copy()
    active_cells_df = df_raw_nuclei[(df_raw_nuclei['roi_region'] == 'C') & (df_raw_nuclei['image_name'].isin(valid_images_df['image_name']))].copy()
else: # Toda la imagen
    valid_images_df = df_all_images.copy()
    active_cells_df = df_raw_nuclei.copy()

# Build image level summaries for valid images
def _img_summary_for_cells(group_cells):
    total_pv  = int(group_cells['is_pv_plus'].sum()) if not group_cells.empty else 0
    total_pnn = int(group_cells['is_pnn_plus'].sum()) if not group_cells.empty else 0
    total_occ = int((group_cells['cell_type'] == 'PV+/PNN+').sum()) if not group_cells.empty else 0
    total_hol = int((group_cells['cell_type'] == 'PV-/PNN+').sum()) if not group_cells.empty else 0
    pv_cells  = group_cells[group_cells['is_pv_plus']  == True] if not group_cells.empty else pd.DataFrame()
    pnn_cells = group_cells[group_cells['is_pnn_plus'] == True] if not group_cells.empty else pd.DataFrame()
    def safe_mean(s): return float(s.mean()) if len(s) > 0 and pd.notna(s.mean()) else 0.0
    row = {
        'pv_count': total_pv, 'pnn_count': total_pnn,
        'pnn_count_filled': total_occ, 'pnn_count_hollow': total_hol,
        'pct_pnn_plus':   (total_occ / total_pv  * 100) if total_pv  > 0 else 0.0,
        'pct_pnn_hollow': (total_hol / total_pnn * 100) if total_pnn > 0 else 0.0,
        'mean_pv_area_um2':    safe_mean(pv_cells['pv_area_um2']) if not pv_cells.empty else 0.0,
        'mean_pv_diameter_um': safe_mean(pv_cells['pv_diameter_um']) if not pv_cells.empty else 0.0,
        'mean_pnn_area_um2':   safe_mean(pnn_cells['pnn_area_um2']) if not pnn_cells.empty else 0.0,
        'mean_pnn_diameter_um':safe_mean(pnn_cells['pnn_diameter_um']) if not pnn_cells.empty else 0.0,
        'mean_soma_area_um2':   safe_mean(pv_cells['pv_area_um2']) if not pv_cells.empty else 0.0,
        'mean_soma_diameter_um':safe_mean(pv_cells['pv_diameter_um']) if not pv_cells.empty else 0.0,
    }
    for col in SKEL_COLS:
        row[f'mean_{col}'] = safe_mean(pnn_cells[col]) if not pnn_cells.empty and col in pnn_cells.columns else 0.0
    return row

img_summaries = []
for _, img_row in valid_images_df.iterrows():
    g = img_row['group']
    s = img_row['section']
    fn = img_row['image_name']
    aid = img_row['animal_id']
    cnum = img_row['corte_num']
    
    cells_in_img = active_cells_df[active_cells_df['image_name'] == fn] if not active_cells_df.empty else pd.DataFrame()
    base = {
        'group': g, 'section': s, 'image_name': fn, 'animal_id': aid, 'corte_num': cnum
    }
    base.update(_img_summary_for_cells(cells_in_img))
    img_summaries.append(base)

df_img = pd.DataFrame(img_summaries) if img_summaries else pd.DataFrame()

# Build subject level summaries
def aggregate_subject_level_clean(df_image_level):
    if df_image_level.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in df_image_level.columns
                    if c not in ['group','section','image_name','animal_id','corte_num']
                    and pd.api.types.is_numeric_dtype(df_image_level[c])]
    summaries = []
    for vals, grp in df_image_level.groupby(['group','section','animal_id']):
        base = dict(zip(['group','section','animal_id'], vals))
        base['n_cortes'] = len(grp)
        for col in numeric_cols:
            base[col] = grp[col].mean()
        summaries.append(base)
    return pd.DataFrame(summaries)

df_subj = aggregate_subject_level_clean(df_img)

# ─── EXPLICIT N AND ROI COVERAGE CARD ───
total_imgs_all = len(df_all_images)
valid_imgs_cnt = len(valid_images_df)
excluded_imgs_cnt = total_imgs_all - valid_imgs_cnt
total_cells_cnt = len(active_cells_df)
total_subjs_cnt = valid_images_df['animal_id'].nunique() if not valid_images_df.empty else 0

st.markdown('<div class="stats-box">', unsafe_allow_html=True)
c_box1, c_box2, c_box3, c_box4 = st.columns(4)
c_box1.metric("📐 Área Seleccionada", area_scope.split(" ")[1] if " " in area_scope else area_scope)
c_box2.metric("🐀 Sujetos Activos (N_suj)", f"{total_subjs_cnt}")
c_box3.metric("🔬 Preparados Validados (N_prep)", f"{valid_imgs_cnt} / {total_imgs_all}")
c_box4.metric("🧫 Células en ROI (N_células)", f"{total_cells_cnt:,}")

if excluded_imgs_cnt > 0:
    st.warning(f"⚠️ **Atención sobre Tamaño Muestral ($N$):** Se excluyeron **{excluded_imgs_cnt}** preparados del análisis porque no tenían la ROI seleccionada (`{area_scope}`) trazada. Se analizan **{valid_imgs_cnt}** preparados para asegurar que los promedios no se distorsionen con ceros falsos.")
else:
    st.success(f"✅ **Consistencia del Muestreo ($N$):** Se están considerando los **{valid_imgs_cnt}** preparados completos del dataset.")

st.markdown('</div>', unsafe_allow_html=True)

# Dynamic Variable Label generator
def get_dynamic_labels(level):
    suffix = "por sujeto" if level == "Por Sujeto (animal, promediando cortes)" else "por preparado" if level == "Por Preparado/Corte (imagen)" else "por célula"
    
    pv_vars = {
        "mean_pv_area_um2": "Área Promedio Soma PV+ (µm²)",
        "mean_pv_diameter_um": "Diámetro Promedio Soma PV+ (µm)",
        "pv_area_um2": "Área del Soma PV+ (µm²) — por Célula",
        "pv_diameter_um": "Diámetro del Soma PV+ (µm) — por Célula"
    }
    
    pnn_vars = {
        "pnn_count": f"N° de Redes PNN+ Totales {suffix}",
        "pnn_count_filled": f"N° de PNN+ Ocupadas (PV+/PNN+) {suffix}",
        "pnn_count_hollow": f"N° de PNN+ Huecas (PNN+/PV-) {suffix}",
        "pct_pnn_plus": "% de PV+ con Red PNN (Coexpresión)",
        "pct_pnn_hollow": "% de PNN+ que son Huecas",
        "mean_pnn_area_um2": "Área Promedio de PNN (µm²)",
        "mean_pnn_diameter_um": "Diámetro Promedio de PNN (µm)",
        "mean_score": "Confianza Promedio (PNNscore)",
        "pnn_area_um2": "Área de PNN (µm²) — por Célula",
        "pnn_diameter_um": "Diámetro de PNN (µm) — por Célula",
        "score": "Confianza PNNscore (IA) — por Célula"
    }
    return pv_vars, pnn_vars

PV_VARS, PNN_VARS = get_dynamic_labels(level_type)

tab_pv, tab_pnn, tab_lupori, tab_global_wfa = st.tabs([
    "🧪 Interneuronas PV+ (Paso 1)",
    "🧠 Redes Perineuronales PNN (Paso 2)",
    "⚡ Métricas Lupori (Potencia y Coexpresión)",
    "🌐 Señal Global Integrada WFA"
])

def run_stats_layout(df_base, var_options, selected_var_key, title_lbl):
    if df_base.empty or selected_var_key not in df_base.columns:
        st.warning(f"No hay datos suficientes para la variable {selected_var_key}.")
        return

    df_base = df_base.copy()
    all_groups = df_base['group'].dropna().unique().tolist()
    group_meta = {}
    for g in all_groups:
        sex, cond, order = parse_group(g)
        if sex is not None:
            group_meta[g] = {'sex': sex, 'condition': cond, 'order': order}

    df_base['sex']       = df_base['group'].map(lambda g: group_meta.get(g, {}).get('sex'))
    df_base['condition'] = df_base['group'].map(lambda g: group_meta.get(g, {}).get('condition'))
    df_base = df_base[df_base['sex'].notna() & df_base['condition'].notna() & df_base['section'].isin(['IPSI', 'CONTRA'])]

    if df_base.empty:
        st.warning("No hay grupos con metadatos válidos de sexo (MACHO/HEMBRA) y condición (NONE/3 DÍAS/7 DÍAS/14 DÍAS).")
        return

    st.subheader(f"📈 {var_options[selected_var_key]} — {title_lbl}")

    scope_lbl = f"Área: {area_scope}"
    if level_type == "Por Sujeto (animal, promediando cortes)":
        st.markdown(f"""
        <div class="level-info">
        📌 <b>Nivel: Sujeto/Animal [{scope_lbl}]</b> — Cada punto representa el promedio de cortes para un animal individual en el hemisferio correspondiente (IPSI o CONTRA).
        </div>
        """, unsafe_allow_html=True)
    elif level_type == "Por Preparado/Corte (imagen)":
        st.markdown(f"""
        <div class="level-info">
        📌 <b>Nivel: Preparado/Corte [{scope_lbl}]</b> — Cada punto representa la métrica total/promedio de una imagen TIFF.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="level-info">
        📌 <b>Nivel: Célula Individual [{scope_lbl}]</b> — Cada punto representa una célula individual detectada dentro de la ROI seleccionada.
        </div>
        """, unsafe_allow_html=True)

    SECTION_COLORS = {'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'}

    # ═════════════════════════════════════════════════════════════════════
    # MODO 1: ANOVA DE 1 VÍA POR CONDICIÓN EXPERIMENTAL (FILTRABLE EN SIDEBAR)
    # ═════════════════════════════════════════════════════════════════════
    if stats_mode == "🔬 ANOVA de 1 Vía (Condiciones) + Post-Hoc":
        df_filtered = df_base.copy()
        if filter_sex == "Solo MACHO":
            df_filtered = df_filtered[df_filtered['sex'] == 'MACHO']
        elif filter_sex == "Solo HEMBRA":
            df_filtered = df_filtered[df_filtered['sex'] == 'HEMBRA']

        if filter_sec == "Solo IPSI":
            df_filtered = df_filtered[df_filtered['section'] == 'IPSI']
        elif filter_sec == "Solo CONTRA":
            df_filtered = df_filtered[df_filtered['section'] == 'CONTRA']

        if df_filtered.empty or len(df_filtered) < 3:
            st.warning(f"No hay suficientes datos tras aplicar los filtros de cohorte (Sexo: {filter_sex}, Hemisferio: {filter_sec}).")
            return

        df_filtered['condition'] = pd.Categorical(df_filtered['condition'], categories=COND_ORDER, ordered=True)
        df_filtered = df_filtered.sort_values('condition').dropna(subset=[selected_var_key])

        present_conds = [c for c in COND_ORDER if c in df_filtered['condition'].unique()]
        groups_data = {c: df_filtered[df_filtered['condition'] == c][selected_var_key].dropna().values for c in present_conds}
        groups_data = {c: vals for c, vals in groups_data.items() if len(vals) > 0}

        # ─── CÁLCULO DE ANOVA Y TAMAÑO DE EFECTO ───
        f_stat, p_anova = np.nan, np.nan
        eta_sq = np.nan
        h_stat, p_kruskal = np.nan, np.nan
        w_stat, p_levene = np.nan, np.nan

        k_groups = len(groups_data)
        n_total = sum(len(v) for v in groups_data.values())
        df_between = k_groups - 1
        df_within = n_total - k_groups

        if k_groups >= 2 and n_total >= k_groups + 1:
            try:
                f_stat, p_anova = f_oneway(*groups_data.values())
                all_v = np.concatenate(list(groups_data.values()))
                g_mean = np.mean(all_v)
                ss_tot = np.sum((all_v - g_mean) ** 2)
                ss_btw = sum(len(v) * (np.mean(v) - g_mean) ** 2 for v in groups_data.values())
                eta_sq = float(ss_btw / (ss_tot + 1e-8))
            except Exception:
                pass

            try:
                h_stat, p_kruskal = kruskal(*groups_data.values())
            except Exception:
                pass

            try:
                w_stat, p_levene = levene(*groups_data.values())
            except Exception:
                pass

        # ─── KPI METRIC CARDS ───
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        c_kpi1.metric(
            "🧪 ANOVA de 1 Vía (F)",
            f"F({df_between}, {df_within}) = {f_stat:.2f}" if not np.isnan(f_stat) else "N/A",
            delta=f"p = {p_anova:.4e}" if not np.isnan(p_anova) else "N/A",
            delta_color="normal" if (not np.isnan(p_anova) and p_anova < 0.05) else "off"
        )
        
        eta_desc = "Fuerte" if eta_sq >= 0.14 else ("Moderado" if eta_sq >= 0.06 else "Pequeño")
        c_kpi2.metric(
            "📐 Tamaño del Efecto (η²)",
            f"η² = {eta_sq:.4f}" if not np.isnan(eta_sq) else "N/A",
            f"Efecto {eta_desc}" if not np.isnan(eta_sq) else ""
        )

        c_kpi3.metric(
            "📊 Kruskal-Wallis (H)",
            f"H = {h_stat:.2f}" if not np.isnan(h_stat) else "N/A",
            delta=f"p = {p_kruskal:.4e}" if not np.isnan(p_kruskal) else "N/A",
            delta_color="normal" if (not np.isnan(p_kruskal) and p_kruskal < 0.05) else "off"
        )

        levene_ok = (not np.isnan(p_levene) and p_levene > 0.05)
        c_kpi4.metric(
            "⚖️ Homocedasticidad (Levene)",
            f"W = {w_stat:.2f}" if not np.isnan(w_stat) else "N/A",
            "Varianzas Homogéneas" if levene_ok else ("Varianzas Heterogéneas" if not np.isnan(p_levene) else "N/A")
        )

        st.caption(f"🎯 **Cohorte Activa:** Sexo: `{filter_sex}` | Hemisferio: `{filter_sec}` | $N_{{\\text{{total}}}} = {n_total}$ | $N_{{\\text{{sujetos}}}} = {df_filtered['animal_id'].nunique()}$")

        # ─── GRÁFICO ANOVA (BOXPLOT + VIOLIN + JITTER + LÍNEA DE MEDIAS) ───
        fig_anova = go.Figure()
        cond_idx_map = {c: i for i, c in enumerate(present_conds)}
        mean_points_x = []
        mean_points_y = []

        for cond in present_conds:
            df_c = df_filtered[df_filtered['condition'] == cond]
            vals = df_c[selected_var_key].dropna().values
            color = COND_COLORS.get(cond, '#00f2fe')

            fig_anova.add_trace(go.Box(
                x=[cond] * len(vals),
                y=vals,
                name=cond,
                marker_color=color,
                boxmean=True,
                boxpoints='all',
                jitter=0.35,
                pointpos=0,
                marker=dict(size=6.5, opacity=0.8, color=color),
                line=dict(color=color, width=2.2),
                text=df_c['animal_id'] if 'animal_id' in df_c.columns else None
            ))

            if len(vals) > 0:
                mean_points_x.append(cond)
                mean_points_y.append(np.mean(vals))

        # Add connecting mean line
        if len(mean_points_x) >= 2:
            fig_anova.add_trace(go.Scatter(
                x=mean_points_x,
                y=mean_points_y,
                mode='lines+markers',
                name='Media Temporal',
                line=dict(color='#ffffff', width=2, dash='dot'),
                marker=dict(size=8, symbol='diamond', color='#ffffff')
            ))

        # ─── OVERHEAD POST-HOC COMPARISON BARS ───
        all_y = df_filtered[selected_var_key].dropna().values
        min_y = float(np.min(all_y)) if len(all_y) > 0 else 0.0
        max_y = float(np.max(all_y)) if len(all_y) > 0 else 1.0
        y_span = max_y - min_y if max_y != min_y else (max_y * 0.2 if max_y != 0 else 1.0)
        step_h = y_span * 0.08
        current_y_offset = max_y + step_h

        # Post-hoc Tukey HSD
        tukey_table = []
        try:
            clean_df = df_filtered[[selected_var_key, 'condition']].dropna()
            clean_df['condition'] = clean_df['condition'].astype(str)
            if clean_df['condition'].nunique() >= 2 and len(clean_df) > clean_df['condition'].nunique():
                thsd = pairwise_tukeyhsd(clean_df[selected_var_key], clean_df['condition'], alpha=0.05)
                for row_t in thsd._results_table.data[1:]:
                    g1, g2, meandiff, p_adj, lower, upper, reject = row_t
                    stars = sig_stars(p_adj)
                    tukey_table.append({
                        "Comparación": f"{g1} vs {g2}",
                        "Diferencia (Δ Media)": f"{meandiff:.4f}",
                        "IC 95%": f"[{lower:.3f}, {upper:.3f}]",
                        "p-valor (Tukey HSD)": f"{p_adj:.4e}",
                        "Significancia": stars
                    })

                    # Draw significant overhead annotations
                    if p_adj < 0.05 and g1 in cond_idx_map and g2 in cond_idx_map:
                        xa = cond_idx_map[g1]
                        xb = cond_idx_map[g2]
                        bar_y = current_y_offset
                        tick_h = y_span * 0.015

                        fig_anova.add_shape(type='line', x0=xa, x1=xb, y0=bar_y, y1=bar_y, line=dict(color='#00ff88', width=1.8))
                        fig_anova.add_shape(type='line', x0=xa, x1=xa, y0=bar_y - tick_h, y1=bar_y, line=dict(color='#00ff88', width=1.8))
                        fig_anova.add_shape(type='line', x0=xb, x1=xb, y0=bar_y - tick_h, y1=bar_y, line=dict(color='#00ff88', width=1.8))
                        fig_anova.add_annotation(
                            x=(xa + xb)/2, y=bar_y + tick_h * 0.8,
                            text=f"<b>{stars}</b>", showarrow=False,
                            font=dict(size=12, color='#00ff88'),
                            xref='x', yref='y'
                        )
                        current_y_offset += step_h * 1.2
        except Exception:
            pass

        # Pairwise non-parametric comparisons (Mann-Whitney U / Bonferroni)
        pw_nonparam = []
        cond_keys = list(groups_data.keys())
        pairs = [(cond_keys[i], cond_keys[j]) for i in range(len(cond_keys)) for j in range(i+1, len(cond_keys))]
        for ca, cb in pairs:
            stat, p_m = run_mwu(groups_data[ca], groups_data[cb])
            p_m_adj = p_m * len(pairs) if use_bonferroni and not np.isnan(p_m) else p_m
            stars_m = sig_stars(p_m_adj if use_bonferroni else p_m)
            m_a, s_a = np.mean(groups_data[ca]), np.std(groups_data[ca])
            m_b, s_b = np.mean(groups_data[cb]), np.std(groups_data[cb])
            pw_nonparam.append({
                "Comparación": f"{ca} vs {cb}",
                f"{ca} (Media ± DE)": f"{m_a:.2f} ± {s_a:.2f} (N={len(groups_data[ca])})",
                f"{cb} (Media ± DE)": f"{m_b:.2f} ± {s_b:.2f} (N={len(groups_data[cb])})",
                "p-valor (Sin Ajustar)": f"{p_m:.4f}" if not np.isnan(p_m) else "N/A",
                "p-valor (Bonferroni)" if use_bonferroni else "p-valor": f"{p_m_adj:.4f}" if not np.isnan(p_m_adj) else "N/A",
                "Significancia": stars_m
            })

        fig_anova.update_layout(
            title=dict(
                text=f"<b>ANOVA de 1 Vía: {var_options[selected_var_key]} por Condición ({filter_sex}, {filter_sec})</b>",
                font=dict(size=16, color='#bb86fc')
            ),
            yaxis_title=var_options[selected_var_key],
            xaxis_title="Condición Experimental",
            template='plotly_dark',
            height=540,
            showlegend=False,
            yaxis=dict(range=[min_y - step_h * 0.5, current_y_offset + step_h * 0.5])
        )

        st.plotly_chart(fig_anova, use_container_width=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown('<div class="table-caption">⏱️ Comparaciones Post-Hoc (Tukey HSD)</div>', unsafe_allow_html=True)
            if tukey_table:
                st.dataframe(pd.DataFrame(tukey_table), use_container_width=True)
            else:
                st.info("No se requirieron o no se pudieron calcular comparaciones de Tukey HSD.")

        with col_t2:
            st.markdown('<div class="table-caption">🧪 Comparaciones Pareadas No Paramétricas (Mann-Whitney U)</div>', unsafe_allow_html=True)
            if pw_nonparam:
                st.dataframe(pd.DataFrame(pw_nonparam), use_container_width=True)

        st.markdown('<div class="table-caption">📋 Estadísticos Descriptivos por Condición</div>', unsafe_allow_html=True)
        desc_rows = []
        for c in present_conds:
            v = groups_data.get(c, [])
            if len(v) > 0:
                desc_rows.append({
                    "Condición": c,
                    "N Observaciones": len(v),
                    "N Sujetos": df_filtered[df_filtered['condition'] == c]['animal_id'].nunique(),
                    "Media": f"{np.mean(v):.4f}",
                    "Desv. Estándar (DE)": f"{np.std(v):.4f}",
                    "Mediana": f"{np.median(v):.4f}",
                    "IQR (Q3 - Q1)": f"{np.percentile(v, 75) - np.percentile(v, 25):.4f}",
                    "Mínimo": f"{np.min(v):.4f}",
                    "Máximo": f"{np.max(v):.4f}"
                })
        if desc_rows:
            st.dataframe(pd.DataFrame(desc_rows), use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════
    # MODO 2: DESGLOSE TRADICIONAL (PANEL POR SEXO E IPSI/CONTRA)
    # ═════════════════════════════════════════════════════════════════════
    else:
        sexes_in_data = [s for s in ['MACHO', 'HEMBRA'] if s in df_base['sex'].unique()]
        if not sexes_in_data:
            sexes_in_data = df_base['sex'].dropna().unique().tolist()
            
        sex_groups = [(s, df_base[df_base['sex'] == s]) for s in sexes_in_data]
        cols = st.columns(len(sex_groups))

        for col_idx, (group_title, df_sub) in enumerate(sex_groups):
            if df_sub.empty:
                continue

            df_sub = df_sub.copy()
            df_sub['condition'] = pd.Categorical(df_sub['condition'], categories=COND_ORDER, ordered=True)
            df_plot = df_sub.sort_values('condition')

            fig = go.Figure()

            present_conds = [c for c in COND_ORDER if c in df_plot['condition'].unique()]
            present_secs  = [s for s in ['IPSI', 'CONTRA'] if s in df_plot['section'].unique()]
            cond_idx_map  = {c: i for i, c in enumerate(present_conds)}

            for sec in present_secs:
                df_sec = df_plot[df_plot['section'] == sec]
                color = SECTION_COLORS.get(sec, '#ffffff')
                fig.add_trace(go.Box(
                    x=df_sec['condition'],
                    y=df_sec[selected_var_key],
                    name=f"{sec}",
                    marker_color=color,
                    boxmean='sd',
                    boxpoints='all',
                    jitter=0.35,
                    pointpos=0,
                    marker=dict(size=6, opacity=0.8, color=color),
                    line=dict(color=color, width=2),
                    text=df_sec['animal_id'] if 'animal_id' in df_sec.columns else None
                ))

            # ─── COMPUTE Y BOUNDS FOR ANNOTATIONS ───
            all_y = df_sub[selected_var_key].dropna().values
            min_y = float(np.min(all_y)) if len(all_y) > 0 else 0.0
            max_y = float(np.max(all_y)) if len(all_y) > 0 else 1.0
            y_span = max_y - min_y if max_y != min_y else (max_y * 0.2 if max_y != 0 else 1.0)
            
            step_h = y_span * 0.08
            current_y_offset = max_y + step_h

            table_intra = []

            # 1. INTRA-GROUP COMPARISONS (IPSI vs CONTRA in each condition)
            for cond in present_conds:
                if cond not in cond_idx_map:
                    continue
                c_idx = cond_idx_map[cond]
                df_cond = df_sub[df_sub['condition'] == cond]
                val_ipsi = df_cond[df_cond['section'] == 'IPSI'][selected_var_key].dropna().values
                val_contra = df_cond[df_cond['section'] == 'CONTRA'][selected_var_key].dropna().values

                df_ipsi_cond = df_cond[df_cond['section'] == 'IPSI']
                df_contra_cond = df_cond[df_cond['section'] == 'CONTRA']

                n_suj_ipsi = df_ipsi_cond['animal_id'].nunique() if 'animal_id' in df_ipsi_cond.columns else len(val_ipsi)
                n_suj_contra = df_contra_cond['animal_id'].nunique() if 'animal_id' in df_contra_cond.columns else len(val_contra)

                n_ipsi, m_ipsi, std_ipsi = len(val_ipsi), (np.mean(val_ipsi) if len(val_ipsi)>0 else 0.0), (np.std(val_ipsi) if len(val_ipsi)>0 else 0.0)
                n_contra, m_contra, std_contra = len(val_contra), (np.mean(val_contra) if len(val_contra)>0 else 0.0), (np.std(val_contra) if len(val_contra)>0 else 0.0)

                stat_name = "Mann-Whitney U"
                p = np.nan

                if len(val_ipsi) >= 2 and len(val_contra) >= 2:
                    if 'animal_id' in df_cond.columns and level_type != "Por Célula (distribuciones individuales)":
                        merged_p = pd.merge(
                            df_cond[df_cond['section'] == 'IPSI'][['animal_id', selected_var_key]].rename(columns={selected_var_key: 'v_ipsi'}),
                            df_cond[df_cond['section'] == 'CONTRA'][['animal_id', selected_var_key]].rename(columns={selected_var_key: 'v_contra'}),
                            on='animal_id'
                        ).dropna()
                        if len(merged_p) >= 2:
                            try:
                                stat, p = wilcoxon(merged_p['v_ipsi'].values, merged_p['v_contra'].values)
                                stat_name = "Wilcoxon Pareado"
                            except Exception:
                                stat, p = run_mwu(val_ipsi, val_contra)
                        else:
                            stat, p = run_mwu(val_ipsi, val_contra)
                    else:
                        stat, p = run_mwu(val_ipsi, val_contra)

                    stars = sig_stars(p)
                    
                    x0 = c_idx - 0.18
                    x1 = c_idx + 0.18
                    bar_y = current_y_offset
                    tick_h = y_span * 0.015

                    fig.add_shape(type='line', x0=x0, x1=x1, y0=bar_y, y1=bar_y, line=dict(color='#ffffff', width=1.5))
                    fig.add_shape(type='line', x0=x0, x1=x0, y0=bar_y - tick_h, y1=bar_y, line=dict(color='#ffffff', width=1.5))
                    fig.add_shape(type='line', x0=x1, x1=x1, y0=bar_y - tick_h, y1=bar_y, line=dict(color='#ffffff', width=1.5))
                    fig.add_annotation(
                        x=c_idx, y=bar_y + tick_h * 0.8,
                        text=f"<b>{stars}</b>", showarrow=False,
                        font=dict(size=12, color='#00ff88' if stars != 'ns' else '#aaaaaa'),
                        xref='x', yref='y'
                    )
                    current_y_offset += step_h * 1.2
                else:
                    stars = "ns"

                table_intra.append({
                    "Condición": cond,
                    "IPSI (Media ± DE)": f"{m_ipsi:.2f} ± {std_ipsi:.2f} (N_suj={n_suj_ipsi}, N_puntos={n_ipsi})",
                    "CONTRA (Media ± DE)": f"{m_contra:.2f} ± {std_contra:.2f} (N_suj={n_suj_contra}, N_puntos={n_contra})",
                    "Prueba": stat_name,
                    "p-valor": f"{p:.4f}" if not np.isnan(p) else "N/A",
                    "Significancia": stars
                })

            # 2. INTER-CONDITION COMPARISONS (Temporal evolution for IPSI and CONTRA)
            table_inter = []

            for sec in present_secs:
                df_sec = df_sub[df_sub['section'] == sec]
                sec_color = SECTION_COLORS.get(sec, '#ffffff')
                sec_offset = -0.18 if sec == 'IPSI' else 0.18

                cond_data = {}
                for c in present_conds:
                    vals = df_sec[df_sec['condition'] == c][selected_var_key].dropna().values
                    if len(vals) > 0:
                        cond_data[c] = vals

                cond_present = [c for c in present_conds if c in cond_data]
                pairs = [(cond_present[i], cond_present[j]) for i in range(len(cond_present)) for j in range(i+1, len(cond_present))]

                for ca, cb in pairs:
                    stat, p = run_mwu(cond_data[ca], cond_data[cb])
                    p_adj = p * len(pairs) if use_bonferroni and not np.isnan(p) else p
                    stars = sig_stars(p_adj if use_bonferroni else p)

                    xa = cond_idx_map[ca] + sec_offset
                    xb = cond_idx_map[cb] + sec_offset
                    bar_y = current_y_offset
                    tick_h = y_span * 0.015

                    fig.add_shape(type='line', x0=xa, x1=xb, y0=bar_y, y1=bar_y, line=dict(color=sec_color, width=1.5, dash='dot'))
                    fig.add_shape(type='line', x0=xa, x1=xa, y0=bar_y - tick_h, y1=bar_y, line=dict(color=sec_color, width=1.5))
                    fig.add_shape(type='line', x0=xb, x1=xb, y0=bar_y - tick_h, y1=bar_y, line=dict(color=sec_color, width=1.5))
                    fig.add_annotation(
                        x=(xa + xb)/2, y=bar_y + tick_h * 0.8,
                        text=f"<b>{sec}: {stars}</b>", showarrow=False,
                        font=dict(size=11, color=sec_color if stars != 'ns' else '#aaaaaa'),
                        xref='x', yref='y'
                    )
                    current_y_offset += step_h * 1.3

                    table_inter.append({
                        "Comparación Temporal": f"{ca} vs {cb}",
                        "Hemisferio": sec,
                        "p-valor (Sin Ajustar)": f"{p:.4f}" if not np.isnan(p) else "N/A",
                        "p-valor (Bonferroni)" if use_bonferroni else "p-valor": f"{p_adj:.4f}" if not np.isnan(p_adj) else "N/A",
                        "Significancia": stars
                    })

            n_subjs_tot = df_sub['animal_id'].nunique() if 'animal_id' in df_sub.columns else len(df_sub)
            n_points_tot = len(df_sub)

            fig.update_layout(
                title=dict(text=f"<b>{group_title} (N_suj = {n_subjs_tot}, N_puntos = {n_points_tot})</b>", font=dict(size=15, color=SEX_COLORS.get(group_title, '#bb86fc'))),
                yaxis_title=var_options[selected_var_key],
                xaxis_title="Condición Experimental",
                boxmode='group',
                template='plotly_dark',
                height=500,
                yaxis=dict(range=[min_y - step_h * 0.5, current_y_offset + step_h * 0.5]),
                legend=dict(title="Hemisferio", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            with cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f'<div class="table-caption">📋 Tabla 1: Comparación Intragrupo ({group_title} — IPSI vs CONTRA)</div>', unsafe_allow_html=True)
                if table_intra:
                    df_t_intra = pd.DataFrame(table_intra)
                    st.dataframe(df_t_intra, use_container_width=True)
                
                st.markdown(f'<div class="table-caption">⏱️ Tabla 2: Evolución Temporal Intergrupo ({group_title})</div>', unsafe_allow_html=True)
                if table_inter:
                    df_t_inter = pd.DataFrame(table_inter)
                    st.dataframe(df_t_inter, use_container_width=True)


# 1. TAB PV+
with tab_pv:
    if df_img.empty:
        st.info("No hay datos de PV+ disponibles.")
    else:
        st.header("Análisis de Interneuronas PV+ (Paso 1)")
        selected_pv_var = st.selectbox("Variable PV+:", list(PV_VARS.keys()), format_func=lambda x: PV_VARS[x], key="pv_var")

        is_cell_var = selected_pv_var in ["pv_area_um2", "pv_diameter_um"]
        if is_cell_var and level_type == "Por Célula (distribuciones individuales)":
            df_pv_base = active_cells_df[active_cells_df['is_pv_plus'] == True].copy()
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_pv_base = df_subj.copy()
            if selected_pv_var == "pv_area_um2": df_pv_base['pv_area_um2'] = df_pv_base['mean_pv_area_um2']
            if selected_pv_var == "pv_diameter_um": df_pv_base['pv_diameter_um'] = df_pv_base['mean_pv_diameter_um']
        else: # Por Preparado/Corte (imagen)
            df_pv_base = df_img.copy()
            if selected_pv_var == "pv_area_um2": df_pv_base['pv_area_um2'] = df_pv_base['mean_pv_area_um2']
            if selected_pv_var == "pv_diameter_um": df_pv_base['pv_diameter_um'] = df_pv_base['mean_pv_diameter_um']

        run_stats_layout(df_pv_base, PV_VARS, selected_pv_var, "Interneuronas PV+")

# 2. TAB PNN
with tab_pnn:
    if df_img.empty:
        st.info("No hay datos de PNN disponibles.")
    else:
        st.header("Análisis de Redes Perineuronales PNN (Paso 2)")
        selected_pnn_var = st.selectbox("Variable PNN:", list(PNN_VARS.keys()), format_func=lambda x: PNN_VARS[x], key="pnn_var")

        is_cell_var = selected_pnn_var in ["pnn_area_um2", "pnn_diameter_um", "score"]
        if is_cell_var and level_type == "Por Célula (distribuciones individuales)":
            df_pnn_base = active_cells_df[active_cells_df['is_pnn_plus'] == True].copy()
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_pnn_base = df_subj.copy()
            if selected_pnn_var == "pnn_area_um2": df_pnn_base['pnn_area_um2'] = df_pnn_base['mean_pnn_area_um2']
            if selected_pnn_var == "pnn_diameter_um": df_pnn_base['pnn_diameter_um'] = df_pnn_base['mean_pnn_diameter_um']
            if selected_pnn_var == "score": df_pnn_base['score'] = df_pnn_base['mean_score']
        else: # Por Preparado/Corte (imagen)
            df_pnn_base = df_img.copy()
            if selected_pnn_var == "pnn_area_um2": df_pnn_base['pnn_area_um2'] = df_pnn_base['mean_pnn_area_um2']
            if selected_pnn_var == "pnn_diameter_um": df_pnn_base['pnn_diameter_um'] = df_pnn_base['mean_pnn_diameter_um']
            if selected_pnn_var == "score": df_pnn_base['score'] = df_pnn_base['mean_score']

        run_stats_layout(df_pnn_base, PNN_VARS, selected_pnn_var, "Redes Perineuronales PNN")

@st.cache_data
def get_image_wfa_component_metrics(mask_file, roi_json_path, area_scope, px_size=0.65):
    if not os.path.exists(mask_file):
        return None
    try:
        stk = tiff.imread(mask_file)
        if len(stk.shape) < 3 or stk.shape[0] < 5:
            return None
        m_wfa = stk[2]
        w_raw = stk[4]

        if area_scope != "🌐 Toda la Imagen (Global)" and os.path.exists(roi_json_path):
            d_roi = load_rois(roi_json_path)
            target_reg = 'ALL' if 'combinadas' in area_scope else area_scope.split(' ')[-1]
            roi_mask = create_roi_mask(w_raw.shape, d_roi, target_region=target_reg)
        else:
            roi_mask = np.ones_like(w_raw, dtype=bool)

        diff_mask = (m_wfa == 0) & roi_mask
        pnn_mask = (m_wfa > 0) & roi_mask

        diff_mean = float(np.mean(w_raw[diff_mask])) if np.sum(diff_mask) > 0 else 0.0
        diff_sum = float(np.sum(w_raw[diff_mask]))
        tot_mean = float(np.mean(w_raw[roi_mask])) if np.sum(roi_mask) > 0 else 0.0
        tot_sum = float(np.sum(w_raw[roi_mask]))
        pnn_mean = float(np.mean(w_raw[pnn_mask])) if np.sum(pnn_mask) > 0 else 0.0
        pnn_sum = float(np.sum(w_raw[pnn_mask]))

        diff_norm = float(diff_mean / (np.mean(w_raw) + 1e-8))
        ratio_pnn_diff = float(pnn_mean / (diff_mean + 1e-8)) if diff_mean > 0 else 0.0

        return {
            'mean_diffuse_wfa_intensity_raw': diff_mean,
            'diffuse_wfa_fluorescence_norm': diff_norm,
            'total_diffuse_wfa_signal': diff_sum,
            'mean_pnn_wfa_intensity_raw': pnn_mean,
            'total_pnn_wfa_signal': pnn_sum,
            'ratio_pnn_to_diffuse': ratio_pnn_diff,
            'mean_wfa_intensity_raw': tot_mean,
            'total_integrated_wfa_signal': tot_sum
        }
    except Exception:
        return None

def compute_dynamic_lupori_metrics(valid_imgs_df, active_cells_df, area_scope):
    px_size = 0.65
    rows = []

    for idx, img_row in valid_imgs_df.iterrows():
        img_name = img_row['image_name']
        grp = img_row['group']
        sec = img_row['section']
        anim = img_row['animal_id']
        base_name = img_row.get('base_name', os.path.splitext(img_name)[0])

        img_cells = active_cells_df[active_cells_df['image_name'] == img_name]

        n_pnn = int((img_cells['is_pnn_plus'] == True).sum()) if not img_cells.empty else 0
        n_pv = int((img_cells['is_pv_plus'] == True).sum()) if not img_cells.empty else 0
        n_coloc = int((img_cells['cell_type'] == 'PV+/PNN+').sum()) if not img_cells.empty else 0

        area_mm2 = float((2048 * px_size * 2048 * px_size) / 1e6)
        roi_json_path = os.path.join(METRICS_BASE_DIR, grp, sec, f"{base_name}_rois.json")
        if not os.path.exists(roi_json_path):
            roi_json_path = get_roi_json_path(METRICS_BASE_DIR, img_name)

        if area_scope != "🌐 Toda la Imagen (Global)" and os.path.exists(roi_json_path):
            d_roi = load_rois(roi_json_path)
            target_regs = ['A', 'B', 'C'] if 'combinadas' in area_scope else ([area_scope.split(' ')[-1]] if 'Región' in area_scope else ['A', 'B', 'C'])
            total_um2 = 0.0
            for r_k in target_regs:
                pts_list = d_roi.get(r_k, [])
                for poly in pts_list:
                    pts = np.array(poly)
                    if len(pts) >= 3:
                        x, y = pts[:, 1], pts[:, 0]
                        a_px = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        total_um2 += a_px * (px_size ** 2)
            if total_um2 > 0:
                area_mm2 = total_um2 / 1e6

        pnn_density = float(n_pnn / area_mm2) if area_mm2 > 0 else 0.0
        pv_density = float(n_pv / area_mm2) if area_mm2 > 0 else 0.0
        coloc_density = float(n_coloc / area_mm2) if area_mm2 > 0 else 0.0
        pct_pv = float((n_coloc / n_pv * 100.0)) if n_pv > 0 else 0.0

        pnn_cells = img_cells[img_cells['is_pnn_plus'] == True] if not img_cells.empty else pd.DataFrame()
        wfa_pericell = float(pnn_cells['wfa_pericellular_norm'].mean()) if not pnn_cells.empty and 'wfa_pericellular_norm' in pnn_cells.columns and pd.notna(pnn_cells['wfa_pericellular_norm'].mean()) else 0.0

        pnn_energy = float(pnn_density * wfa_pericell)
        coloc_energy = float(coloc_density * wfa_pericell)

        # Check segmented masks for precise diffuse WFA calculation (outside PNN circles)
        mask_file = os.path.join("data/processed/segmented", grp, sec, f"{base_name}_masks.tif")
        wfa_comp = get_image_wfa_component_metrics(mask_file, roi_json_path, area_scope, px_size)

        if wfa_comp is not None:
            diffuse_wfa_raw = wfa_comp['mean_diffuse_wfa_intensity_raw']
            diffuse_wfa_norm = wfa_comp['diffuse_wfa_fluorescence_norm']
            diffuse_wfa_sum = wfa_comp['total_diffuse_wfa_signal']
            pnn_wfa_raw = wfa_comp['mean_pnn_wfa_intensity_raw']
            ratio_pnn_diff = wfa_comp['ratio_pnn_to_diffuse']
            total_wfa_sum = wfa_comp['total_integrated_wfa_signal']
            mean_wfa_raw = wfa_comp['mean_wfa_intensity_raw']
        else:
            wfa_sum = float(img_cells['wfa_sum_intensity'].sum()) if not img_cells.empty and 'wfa_sum_intensity' in img_cells.columns else 0.0
            diffuse_wfa_raw = float(img_cells['wfa_mean_intensity'].mean()) if not img_cells.empty and 'wfa_mean_intensity' in img_cells.columns else 0.0
            diffuse_wfa_norm = diffuse_wfa_raw / 255.0
            diffuse_wfa_sum = wfa_sum
            pnn_wfa_raw = float(pnn_cells['wfa_mean_intensity'].mean()) if not pnn_cells.empty and 'wfa_mean_intensity' in pnn_cells.columns else diffuse_wfa_raw
            ratio_pnn_diff = (pnn_wfa_raw / (diffuse_wfa_raw + 1e-8)) if diffuse_wfa_raw > 0 else 1.0
            total_wfa_sum = wfa_sum
            mean_wfa_raw = diffuse_wfa_raw

        sex, cond, order = parse_group(grp)

        rows.append({
            'group': grp, 'section': sec, 'image_name': img_name, 'base_name': base_name, 'animal_id': anim,
            'sex': sex, 'condition': cond,
            'pnn_density_mm2': pnn_density, 'pv_density_mm2': pv_density,
            'coloc_density_mm2': coloc_density, 'pct_pv_surrounded_by_pnn': pct_pv,
            'mean_pnn_pericellular_wfa_norm': wfa_pericell,
            'pnn_energy': pnn_energy, 'coloc_energy': coloc_energy,
            'mean_diffuse_wfa_intensity_raw': diffuse_wfa_raw,
            'diffuse_wfa_fluorescence_norm': diffuse_wfa_norm,
            'total_diffuse_wfa_signal': diffuse_wfa_sum,
            'diffuse_wfa_density_mm2': float(diffuse_wfa_sum / area_mm2) if area_mm2 > 0 else 0.0,
            'mean_pnn_wfa_intensity_raw': pnn_wfa_raw,
            'ratio_pnn_to_diffuse': ratio_pnn_diff,
            'total_integrated_wfa_signal': total_wfa_sum,
            'integrated_wfa_density_mm2': float(total_wfa_sum / area_mm2) if area_mm2 > 0 else 0.0,
            'mean_wfa_intensity_raw': mean_wfa_raw,
            'diffuse_wfa_fluorescence': diffuse_wfa_norm,
            'image_area_mm2': area_mm2
        })

    return pd.DataFrame(rows)

# 3. TAB LUPORI METRICS (POTENCIA, DENSIDAD, ENERGÍA Y COEXPRESIÓN)
with tab_lupori:
    st.header("⚡ Cuantificación Método Lupori et al. (2023)")
    st.markdown("""
    **Métricas Oficiales:**
    * **Densidad (Density):** $N^\circ \text{de células o PNNs / mm}^2$
    * **Potencia (Energy):** $\text{Densidad} \times \text{Intensidad Promedio Normalizada (0-1)} = \frac{\sum \text{intensity}_i}{\text{Área mm}^2}$
    * **Intensidad Circundante WFA:** Intensidad medida en el anillo de expansión perineuronal ($3\text{--}5\,\mu\text{m}$) alrededor del soma.
    * **Coexpresión:** $\%$ de células PV+ que están rodeadas por una red PNN+.
    """)

    df_lup_filtered = compute_dynamic_lupori_metrics(valid_images_df, active_cells_df, area_scope)

    if df_lup_filtered.empty:
        st.info("No se encontraron métricas Lupori disponibles para la selección actual.")
    else:
        st.success(f"✅ Cálculo dinámico de la cuantificación Lupori con **{len(df_lup_filtered)}** preparados válidos para {area_scope}.")
        
        LUPORI_VARS = {
            "pnn_energy": "Potencia PNN (Energy)",
            "coloc_energy": "Potencia Coexpresión PV+/PNN+ (Energy)",
            "pnn_density_mm2": "Densidad de PNN+ (redes / mm²)",
            "pv_density_mm2": "Densidad de PV+ (células / mm²)",
            "coloc_density_mm2": "Densidad de Coexpresión (células PV+/PNN+ / mm²)",
            "pct_pv_surrounded_by_pnn": "% PV+ rodeadas por PNN+ (Coexpresión)",
            "mean_pnn_pericellular_wfa_norm": "Intensidad WFA Circundante (Norm 0-1)",
            "mean_diffuse_wfa_intensity_raw": "Intensidad Media WFA Difuso (Fuera de PNN - Raw)",
            "diffuse_wfa_fluorescence": "Fluorescencia Difusa WFA (Norm 0-1)"
        }
        
        sel_lup_var = st.selectbox("Seleccionar Métricas Lupori:", list(LUPORI_VARS.keys()), format_func=lambda x: LUPORI_VARS[x], key="lup_var")

        if level_type == "Por Célula (distribuciones individuales)":
            df_lup_base = active_cells_df[active_cells_df['is_pnn_plus'] == True].copy()
            if 'pnn_energy' not in df_lup_base.columns: df_lup_base['pnn_energy'] = df_lup_base['wfa_pericellular_norm']
            if 'coloc_energy' not in df_lup_base.columns: df_lup_base['coloc_energy'] = df_lup_base['wfa_pericellular_norm']
            if 'mean_pnn_pericellular_wfa_norm' not in df_lup_base.columns: df_lup_base['mean_pnn_pericellular_wfa_norm'] = df_lup_base['wfa_pericellular_norm']
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_lup_base = df_lup_filtered.groupby(['group', 'section', 'animal_id']).mean(numeric_only=True).reset_index()
            df_lup_base['sex']       = df_lup_base['group'].map(lambda g: parse_group(g)[0])
            df_lup_base['condition'] = df_lup_base['group'].map(lambda g: parse_group(g)[1])
        else:
            df_lup_base = df_lup_filtered.copy()

        run_stats_layout(df_lup_base, LUPORI_VARS, sel_lup_var, "Métricas Lupori et al.")
        
        st.markdown("### 📋 Tabla Completa Consolidada Lupori:")
        st.dataframe(df_lup_filtered, use_container_width=True)

# 4. TAB SEÑAL GLOBAL Y COMPONENTE DIFUSO WFA
with tab_global_wfa:
    st.header("🌐 Señal Global y Componente Difuso de WFA")
    st.markdown("""
    **Medición de Componentes de Fluorescencia WFA:**
    * **Intensidad Media WFA Difusa (Raw):** Señal promedio de WFA en todo el tejido o ROI que se encuentra **fuera del círculo de cuantificación de las PNNs** ($m_{\\text{wfa}} = 0$). Mide la matriz extracelular difusa libre de condensaciones perineuronales.
    * **Fluorescencia Difusa WFA Normalizada:** Señal difusa normalizada respecto a la fluorescencia global del corte ($0\\text{--}1$).
    * **Intensidad Media WFA Perineuronal (Raw):** Señal promedio de WFA **dentro de los círculos de cuantificación de las PNNs**.
    * **Ratio de Contraste (PNN / Difuso):** Cociente $\\frac{\\text{Intensidad Media PNN}}{\\text{Intensidad Media Difusa}}$, indicando la definición o nitidez de las redes sobre el fondo.
    * **Señal Global Integrada WFA ($\sum \\text{Intensidad}$):** Suma total de fluorescencia acumulada de WFA en todo el tejido/ROI.
    * **Densidad de Señal Integrada por $\\text{mm}^2$:** Intensidad WFA integrada dividida entre el área del preparado o ROI en $\\text{mm}^2$.
    """)

    df_gwfa_filtered = compute_dynamic_lupori_metrics(valid_images_df, active_cells_df, area_scope)

    if df_gwfa_filtered.empty:
        st.info("No se encontraron métricas de señal global WFA disponibles.")
    else:
        GLOBAL_WFA_VARS = {
            "mean_diffuse_wfa_intensity_raw": "Intensidad Media WFA Difusa (Fuera de Círculos PNN - Raw)",
            "diffuse_wfa_fluorescence_norm": "Fluorescencia Difusa WFA Normalizada (Norm 0-1)",
            "mean_pnn_wfa_intensity_raw": "Intensidad Media WFA Perineuronal (Dentro de Círculos PNN - Raw)",
            "ratio_pnn_to_diffuse": "Ratio de Contraste (Intensidad PNN / Intensidad Difusa)",
            "total_diffuse_wfa_signal": "Señal Integrada Difusa WFA (Suma Fuera de PNN)",
            "diffuse_wfa_density_mm2": "Densidad de Señal Difusa WFA por mm²",
            "total_integrated_wfa_signal": "Señal Global Integrada WFA (Suma Total Tejido)",
            "integrated_wfa_density_mm2": "Densidad de Señal Integrada WFA por mm²",
            "mean_wfa_intensity_raw": "Intensidad Media Global WFA (Tejido Completo - Raw)"
        }

        sel_gwfa_var = st.selectbox("Seleccionar Métrica de Señal Global / Difusa WFA:", list(GLOBAL_WFA_VARS.keys()), format_func=lambda x: GLOBAL_WFA_VARS[x], key="gwfa_var")

        if level_type == "Por Célula (distribuciones individuales)":
            df_gwfa_base = active_cells_df.copy()
            for k in GLOBAL_WFA_VARS.keys():
                if k not in df_gwfa_base.columns and k in df_gwfa_filtered.columns:
                    mapping = df_gwfa_filtered.set_index('image_name')[k].to_dict()
                    df_gwfa_base[k] = df_gwfa_base['image_name'].map(mapping)
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_gwfa_base = df_gwfa_filtered.groupby(['group', 'section', 'animal_id']).mean(numeric_only=True).reset_index()
            df_gwfa_base['sex']       = df_gwfa_base['group'].map(lambda g: parse_group(g)[0])
            df_gwfa_base['condition'] = df_gwfa_base['group'].map(lambda g: parse_group(g)[1])
        else:
            df_gwfa_base = df_gwfa_filtered.copy()

        run_stats_layout(df_gwfa_base, GLOBAL_WFA_VARS, sel_gwfa_var, "Señal Global y Difusa WFA")

        st.markdown("### 📋 Tabla Consolidada de Señal Global y Difusa WFA:")
        st.dataframe(df_gwfa_filtered, use_container_width=True)
