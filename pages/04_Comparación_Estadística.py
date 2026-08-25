import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import mannwhitneyu, wilcoxon, f_oneway, kruskal, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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
        has_roi = len(regions_dict.get('ROI', [])) > 0 or len(regions_dict.get('B', [])) > 0
        from roi import load_roi_details
        _, is_adj, roi_area_mm2, _ = load_roi_details(roi_file) if os.path.exists(roi_file) else ({}, False, 0.0, "")

        animal_id = extract_animal_id(fname)
        m2 = re.search(r'~(\d+)$', fname)
        corte_num = int(m2.group(1)) if m2 else 1

        image_catalog.append({
            'group': group, 'section': sec, 'image_name': fname, 'base_name': base_name,
            'animal_id': animal_id, 'corte_num': corte_num,
            'has_roi': has_roi,
            'has_roi_b': has_roi,
            'has_roi_any': has_roi,
            'is_roi_adjusted': is_adj,
            'roi_area_mm2': roi_area_mm2
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
                    df_c['is_in_roi']  = points_in_rois(coords, regions_dict) if has_roi else False
                    df_c['roi_region'] = get_point_region_assignment(coords, regions_dict) if has_roi else 'NONE'
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
        "🏆 ANOVA Factorial de 3 Vías (Condición × Sexo × Hemisferio)",
        "🔬 ANOVA de 1 Vía por Condición (Filtro de Cohorte)",
        "📊 Desglose Tradicional (Panel por Sexo)"
    ],
    index=0,
    key="stats_mode_select"
)

# ─── DATA PREPARATION (ROI & GLOBAL COMPREHENSIVE CALCULATION) ───
valid_images_df = df_all_images.copy()
active_cells_df = df_raw_nuclei.copy()

img_summaries = []
for _, img_row in valid_images_df.iterrows():
    g = img_row['group']
    s = img_row['section']
    fn = img_row['image_name']
    aid = img_row['animal_id']
    cnum = img_row['corte_num']
    base_name = img_row.get('base_name', os.path.splitext(fn)[0])
    
    px_size = 0.8913
    area_global_mm2 = float((2048 * px_size * 2048 * px_size) / 1e6)
    area_roi_mm2 = 0.3480150625
    
    roi_json_path = os.path.join(METRICS_BASE_DIR, g, s, f"{base_name}_rois.json")
    if not os.path.exists(roi_json_path):
        roi_json_path = get_roi_json_path(METRICS_BASE_DIR, fn)
    if os.path.exists(roi_json_path):
        d_roi = load_rois(roi_json_path)
        polys = d_roi.get("ROI", [])
        total_um2 = 0.0
        for poly in polys:
            pts = np.array(poly)
            if len(pts) >= 3:
                x, y = pts[:, 1], pts[:, 0]
                a_px = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                total_um2 += a_px * (px_size ** 2)
        if total_um2 > 0:
            area_roi_mm2 = total_um2 / 1e6
            
    img_cells = active_cells_df[active_cells_df['image_name'] == fn] if not active_cells_df.empty else pd.DataFrame()
    cells_roi = img_cells[img_cells['is_in_roi'] == True] if not img_cells.empty else pd.DataFrame()
    cells_global = img_cells

    def _calc_metrics_subset(c_df, area_mm2, tag):
        pnn_cells = c_df[c_df['is_pnn_plus'] == True] if not c_df.empty else pd.DataFrame()
        pv_cells = c_df[c_df['is_pv_plus'] == True] if not c_df.empty else pd.DataFrame()
        
        coloc_cells = c_df[c_df['cell_type'] == 'PV+/PNN+'] if not c_df.empty else pd.DataFrame()
        hollow_cells = c_df[c_df['cell_type'] == 'PV-/PNN+'] if not c_df.empty else pd.DataFrame()
        isolated_cells = c_df[c_df['cell_type'] == 'PV+/PNN-'] if not c_df.empty else pd.DataFrame()
        
        n_pnn = len(pnn_cells)
        n_pv = len(pv_cells)
        n_coloc = len(coloc_cells)
        n_hollow = len(hollow_cells)
        n_isolated = len(isolated_cells)
        
        dens_pnn = float(n_pnn / area_mm2) if area_mm2 > 0 else 0.0
        dens_pv = float(n_pv / area_mm2) if area_mm2 > 0 else 0.0
        dens_coloc = float(n_coloc / area_mm2) if area_mm2 > 0 else 0.0
        dens_hollow = float(n_hollow / area_mm2) if area_mm2 > 0 else 0.0
        dens_isolated = float(n_isolated / area_mm2) if area_mm2 > 0 else 0.0
        
        pct_pv_coloc = float(n_coloc / n_pv * 100.0) if n_pv > 0 else 0.0
        pct_pv_isolated = float(n_isolated / n_pv * 100.0) if n_pv > 0 else 0.0
        pct_pnn_hollow = float(n_hollow / n_pnn * 100.0) if n_pnn > 0 else 0.0
        
        def _sm(s): return float(s.mean()) if len(s) > 0 and pd.notna(s.mean()) else 0.0
        
        mean_pnn_area = _sm(pnn_cells['pnn_area_um2']) if 'pnn_area_um2' in pnn_cells.columns else _sm(pnn_cells['area_um2'])
        mean_pnn_diam = _sm(pnn_cells['pnn_diameter_um']) if 'pnn_diameter_um' in pnn_cells.columns else _sm(pnn_cells['diameter_um'])
        mean_pv_area = _sm(pv_cells['pv_area_um2']) if 'pv_area_um2' in pv_cells.columns else _sm(pv_cells['area_um2'])
        mean_pv_diam = _sm(pv_cells['pv_diameter_um']) if 'pv_diameter_um' in pv_cells.columns else _sm(pv_cells['diameter_um'])
        
        wfa_sum = _sm(pnn_cells['wfa_sum_intensity']) if not pnn_cells.empty and 'wfa_sum_intensity' in pnn_cells.columns else 0.0
        wfa_mean = _sm(pnn_cells['wfa_mean_intensity']) if not pnn_cells.empty and 'wfa_mean_intensity' in pnn_cells.columns else 0.0
        wfa_peri = _sm(pnn_cells['wfa_pericellular_intensity']) if not pnn_cells.empty and 'wfa_pericellular_intensity' in pnn_cells.columns else 0.0

        return {
            f'pnn_density_{tag}': dens_pnn,
            f'pnn_count_{tag}': n_pnn,
            f'pnn_filled_density_{tag}': dens_coloc,
            f'pnn_count_filled_{tag}': n_coloc,
            f'pnn_hollow_density_{tag}': dens_hollow,
            f'pnn_count_hollow_{tag}': n_hollow,
            f'pct_pnn_hollow_{tag}': pct_pnn_hollow,
            
            f'pv_density_{tag}': dens_pv,
            f'pv_count_{tag}': n_pv,
            f'pv_coloc_density_{tag}': dens_coloc,
            f'pv_count_coloc_{tag}': n_coloc,
            f'pv_isolated_density_{tag}': dens_isolated,
            f'pv_count_isolated_{tag}': n_isolated,
            f'pct_pv_wrapped_{tag}': pct_pv_coloc,
            f'pct_pv_isolated_{tag}': pct_pv_isolated,
            
            f'mean_pnn_area_{tag}': mean_pnn_area,
            f'mean_pnn_diameter_{tag}': mean_pnn_diam,
            f'mean_pv_area_{tag}': mean_pv_area,
            f'mean_pv_diameter_{tag}': mean_pv_diam,
            
            f'mean_wfa_int_density_{tag}': wfa_sum,
            f'mean_wfa_mean_intensity_{tag}': wfa_mean,
            f'mean_wfa_pericellular_intensity_{tag}': wfa_peri,
            f'image_area_mm2_{tag}': area_mm2
        }

    base = {
        'group': g, 'section': s, 'image_name': fn, 'animal_id': aid, 'corte_num': cnum, 'base_name': base_name
    }
    base.update(_calc_metrics_subset(cells_roi, area_roi_mm2, 'roi'))
    base.update(_calc_metrics_subset(cells_global, area_global_mm2, 'global'))
    
    # Diffuse WFA Neuropil Component
    mask_file = os.path.join("data/processed/segmented", g, s, f"{base_name}_masks.tif")
    if os.path.exists(mask_file):
        try:
            stk = tiff.imread(mask_file)
            m_wfa = stk[2] if len(stk.shape) >= 3 and stk.shape[0] >= 3 else np.zeros((2048, 2048), dtype=np.uint16)
            w_raw = stk[4] if len(stk.shape) >= 3 and stk.shape[0] >= 5 else (stk[3] if stk.shape[0] >= 4 else np.zeros_like(m_wfa))
            
            roi_mask = np.zeros_like(m_wfa, dtype=bool)
            if os.path.exists(roi_json_path):
                d_roi = load_rois(roi_json_path)
                roi_mask = create_roi_mask(w_raw.shape, d_roi, target_region='ALL')
            else:
                roi_mask = np.ones_like(m_wfa, dtype=bool)
                
            pnn_mask = (m_wfa > 0)
            diffuse_mask_global = ~pnn_mask
            diffuse_mask_roi = diffuse_mask_global & roi_mask
            diffuse_mask_outside = diffuse_mask_global & (~roi_mask)
            
            wfa_diff_roi = float(np.mean(w_raw[diffuse_mask_roi])) if np.sum(diffuse_mask_roi) > 0 else 0.0
            wfa_diff_outside = float(np.mean(w_raw[diffuse_mask_outside])) if np.sum(diffuse_mask_outside) > 0 else 0.0
            wfa_diff_global = float(np.mean(w_raw[diffuse_mask_global])) if np.sum(diffuse_mask_global) > 0 else 0.0
            
            pnn_mean_roi = float(np.mean(w_raw[pnn_mask & roi_mask])) if np.sum(pnn_mask & roi_mask) > 0 else 0.0
            pnn_mean_global = float(np.mean(w_raw[pnn_mask])) if np.sum(pnn_mask) > 0 else 0.0
            
            base['wfa_diffuse_mean_roi'] = wfa_diff_roi
            base['wfa_diffuse_mean_outside_roi'] = wfa_diff_outside
            base['wfa_diffuse_mean_global'] = wfa_diff_global
            base['wfa_signal_to_diffuse_ratio_roi'] = float(pnn_mean_roi / (wfa_diff_roi + 1e-8)) if wfa_diff_roi > 0 else 1.0
            base['wfa_signal_to_diffuse_ratio_global'] = float(pnn_mean_global / (wfa_diff_global + 1e-8)) if wfa_diff_global > 0 else 1.0
        except Exception:
            base['wfa_diffuse_mean_roi'] = 0.0
            base['wfa_diffuse_mean_outside_roi'] = 0.0
            base['wfa_diffuse_mean_global'] = 0.0
            base['wfa_signal_to_diffuse_ratio_roi'] = 1.0
            base['wfa_signal_to_diffuse_ratio_global'] = 1.0
    else:
        base['wfa_diffuse_mean_roi'] = 0.0
        base['wfa_diffuse_mean_outside_roi'] = 0.0
        base['wfa_diffuse_mean_global'] = 0.0
        base['wfa_signal_to_diffuse_ratio_roi'] = 1.0
        base['wfa_signal_to_diffuse_ratio_global'] = 1.0

    img_summaries.append(base)

df_img = pd.DataFrame(img_summaries) if img_summaries else pd.DataFrame()

# Build subject level summaries
def aggregate_subject_level_clean(df_image_level):
    if df_image_level.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in df_image_level.columns
                    if c not in ['group','section','image_name','animal_id','corte_num','base_name']
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

# ─── EXPLICIT N AND DATA CARD ───
total_imgs_all = len(df_all_images)
total_cells_cnt = len(active_cells_df)
total_subjs_cnt = valid_images_df['animal_id'].nunique() if not valid_images_df.empty else 0

st.markdown('<div class="stats-box">', unsafe_allow_html=True)
c_box1, c_box2, c_box3, c_box4 = st.columns(4)
c_box1.metric("📐 Métricas Disponibles", "ROI (0.348 mm²) y Globales")
c_box2.metric("🐀 Sujetos Activos (N_suj)", f"{total_subjs_cnt}")
c_box3.metric("🔬 Preparados Totales (N_prep)", f"{total_imgs_all}")
c_box4.metric("🧫 Células Totales (N_células)", f"{total_cells_cnt:,}")

with st.expander("📐 Desglose de Área y Consistencia de ROIs por Grupo Experimental", expanded=False):
    roi_grp_rows = []
    for (grp, sec), g_df in df_img.groupby(['group', 'section']):
        s_cnt = g_df['animal_id'].nunique()
        i_cnt = len(g_df)
        a_mean = g_df['image_area_mm2_roi'].mean()
        a_std = g_df['image_area_mm2_roi'].std()
        roi_grp_rows.append({
            "Grupo Experimental": grp,
            "Hemisferio": sec,
            "N Sujetos": s_cnt,
            "N Preparados": i_cnt,
            "Área Media ROI (mm²)": f"{a_mean:.4f}" if pd.isna(a_std) or a_std < 1e-4 else f"{a_mean:.4f} ± {a_std:.4f}",
            "Área Media ROI (µm²)": f"{(a_mean * 1e6):,.0f}",
            "Área Global Media (mm²)": f"{g_df['image_area_mm2_global'].mean():.2f}"
        })
    df_roi_table = pd.DataFrame(roi_grp_rows)
    st.dataframe(df_roi_table, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Dynamic Variable Label generator
def get_dynamic_labels(level):
    suffix = "— Sujeto" if level == "Por Sujeto (animal, promediando cortes)" else "— Preparado" if level == "Por Preparado/Corte (imagen)" else "— Célula"
    
    if level == "Por Célula (distribuciones individuales)":
        pv_vars = {
            "pv_area_um2": "Área del Soma PV+ (µm²) — por Célula",
            "pv_diameter_um": "Diámetro del Soma PV+ (µm) — por Célula",
        }
        pnn_vars = {
            "pnn_area_um2": "Área Malla PNN (µm²) — por Célula",
            "pnn_diameter_um": "Diámetro Malla PNN (µm) — por Célula",
            "score": "Confianza PNNscore (IA) — por Célula",
        }
        intensity_vars = {
            "wfa_sum_intensity": "Densidad Integrada Fluorescencia WFA (Suma) — por Célula",
            "wfa_mean_intensity": "Intensidad Media Fluorescencia WFA en PNN — por Célula",
            "wfa_pericellular_intensity": "Intensidad Pericelular Anillo WFA — por Célula"
        }
        diffuse_vars = {
            "wfa_diffuse_mean_roi": "Intensidad Fondo Difuso WFA (Dentro de ROI)",
            "wfa_diffuse_mean_outside_roi": "Intensidad Fondo Difuso WFA (Fuera de ROI)",
            "wfa_diffuse_mean_global": "Intensidad Fondo Difuso WFA (Global / Toda la Imagen)"
        }
    else:
        pv_vars = {
            "pv_density_roi": f"Densidad PV+ Totales (/mm²) (ROI) {suffix}",
            "pv_density_global": f"Densidad PV+ Totales (/mm²) (Global) {suffix}",
            "pv_count_roi": f"N° Interneuronas PV+ Totales (ROI) {suffix}",
            "pv_count_global": f"N° Interneuronas PV+ Totales (Global) {suffix}",
            "pv_coloc_density_roi": f"Densidad PV+/PNN+ (/mm²) (ROI) {suffix}",
            "pv_coloc_density_global": f"Densidad PV+/PNN+ (/mm²) (Global) {suffix}",
            "pv_coloc_count_roi": f"N° PV+/PNN+ Colocalizadas (ROI) {suffix}",
            "pv_coloc_count_global": f"N° PV+/PNN+ Colocalizadas (Global) {suffix}",
            "pv_isolated_density_roi": f"Densidad PV+/PNN- Libres de PNN (/mm²) (ROI) {suffix}",
            "pv_isolated_density_global": f"Densidad PV+/PNN- Libres de PNN (/mm²) (Global) {suffix}",
            "pv_isolated_count_roi": f"N° PV+/PNN- Libres de PNN (ROI) {suffix}",
            "pv_isolated_count_global": f"N° PV+/PNN- Libres de PNN (Global) {suffix}",
            "pct_pv_wrapped_roi": f"% de PV+ Envueltas en PNN (PV+/PNN+) (ROI) {suffix}",
            "pct_pv_wrapped_global": f"% de PV+ Envueltas en PNN (PV+/PNN+) (Global) {suffix}",
            "pct_pv_isolated_roi": f"% de PV+ Libres de PNN (PV+/PNN-) (ROI) {suffix}",
            "pct_pv_isolated_global": f"% de PV+ Libres de PNN (PV+/PNN-) (Global) {suffix}",
            "mean_pv_area_roi": f"Área Promedio Soma PV+ (µm²) (ROI) {suffix}",
            "mean_pv_area_global": f"Área Promedio Soma PV+ (µm²) (Global) {suffix}",
            "mean_pv_diameter_roi": f"Diámetro Promedio Soma PV+ (µm²) (ROI) {suffix}",
            "mean_pv_diameter_global": f"Diámetro Promedio Soma PV+ (µm²) (Global) {suffix}"
        }
        
        pnn_vars = {
            "pnn_density_roi": f"Densidad PNNs Totales (/mm²) (ROI) {suffix}",
            "pnn_density_global": f"Densidad PNNs Totales (/mm²) (Global) {suffix}",
            "pnn_count_roi": f"N° Redes PNN+ Totales (ROI) {suffix}",
            "pnn_count_global": f"N° Redes PNN+ Totales (Global) {suffix}",
            "pnn_filled_density_roi": f"Densidad PNNs Rellenas PV+/PNN+ (/mm²) (ROI) {suffix}",
            "pnn_filled_density_global": f"Densidad PNNs Rellenas PV+/PNN+ (/mm²) (Global) {suffix}",
            "pnn_count_filled_roi": f"N° PNNs Rellenas PV+/PNN+ (ROI) {suffix}",
            "pnn_count_filled_global": f"N° PNNs Rellenas PV+/PNN+ (Global) {suffix}",
            "pnn_hollow_density_roi": f"Densidad PNNs Huecas PV-/PNN+ (/mm²) (ROI) {suffix}",
            "pnn_hollow_density_global": f"Densidad PNNs Huecas PV-/PNN+ (/mm²) (Global) {suffix}",
            "pnn_count_hollow_roi": f"N° PNNs Huecas PV-/PNN+ (ROI) {suffix}",
            "pnn_count_hollow_global": f"N° PNNs Huecas PV-/PNN+ (Global) {suffix}",
            "pct_pnn_hollow_roi": f"% PNNs Huecas (PV-/PNN+) (ROI) {suffix}",
            "pct_pnn_hollow_global": f"% PNNs Huecas (PV-/PNN+) (Global) {suffix}",
            "mean_pnn_area_roi": f"Área Promedio Malla PNN (µm²) (ROI) {suffix}",
            "mean_pnn_area_global": f"Área Promedio Malla PNN (µm²) (Global) {suffix}",
            "mean_pnn_diameter_roi": f"Diámetro Promedio Malla PNN (µm) (ROI) {suffix}",
            "mean_pnn_diameter_global": f"Diámetro Promedio Malla PNN (µm) (Global) {suffix}"
        }

        intensity_vars = {
            "mean_wfa_int_density_roi": f"Media Integrada Fluorescencia WFA (Densidad Integrada) (ROI) {suffix}",
            "mean_wfa_int_density_global": f"Media Integrada Fluorescencia WFA (Densidad Integrada) (Global) {suffix}",
            "mean_wfa_mean_intensity_roi": f"Intensidad Media Fluorescencia WFA en PNN (ROI) {suffix}",
            "mean_wfa_mean_intensity_global": f"Intensidad Media Fluorescencia WFA en PNN (Global) {suffix}",
            "mean_wfa_pericellular_intensity_roi": f"Intensidad Anillo Pericelular WFA (ROI) {suffix}",
            "mean_wfa_pericellular_intensity_global": f"Intensidad Anillo Pericelular WFA (Global) {suffix}"
        }

        diffuse_vars = {
            "wfa_diffuse_mean_roi": f"Intensidad Fondo Difuso WFA (Dentro de ROI) {suffix}",
            "wfa_diffuse_mean_outside_roi": f"Intensidad Fondo Difuso WFA (Fuera de ROI) {suffix}",
            "wfa_diffuse_mean_global": f"Intensidad Fondo Difuso WFA (Global / Toda la Imagen) {suffix}",
            "wfa_signal_to_diffuse_ratio_roi": f"Ratio Señal PNN / Fondo Difuso (ROI) {suffix}",
            "wfa_signal_to_diffuse_ratio_global": f"Ratio Señal PNN / Fondo Difuso (Global) {suffix}"
        }
        
    return pv_vars, pnn_vars, intensity_vars, diffuse_vars

PV_VARS, PNN_VARS, INTENSITY_VARS, DIFFUSE_VARS = get_dynamic_labels(level_type)

tab_pv, tab_pnn, tab_wfa_int, tab_diffuse = st.tabs([
    "🧪 Interneuronas PV+ (Subpoblaciones)",
    "🧠 Redes Perineuronales PNN (Huecas / Rellenas)",
    "🔬 Intensidades y Medias Integradas WFA",
    "🌫️ Espacio Difuso WFA (Dentro / Fuera de ROI)"
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

    if level_type == "Por Sujeto (animal, promediando cortes)":
        st.markdown("""
        <div class="level-info">
        📌 <b>Nivel: Sujeto/Animal</b> — Cada punto representa el promedio de cortes para un animal individual en el hemisferio correspondiente (IPSI o CONTRA).
        </div>
        """, unsafe_allow_html=True)
    elif level_type == "Por Preparado/Corte (imagen)":
        st.markdown("""
        <div class="level-info">
        📌 <b>Nivel: Preparado/Corte</b> — Cada punto representa la métrica total/promedio de una imagen TIFF.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="level-info">
        📌 <b>Nivel: Célula Individual</b> — Cada punto representa una célula individual detectada.
        </div>
        """, unsafe_allow_html=True)

    SECTION_COLORS = {'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'}

    # ═════════════════════════════════════════════════════════════════════
    # MODO 0: ANOVA FACTORIAL DE 3 VÍAS (CONDICIÓN × SEXO × HEMISFERIO)
    # ═════════════════════════════════════════════════════════════════════
    if "3 Vías" in stats_mode or "Factorial" in stats_mode:
        df_3way = df_base.dropna(subset=[selected_var_key, 'condition', 'sex', 'section']).copy()
        df_3way['condition'] = pd.Categorical(df_3way['condition'], categories=COND_ORDER, ordered=True)
        
        if len(df_3way) < 8 or df_3way['condition'].nunique() < 2:
            st.warning("No hay suficientes datos celulares o de sujetos para ajustar el modelo ANOVA Factorial de 3 Vías.")
            return
            
        formula = f"{selected_var_key} ~ C(condition, Sum) * C(sex, Sum) * C(section, Sum)"
        
        try:
            model_3way = ols(formula, data=df_3way).fit()
            aov_raw = anova_lm(model_3way, typ=3)
            
            ss_res = aov_raw.loc['Residual', 'sum_sq']
            aov_raw['eta_p2'] = aov_raw['sum_sq'] / (aov_raw['sum_sq'] + ss_res)
            
            name_map = {
                'Intercept': 'Intersección (Media Global)',
                'C(condition, Sum)': 'Condición Temporal (NONE, 3d, 7d, 14d)',
                'C(sex, Sum)': 'Sexo (MACHO vs HEMBRA)',
                'C(section, Sum)': 'Hemisferio (IPSI vs CONTRA)',
                'C(condition, Sum):C(sex, Sum)': 'Condición × Sexo',
                'C(condition, Sum):C(section, Sum)': 'Condición × Hemisferio',
                'C(sex, Sum):C(section, Sum)': 'Sexo × Hemisferio',
                'C(condition, Sum):C(sex, Sum):C(section, Sum)': 'Condición × Sexo × Hemisferio (Triple)',
                'Residual': 'Varianza Residual (Error)'
            }
            
            aov_rows = []
            for idx_name, r in aov_raw.iterrows():
                if idx_name == 'Intercept':
                    continue
                clean_name = name_map.get(idx_name, idx_name)
                ss_val = r['sum_sq']
                df_val = int(r['df'])
                ms_val = ss_val / df_val if df_val > 0 else 0.0
                f_val = r['F']
                p_val = r['PR(>F)']
                eta_p2 = r.get('eta_p2', 0.0)
                
                if idx_name == 'Residual':
                    aov_rows.append({
                        "Fuente de Variación": clean_name,
                        "Suma Cuadrados (SS)": f"{ss_val:.3f}",
                        "Grados Libertad (df)": f"{df_val}",
                        "Cuadrado Medio (MS)": f"{ms_val:.3f}",
                        "F": "-",
                        "p-valor": "-",
                        "Significancia": "-",
                        "Eta Parcial (ηp²)": "-",
                        "Impacto": "-"
                    })
                else:
                    impact = "Fuerte (Marcado)" if eta_p2 >= 0.14 else ("Moderado" if eta_p2 >= 0.06 else ("Pequeño" if eta_p2 >= 0.01 else "Despreciable"))
                    aov_rows.append({
                        "Fuente de Variación": clean_name,
                        "Suma Cuadrados (SS)": f"{ss_val:.3f}",
                        "Grados Libertad (df)": f"{df_val}",
                        "Cuadrado Medio (MS)": f"{ms_val:.3f}",
                        "F": f"{f_val:.3f}" if pd.notna(f_val) else "-",
                        "p-valor": f"{p_val:.4e}" if pd.notna(p_val) else "-",
                        "Significancia": sig_stars(p_val) if pd.notna(p_val) else "-",
                        "Eta Parcial (ηp²)": f"{eta_p2:.4f}" if pd.notna(eta_p2) else "-",
                        "Impacto": impact if pd.notna(eta_p2) else "-"
                    })
                    
            st.markdown('<div class="table-caption">📋 Tabla ANOVA Factorial de Tres Vías (Tipo III - Sumas de Cuadrados Parciales)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(aov_rows), use_container_width=True)
            
            # Key KPI highlights
            c_a1, c_a2, c_a3 = st.columns(3)
            r_cond = aov_raw.loc['C(condition, Sum)'] if 'C(condition, Sum)' in aov_raw.index else None
            r_sec = aov_raw.loc['C(section, Sum)'] if 'C(section, Sum)' in aov_raw.index else None
            r_inter = aov_raw.loc['C(condition, Sum):C(section, Sum)'] if 'C(condition, Sum):C(section, Sum)' in aov_raw.index else None
            
            if r_cond is not None and pd.notna(r_cond['F']):
                c_a1.metric("⏱️ Efecto Condición", f"F = {r_cond['F']:.2f}", delta=f"p = {r_cond['PR(>F)']:.4e}", delta_color="normal" if r_cond['PR(>F)'] < 0.05 else "off")
            if r_sec is not None and pd.notna(r_sec['F']):
                c_a2.metric("🛡️ Efecto Hemisferio", f"F = {r_sec['F']:.2f}", delta=f"p = {r_sec['PR(>F)']:.4e}", delta_color="normal" if r_sec['PR(>F)'] < 0.05 else "off")
            if r_inter is not None and pd.notna(r_inter['F']):
                c_a3.metric("⚡ Interacción Condición × Hemisferio", f"F = {r_inter['F']:.2f}", delta=f"p = {r_inter['PR(>F)']:.4e}", delta_color="normal" if r_inter['PR(>F)'] < 0.05 else "off")
                
            # ─── INTERACTION PLOT (MEDIAS ESTIMADAS 3 VÍAS) ───
            st.divider()
            st.subheader("📈 Perfil de Interacción Factorial (Medias Estimadas Condición × Hemisferio × Sexo)")
            
            marginal_stats = df_3way.groupby(['condition', 'section', 'sex'])[selected_var_key].agg(['mean', 'std', 'count']).reset_index()
            marginal_stats['sem'] = marginal_stats['std'] / np.sqrt(marginal_stats['count'])
            
            fig_3w = px.line(
                marginal_stats,
                x='condition',
                y='mean',
                color='section',
                facet_col='sex',
                error_y='sem',
                markers=True,
                color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                title=f"Perfil de Interacción 3 Vías: {var_options[selected_var_key]} (Media ± SEM)",
                labels={'condition': 'Condición', 'mean': var_options[selected_var_key], 'section': 'Hemisferio'},
                template='plotly_dark'
            )
            fig_3w.update_traces(line=dict(width=2.5), marker=dict(size=8))
            fig_3w.update_layout(height=480)
            st.plotly_chart(fig_3w, use_container_width=True)
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.download_button(
                    "📥 Descargar Medias Estimadas (Gráfica 3-Vías) (.csv)",
                    data=marginal_stats.to_csv(index=False).encode('utf-8'),
                    file_name=f"grafica_anova_3vias_medias_{selected_var_key}.csv",
                    mime="text/csv",
                    key=f"dl_3w_stats_{selected_var_key}_{title_lbl}".replace(" ", "_")
                )
            with c_d2:
                cols_3w = [c for c in ['animal_id', 'condition', 'sex', 'section', selected_var_key] if c in df_3way.columns]
                st.download_button(
                    "📥 Descargar Datos Brutos (Gráfica 3-Vías) (.csv)",
                    data=df_3way[cols_3w].to_csv(index=False).encode('utf-8'),
                    file_name=f"grafica_anova_3vias_datos_brutos_{selected_var_key}.csv",
                    mime="text/csv",
                    key=f"dl_3w_raw_{selected_var_key}_{title_lbl}".replace(" ", "_")
                )
            
        except Exception as e_aov:
            st.error(f"No se pudo ajustar el modelo ANOVA de 3 Vías: {e_aov}")

    # ═════════════════════════════════════════════════════════════════════
    # MODO 1: ANOVA DE 1 VÍA POR CONDICIÓN EXPERIMENTAL (FILTRABLE EN SIDEBAR)
    # ═════════════════════════════════════════════════════════════════════
    elif "1 Vía" in stats_mode:
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

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            cols_export_1w = [c for c in ['animal_id', 'condition', 'sex', 'section', selected_var_key] if c in df_filtered.columns]
            st.download_button(
                "📥 Descargar Datos de la Gráfica (.csv)",
                data=df_filtered[cols_export_1w].to_csv(index=False).encode('utf-8'),
                file_name=f"grafica_anova_1via_{selected_var_key}_{filter_sex}_{filter_sec}.csv".replace(" ", "_"),
                mime="text/csv",
                key=f"dl_1w_data_{selected_var_key}_{title_lbl}_{filter_sex}_{filter_sec}".replace(" ", "_")
            )
        with c_d2:
            if tukey_table:
                st.download_button(
                    "📥 Descargar Resultados Post-Hoc Tukey (.csv)",
                    data=pd.DataFrame(tukey_table).to_csv(index=False).encode('utf-8'),
                    file_name=f"posthoc_tukey_{selected_var_key}_{filter_sex}_{filter_sec}.csv".replace(" ", "_"),
                    mime="text/csv",
                    key=f"dl_1w_tukey_{selected_var_key}_{title_lbl}".replace(" ", "_")
                )

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

                cols_export_sub = [c for c in ['animal_id', 'condition', 'sex', 'section', selected_var_key] if c in df_sub.columns]
                st.download_button(
                    f"📥 Descargar Datos de la Gráfica ({group_title}) (.csv)",
                    data=df_sub[cols_export_sub].to_csv(index=False).encode('utf-8'),
                    file_name=f"grafica_{group_title.lower()}_{selected_var_key}.csv".replace(" ", "_"),
                    mime="text/csv",
                    key=f"dl_classic_{group_title}_{selected_var_key}_{title_lbl}".replace(" ", "_")
                )

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
        st.header("🧪 Análisis de Interneuronas PV+ (Subpoblaciones)")
        selected_pv_var = st.selectbox("Variable PV+:", list(PV_VARS.keys()), format_func=lambda x: PV_VARS[x], key="pv_var")

        is_cell_var = selected_pv_var in ["pv_area_um2", "pv_diameter_um"]
        if is_cell_var and level_type == "Por Célula (distribuciones individuales)":
            df_pv_base = active_cells_df[active_cells_df['is_pv_plus'] == True].copy()
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_pv_base = df_subj.copy()
        else: # Por Preparado/Corte (imagen)
            df_pv_base = df_img.copy()

        run_stats_layout(df_pv_base, PV_VARS, selected_pv_var, "Interneuronas PV+")

# 2. TAB PNN
with tab_pnn:
    if df_img.empty:
        st.info("No hay datos de PNN disponibles.")
    else:
        st.header("🧠 Análisis de Redes Perineuronales PNN (Huecas y Rellenas)")
        selected_pnn_var = st.selectbox("Variable PNN:", list(PNN_VARS.keys()), format_func=lambda x: PNN_VARS[x], key="pnn_var")

        is_cell_var = selected_pnn_var in ["pnn_area_um2", "pnn_diameter_um", "score"]
        if is_cell_var and level_type == "Por Célula (distribuciones individuales)":
            df_pnn_base = active_cells_df[active_cells_df['is_pnn_plus'] == True].copy()
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_pnn_base = df_subj.copy()
        else: # Por Preparado/Corte (imagen)
            df_pnn_base = df_img.copy()

        run_stats_layout(df_pnn_base, PNN_VARS, selected_pnn_var, "Redes Perineuronales PNN")

# 3. TAB INTENSIDADES Y MEDIAS INTEGRADAS WFA
with tab_wfa_int:
    if df_img.empty:
        st.info("No hay datos de intensidades WFA disponibles.")
    else:
        st.header("🔬 Intensidades y Medias Integradas de Fluorescencia WFA")
        st.markdown("""
        * **Media Integrada de Fluorescencia WFA:** Suma total de intensidad de señal WFA condensada en la red perineuronal (Densidad Integrada).
        * **Intensidad Media WFA en PNN:** Nivel medio de fluorescencia dentro del área de la PNN.
        * **Intensidad Pericelular:** Fluorescencia específica medida en el anillo perisomático de la red.
        """)
        selected_wfa_var = st.selectbox("Variable de Intensidad WFA:", list(INTENSITY_VARS.keys()), format_func=lambda x: INTENSITY_VARS[x], key="wfa_int_var")

        if level_type == "Por Célula (distribuciones individuales)":
            df_wfa_base = active_cells_df[active_cells_df['is_pnn_plus'] == True].copy()
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_wfa_base = df_subj.copy()
        else:
            df_wfa_base = df_img.copy()

        run_stats_layout(df_wfa_base, INTENSITY_VARS, selected_wfa_var, "Intensidades Fluorescentes WFA")

# 4. TAB ESPACIO DIFUSO WFA (DENTRO Y FUERA DEL ROI)
with tab_diffuse:
    if df_img.empty:
        st.info("No hay datos de espacio difuso disponibles.")
    else:
        st.header("🌫️ Espacio Difuso y Neuropilo WFA (Dentro y Fuera del ROI)")
        st.markdown("""
        * **Fondo Difuso Dentro del ROI:** Fluorescencia media del neuropilo extracelular libre de somas y condensaciones PNN dentro del área estándar ROI.
        * **Fondo Difuso Fuera del ROI:** Fluorescencia media del neuropilo en el resto de la corteza.
        * **Fondo Difuso Global:** Media en todo el corte histológico.
        * **Ratio Señal PNN / Fondo Difuso:** Contraste y definición de las redes perineuronales sobre el fondo de la matriz.
        """)
        selected_diff_var = st.selectbox("Variable de Espacio Difuso WFA:", list(DIFFUSE_VARS.keys()), format_func=lambda x: DIFFUSE_VARS[x], key="diff_var")

        if level_type == "Por Célula (distribuciones individuales)":
            df_diff_base = active_cells_df.copy()
            for k in DIFFUSE_VARS.keys():
                if k not in df_diff_base.columns and k in df_img.columns:
                    mapping = df_img.set_index('image_name')[k].to_dict()
                    df_diff_base[k] = df_diff_base['image_name'].map(mapping)
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_diff_base = df_subj.copy()
        else:
            df_diff_base = df_img.copy()

        run_stats_layout(df_diff_base, DIFFUSE_VARS, selected_diff_var, "Espacio Difuso WFA")

        st.markdown("### 📋 Resumen Consolidado de Espacio Difuso y Ratios por Preparado:")
        cols_show = ['group', 'section', 'image_name', 'wfa_diffuse_mean_roi', 'wfa_diffuse_mean_outside_roi', 'wfa_diffuse_mean_global', 'wfa_signal_to_diffuse_ratio_roi', 'wfa_signal_to_diffuse_ratio_global']
        cols_present = [c for c in cols_show if c in df_img.columns]
        st.dataframe(df_img[cols_present], use_container_width=True)
        st.download_button(
            "📥 Descargar Datos de Espacio Difuso (.csv)",
            data=df_img[cols_present].to_csv(index=False).encode('utf-8'),
            file_name="espacio_difuso_wfa_roi_global.csv",
            mime="text/csv",
            key="dl_diffuse_wfa_csv"
        )
