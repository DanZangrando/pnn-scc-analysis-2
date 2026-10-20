import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import mannwhitneyu, wilcoxon, f_oneway, kruskal, levene, ttest_rel, ttest_ind, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import sys
import io
import datetime
import tifffile as tiff
sys.path.append(os.path.abspath("src"))
from roi import load_rois, get_roi_json_path, points_in_rois, get_point_region_assignment, create_roi_mask
from image_io import extract_animal_id
import subprocess
from export_utils import (
    get_pipeline_parameters_dict,
    generate_prism_column_matrix,
    generate_prism_grouped_2way_matrix,
    build_full_export_zip,
    build_all_graphics_and_data_zip,
    export_all_graphics_and_data_to_directory
)

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
    if 'wfa_int_per_mesh_um2' not in df_raw_nuclei.columns and 'wfa_sum_intensity' in df_raw_nuclei.columns:
        df_raw_nuclei['wfa_int_per_mesh_um2'] = df_raw_nuclei.apply(
            lambda r: (r['wfa_sum_intensity'] / (r['pnn_area_um2'] + 1e-8)) if (r.get('is_pnn_plus', False) and r.get('pnn_area_um2', 0.0) > 0) else 0.0,
            axis=1
        )
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
        
        # Intensidades del Anillo de la PNN (Máscara de Señal WFA)
        wfa_peri = _sm(pnn_cells['wfa_pericellular_intensity']) if not pnn_cells.empty and 'wfa_pericellular_intensity' in pnn_cells.columns else (_sm(pnn_cells['wfa_mean_intensity']) if not pnn_cells.empty and 'wfa_mean_intensity' in pnn_cells.columns else 0.0)
        wfa_sum = _sm(pnn_cells['wfa_sum_intensity']) if not pnn_cells.empty and 'wfa_sum_intensity' in pnn_cells.columns else 0.0

        # Subpoblaciones PNN PV+ (Rellenas) y PNN PV- (Huecas)
        wfa_peri_filled = _sm(coloc_cells['wfa_pericellular_intensity']) if not coloc_cells.empty and 'wfa_pericellular_intensity' in coloc_cells.columns else (_sm(coloc_cells['wfa_mean_intensity']) if not coloc_cells.empty and 'wfa_mean_intensity' in coloc_cells.columns else 0.0)
        wfa_sum_filled = _sm(coloc_cells['wfa_sum_intensity']) if not coloc_cells.empty and 'wfa_sum_intensity' in coloc_cells.columns else 0.0
        pnn_area_filled = _sm(coloc_cells['pnn_area_um2']) if not coloc_cells.empty and 'pnn_area_um2' in coloc_cells.columns else _sm(coloc_cells['area_um2'])

        wfa_peri_hollow = _sm(hollow_cells['wfa_pericellular_intensity']) if not hollow_cells.empty and 'wfa_pericellular_intensity' in hollow_cells.columns else (_sm(hollow_cells['wfa_mean_intensity']) if not hollow_cells.empty and 'wfa_mean_intensity' in hollow_cells.columns else 0.0)
        wfa_sum_hollow = _sm(hollow_cells['wfa_sum_intensity']) if not hollow_cells.empty and 'wfa_sum_intensity' in hollow_cells.columns else 0.0
        pnn_area_hollow = _sm(hollow_cells['pnn_area_um2']) if not hollow_cells.empty and 'pnn_area_um2' in hollow_cells.columns else _sm(hollow_cells['area_um2'])

        # Suma total acumulada de intensidad integrada en todo el ROI
        total_wfa_sum = float(pnn_cells['wfa_sum_intensity'].sum()) if not pnn_cells.empty and 'wfa_sum_intensity' in pnn_cells.columns else 0.0
        total_wfa_sum_filled = float(coloc_cells['wfa_sum_intensity'].sum()) if not coloc_cells.empty and 'wfa_sum_intensity' in coloc_cells.columns else 0.0
        total_wfa_sum_hollow = float(hollow_cells['wfa_sum_intensity'].sum()) if not hollow_cells.empty and 'wfa_sum_intensity' in hollow_cells.columns else 0.0

        # Superficie total ocupada por mallas PNN (µm²)
        total_pnn_mesh_area_um2 = float(pnn_cells['pnn_area_um2'].sum()) if not pnn_cells.empty and 'pnn_area_um2' in pnn_cells.columns else (float(pnn_cells['area_um2'].sum()) if not pnn_cells.empty and 'area_um2' in pnn_cells.columns else 0.0)
        total_pnn_mesh_area_filled_um2 = float(coloc_cells['pnn_area_um2'].sum()) if not coloc_cells.empty and 'pnn_area_um2' in coloc_cells.columns else (float(coloc_cells['area_um2'].sum()) if not coloc_cells.empty and 'area_um2' in coloc_cells.columns else 0.0)
        total_pnn_mesh_area_hollow_um2 = float(hollow_cells['pnn_area_um2'].sum()) if not hollow_cells.empty and 'pnn_area_um2' in hollow_cells.columns else (float(hollow_cells['area_um2'].sum()) if not hollow_cells.empty and 'area_um2' in hollow_cells.columns else 0.0)

        # 1. Densidad Integrada WFA por mm² de ROI (Carga total de fluorescencia ponderada por área de tejido)
        wfa_int_density_per_mm2 = float(total_wfa_sum / area_mm2) if area_mm2 > 0 else 0.0
        wfa_int_density_filled_per_mm2 = float(total_wfa_sum_filled / area_mm2) if area_mm2 > 0 else 0.0
        wfa_int_density_hollow_per_mm2 = float(total_wfa_sum_hollow / area_mm2) if area_mm2 > 0 else 0.0

        # 2. Densidad Integrada WFA Ponderada por Área de Malla PNN (Intensidad / µm² de Red)
        wfa_int_per_mesh_um2 = float(total_wfa_sum / (total_pnn_mesh_area_um2 + 1e-8)) if total_pnn_mesh_area_um2 > 0 else 0.0
        wfa_int_filled_per_mesh_um2 = float(total_wfa_sum_filled / (total_pnn_mesh_area_filled_um2 + 1e-8)) if total_pnn_mesh_area_filled_um2 > 0 else 0.0
        wfa_int_hollow_per_mesh_um2 = float(total_wfa_sum_hollow / (total_pnn_mesh_area_hollow_um2 + 1e-8)) if total_pnn_mesh_area_hollow_um2 > 0 else 0.0

        # 3. Fracción de Área / Cobertura de PNNs en el ROI (% superficie cortical cubierta por PNNs)
        pnn_area_fraction_pct = float((total_pnn_mesh_area_um2 / (area_mm2 * 1e6)) * 100.0) if area_mm2 > 0 else 0.0
        pnn_filled_area_fraction_pct = float((total_pnn_mesh_area_filled_um2 / (area_mm2 * 1e6)) * 100.0) if area_mm2 > 0 else 0.0
        pnn_hollow_area_fraction_pct = float((total_pnn_mesh_area_hollow_um2 / (area_mm2 * 1e6)) * 100.0) if area_mm2 > 0 else 0.0

        # 4. Potencia / Energía WFA (Lupori et al.) y Potencia Ponderada por Fracción de Área
        pnn_energy = float(dens_pnn * wfa_peri)
        pnn_filled_energy = float(dens_coloc * wfa_peri_filled)
        pnn_hollow_energy = float(dens_hollow * wfa_peri_hollow)

        pnn_energy_area_weighted = float((total_pnn_mesh_area_um2 / (area_mm2 * 1e6)) * wfa_peri)
        pnn_filled_energy_area_weighted = float((total_pnn_mesh_area_filled_um2 / (area_mm2 * 1e6)) * wfa_peri_filled)
        pnn_hollow_energy_area_weighted = float((total_pnn_mesh_area_hollow_um2 / (area_mm2 * 1e6)) * wfa_peri_hollow)

        # Ratios Comparativos (PV+PNN+ vs PV-PNN+)
        ratio_intensity = float(wfa_peri_filled / (wfa_peri_hollow + 1e-8)) if wfa_peri_hollow > 0 else 1.0
        ratio_int_density = float(wfa_sum_filled / (wfa_sum_hollow + 1e-8)) if wfa_sum_hollow > 0 else 1.0
        ratio_int_density_per_mm2 = float(wfa_int_density_filled_per_mm2 / (wfa_int_density_hollow_per_mm2 + 1e-8)) if wfa_int_density_hollow_per_mm2 > 0 else 1.0
        ratio_int_per_mesh_um2 = float(wfa_int_filled_per_mesh_um2 / (wfa_int_hollow_per_mesh_um2 + 1e-8)) if wfa_int_hollow_per_mesh_um2 > 0 else 1.0
        ratio_energy = float(pnn_filled_energy / (pnn_hollow_energy + 1e-8)) if pnn_hollow_energy > 0 else 1.0

        return {
            # Conteo y Densidades PNN
            f'pnn_density_{tag}': dens_pnn,
            f'pnn_count_{tag}': n_pnn,
            f'pnn_filled_density_{tag}': dens_coloc,
            f'pnn_count_filled_{tag}': n_coloc,
            f'pnn_filled_count_{tag}': n_coloc,
            f'pnn_hollow_density_{tag}': dens_hollow,
            f'pnn_count_hollow_{tag}': n_hollow,
            f'pnn_hollow_count_{tag}': n_hollow,
            f'pct_pnn_hollow_{tag}': pct_pnn_hollow,
            
            # Conteo y Densidades PV
            f'pv_density_{tag}': dens_pv,
            f'pv_count_{tag}': n_pv,
            f'pv_coloc_density_{tag}': dens_coloc,
            f'pv_count_coloc_{tag}': n_coloc,
            f'pv_coloc_count_{tag}': n_coloc,
            f'pv_isolated_density_{tag}': dens_isolated,
            f'pv_count_isolated_{tag}': n_isolated,
            f'pv_isolated_count_{tag}': n_isolated,
            f'pct_pv_wrapped_{tag}': pct_pv_coloc,
            f'pct_pv_isolated_{tag}': pct_pv_isolated,
            
            # Morfometría
            f'mean_pnn_area_{tag}': mean_pnn_area,
            f'mean_pnn_diameter_{tag}': mean_pnn_diam,
            f'mean_pnn_area_filled_{tag}': pnn_area_filled,
            f'mean_pnn_area_hollow_{tag}': pnn_area_hollow,
            f'mean_pv_area_{tag}': mean_pv_area,
            f'mean_pv_diameter_{tag}': mean_pv_diam,

            # Cobertura de Área (% de Área Cortical Ocupada por PNNs)
            f'pnn_area_fraction_pct_{tag}': pnn_area_fraction_pct,
            f'pnn_filled_area_fraction_pct_{tag}': pnn_filled_area_fraction_pct,
            f'pnn_hollow_area_fraction_pct_{tag}': pnn_hollow_area_fraction_pct,
            
            # Potencia WFA (Energy)
            f'pnn_energy_{tag}': pnn_energy,
            f'pnn_filled_energy_{tag}': pnn_filled_energy,
            f'pnn_hollow_energy_{tag}': pnn_hollow_energy,
            f'pnn_energy_area_weighted_{tag}': pnn_energy_area_weighted,
            f'pnn_filled_energy_area_weighted_{tag}': pnn_filled_energy_area_weighted,
            f'pnn_hollow_energy_area_weighted_{tag}': pnn_hollow_energy_area_weighted,

            # Intensidad de la Red PNN (Anillo WFA)
            f'mean_wfa_intensity_{tag}': wfa_peri,
            f'mean_wfa_intensity_filled_{tag}': wfa_peri_filled,
            f'mean_wfa_intensity_hollow_{tag}': wfa_peri_hollow,

            # 1. Densidad Integrada WFA por mm² de ROI (Ponderada por Área de Tejido)
            f'wfa_int_density_per_mm2_{tag}': wfa_int_density_per_mm2,
            f'wfa_int_density_filled_per_mm2_{tag}': wfa_int_density_filled_per_mm2,
            f'wfa_int_density_hollow_per_mm2_{tag}': wfa_int_density_hollow_per_mm2,

            # 2. Densidad Integrada WFA por µm² de Malla PNN (Ponderada por Superficie de Red)
            f'wfa_int_per_mesh_um2_{tag}': wfa_int_per_mesh_um2,
            f'wfa_int_filled_per_mesh_um2_{tag}': wfa_int_filled_per_mesh_um2,
            f'wfa_int_hollow_per_mesh_um2_{tag}': wfa_int_hollow_per_mesh_um2,

            # 3. Suma Total de Densidad Integrada WFA en PNNs dentro de ROI
            f'total_wfa_int_density_{tag}': total_wfa_sum,
            f'total_wfa_int_density_filled_{tag}': total_wfa_sum_filled,
            f'total_wfa_int_density_hollow_{tag}': total_wfa_sum_hollow,

            # 4. Densidad Integrada Promedio por Red PNN (Suma Celular)
            f'mean_wfa_int_density_{tag}': wfa_sum,
            f'mean_wfa_int_density_filled_{tag}': wfa_sum_filled,
            f'mean_wfa_int_density_hollow_{tag}': wfa_sum_hollow,

            # Coeficientes Comparativos (PV+/PNN+ vs PV-/PNN+)
            f'ratio_wfa_int_density_per_mm2_{tag}': ratio_int_density_per_mm2,
            f'ratio_wfa_int_per_mesh_um2_{tag}': ratio_int_per_mesh_um2,
            f'ratio_wfa_intensity_filled_to_hollow_{tag}': ratio_intensity,
            f'ratio_wfa_int_density_filled_to_hollow_{tag}': ratio_int_density,
            f'ratio_wfa_energy_filled_to_hollow_{tag}': ratio_energy,
            
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
            "wfa_pericellular_intensity": "Intensidad Media de la Red PNN (Anillo WFA) — por Célula",
            "wfa_sum_intensity": "Densidad Integrada Fluorescencia WFA (Suma) — por Célula",
            "wfa_int_per_mesh_um2": "Densidad Integrada WFA por µm² de Malla PNN (Ponderada por Área Celular) — por Célula"
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
            "pv_coloc_density_roi": f"Densidad PV+/PNN+ Envueltas (/mm²) (ROI) {suffix}",
            "pv_coloc_density_global": f"Densidad PV+/PNN+ Envueltas (/mm²) (Global) {suffix}",
            "pv_coloc_count_roi": f"N° PV+/PNN+ Envueltas (ROI) {suffix}",
            "pv_coloc_count_global": f"N° PV+/PNN+ Envueltas (Global) {suffix}",
            "pv_isolated_density_roi": f"Densidad PV+/PNN- Libres (/mm²) (ROI) {suffix}",
            "pv_isolated_density_global": f"Densidad PV+/PNN- Libres (/mm²) (Global) {suffix}",
            "pv_isolated_count_roi": f"N° PV+/PNN- Libres (ROI) {suffix}",
            "pv_isolated_count_global": f"N° PV+/PNN- Libres (Global) {suffix}",
            "pct_pv_wrapped_roi": f"% de PV+ Envueltas en PNN (ROI) {suffix}",
            "pct_pv_wrapped_global": f"% de PV+ Envueltas en PNN (Global) {suffix}",
            "pct_pv_isolated_roi": f"% de PV+ Libres de PNN (ROI) {suffix}",
            "pct_pv_isolated_global": f"% de PV+ Libres de PNN (Global) {suffix}",
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
            "pnn_area_fraction_pct_roi": f"Cobertura PNNs en ROI (% Área Cortical) (ROI) {suffix}",
            "pnn_area_fraction_pct_global": f"Cobertura PNNs Global (% Área Total) (Global) {suffix}",
            "pnn_filled_area_fraction_pct_roi": f"Cobertura PNNs PV+/PNN+ (% Área Cortical) (ROI) {suffix}",
            "pnn_filled_area_fraction_pct_global": f"Cobertura PNNs PV+/PNN+ (% Área Total) (Global) {suffix}",
            "pnn_hollow_area_fraction_pct_roi": f"Cobertura PNNs PV-/PNN+ (% Área Cortical) (ROI) {suffix}",
            "pnn_hollow_area_fraction_pct_global": f"Cobertura PNNs PV-/PNN+ (% Área Total) (Global) {suffix}",
            "mean_pnn_area_roi": f"Área Promedio Malla PNN Totales (µm²) (ROI) {suffix}",
            "mean_pnn_area_global": f"Área Promedio Malla PNN Totales (µm²) (Global) {suffix}",
            "mean_pnn_area_filled_roi": f"Área Promedio Malla PNN en PV+/PNN+ (µm²) (ROI) {suffix}",
            "mean_pnn_area_filled_global": f"Área Promedio Malla PNN en PV+/PNN+ (µm²) (Global) {suffix}",
            "mean_pnn_area_hollow_roi": f"Área Promedio Malla PNN en PV-/PNN+ (µm²) (ROI) {suffix}",
            "mean_pnn_area_hollow_global": f"Área Promedio Malla PNN en PV-/PNN+ (µm²) (Global) {suffix}",
            "mean_pnn_diameter_roi": f"Diámetro Promedio Malla PNN (µm) (ROI) {suffix}",
            "mean_pnn_diameter_global": f"Diámetro Promedio Malla PNN (µm) (Global) {suffix}"
        }

        intensity_vars = {
            # ─── 1. DENSIDAD INTEGRADA DE REDES PNN PONDERADA POR ÁREA DE TEJIDO (por mm² de ROI) ───
            "wfa_int_density_per_mm2_roi": f"Densidad Integrada de Redes PNN por mm² de ROI [PONDERADA Tejido] {suffix}",
            "wfa_int_density_per_mm2_global": f"Densidad Integrada de Redes PNN por mm² Global [PONDERADA Tejido] {suffix}",
            "wfa_int_density_filled_per_mm2_roi": f"Densidad Integrada de PNNs PV+/PNN+ por mm² de ROI [PONDERADA] {suffix}",
            "wfa_int_density_filled_per_mm2_global": f"Densidad Integrada de PNNs PV+/PNN+ por mm² Global [PONDERADA] {suffix}",
            "wfa_int_density_hollow_per_mm2_roi": f"Densidad Integrada de PNNs PV-/PNN+ por mm² de ROI [PONDERADA] {suffix}",
            "wfa_int_density_hollow_per_mm2_global": f"Densidad Integrada de PNNs PV-/PNN+ por mm² Global [PONDERADA] {suffix}",

            # ─── 2. DENSIDAD INTEGRADA DE REDES PNN PONDERADA POR SUPERFICIE DE MALLA (por µm² de Red) ───
            "wfa_int_per_mesh_um2_roi": f"Densidad Integrada de PNNs por µm² de Malla [PONDERADA Superficie Red] (ROI) {suffix}",
            "wfa_int_per_mesh_um2_global": f"Densidad Integrada de PNNs por µm² de Malla [PONDERADA Superficie Red] (Global) {suffix}",
            "wfa_int_filled_per_mesh_um2_roi": f"Densidad Integrada en PNNs PV+/PNN+ por µm² de Malla [PONDERADA Red] (ROI) {suffix}",
            "wfa_int_filled_per_mesh_um2_global": f"Densidad Integrada en PNNs PV+/PNN+ por µm² de Malla [PONDERADA Red] (Global) {suffix}",
            "wfa_int_hollow_per_mesh_um2_roi": f"Densidad Integrada en PNNs PV-/PNN+ por µm² de Malla [PONDERADA Red] (ROI) {suffix}",
            "wfa_int_hollow_per_mesh_um2_global": f"Densidad Integrada en PNNs PV-/PNN+ por µm² de Malla [PONDERADA Red] (Global) {suffix}",

            # ─── 3. DENSIDAD INTEGRADA DE REDES PNN NO PONDERADA (Suma Neta en Redes en ROI / Imagen) ───
            "total_wfa_int_density_roi": f"Suma Total de Fluorescencia en Redes PNN [NO PONDERADA - Suma Neta en PNNs en ROI] {suffix}",
            "total_wfa_int_density_global": f"Suma Total de Fluorescencia en Redes PNN [NO PONDERADA - Suma Neta en PNNs Global] {suffix}",
            "total_wfa_int_density_filled_roi": f"Suma de Fluorescencia en PNNs PV+/PNN+ [NO PONDERADA - Suma Neta en PNNs] {suffix}",
            "total_wfa_int_density_hollow_roi": f"Suma de Fluorescencia en PNNs PV-/PNN+ [NO PONDERADA - Suma Neta en PNNs] {suffix}",

            # ─── 4. DENSIDAD INTEGRADA DE REDES PNN NO PONDERADA (Promedio Individual por Red PNN) ───
            "mean_wfa_int_density_roi": f"Densidad Integrada Promedio por Red PNN [NO PONDERADA - Fluorescencia Media por Red] (ROI) {suffix}",
            "mean_wfa_int_density_global": f"Densidad Integrada Promedio por Red PNN [NO PONDERADA - Fluorescencia Media por Red] (Global) {suffix}",
            "mean_wfa_int_density_filled_roi": f"Densidad Integrada Promedio en PNNs PV+/PNN+ [NO PONDERADA] (ROI) {suffix}",
            "mean_wfa_int_density_filled_global": f"Densidad Integrada Promedio en PNNs PV+/PNN+ [NO PONDERADA] (Global) {suffix}",
            "mean_wfa_int_density_hollow_roi": f"Densidad Integrada Promedio en PNNs PV-/PNN+ [NO PONDERADA] (ROI) {suffix}",
            "mean_wfa_int_density_hollow_global": f"Densidad Integrada Promedio en PNNs PV-/PNN+ [NO PONDERADA] (Global) {suffix}",

            # ─── 5. POTENCIA WFA EN PNNS (Lupori Energy) Y POTENCIA PONDERADA POR COBERTURA ───
            "pnn_energy_roi": f"Potencia WFA en PNNs Totales (Densidad x Intensidad) (ROI) {suffix}",
            "pnn_energy_global": f"Potencia WFA en PNNs Totales (Densidad x Intensidad) (Global) {suffix}",
            "pnn_filled_energy_roi": f"Potencia WFA en PNNs PV+/PNN+ (ROI) {suffix}",
            "pnn_filled_energy_global": f"Potencia WFA en PNNs PV+/PNN+ (Global) {suffix}",
            "pnn_hollow_energy_roi": f"Potencia WFA en PNNs PV-/PNN+ (ROI) {suffix}",
            "pnn_hollow_energy_global": f"Potencia WFA en PNNs PV-/PNN+ (Global) {suffix}",
            "pnn_energy_area_weighted_roi": f"Potencia WFA Ponderada por Cobertura de Área (ROI) {suffix}",
            "pnn_energy_area_weighted_global": f"Potencia WFA Ponderada por Cobertura de Área (Global) {suffix}",
            
            # ─── 6. INTENSIDAD MEDIA DE FLUORESCENCIA DE RED (Anillo WFA - NO Ponderada por Área) ───
            "mean_wfa_intensity_roi": f"Intensidad Media de Red en PNNs Totales [NO PONDERADA - Brillo Medio] (ROI) {suffix}",
            "mean_wfa_intensity_global": f"Intensidad Media de Red en PNNs Totales [NO PONDERADA - Brillo Medio] (Global) {suffix}",
            "mean_wfa_intensity_filled_roi": f"Intensidad Media de Red en PV+/PNN+ [NO PONDERADA - Brillo Medio] (ROI) {suffix}",
            "mean_wfa_intensity_filled_global": f"Intensidad Media de Red en PV+/PNN+ [NO PONDERADA - Brillo Medio] (Global) {suffix}",
            "mean_wfa_intensity_hollow_roi": f"Intensidad Media de Red en PV-/PNN+ [NO PONDERADA - Brillo Medio] (ROI) {suffix}",
            "mean_wfa_intensity_hollow_global": f"Intensidad Media de Red en PV-/PNN+ [NO PONDERADA - Brillo Medio] (Global) {suffix}",
            
            # ─── 7. RATIOS COMPARATIVOS (PV+/PNN+ vs PV-/PNN+) ───
            "ratio_wfa_int_density_per_mm2_roi": f"Ratio Densidad Integrada de PNNs por mm² [PONDERADA ROI] (PV+/PNN+ vs PV-/PNN+) {suffix}",
            "ratio_wfa_int_density_per_mm2_global": f"Ratio Densidad Integrada de PNNs por mm² [PONDERADA Global] (PV+/PNN+ vs PV-/PNN+) {suffix}",
            "ratio_wfa_int_per_mesh_um2_roi": f"Ratio Densidad Integrada de PNNs por µm² de Malla [PONDERADA Malla] (PV+/PNN+ vs PV-/PNN+) {suffix}",
            "ratio_wfa_int_per_mesh_um2_global": f"Ratio Densidad Integrada de PNNs por µm² de Malla [PONDERADA Malla] (PV+/PNN+ vs PV-/PNN+) {suffix}",
            "ratio_wfa_int_density_filled_to_hollow_roi": f"Ratio Densidad Integrada Suma en Redes [NO PONDERADA] (PV+/PNN+ vs PV-/PNN+) {suffix}",
            "ratio_wfa_intensity_filled_to_hollow_roi": f"Ratio Intensidad de Red [NO PONDERADA] (PV+/PNN+ vs PV-/PNN+) (ROI) {suffix}",
            "ratio_wfa_intensity_filled_to_hollow_global": f"Ratio Intensidad de Red [NO PONDERADA] (PV+/PNN+ vs PV-/PNN+) (Global) {suffix}",
            "ratio_wfa_energy_filled_to_hollow_roi": f"Ratio Potencia WFA (PV+/PNN+ vs PV-/PNN+) (ROI) {suffix}",
            "ratio_wfa_energy_filled_to_hollow_global": f"Ratio Potencia WFA (PV+/PNN+ vs PV-/PNN+) (Global) {suffix}"
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

tab_pv, tab_pnn, tab_wfa_int, tab_wfa_roi_none_acf, tab_diffuse, tab_nones, tab_export = st.tabs([
    "🧪 Interneuronas PV+ (Subpoblaciones)",
    "🧠 Redes Perineuronales PNN (Huecas / Rellenas)",
    "⚡ Potencia e Intensidades WFA",
    "✨ WFA Integrada PNNs Totales en ROI (NONE vs ACF)",
    "🌫️ Espacio Difuso WFA (Dentro / Fuera de ROI)",
    "🔬 Control Basal (NONE): Hemisferios & Sexo (t-Student)",
    "📦 Exportación Integral (Formato Prism & Parámetros)"
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
        with st.expander("📖 ¿Cómo se miden las variables de Interneuronas PV+?", expanded=False):
            st.markdown(r"""
            * **Densidad de Interneuronas PV+ ($\text{células/mm}^2$):** Conteo total de somas $\text{PV}^+$ segmentados por *Cellpose*, dividido entre el área estándar del ROI ($0.3480\,\text{mm}^2$) o el área total de la imagen ($\approx 3.34\,\text{mm}^2$).
            * **Subpoblación $\text{PV}^+/\text{PNN}^+$ (Envueltas / Colocalizadas):** Neuronas $\text{PV}^+$ cuyo soma se encuentra físicamente envuelto por una red perineuronal de WFA.
            * **Subpoblación $\text{PV}^+/\text{PNN}^-$ (Libres / Desnudas):** Neuronas $\text{PV}^+$ que no poseen red perineuronal condensada a su alrededor.
            * **$\% \text{ de PV}^+ \text{ con Red PNN (Coexpresión)}$:** $\frac{\text{Conteo } \text{PV}^+/\text{PNN}^+}{\text{Conteo Total } \text{PV}^+} \times 100$, indicador clave del nivel de maduración y cierre del período crítico plástico.
            * **Área y Diámetro del Soma ($\mu\text{m}^2$ y $\mu\text{m}$):** Morfometría del cuerpo neuronal calculada a partir de los píxeles segmentados por Cellpose multiplicados por la calibración espacial ($0.8913\,\mu\text{m/px}$).
            """)

        selected_pv_var = st.selectbox("Variable PV+:", list(PV_VARS.keys()), format_func=lambda x: PV_VARS[x], key="pv_var")

        is_cell_var = selected_pv_var in ["pv_area_um2", "pv_diameter_um"]
        if level_type == "Por Célula (distribuciones individuales)":
            df_pv_base = active_cells_df[active_cells_df['is_pv_plus'] == True].copy()
            if not is_cell_var:
                for k in PV_VARS.keys():
                    if k not in df_pv_base.columns and k in df_img.columns:
                        mapping = df_img.set_index('image_name')[k].to_dict()
                        df_pv_base[k] = df_pv_base['image_name'].map(mapping)
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
        with st.expander("📖 ¿Cómo se miden las variables de Redes Perineuronales PNN?", expanded=False):
            st.markdown(r"""
            * **Densidad de PNNs ($\text{redes/mm}^2$):** Número de redes perineuronales detectadas mediante el filtro de IA/DoG en el canal de WFA, normalizado por área en $\text{mm}^2$.
            * **PNNs Rellenas ($\text{PV}^+/\text{PNN}^+$):** Redes perineuronales que encierran un soma positivo para parvalbúmina en su interior.
            * **PNNs Huecas ($\text{PV}^-/\text{PNN}^+$):** Redes perineuronales que rodean somas $\text{PV}^-$ (p. ej., neuronas piramidales u otros subtipos GABAérgicos).
            * **$\% \text{ de PNNs Huecas}$:** $\frac{\text{Conteo } \text{PV}^-/\text{PNN}^+}{\text{Conteo Total PNNs}} \times 100$.
            * **Área y Diámetro de Malla PNN ($\mu\text{m}^2$ y $\mu\text{m}$):** Superficie biológica real ocupada por la condensación de WFA, determinada por umbralización adaptativa local (Otsu) sobre el fondo circundante.
            """)

        selected_pnn_var = st.selectbox("Variable PNN:", list(PNN_VARS.keys()), format_func=lambda x: PNN_VARS[x], key="pnn_var")

        if level_type == "Por Célula (distribuciones individuales)":
            if "filled" in selected_pnn_var:
                df_pnn_base = active_cells_df[active_cells_df['cell_type'] == 'PV+/PNN+'].copy()
            elif "hollow" in selected_pnn_var:
                df_pnn_base = active_cells_df[active_cells_df['cell_type'] == 'PV-/PNN+'].copy()
            else:
                df_pnn_base = active_cells_df[active_cells_df['is_pnn_plus'] == True].copy()

            for k in PNN_VARS.keys():
                if k not in df_pnn_base.columns and k in df_img.columns:
                    mapping = df_img.set_index('image_name')[k].to_dict()
                    df_pnn_base[k] = df_pnn_base['image_name'].map(mapping)
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_pnn_base = df_subj.copy()
        else: # Por Preparado/Corte (imagen)
            df_pnn_base = df_img.copy()

        run_stats_layout(df_pnn_base, PNN_VARS, selected_pnn_var, "Redes Perineuronales PNN")

# 3. TAB POTENCIA E INTENSIDADES WFA
with tab_wfa_int:
    if df_img.empty:
        st.info("No hay datos de intensidades WFA disponibles.")
    else:
        st.header("⚡ Potencia e Intensidad de Redes PNN (WFA)")
        with st.expander("📖 ¿Cómo se interpretan las Métricas Ponderadas vs NO Ponderadas por Área?", expanded=False):
            st.markdown(r"""
            ### 📐 Diferenciación Metodológica: Ponderadas vs NO Ponderadas por Área

            #### 🟢 1. Métricas PONDERADAS por el Área (Normalizadas):
            * **Densidad Integrada WFA por $\text{mm}^2$ de ROI (Ponderada por Área de Tejido):**
              $$\text{Densidad Integrada por mm}^2 = \frac{\sum_{\text{PNNs}} \text{wfa\_sum\_intensity}}{\text{Área}_{\text{ROI}}\ (\text{mm}^2)}$$
              *Propósito:* Mide la **carga biofísica total de matriz perineuronal depositada por unidad de superficie cortical**. Neutraliza las diferencias en el área del ROI entre preparados y permite comparar directamente la cantidad neta de matriz por $\text{mm}^2$ de parénquima.
            * **Densidad Integrada WFA por $\mu\text{m}^2$ de Malla PNN (Ponderada por Área de Red):**
              $$\text{Densidad por }\mu\text{m}^2\text{ de Malla} = \frac{\sum_{\text{PNNs}} \text{wfa\_sum\_intensity}}{\sum_{\text{PNNs}} \text{Área Malla PNN}\ (\mu\text{m}^2)}$$
              *Propósito:* Mide la **densidad óptica / concentración intrínseca de condroitínsulfato** por cada micra cuadrada de malla física real, independientemente de si hay muchas o pocas redes.
            * **Cobertura de Área de PNNs en el ROI (%):**
              $$\text{Cobertura PNN (\%)} = \frac{\sum \text{Área Malla PNN}\ (\mu\text{m}^2)}{\text{Área}_{\text{ROI}}\ (\mu\text{m}^2)} \times 100$$
              *Propósito:* Fracción física del parénquima cortical envuelto en redes perineuronales.
            * **Potencia Ponderada por Cobertura:** $\text{Cobertura (\%)} \times \text{Intensidad Media de Red}$.

            ---

            #### 🔴 2. Métricas NO PONDERADAS por el Área (Valores Crudos / Promedios Celulares):
            * **Suma Total de Fluorescencia WFA en ROI (Carga Bruta NO Ponderada):**
              $$\text{Suma Total WFA} = \sum_{\text{PNNs en ROI}} \text{wfa\_sum\_intensity}$$
              *Propósito:* Cuantifica el volumen acumulado de emisión fluorescente dentro del campo sin normalizar por el área del polígono.
            * **Densidad Integrada Promedio por Red PNN (Suma Media Celular NO Ponderada):**
              $$\text{Densidad Promedio por Red} = \frac{1}{N_{\text{PNN}}} \sum_{\text{PNNs}} \text{wfa\_sum\_intensity}$$
              *Propósito:* Intensidad integrada promedio que porta una célula individual promedio.
            * **Intensidad Media de Red (Brillo Pericelular NO Ponderado):** Promedio de los niveles de gris de los píxeles perasomáticos de WFA.
            """)

        selected_wfa_var = st.selectbox("Variable de Intensidad / Potencia WFA:", list(INTENSITY_VARS.keys()), format_func=lambda x: INTENSITY_VARS[x], key="wfa_int_var")

        if level_type == "Por Célula (distribuciones individuales)":
            if "filled" in selected_wfa_var:
                df_wfa_base = active_cells_df[active_cells_df['cell_type'] == 'PV+/PNN+'].copy()
            elif "hollow" in selected_wfa_var:
                df_wfa_base = active_cells_df[active_cells_df['cell_type'] == 'PV-/PNN+'].copy()
            else:
                df_wfa_base = active_cells_df[active_cells_df['is_pnn_plus'] == True].copy()
            
            for k in INTENSITY_VARS.keys():
                if k not in df_wfa_base.columns and k in df_img.columns:
                    mapping = df_img.set_index('image_name')[k].to_dict()
                    df_wfa_base[k] = df_wfa_base['image_name'].map(mapping)
        elif level_type == "Por Sujeto (animal, promediando cortes)":
            df_wfa_base = df_subj.copy()
        else:
            df_wfa_base = df_img.copy()

        run_stats_layout(df_wfa_base, INTENSITY_VARS, selected_wfa_var, "Potencia e Intensidades WFA")

        # ─── RESUMEN COMPARATIVO DIRECTO: DENSIDAD INTEGRADA WFA EN PNNS (NONE vs ACF) ───
        st.divider()
        st.subheader("🔬 Comparación Directa de Densidad Integrada WFA: Control (NONE) vs Lesión (ACF)")
        st.markdown("""
        Tabla comparativa que presenta en paralelo las **métricas ponderadas por área** ($\text{mm}^2$ de ROI y $\mu\text{m}^2$ de malla) y las **métricas NO ponderadas** (suma bruta y promedio celular), desglosadas por condición temporal (**NONE**, **ACF 3 DÍAS**, **ACF 14 DÍAS**), sexo y hemisferio:
        """)

        df_wfa_eval = df_img.copy() if level_type == "Por Preparado/Corte (imagen)" else df_subj.copy()
        
        all_grps_w = df_wfa_eval['group'].dropna().unique().tolist()
        grp_meta_w = {}
        for g in all_grps_w:
            sex, cond, order = parse_group(g)
            if sex is not None:
                grp_meta_w[g] = {'sex': sex, 'condition': cond, 'order': order}

        df_wfa_eval['sex'] = df_wfa_eval['group'].map(lambda g: grp_meta_w.get(g, {}).get('sex'))
        df_wfa_eval['condition'] = df_wfa_eval['group'].map(lambda g: grp_meta_w.get(g, {}).get('condition'))
        df_wfa_eval = df_wfa_eval[df_wfa_eval['sex'].notna() & df_wfa_eval['condition'].notna() & df_wfa_eval['section'].isin(['IPSI', 'CONTRA'])].copy()

        if not df_wfa_eval.empty:
            wfa_summary_rows = []
            
            # Baseline reference NONE means
            none_ref = {}
            for (sec_k, sex_k), sub_none in df_wfa_eval[df_wfa_eval['condition'] == 'NONE'].groupby(['section', 'sex']):
                none_ref[(sec_k, sex_k)] = {
                    'wfa_mm2': sub_none['wfa_int_density_per_mm2_roi'].mean() if 'wfa_int_density_per_mm2_roi' in sub_none.columns else np.nan,
                    'wfa_tot_raw': sub_none['total_wfa_int_density_roi'].mean() if 'total_wfa_int_density_roi' in sub_none.columns else np.nan,
                    'wfa_mesh': sub_none['wfa_int_per_mesh_um2_roi'].mean() if 'wfa_int_per_mesh_um2_roi' in sub_none.columns else np.nan,
                    'pnn_dens': sub_none['pnn_density_roi'].mean() if 'pnn_density_roi' in sub_none.columns else np.nan
                }

            # Sort conditions
            df_wfa_eval['cond_order'] = df_wfa_eval['condition'].map(lambda c: 0 if c == 'NONE' else (1 if '3' in str(c) else (2 if '14' in str(c) else 3)))
            df_wfa_eval = df_wfa_eval.sort_values(by=['cond_order', 'sex', 'section'])

            for (cond_k, sex_k, sec_k), g_df in df_wfa_eval.groupby(['condition', 'sex', 'section'], sort=False):
                n_count = len(g_df)
                
                # Ponderadas
                w_mm2_m = g_df['wfa_int_density_per_mm2_roi'].mean() if 'wfa_int_density_per_mm2_roi' in g_df.columns else 0.0
                w_mm2_sem = (g_df['wfa_int_density_per_mm2_roi'].std() / np.sqrt(n_count)) if n_count > 1 and 'wfa_int_density_per_mm2_roi' in g_df.columns else 0.0
                
                w_mesh_m = g_df['wfa_int_per_mesh_um2_roi'].mean() if 'wfa_int_per_mesh_um2_roi' in g_df.columns else 0.0
                w_mesh_sem = (g_df['wfa_int_per_mesh_um2_roi'].std() / np.sqrt(n_count)) if n_count > 1 and 'wfa_int_per_mesh_um2_roi' in g_df.columns else 0.0

                cov_m = g_df['pnn_area_fraction_pct_roi'].mean() if 'pnn_area_fraction_pct_roi' in g_df.columns else 0.0
                cov_sem = (g_df['pnn_area_fraction_pct_roi'].std() / np.sqrt(n_count)) if n_count > 1 and 'pnn_area_fraction_pct_roi' in g_df.columns else 0.0

                # NO Ponderadas
                w_tot_raw_m = g_df['total_wfa_int_density_roi'].mean() if 'total_wfa_int_density_roi' in g_df.columns else 0.0
                w_tot_raw_sem = (g_df['total_wfa_int_density_roi'].std() / np.sqrt(n_count)) if n_count > 1 and 'total_wfa_int_density_roi' in g_df.columns else 0.0

                w_mean_m = g_df['mean_wfa_int_density_roi'].mean() if 'mean_wfa_int_density_roi' in g_df.columns else 0.0
                w_mean_sem = (g_df['mean_wfa_int_density_roi'].std() / np.sqrt(n_count)) if n_count > 1 and 'mean_wfa_int_density_roi' in g_df.columns else 0.0

                pnn_dens_m = g_df['pnn_density_roi'].mean() if 'pnn_density_roi' in g_df.columns else 0.0

                # Deltas vs NONE
                base_mm2 = none_ref.get((sec_k, sex_k), {}).get('wfa_mm2', np.nan)
                base_raw = none_ref.get((sec_k, sex_k), {}).get('wfa_tot_raw', np.nan)
                
                if pd.notna(base_mm2) and base_mm2 > 0 and cond_k != 'NONE':
                    pct_delta_mm2 = ((w_mm2_m - base_mm2) / base_mm2) * 100.0
                    delta_mm2_str = f"{pct_delta_mm2:+.2f}%"
                else:
                    delta_mm2_str = "Control (0.0%)" if cond_k == "NONE" else "N/A"

                if pd.notna(base_raw) and base_raw > 0 and cond_k != 'NONE':
                    pct_delta_raw = ((w_tot_raw_m - base_raw) / base_raw) * 100.0
                    delta_raw_str = f"{pct_delta_raw:+.2f}%"
                else:
                    delta_raw_str = "Control (0.0%)" if cond_k == "NONE" else "N/A"

                wfa_summary_rows.append({
                    'Condición': cond_k,
                    'Sexo': sex_k,
                    'Hemisferio': sec_k,
                    'N Réplicas': n_count,
                    'Densidad PNNs (/mm²)': f"{pnn_dens_m:.1f}",
                    'WFA / mm² ROI [Ponderada Tejido]': f"{w_mm2_m:,.0f} ± {w_mm2_sem:,.0f}",
                    'Δ % vs NONE [Ponderada mm²]': delta_mm2_str,
                    'WFA / µm² Malla [Ponderada Red]': f"{w_mesh_m:.2f} ± {w_mesh_sem:.2f}",
                    'Cobertura PNN ROI (%)': f"{cov_m:.2f}% ± {cov_sem:.2f}%",
                    'Suma Total WFA en ROI [NO Ponderada]': f"{w_tot_raw_m:,.0f} ± {w_tot_raw_sem:,.0f}",
                    'Δ % vs NONE [Suma Bruta NO Ponderada]': delta_raw_str,
                    'WFA Medio / Red [NO Ponderada]': f"{w_mean_m:,.0f} ± {w_mean_sem:,.0f}"
                })

            df_wfa_none_acf_summary = pd.DataFrame(wfa_summary_rows)
            st.dataframe(df_wfa_none_acf_summary, use_container_width=True)

            st.download_button(
                "📥 Descargar Tabla Comparativa WFA (NONE vs ACF) (.csv)",
                data=df_wfa_none_acf_summary.to_csv(index=False).encode('utf-8'),
                file_name="comparacion_densidad_integrada_wfa_pnn_none_vs_acf.csv",
                mime="text/csv",
                key="dl_wfa_none_acf_csv"
            )

# 4. TAB WFA INTEGRADA EN PNNS TOTALES DENTRO DE ROI (NONE vs ACF: PONDERADA & NO PONDERADA)
with tab_wfa_roi_none_acf:
    if df_img.empty:
        st.info("No hay datos de imágenes o intensidades WFA disponibles.")
    else:
        st.header("✨ Densidad Integrada de WFA en Redes PNN: Controles (NONE) vs Lesión (ACF)")
        st.markdown("""
        <div class="stats-box">
        <b>Sección Especializada</b> para evaluar la fluorescencia y carga total de matriz WFA asociada <b>exclusivamente a las Redes Perineuronales (PNNs)</b> contenidas dentro del <b>ROI cortical estándar</b> (cuantificada en los enrejados perasomáticos confirmados, sin incluir parénquima de fondo libre).
        Permite contrastar animales basales <b>Control (NONE)</b> contra animales con lesión excitotóxica <b>ACF (3 DÍAS y 14 DÍAS)</b>, comparando simultáneamente las versiones <b>PONDERADAS por Área</b> (por mm² de ROI y por µm² de malla) y <b>NO PONDERADAS</b> (suma neta en redes y promedio por PNN), con descarga completa de <b>gráficos</b> y <b>raw data para GraphPad Prism</b>.
        </div>
        """, unsafe_allow_html=True)

        # ─── CONTROLES DE FILTRADO Y NIVEL ───
        c_lvl1, c_lvl2, c_lvl3 = st.columns(3)
        with c_lvl1:
            sel_wfa_level = st.selectbox(
                "Nivel de Análisis:",
                ["Por Sujeto (Promedio Animal)", "Por Preparado / Corte (Imagen)"],
                key="tab_spec_wfa_level"
            )
        with c_lvl2:
            sel_wfa_sex = st.selectbox(
                "Filtrar por Sexo:",
                ["Todos los Sexos (Ambos)", "Solo MACHOS", "Solo HEMBRAS"],
                key="tab_spec_wfa_sex"
            )
        with c_lvl3:
            sel_wfa_sec = st.selectbox(
                "Filtrar por Hemisferio:",
                ["Ambos Hemisferios (IPSI y CONTRA)", "Solo IPSI (Lesionado / Tratado)", "Solo CONTRA (Intacto)"],
                key="tab_spec_wfa_sec"
            )

        df_target_base = df_subj.copy() if sel_wfa_level.startswith("Por Sujeto") else df_img.copy()
        
        # Meta mapping
        all_grps_sp = df_target_base['group'].dropna().unique().tolist()
        grp_meta_sp = {}
        for g in all_grps_sp:
            sex, cond, order = parse_group(g)
            if sex is not None:
                grp_meta_sp[g] = {'sex': sex, 'condition': cond, 'order': order}

        df_target_base['sex'] = df_target_base['group'].map(lambda g: grp_meta_sp.get(g, {}).get('sex'))
        df_target_base['condition'] = df_target_base['group'].map(lambda g: grp_meta_sp.get(g, {}).get('condition'))
        df_target_base = df_target_base[df_target_base['sex'].notna() & df_target_base['condition'].notna() & df_target_base['section'].isin(['IPSI', 'CONTRA'])].copy()

        # Apply sex & section filters for graphs
        df_filtered = df_target_base.copy()
        if sel_wfa_sex == "Solo MACHOS":
            df_filtered = df_filtered[df_filtered['sex'] == 'MACHO']
        elif sel_wfa_sex == "Solo HEMBRAS":
            df_filtered = df_filtered[df_filtered['sex'] == 'HEMBRA']

        if sel_wfa_sec == "Solo IPSI (Lesionado / Tratado)":
            df_filtered = df_filtered[df_filtered['section'] == 'IPSI']
        elif sel_wfa_sec == "Solo CONTRA (Intacto)":
            df_filtered = df_filtered[df_filtered['section'] == 'CONTRA']

        # Ensure conditions ordered
        df_filtered['cond_order'] = df_filtered['condition'].map(lambda c: 0 if c == 'NONE' else (1 if '3' in str(c) else (2 if '14' in str(c) else 3)))
        df_filtered = df_filtered.sort_values(by=['cond_order', 'sex', 'section'])

        # ─── KPI CARDS COMPARATIVAS NONE vs ACF ───
        df_none_kpi = df_target_base[df_target_base['condition'] == 'NONE']
        df_acf3_kpi = df_target_base[df_target_base['condition'] == 'ACF 3 DÍAS']
        df_acf14_kpi = df_target_base[df_target_base['condition'] == 'ACF 14 DÍAS']

        k1, k2, k3, k4 = st.columns(4)
        
        m_none_mm2 = df_none_kpi['wfa_int_density_per_mm2_roi'].mean() if not df_none_kpi.empty else 0.0
        m_acf3_mm2 = df_acf3_kpi['wfa_int_density_per_mm2_roi'].mean() if not df_acf3_kpi.empty else 0.0
        m_acf14_mm2 = df_acf14_kpi['wfa_int_density_per_mm2_roi'].mean() if not df_acf14_kpi.empty else 0.0
        delta_acf3_mm2 = ((m_acf3_mm2 - m_none_mm2)/m_none_mm2)*100 if m_none_mm2 > 0 else 0.0
        delta_acf14_mm2 = ((m_acf14_mm2 - m_none_mm2)/m_none_mm2)*100 if m_none_mm2 > 0 else 0.0

        k1.metric(
            "🟢 PNN WFA / mm² ROI (Ponderada)",
            f"{m_none_mm2:,.0f} (NONE)",
            f"3D: {delta_acf3_mm2:+.1f}% | 14D: {delta_acf14_mm2:+.1f}%",
            delta_color="normal"
        )

        m_none_mesh = df_none_kpi['wfa_int_per_mesh_um2_roi'].mean() if not df_none_kpi.empty else 0.0
        m_acf3_mesh = df_acf3_kpi['wfa_int_per_mesh_um2_roi'].mean() if not df_acf3_kpi.empty else 0.0
        m_acf14_mesh = df_acf14_kpi['wfa_int_per_mesh_um2_roi'].mean() if not df_acf14_kpi.empty else 0.0
        delta_acf3_mesh = ((m_acf3_mesh - m_none_mesh)/m_none_mesh)*100 if m_none_mesh > 0 else 0.0
        delta_acf14_mesh = ((m_acf14_mesh - m_none_mesh)/m_none_mesh)*100 if m_none_mesh > 0 else 0.0

        k2.metric(
            "🔵 PNN WFA / µm² Malla (Ponderada)",
            f"{m_none_mesh:.2f} (NONE)",
            f"3D: {delta_acf3_mesh:+.1f}% | 14D: {delta_acf14_mesh:+.1f}%",
            delta_color="normal"
        )

        m_none_raw = df_none_kpi['total_wfa_int_density_roi'].mean() if not df_none_kpi.empty else 0.0
        m_acf3_raw = df_acf3_kpi['total_wfa_int_density_roi'].mean() if not df_acf3_kpi.empty else 0.0
        m_acf14_raw = df_acf14_kpi['total_wfa_int_density_roi'].mean() if not df_acf14_kpi.empty else 0.0
        delta_acf3_raw = ((m_acf3_raw - m_none_raw)/m_none_raw)*100 if m_none_raw > 0 else 0.0
        delta_acf14_raw = ((m_acf14_raw - m_none_raw)/m_none_raw)*100 if m_none_raw > 0 else 0.0

        k3.metric(
            "🔴 Suma Neta en PNNs (NO Ponderada)",
            f"{m_none_raw:,.0f} (NONE)",
            f"3D: {delta_acf3_raw:+.1f}% | 14D: {delta_acf14_raw:+.1f}%",
            delta_color="normal"
        )

        m_none_cell = df_none_kpi['mean_wfa_int_density_roi'].mean() if not df_none_kpi.empty else 0.0
        m_acf3_cell = df_acf3_kpi['mean_wfa_int_density_roi'].mean() if not df_acf3_kpi.empty else 0.0
        m_acf14_cell = df_acf14_kpi['mean_wfa_int_density_roi'].mean() if not df_acf14_kpi.empty else 0.0
        delta_acf3_cell = ((m_acf3_cell - m_none_cell)/m_none_cell)*100 if m_none_cell > 0 else 0.0
        delta_acf14_cell = ((m_acf14_cell - m_none_cell)/m_none_cell)*100 if m_none_cell > 0 else 0.0

        k4.metric(
            "🟠 WFA Media por Red PNN (NO Ponderada)",
            f"{m_none_cell:,.0f} (NONE)",
            f"3D: {delta_acf3_cell:+.1f}% | 14D: {delta_acf14_cell:+.1f}%",
            delta_color="normal"
        )

        # ─── SECCIÓN 1: VISUALIZACIONES GRÁFICAS COMPARATIVAS PARALELAS ───
        st.divider()
        st.subheader("📊 1. Visualizaciones Gráficas de Densidad Integrada de Redes PNN")

        # Helper function for flexible SEM bar chart
        def make_wfa_bar_fig(df_data, metric_col, title_txt, y_axis_txt, grouping_mode="Desglose Completo (Condición × Sexo × Hemisferio)"):
            if df_data.empty:
                return go.Figure()
            
            if grouping_mode == "Agrupado solo por Condición (Combinar Sexos y Hemisferios)":
                agg_df = df_data.groupby(['condition'], as_index=False).agg(
                    mean_val=(metric_col, 'mean'),
                    std_val=(metric_col, 'std'),
                    n_count=(metric_col, 'count')
                )
                agg_df['sem_val'] = agg_df.apply(lambda r: r['std_val'] / np.sqrt(r['n_count']) if r['n_count'] > 1 else 0.0, axis=1)
                fig = px.bar(
                    agg_df,
                    x='condition',
                    y='mean_val',
                    color='condition',
                    error_y='sem_val',
                    color_discrete_map={'NONE': '#00ff88', 'ACF 3 DÍAS': '#ff3366', 'ACF 14 DÍAS': '#ffbb00'},
                    title=title_txt,
                    template='plotly_dark'
                )
                fig.update_layout(height=480, yaxis_title=y_axis_txt, xaxis_title="Condición Experimental", showlegend=False)
                return fig
                
            elif grouping_mode == "Por Condición y Hemisferio (Combinar Sexos)":
                agg_df = df_data.groupby(['condition', 'section'], as_index=False).agg(
                    mean_val=(metric_col, 'mean'),
                    std_val=(metric_col, 'std'),
                    n_count=(metric_col, 'count')
                )
                agg_df['sem_val'] = agg_df.apply(lambda r: r['std_val'] / np.sqrt(r['n_count']) if r['n_count'] > 1 else 0.0, axis=1)
                fig = px.bar(
                    agg_df,
                    x='condition',
                    y='mean_val',
                    color='section',
                    barmode='group',
                    error_y='sem_val',
                    color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                    title=title_txt,
                    template='plotly_dark'
                )
                fig.update_layout(height=480, yaxis_title=y_axis_txt, xaxis_title="Condición Experimental", legend_title="Hemisferio")
                return fig

            elif grouping_mode == "Por Condición y Sexo (Combinar Hemisferios)":
                agg_df = df_data.groupby(['condition', 'sex'], as_index=False).agg(
                    mean_val=(metric_col, 'mean'),
                    std_val=(metric_col, 'std'),
                    n_count=(metric_col, 'count')
                )
                agg_df['sem_val'] = agg_df.apply(lambda r: r['std_val'] / np.sqrt(r['n_count']) if r['n_count'] > 1 else 0.0, axis=1)
                fig = px.bar(
                    agg_df,
                    x='condition',
                    y='mean_val',
                    color='sex',
                    barmode='group',
                    error_y='sem_val',
                    color_discrete_map={'MACHO': '#00f2fe', 'HEMBRA': '#ff7b00'},
                    title=title_txt,
                    template='plotly_dark'
                )
                fig.update_layout(height=480, yaxis_title=y_axis_txt, xaxis_title="Condición Experimental", legend_title="Sexo")
                return fig

            else:
                # Desglose Completo
                agg_df = df_data.groupby(['condition', 'section', 'sex'], as_index=False).agg(
                    mean_val=(metric_col, 'mean'),
                    std_val=(metric_col, 'std'),
                    n_count=(metric_col, 'count')
                )
                agg_df['sem_val'] = agg_df.apply(lambda r: r['std_val'] / np.sqrt(r['n_count']) if r['n_count'] > 1 else 0.0, axis=1)
                agg_df['group_lbl'] = agg_df['condition'] + " (" + agg_df['sex'] + ")"
                
                fig = px.bar(
                    agg_df,
                    x='group_lbl',
                    y='mean_val',
                    color='section',
                    barmode='group',
                    error_y='sem_val',
                    color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                    title=title_txt,
                    template='plotly_dark'
                )
                fig.update_layout(height=480, yaxis_title=y_axis_txt, xaxis_title="Grupo Experimental", legend_title="Hemisferio")
                return fig

        # Helper function for flexible Boxplot with individual points
        def make_wfa_box_fig(df_data, metric_col, title_txt, y_axis_txt, grouping_mode="Desglose Completo (Condición × Sexo × Hemisferio)"):
            if df_data.empty:
                return go.Figure()
            
            fig = go.Figure()
            if grouping_mode == "Agrupado solo por Condición (Combinar Sexos y Hemisferios)":
                for cond_name, col_cond in [('NONE', '#00ff88'), ('ACF 3 DÍAS', '#ff3366'), ('ACF 14 DÍAS', '#ffbb00')]:
                    df_c = df_data[df_data['condition'] == cond_name]
                    if not df_c.empty:
                        fig.add_trace(go.Box(
                            x=df_c['condition'],
                            y=df_c[metric_col],
                            name=cond_name,
                            marker_color=col_cond,
                            boxmean='sd',
                            boxpoints='all',
                            jitter=0.3,
                            marker=dict(size=7),
                            line=dict(width=2)
                        ))
                fig.update_layout(height=480, title=title_txt, yaxis_title=y_axis_txt, template='plotly_dark', xaxis_title="Condición Experimental", showlegend=False)
                return fig
                
            elif grouping_mode == "Por Condición y Hemisferio (Combinar Sexos)":
                for sec_name, col_sec in [('IPSI', '#00f2fe'), ('CONTRA', '#ff7b00')]:
                    df_s = df_data[df_data['section'] == sec_name]
                    if not df_s.empty:
                        fig.add_trace(go.Box(
                            x=df_s['condition'],
                            y=df_s[metric_col],
                            name=sec_name,
                            marker_color=col_sec,
                            boxmean='sd',
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-0.2 if sec_name == 'IPSI' else 0.2,
                            marker=dict(size=7),
                            line=dict(width=2)
                        ))
                fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_axis_txt, template='plotly_dark', xaxis_title="Condición Experimental")
                return fig

            elif grouping_mode == "Por Condición y Sexo (Combinar Hemisferios)":
                for sx_name, col_sx in [('MACHO', '#00f2fe'), ('HEMBRA', '#ff7b00')]:
                    df_sx = df_data[df_data['sex'] == sx_name]
                    if not df_sx.empty:
                        fig.add_trace(go.Box(
                            x=df_sx['condition'],
                            y=df_sx[metric_col],
                            name=sx_name,
                            marker_color=col_sx,
                            boxmean='sd',
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-0.2 if sx_name == 'MACHO' else 0.2,
                            marker=dict(size=7),
                            line=dict(width=2)
                        ))
                fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_axis_txt, template='plotly_dark', xaxis_title="Condición Experimental")
                return fig

            else:
                # Desglose Completo
                for sec_name, col_sec in [('IPSI', '#00f2fe'), ('CONTRA', '#ff7b00')]:
                    df_s = df_data[df_data['section'] == sec_name]
                    if not df_s.empty:
                        fig.add_trace(go.Box(
                            x=df_s['condition'] + " - " + df_s['sex'],
                            y=df_s[metric_col],
                            name=sec_name,
                            marker_color=col_sec,
                            boxmean='sd',
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-0.2 if sec_name == 'IPSI' else 0.2,
                            marker=dict(size=7),
                            line=dict(width=2)
                        ))
                fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_axis_txt, template='plotly_dark', xaxis_title="Condición - Sexo")
                return fig

        # Helper function to render a 4-chart panel with download buttons for both HTML and CSV
        def render_wfa_4plot_panel(df_plot_source, panel_key_prefix):
            c_grp1, c_grp2 = st.columns([2, 2])
            with c_grp1:
                sel_grouping = st.selectbox(
                    "Modo de Agrupación / Desglose:",
                    [
                        "Desglose Completo (Condición × Sexo × Hemisferio)",
                        "Agrupado solo por Condición (Combinar Sexos y Hemisferios)",
                        "Por Condición y Hemisferio (Combinar Sexos)",
                        "Por Condición y Sexo (Combinar Hemisferios)"
                    ],
                    index=0,
                    key=f"{panel_key_prefix}_grouping"
                )
            with c_grp2:
                st.info(f"📊 Visualizando **{len(df_plot_source)}** observaciones con agrupación: **{sel_grouping}**")

            col_p1, col_p2 = st.columns(2)
            # ─── GRÁFICO 1: PONDERADA POR MM² ROI ───
            with col_p1:
                st.markdown("#### 🟢 A. Densidad Integrada de Redes PNN [PONDERADA por mm² de ROI]")
                fig_mm2 = make_wfa_bar_fig(
                    df_plot_source,
                    'wfa_int_density_per_mm2_roi',
                    "Densidad Integrada de Redes PNN por mm² ROI (Media ± SEM)",
                    "Densidad Integrada de PNNs / mm² ROI",
                    sel_grouping
                )
                st.plotly_chart(fig_mm2, use_container_width=True)
                
                cb1, cb2 = st.columns(2)
                with cb1:
                    st.download_button(
                        "📥 Descargar Gráfico (.html)",
                        data=fig_mm2.to_html(include_plotlyjs='cdn').encode('utf-8'),
                        file_name=f"grafico_wfa_pnn_ponderada_mm2_{panel_key_prefix}.html",
                        mime="text/html",
                        key=f"dl_fig_mm2_{panel_key_prefix}",
                        use_container_width=True
                    )
                with cb2:
                    cols_mm2 = [c for c in ['animal_id', 'image_name', 'condition', 'sex', 'section', 'group', 'wfa_int_density_per_mm2_roi'] if c in df_plot_source.columns]
                    st.download_button(
                        "📄 Descargar Datos (.csv)",
                        data=df_plot_source[cols_mm2].to_csv(index=False).encode('utf-8'),
                        file_name=f"datos_wfa_pnn_ponderada_mm2_{panel_key_prefix}.csv",
                        mime="text/csv",
                        key=f"dl_data_mm2_{panel_key_prefix}",
                        use_container_width=True
                    )

            # ─── GRÁFICO 2: NO PONDERADA (SUMA NETA EN PNNS) ───
            with col_p2:
                st.markdown("#### 🔴 B. Suma Total de Fluorescencia en Redes PNN [NO PONDERADA - Neta en PNNs]")
                fig_raw = make_wfa_bar_fig(
                    df_plot_source,
                    'total_wfa_int_density_roi',
                    "Suma Total de Fluorescencia en Redes PNN en ROI (Media ± SEM)",
                    "Suma de Fluorescencia en PNNs",
                    sel_grouping
                )
                st.plotly_chart(fig_raw, use_container_width=True)
                
                cb3, cb4 = st.columns(2)
                with cb3:
                    st.download_button(
                        "📥 Descargar Gráfico (.html)",
                        data=fig_raw.to_html(include_plotlyjs='cdn').encode('utf-8'),
                        file_name=f"grafico_wfa_pnn_no_ponderada_suma_{panel_key_prefix}.html",
                        mime="text/html",
                        key=f"dl_fig_raw_{panel_key_prefix}",
                        use_container_width=True
                    )
                with cb4:
                    cols_raw = [c for c in ['animal_id', 'image_name', 'condition', 'sex', 'section', 'group', 'total_wfa_int_density_roi'] if c in df_plot_source.columns]
                    st.download_button(
                        "📄 Descargar Datos (.csv)",
                        data=df_plot_source[cols_raw].to_csv(index=False).encode('utf-8'),
                        file_name=f"datos_wfa_pnn_no_ponderada_suma_{panel_key_prefix}.csv",
                        mime="text/csv",
                        key=f"dl_data_raw_{panel_key_prefix}",
                        use_container_width=True
                    )

            col_p3, col_p4 = st.columns(2)
            # ─── GRÁFICO 3: PONDERADA POR ÁREA DE MALLA (µm²) ───
            with col_p3:
                st.markdown("#### 🔵 C. Densidad Integrada de PNNs [PONDERADA por µm² de Malla]")
                fig_mesh = make_wfa_bar_fig(
                    df_plot_source,
                    'wfa_int_per_mesh_um2_roi',
                    "Densidad Integrada por µm² de Malla PNN (Concentración Intrínseca)",
                    "Densidad Integrada de PNNs / µm² Malla",
                    sel_grouping
                )
                st.plotly_chart(fig_mesh, use_container_width=True)
                
                cb5, cb6 = st.columns(2)
                with cb5:
                    st.download_button(
                        "📥 Descargar Gráfico (.html)",
                        data=fig_mesh.to_html(include_plotlyjs='cdn').encode('utf-8'),
                        file_name=f"grafico_wfa_pnn_ponderada_malla_{panel_key_prefix}.html",
                        mime="text/html",
                        key=f"dl_fig_mesh_{panel_key_prefix}",
                        use_container_width=True
                    )
                with cb6:
                    cols_mesh = [c for c in ['animal_id', 'image_name', 'condition', 'sex', 'section', 'group', 'wfa_int_per_mesh_um2_roi'] if c in df_plot_source.columns]
                    st.download_button(
                        "📄 Descargar Datos (.csv)",
                        data=df_plot_source[cols_mesh].to_csv(index=False).encode('utf-8'),
                        file_name=f"datos_wfa_pnn_ponderada_malla_{panel_key_prefix}.csv",
                        mime="text/csv",
                        key=f"dl_data_mesh_{panel_key_prefix}",
                        use_container_width=True
                    )

            # ─── GRÁFICO 4: DISTRIBUCIÓN INDIVIDUAL / BOXPLOT ───
            with col_p4:
                st.markdown("#### 📦 D. Distribución Individual por Réplicas Biológicas (Boxplot)")
                fig_box = make_wfa_box_fig(
                    df_plot_source,
                    'wfa_int_density_per_mm2_roi',
                    "Distribución de Réplicas: WFA en PNNs / mm² ROI",
                    "Densidad Integrada de PNNs / mm² ROI",
                    sel_grouping
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                cb7, cb8 = st.columns(2)
                with cb7:
                    st.download_button(
                        "📥 Descargar Gráfico (.html)",
                        data=fig_box.to_html(include_plotlyjs='cdn').encode('utf-8'),
                        file_name=f"grafico_pnn_boxplot_{panel_key_prefix}.html",
                        mime="text/html",
                        key=f"dl_fig_box_{panel_key_prefix}",
                        use_container_width=True
                    )
                with cb8:
                    cols_box = [c for c in ['animal_id', 'image_name', 'condition', 'sex', 'section', 'group', 'wfa_int_density_per_mm2_roi'] if c in df_plot_source.columns]
                    st.download_button(
                        "📄 Descargar Datos (.csv)",
                        data=df_plot_source[cols_box].to_csv(index=False).encode('utf-8'),
                        file_name=f"datos_boxplot_individual_{panel_key_prefix}.csv",
                        mime="text/csv",
                        key=f"dl_data_box_{panel_key_prefix}",
                        use_container_width=True
                    )

        # ─── SUB-PESTAÑAS DE VISUALIZACIÓN: GLOBAL, SOLO NONES, SOLO ACF ───
        subtab_global, subtab_none_only, subtab_acf_only = st.tabs([
            "🌐 1. Comparación Global (NONE vs ACF)",
            "🔬 2. Solo Controles Basales (NONE)",
            "⚡ 3. Solo Tratamientos con Lesión (ACF 3D vs ACF 14D)"
        ])

        with subtab_global:
            st.markdown("##### 🌐 Panel Global: Todos los Grupos Experimentales (NONE vs ACF 3 DÍAS vs ACF 14 DÍAS)")
            render_wfa_4plot_panel(df_filtered, panel_key_prefix="global_all")

        with subtab_none_only:
            st.markdown("##### 🔬 Panel Específico: Solo Controles Basales (NONE)")
            df_none_only = df_filtered[df_filtered['condition'] == 'NONE'].copy()
            if df_none_only.empty:
                st.warning("No hay datos de controles basales (NONE) con los filtros actuales de sexo y hemisferio.")
            else:
                render_wfa_4plot_panel(df_none_only, panel_key_prefix="none_only")

        with subtab_acf_only:
            st.markdown("##### ⚡ Panel Específico: Solo Tratamientos con Lesión (ACF 3 DÍAS vs ACF 14 DÍAS)")
            df_acf_only = df_filtered[df_filtered['condition'].isin(['ACF 3 DÍAS', 'ACF 14 DÍAS'])].copy()
            if df_acf_only.empty:
                st.warning("No hay datos de tratamientos con lesión ACF con los filtros actuales de sexo y hemisferio.")
            else:
                render_wfa_4plot_panel(df_acf_only, panel_key_prefix="acf_only")

        # ─── SECCIÓN 2: TABLA RESUMEN CONSOLIDADA NONE vs ACF ───
        st.divider()
        st.subheader("📋 2. Tabla Resumen Consolidada de Densidades Integradas de PNNs (NONE vs ACF)")
        st.markdown("Comparativa cuantitativa exclusiva de las **Redes Perineuronales (PNNs)** dentro del ROI cortical con $\\text{Media} \\pm \\text{SEM}$, valores basales y deltas de variación porcentual:")

        # Compute full comparison table
        summary_rows_tab = []
        none_ref_tab = {}
        for (sec_k, sex_k), sub_n in df_target_base[df_target_base['condition'] == 'NONE'].groupby(['section', 'sex']):
            none_ref_tab[(sec_k, sex_k)] = {
                'wfa_mm2': sub_n['wfa_int_density_per_mm2_roi'].mean(),
                'wfa_raw': sub_n['total_wfa_int_density_roi'].mean(),
                'wfa_mesh': sub_n['wfa_int_per_mesh_um2_roi'].mean(),
                'pnn_dens': sub_n['pnn_density_roi'].mean()
            }

        df_target_base['cond_order'] = df_target_base['condition'].map(lambda c: 0 if c == 'NONE' else (1 if '3' in str(c) else (2 if '14' in str(c) else 3)))
        df_target_base_sorted = df_target_base.sort_values(by=['cond_order', 'sex', 'section'])

        for (cond_k, sex_k, sec_k), g_df in df_target_base_sorted.groupby(['condition', 'sex', 'section'], sort=False):
            n_count = len(g_df)
            
            # Ponderadas
            w_mm2_m = g_df['wfa_int_density_per_mm2_roi'].mean()
            w_mm2_sem = g_df['wfa_int_density_per_mm2_roi'].std() / np.sqrt(n_count) if n_count > 1 else 0.0
            
            w_mesh_m = g_df['wfa_int_per_mesh_um2_roi'].mean()
            w_mesh_sem = g_df['wfa_int_per_mesh_um2_roi'].std() / np.sqrt(n_count) if n_count > 1 else 0.0

            cov_m = g_df['pnn_area_fraction_pct_roi'].mean()
            cov_sem = g_df['pnn_area_fraction_pct_roi'].std() / np.sqrt(n_count) if n_count > 1 else 0.0

            # NO Ponderadas
            w_raw_m = g_df['total_wfa_int_density_roi'].mean()
            w_raw_sem = g_df['total_wfa_int_density_roi'].std() / np.sqrt(n_count) if n_count > 1 else 0.0

            w_mean_m = g_df['mean_wfa_int_density_roi'].mean()
            w_mean_sem = g_df['mean_wfa_int_density_roi'].std() / np.sqrt(n_count) if n_count > 1 else 0.0

            pnn_dens_m = g_df['pnn_density_roi'].mean()

            # Deltas
            base_mm2 = none_ref_tab.get((sec_k, sex_k), {}).get('wfa_mm2', np.nan)
            base_raw = none_ref_tab.get((sec_k, sex_k), {}).get('wfa_raw', np.nan)

            delta_mm2_str = f"{((w_mm2_m - base_mm2)/base_mm2)*100:+.2f}%" if pd.notna(base_mm2) and base_mm2 > 0 and cond_k != 'NONE' else ("Control Basal (0.0%)" if cond_k == "NONE" else "N/A")
            delta_raw_str = f"{((w_raw_m - base_raw)/base_raw)*100:+.2f}%" if pd.notna(base_raw) and base_raw > 0 and cond_k != 'NONE' else ("Control Basal (0.0%)" if cond_k == "NONE" else "N/A")

            summary_rows_tab.append({
                'Condición': cond_k,
                'Sexo': sex_k,
                'Hemisferio': sec_k,
                'N Réplicas': n_count,
                'Densidad PNNs (/mm²)': f"{pnn_dens_m:.1f}",
                'PNN WFA / mm² ROI [PONDERADA]': f"{w_mm2_m:,.0f} ± {w_mm2_sem:,.0f}",
                'Δ % vs NONE (Ponderada mm²)': delta_mm2_str,
                'PNN WFA / µm² Malla [PONDERADA]': f"{w_mesh_m:.2f} ± {w_mesh_sem:.2f}",
                'Cobertura PNN ROI (%)': f"{cov_m:.2f}% ± {cov_sem:.2f}%",
                'Suma Neta en PNNs en ROI [NO PONDERADA]': f"{w_raw_m:,.0f} ± {w_raw_sem:,.0f}",
                'Δ % vs NONE (Suma Neta PNNs)': delta_raw_str,
                'PNN WFA Media / Red [NO PONDERADA]': f"{w_mean_m:,.0f} ± {w_mean_sem:,.0f}"
            })

        df_summary_tab_view = pd.DataFrame(summary_rows_tab)
        st.dataframe(df_summary_tab_view, use_container_width=True)

        # ─── SECCIÓN 3: DESCARGA DE RAW DATA Y MATRICES GRAPHPAD PRISM ───
        st.divider()
        st.subheader("📥 3. Centro de Descarga de Raw Data y Formatos GraphPad Prism")
        st.markdown("Descarga los datos crudos y matrices directamente estructuradas para análisis estadístico en GraphPad Prism, R o Excel:")

        # Raw data table of this specific analysis
        raw_cols_export = [
            'animal_id', 'group', 'condition', 'sex', 'section', 'pnn_density_roi',
            'wfa_int_density_per_mm2_roi', 'wfa_int_per_mesh_um2_roi', 'pnn_area_fraction_pct_roi',
            'total_wfa_int_density_roi', 'mean_wfa_int_density_roi', 'mean_wfa_intensity_roi'
        ]
        if 'image_name' in df_target_base.columns:
            raw_cols_export.insert(1, 'image_name')
        
        avail_raw_cols = [c for c in raw_cols_export if c in df_target_base.columns]
        df_raw_wfa_export = df_target_base[avail_raw_cols].copy()

        # Prism Matrices
        df_prism_wfa_mm2 = generate_prism_column_matrix(df_target_base, 'wfa_int_density_per_mm2_roi', level='subject' if sel_wfa_level.startswith("Por Sujeto") else 'image')
        df_prism_wfa_raw = generate_prism_column_matrix(df_target_base, 'total_wfa_int_density_roi', level='subject' if sel_wfa_level.startswith("Por Sujeto") else 'image')
        df_prism_wfa_mesh = generate_prism_column_matrix(df_target_base, 'wfa_int_per_mesh_um2_roi', level='subject' if sel_wfa_level.startswith("Por Sujeto") else 'image')
        df_prism_wfa_2way = generate_prism_grouped_2way_matrix(df_target_base, 'wfa_int_density_per_mm2_roi')

        # Excel multi-sheet buffer with everything
        wfa_excel_buffer = io.BytesIO()
        with pd.ExcelWriter(wfa_excel_buffer, engine='openpyxl') as wr:
            df_summary_tab_view.to_excel(wr, sheet_name="Resumen_Comparativo", index=False)
            df_raw_wfa_export.to_excel(wr, sheet_name="Raw_Data_Completa", index=False)
            df_prism_wfa_mm2.to_excel(wr, sheet_name="Prism_Ponderada_mm2_ROI", index=False)
            df_prism_wfa_raw.to_excel(wr, sheet_name="Prism_NoPonderada_SumaRaw", index=False)
            df_prism_wfa_mesh.to_excel(wr, sheet_name="Prism_Ponderada_Malla_um2", index=False)
            df_prism_wfa_2way.to_excel(wr, sheet_name="Prism_2Way_IPSI_vs_CONTRA", index=False)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                "📗 Descargar Libro Completo (.xlsx) [Raw Data + Prism]",
                data=wfa_excel_buffer.getvalue(),
                file_name=f"wfa_integrada_pnn_none_vs_acf_completo_{sel_wfa_level[:7].strip().lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_wfa_spec_xlsx",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                "📄 Descargar Raw Data Cruda (.csv)",
                data=df_raw_wfa_export.to_csv(index=False).encode('utf-8'),
                file_name=f"raw_data_wfa_integrada_pnn_{sel_wfa_level[:7].strip().lower()}.csv",
                mime="text/csv",
                key="dl_wfa_spec_raw_csv",
                use_container_width=True
            )
        with col_d3:
            st.download_button(
                "📊 Descargar Matriz Prism Ponderada mm² (.csv)",
                data=df_prism_wfa_mm2.to_csv(index=False).encode('utf-8'),
                file_name="prism_column_wfa_ponderada_mm2_roi.csv",
                mime="text/csv",
                key="dl_wfa_spec_prism_mm2_csv",
                use_container_width=True
            )

        col_d4, col_d5, col_d6 = st.columns(3)
        with col_d4:
            st.download_button(
                "📊 Descargar Matriz Prism NO Ponderada Suma (.csv)",
                data=df_prism_wfa_raw.to_csv(index=False).encode('utf-8'),
                file_name="prism_column_wfa_no_ponderada_suma_raw.csv",
                mime="text/csv",
                key="dl_wfa_spec_prism_raw_csv",
                use_container_width=True
            )
        with col_d5:
            st.download_button(
                "📊 Descargar Matriz Prism Ponderada Malla (.csv)",
                data=df_prism_wfa_mesh.to_csv(index=False).encode('utf-8'),
                file_name="prism_column_wfa_ponderada_malla_um2.csv",
                mime="text/csv",
                key="dl_wfa_spec_prism_mesh_csv",
                use_container_width=True
            )
        with col_d6:
            st.download_button(
                "📋 Descargar Tabla Resumen Medias & Deltas (.csv)",
                data=df_summary_tab_view.to_csv(index=False).encode('utf-8'),
                file_name="tabla_resumen_wfa_none_vs_acf_medias_deltas.csv",
                mime="text/csv",
                key="dl_wfa_spec_summary_csv",
                use_container_width=True
            )

        # ─── BOTÓN MAESTRO FINAL: EXPORTACIÓN DE CARPETA ORDENADA EN DISCO CON SUBCARPETAS Y ARCHIVOS ───
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 242, 254, 0.20), rgba(187, 134, 252, 0.20)); border: 1.5px solid #00f2fe; border-radius: 12px; padding: 22px; margin-top: 30px; margin-bottom: 20px; text-align: center;">
            <h3 style="color: #00f2fe; margin-top: 0; margin-bottom: 10px;">📁 Centro Maestro de Exportación: Carpeta Estructurada con Subcarpetas</h3>
            <p style="color: #ffffff; margin-bottom: 14px; font-size: 15px;">
                Genera directamente en tu computadora una <b>carpeta ordenada con todas las subcarpetas temáticas</b> (Global, Solo NONE, Solo ACF, Filtros de Sexo y Hemisferio).<br>
                Cada subcarpeta contiene las <b>imágenes PNG de alta resolución (300 DPI)</b> junto con sus <b>archivos CSV de datos con el MISMO NOMBRE</b> y gráficos HTML, más la base <b>RAW completa</b> en la raíz.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        default_export_dir = os.path.join(os.getcwd(), "exports", "graficas_y_datos_completos")
        
        c_exp_dir, c_exp_btn = st.columns([3, 2])
        with c_exp_dir:
            target_export_path = st.text_input(
                "Ruta de destino de la carpeta en disco:",
                value=default_export_dir,
                key="input_target_export_dir"
            )
        with c_exp_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📁 🚀 GENERAR CARPETA CON SUBCARPETAS EN DISCO", key="btn_generate_disk_folders", use_container_width=True):
                with st.spinner("Generando todas las figuras PNG, CSVs y subcarpetas en disco..."):
                    try:
                        final_dir = export_all_graphics_and_data_to_directory(
                            df_target_base,
                            target_directory=target_export_path,
                            level_name=sel_wfa_level[:7].strip().lower()
                        )
                        st.session_state['last_exported_dir'] = final_dir
                        st.success(f"✅ ¡Carpeta generada exitosamente con toda la estructura de subcarpetas en:\n\n`{final_dir}`")
                    except Exception as e:
                        st.error(f"Error al generar la carpeta en disco: {e}")

        # If a folder was generated, show shortcut to open it in Linux/Windows file manager
        if st.session_state.get('last_exported_dir') and os.path.exists(st.session_state['last_exported_dir']):
            cur_dir = st.session_state['last_exported_dir']
            c_open1, c_open2 = st.columns([2, 3])
            with c_open1:
                if st.button("📂 Abrir Carpeta en el Explorador de Archivos", key="btn_open_file_mgr", use_container_width=True):
                    try:
                        subprocess.Popen(["xdg-open", cur_dir])
                        st.info("Abriendo explorador de archivos...")
                    except Exception:
                        pass
            with c_open2:
                st.caption(f"📍 Ruta física actual: `{cur_dir}`")

            # Interactive Folder Tree Explorer
            with st.expander("🌳 Explorar Árbol de Subcarpetas y Archivos Generados en Disco", expanded=True):
                for root, dirs, files in os.walk(cur_dir):
                    rel_path = os.path.relpath(root, cur_dir)
                    level = rel_path.count(os.sep)
                    indent = "    " * level
                    if rel_path == ".":
                        st.markdown(f"**📁 {os.path.basename(cur_dir)}/** (Raíz)")
                    else:
                        st.markdown(f"{indent}**📁 {os.path.basename(root)}/**")
                    
                    for f_name in sorted(files):
                        icon = "🖼️" if f_name.endswith(".png") else ("📄" if f_name.endswith(".csv") else ("🌐" if f_name.endswith(".html") else ("📗" if f_name.endswith(".xlsx") else "📖")))
                        st.markdown(f"{indent}    {icon} `{f_name}`")

        st.divider()
        st.markdown("##### 📥 Opciones Adicionales de Descarga Directa:")
        
        c_m0, c_m1, c_m2 = st.columns(3)
        with c_m0:
            st.download_button(
                "🌟 📥 Descargar Todo en .ZIP",
                data=build_all_graphics_and_data_zip(df_target_base, level_name=sel_wfa_level[:7].strip().lower()),
                file_name=f"todas_las_graficas_y_datos_completos_{sel_wfa_level[:7].strip().lower()}.zip",
                mime="application/zip",
                key="dl_master_all_graphics_and_data_zip",
                use_container_width=True
            )
        with c_m1:
            st.download_button(
                "📄 📥 Raw Data Completa (.CSV)",
                data=df_target_base.to_csv(index=False).encode('utf-8'),
                file_name=f"todos_los_datos_crudos_completos_{sel_wfa_level[:7].strip().lower()}.csv",
                mime="text/csv",
                key="dl_master_all_raw_csv",
                use_container_width=True
            )
        with c_m2:
            st.download_button(
                "📗 📥 Libro Maestro Prism (.XLSX)",
                data=wfa_excel_buffer.getvalue(),
                file_name=f"libro_maestro_datos_crudos_completo_{sel_wfa_level[:7].strip().lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_master_all_raw_xlsx",
                use_container_width=True
            )

# 5. TAB ESPACIO DIFUSO WFA (DENTRO Y FUERA DEL ROI)
with tab_diffuse:
    if df_img.empty:
        st.info("No hay datos de espacio difuso disponibles.")
    else:
        st.header("🌫️ Espacio Difuso y Neuropilo WFA (Dentro y Fuera del ROI)")
        with st.expander("📖 ¿Cómo se mide el Espacio Difuso y el Ratio de Contraste?", expanded=False):
            st.markdown("""
            * **Fondo Difuso WFA (Neuropilo):** Nivel promedio de fluorescencia en el tejido libre de somas y condensaciones perineuronales ($m_{\text{wfa}} = 0$). Refleja la matriz intersticial basal no estructurada.
              - **Dentro de ROI:** Medido estrictamente dentro de la región estándar de análisis.
              - **Fuera de ROI:** Medido en el tejido cortical restante del corte.
              - **Global:** Medido en toda la superficie de la imagen histológica.
            * **Ratio Señal PNN / Fondo Difuso:** Cociente $\\frac{\\text{Intensidad Media de PNN}}{\\text{Intensidad Fondo Difuso}}$, parámetro clave que cuantifica la definición, contraste y nitidez de las redes sobre la matriz circundante.
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

# 5. TAB CONTROL BASAL (NONE): HEMISFERIOS & SEXO (t-Student)
with tab_nones:
    st.header("🔬 Control Basal (NONE): Simetría Interhemisférica y Dimorfismo Sexual")
    st.markdown("""
    <div class="stats-box">
    El grupo control (<b>NONE</b>) permite establecer la línea de base biológica de la corteza somatosensorial sin tratamiento farmacológico.
    En esta sección se evalúa:<br>
    1. <b>Simetría Interhemisférica:</b> Prueba <b>t de Student pareada</b> por sujeto (IPSI vs CONTRA en cada animal) para comprobar si existen asimetrías basales.<br>
    2. <b>Dimorfismo Sexual:</b> Prueba <b>t de Student independiente / Welch</b> (NONE MACHO vs NONE HEMBRA) para detectar diferencias basales ligadas al sexo.
    </div>
    """, unsafe_allow_html=True)

    # Prepare base data for NONE
    ALL_NONE_VARS = {}
    ALL_NONE_VARS.update(PV_VARS)
    ALL_NONE_VARS.update(PNN_VARS)
    ALL_NONE_VARS.update(INTENSITY_VARS)
    ALL_NONE_VARS.update(DIFFUSE_VARS)

    if level_type == "Por Célula (distribuciones individuales)":
        df_none_src = active_cells_df.copy()
        for k in ALL_NONE_VARS.keys():
            if k not in df_none_src.columns and k in df_img.columns:
                mapping = df_img.set_index('image_name')[k].to_dict()
                df_none_src[k] = df_none_src['image_name'].map(mapping)
    elif level_type == "Por Sujeto (animal, promediando cortes)":
        df_none_src = df_subj.copy()
    else:
        df_none_src = df_img.copy()

    # Parse metadata for NONE
    all_grps_n = df_none_src['group'].dropna().unique().tolist()
    grp_meta_n = {}
    for g in all_grps_n:
        sex, cond, order = parse_group(g)
        if sex is not None:
            grp_meta_n[g] = {'sex': sex, 'condition': cond, 'order': order}

    df_none_src['sex'] = df_none_src['group'].map(lambda g: grp_meta_n.get(g, {}).get('sex'))
    df_none_src['condition'] = df_none_src['group'].map(lambda g: grp_meta_n.get(g, {}).get('condition'))
    df_none_all = df_none_src[df_none_src['condition'] == 'NONE'].copy()

    if df_none_all.empty:
        st.warning("⚠️ No se encontraron datos para la condición control basal 'NONE'. Ejecuta el pipeline sobre el grupo NONE primero.")
    else:
        selected_none_var = st.selectbox(
            "Seleccionar Variable para Análisis de Controles Basales:",
            list(ALL_NONE_VARS.keys()),
            format_func=lambda x: ALL_NONE_VARS[x],
            key="none_var_selector"
        )

        # Ensure selected_none_var is present in df_none_all
        if selected_none_var not in df_none_all.columns:
            if selected_none_var in df_img.columns and 'image_name' in df_none_all.columns:
                mapping = df_img.set_index('image_name')[selected_none_var].to_dict()
                df_none_all[selected_none_var] = df_none_all['image_name'].map(mapping)
            elif selected_none_var in df_subj.columns and 'animal_id' in df_none_all.columns:
                mapping = df_subj.set_index('animal_id')[selected_none_var].to_dict()
                df_none_all[selected_none_var] = df_none_all['animal_id'].map(mapping)

        # Filter cell subset if evaluating single-cell distributions
        df_none_eval = df_none_all.copy()
        if level_type == "Por Célula (distribuciones individuales)":
            if selected_none_var in ["pv_area_um2", "pv_diameter_um"]:
                df_none_eval = df_none_eval[df_none_eval['is_pv_plus'] == True]
            elif selected_none_var in ["pnn_area_um2", "pnn_diameter_um", "score", "wfa_pericellular_intensity", "wfa_sum_intensity", "wfa_int_per_mesh_um2"]:
                df_none_eval = df_none_eval[df_none_eval['is_pnn_plus'] == True]

        var_label_none = ALL_NONE_VARS.get(selected_none_var, selected_none_var)

        # ─── SECCIÓN 1: COMPARACIÓN PAREADA INTERHEMISFÉRICA (IPSI vs CONTRA) ───
        st.divider()
        st.subheader("⚖️ 1. Comparación Pareada Interhemisférica (IPSI vs CONTRA por Animal)")

        if selected_none_var not in df_none_eval.columns:
            st.warning(f"La variable '{selected_none_var}' no está disponible en los datos de control basal.")
        else:
            # Prepare paired dataset by animal_id
            df_sub_ip = df_none_eval[df_none_eval['section'] == 'IPSI']
            df_sub_ct = df_none_eval[df_none_eval['section'] == 'CONTRA']
            
            df_ipsi_n = df_sub_ip[['animal_id', 'sex', selected_none_var]].dropna().rename(columns={selected_none_var: 'v_ipsi'})
            df_contra_n = df_sub_ct[['animal_id', selected_none_var]].dropna().rename(columns={selected_none_var: 'v_contra'})
            
            # If multiple records per animal (e.g. at image or cell level), average them to ensure 1 pair per animal
            df_ipsi_n = df_ipsi_n.groupby(['animal_id', 'sex'], as_index=False)['v_ipsi'].mean()
            df_contra_n = df_contra_n.groupby('animal_id', as_index=False)['v_contra'].mean()

            df_paired_n = pd.merge(df_ipsi_n, df_contra_n, on='animal_id').dropna()
            df_paired_n['diff'] = df_paired_n['v_ipsi'] - df_paired_n['v_contra']
            df_paired_n['pct_diff'] = ((df_paired_n['v_ipsi'] - df_paired_n['v_contra']) / (df_paired_n['v_contra'] + 1e-8)) * 100.0

            n_pairs = len(df_paired_n)

            if n_pairs < 2:
                st.info(f"Se encontraron {n_pairs} animales con ambos hemisferios (IPSI y CONTRA) en condición NONE. Se requieren al menos 2 animales para la prueba pareada.")
            else:
                # Paired Student's t-test
                t_pair, p_pair = ttest_rel(df_paired_n['v_ipsi'], df_paired_n['v_contra'])
                df_deg = n_pairs - 1
                mean_ipsi = float(np.mean(df_paired_n['v_ipsi']))
                std_ipsi = float(np.std(df_paired_n['v_ipsi'], ddof=1))
                mean_contra = float(np.mean(df_paired_n['v_contra']))
                std_contra = float(np.std(df_paired_n['v_contra'], ddof=1))
                
                mean_diff = float(np.mean(df_paired_n['diff']))
                std_diff = float(np.std(df_paired_n['diff'], ddof=1))
                sem_diff = std_diff / np.sqrt(n_pairs) if n_pairs > 0 else 0.0

                # 95% Confidence Interval for mean difference
                from scipy.stats import t as t_dist
                t_crit = t_dist.ppf(0.975, df=df_deg)
                ci_lower = mean_diff - t_crit * sem_diff
                ci_upper = mean_diff + t_crit * sem_diff

                # Cohen's dz (paired effect size)
                cohen_dz = float(mean_diff / (std_diff + 1e-8)) if std_diff > 0 else 0.0

                # Normality of differences (Shapiro-Wilk)
                if n_pairs >= 3:
                    sw_stat, sw_p = shapiro(df_paired_n['diff'])
                else:
                    sw_stat, sw_p = np.nan, np.nan

                # Wilcoxon non-parametric confirmation
                try:
                    wx_stat, wx_p = wilcoxon(df_paired_n['v_ipsi'], df_paired_n['v_contra'])
                except Exception:
                    wx_stat, wx_p = np.nan, np.nan

                # KPI Cards
                c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                c_p1.metric(
                    "🧪 t de Student Pareada",
                    f"t({df_deg}) = {t_pair:.3f}",
                    delta=f"p = {p_pair:.4e} ({sig_stars(p_pair)})",
                    delta_color="normal" if p_pair < 0.05 else "off"
                )
                c_p2.metric(
                    "📐 Diferencia Media (Δ)",
                    f"{mean_diff:.3f}",
                    f"IC 95%: [{ci_lower:.2f}, {ci_upper:.2f}]"
                )
                c_p3.metric(
                    "📏 Cohen's dz (Efecto Pareado)",
                    f"dz = {cohen_dz:.3f}",
                    "Fuerte" if abs(cohen_dz)>=0.8 else ("Moderado" if abs(cohen_dz)>=0.5 else "Pequeño / Simétrico")
                )
                c_p4.metric(
                    "🔍 Normalidad de Diferencias (Shapiro)",
                    f"W = {sw_stat:.3f}" if not np.isnan(sw_stat) else "N/A",
                    "Distribución Gaussiana (p > 0.05)" if (not np.isnan(sw_p) and sw_p > 0.05) else ("No Gaussiana" if not np.isnan(sw_p) else "N < 3")
                )

                # ─── SLOPE PLOT (LÍNEAS DE PENDIENTE INDIVIDUAL POR ANIMAL) ───
                col_g1, col_g2 = st.columns([1.2, 1.0])

                with col_g1:
                    fig_slope = go.Figure()

                    # Add individual animal trajectory lines
                    for _, p_row in df_paired_n.iterrows():
                        aid = str(p_row['animal_id'])
                        s_sex = str(p_row.get('sex', 'MACHO'))
                        line_color = SEX_COLORS.get(s_sex, '#00f2fe')
                        y_vals = [p_row['v_ipsi'], p_row['v_contra']]

                        fig_slope.add_trace(go.Scatter(
                            x=['IPSI', 'CONTRA'],
                            y=y_vals,
                            mode='lines+markers',
                            name=f"{aid} ({s_sex})",
                            line=dict(color=line_color, width=2.2, dash='solid'),
                            marker=dict(size=9, color=line_color),
                            text=[f"{aid} (IPSI): {y_vals[0]:.2f}", f"{aid} (CONTRA): {y_vals[1]:.2f}"],
                            hoverinfo='text'
                        ))

                    # Add overall mean trajectory
                    fig_slope.add_trace(go.Scatter(
                        x=['IPSI', 'CONTRA'],
                        y=[mean_ipsi, mean_contra],
                        mode='lines+markers',
                        name='MEDIA POBLACIONAL',
                        line=dict(color='#ffffff', width=4, dash='dash'),
                        marker=dict(size=14, symbol='diamond', color='#ffffff', line=dict(color='#000000', width=1.5))
                    ))

                    # Add overhead significance bracket
                    all_paired_y = list(df_paired_n['v_ipsi']) + list(df_paired_n['v_contra'])
                    max_py = max(all_paired_y)
                    min_py = min(all_paired_y)
                    span_py = max_py - min_py if max_py != min_py else 1.0
                    bar_py = max_py + span_py * 0.12
                    tick_py = span_py * 0.025

                    fig_slope.add_shape(type='line', x0=0, x1=1, y0=bar_py, y1=bar_py, line=dict(color='#00ff88' if p_pair < 0.05 else '#ffffff', width=2))
                    fig_slope.add_shape(type='line', x0=0, x1=0, y0=bar_py - tick_py, y1=bar_py, line=dict(color='#00ff88' if p_pair < 0.05 else '#ffffff', width=2))
                    fig_slope.add_shape(type='line', x0=1, x1=1, y0=bar_py - tick_py, y1=bar_py, line=dict(color='#00ff88' if p_pair < 0.05 else '#ffffff', width=2))
                    fig_slope.add_annotation(
                        x=0.5, y=bar_py + tick_py * 1.5,
                        text=f"<b>t-Student Pareada: t({df_deg}) = {t_pair:.2f} (p = {p_pair:.4f}, {sig_stars(p_pair)})</b>",
                        showarrow=False,
                        font=dict(size=12, color='#00ff88' if p_pair < 0.05 else '#ffffff')
                    )

                    fig_slope.update_layout(
                        title=dict(text=f"<b>Trayectoria Pareada Interhemisférica (IPSI vs CONTRA) — {var_label_none}</b>", font=dict(size=14, color='#bb86fc')),
                        yaxis_title=var_label_none,
                        xaxis_title="Hemisferio",
                        template='plotly_dark',
                        height=460,
                        showlegend=True,
                        yaxis=dict(range=[min_py - span_py * 0.05, bar_py + span_py * 0.15])
                    )
                    st.plotly_chart(fig_slope, use_container_width=True)
                    st.download_button(
                        "📄 Descargar Datos del Gráfico Pareado (.csv)",
                        data=df_paired_n[['animal_id', 'sex', 'v_ipsi', 'v_contra']].to_csv(index=False).encode('utf-8'),
                        file_name=f"datos_grafico_pareado_{selected_none_var}.csv",
                        mime="text/csv",
                        key=f"dl_btn_slope_{selected_none_var}",
                        use_container_width=True
                    )

                with col_g2:
                    # Difference bar chart per animal
                    fig_diff = px.bar(
                        df_paired_n,
                        x='animal_id',
                        y='diff',
                        color='sex',
                        color_discrete_map=SEX_COLORS,
                        title=f"Diferencia Individual por Animal (IPSI − CONTRA)",
                        labels={'diff': 'Δ (IPSI − CONTRA)', 'animal_id': 'Sujeto', 'sex': 'Sexo'},
                        template='plotly_dark'
                    )
                    fig_diff.add_hline(y=0, line_dash="solid", line_color="#888888", line_width=1.5)
                    fig_diff.add_hline(y=mean_diff, line_dash="dash", line_color="#00f2fe", annotation_text=f"Media Δ: {mean_diff:.2f}", annotation_position="top right")
                    fig_diff.update_layout(height=460)
                    st.plotly_chart(fig_diff, use_container_width=True)
                    st.download_button(
                        "📄 Descargar Datos del Gráfico de Diferencias (.csv)",
                        data=df_paired_n[['animal_id', 'sex', 'diff', 'pct_diff']].to_csv(index=False).encode('utf-8'),
                        file_name=f"datos_grafico_diferencias_{selected_none_var}.csv",
                        mime="text/csv",
                        key=f"dl_btn_diff_{selected_none_var}",
                        use_container_width=True
                    )

                # Table of paired differences
                with st.expander("📋 Ver Tabla Detallada Pareada por Sujeto (IPSI, CONTRA, Δ y % Diferencia)", expanded=False):
                    df_show_paired = df_paired_n[['animal_id', 'sex', 'v_ipsi', 'v_contra', 'diff', 'pct_diff']].copy()
                    df_show_paired.columns = ['Sujeto (Animal ID)', 'Sexo', 'IPSI', 'CONTRA', 'Diferencia (Δ)', '% Variación Relativa']
                    st.dataframe(df_show_paired.style.format({
                        'IPSI': '{:.3f}', 'CONTRA': '{:.3f}', 'Diferencia (Δ)': '{:.3f}', '% Variación Relativa': '{:+.2f}%'
                    }), use_container_width=True)
                    st.download_button(
                        "📥 Descargar Datos Pareados IPSI vs CONTRA (.csv)",
                        data=df_paired_n.to_csv(index=False).encode('utf-8'),
                        file_name=f"control_none_pareado_ipsi_contra_{selected_none_var}.csv",
                        mime="text/csv",
                        key=f"dl_none_paired_{selected_none_var}"
                    )

        # ─── SECCIÓN 2: COMPARACIÓN POR SEXO EN CONTROLES BASALES (MACHO vs HEMBRA) ───
        st.divider()
        st.subheader("🚻 2. Comparación por Sexo en Controles Basales (NONE MACHO vs NONE HEMBRA)")

        scope_sex = st.radio(
            "Seleccionar Ámbito de Hemisferios para Comparación de Sexo:",
            ["Promedio por Sujeto (Ambos Hemisferios)", "Solo Hemisferio IPSI", "Solo Hemisferio CONTRA"],
            horizontal=True,
            key="none_sex_scope"
        )

        if selected_none_var not in df_none_eval.columns:
            st.warning(f"La variable '{selected_none_var}' no está disponible para comparación por sexo.")
            vals_macho, vals_hembra = np.array([]), np.array([])
        else:
            if scope_sex == "Solo Hemisferio IPSI":
                df_sex_eval = df_none_eval[df_none_eval['section'] == 'IPSI'].dropna(subset=[selected_none_var]).copy()
            elif scope_sex == "Solo Hemisferio CONTRA":
                df_sex_eval = df_none_eval[df_none_eval['section'] == 'CONTRA'].dropna(subset=[selected_none_var]).copy()
            else:
                # Average across hemispheres per animal
                df_sex_eval = df_none_eval.groupby(['animal_id', 'sex'], as_index=False)[selected_none_var].mean().dropna()

            vals_macho = df_sex_eval[df_sex_eval['sex'] == 'MACHO'][selected_none_var].dropna().values
            vals_hembra = df_sex_eval[df_sex_eval['sex'] == 'HEMBRA'][selected_none_var].dropna().values

        n_m = len(vals_macho)
        n_h = len(vals_hembra)

        if n_m < 2 or n_h < 2:
            st.info(f"Se requieren al menos 2 machos y 2 hembras para la prueba t independiente (Encontrados: Machos N={n_m}, Hembras N={n_h}).")
        else:
            # Independent t-test (Equal variances)
            t_ind_val, p_ind_val = ttest_ind(vals_macho, vals_hembra, equal_var=True)
            # Welch's t-test (Unequal variances)
            t_welch_val, p_welch_val = ttest_ind(vals_macho, vals_hembra, equal_var=False)
            
            # Levene's test of homogeneity
            w_lev_s, p_lev_s = levene(vals_macho, vals_hembra)
            
            # Mann-Whitney U test confirmation
            u_stat_s, p_u_s = mannwhitneyu(vals_macho, vals_hembra)

            m_m, s_m = float(np.mean(vals_macho)), float(np.std(vals_macho, ddof=1))
            m_h, s_h = float(np.mean(vals_hembra)), float(np.std(vals_hembra, ddof=1))

            # Pooled SD & Cohen's d
            s_pooled = np.sqrt(((n_m - 1)*(s_m**2) + (n_h - 1)*(s_h**2)) / (n_m + n_h - 2))
            cohen_d_sex = (m_m - m_h) / (s_pooled + 1e-8) if s_pooled > 0 else 0.0

            # KPI Cards Sex
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric(
                "🧪 t de Student Independiente",
                f"t({n_m+n_h-2}) = {t_ind_val:.3f}",
                delta=f"p = {p_ind_val:.4e} ({sig_stars(p_ind_val)})",
                delta_color="normal" if p_ind_val < 0.05 else "off"
            )
            c_s2.metric(
                "🔬 Welch's t (Sin asumir homocedasticidad)",
                f"t = {t_welch_val:.3f}",
                f"p = {p_welch_val:.4e} ({sig_stars(p_welch_val)})"
            )
            c_s3.metric(
                "📏 Cohen's d (Dimorfismo)",
                f"d = {cohen_d_sex:.3f}",
                "Fuerte" if abs(cohen_d_sex)>=0.8 else ("Moderado" if abs(cohen_d_sex)>=0.5 else "Bajo / Sin sesgo")
            )
            c_s4.metric(
                "⚖️ Homocedasticidad (Levene)",
                f"W = {w_lev_s:.3f}",
                "Varianzas Homogéneas" if p_lev_s > 0.05 else "Varianzas Heterogéneas"
            )

            col_sx1, col_sx2 = st.columns(2)

            with col_sx1:
                # Boxplot + Violín + Jitter
                fig_sex_box = go.Figure()
                for sx_name in ['MACHO', 'HEMBRA']:
                    df_sx = df_sex_eval[df_sex_eval['sex'] == sx_name]
                    v_sx = df_sx[selected_none_var].dropna().values
                    col_sx = SEX_COLORS.get(sx_name, '#ffffff')
                    fig_sex_box.add_trace(go.Box(
                        x=[sx_name] * len(v_sx),
                        y=v_sx,
                        name=sx_name,
                        marker_color=col_sx,
                        boxmean='sd',
                        boxpoints='all',
                        jitter=0.35,
                        pointpos=0,
                        marker=dict(size=8, color=col_sx),
                        line=dict(color=col_sx, width=2.2),
                        text=df_sx['animal_id'] if 'animal_id' in df_sx.columns else None
                    ))

                # Overhead bracket for sex comparison
                all_sy = list(vals_macho) + list(vals_hembra)
                max_sy = max(all_sy)
                min_sy = min(all_sy)
                span_sy = max_sy - min_sy if max_sy != min_sy else 1.0
                bar_sy = max_sy + span_sy * 0.12
                tick_sy = span_sy * 0.025

                fig_sex_box.add_shape(type='line', x0=0, x1=1, y0=bar_sy, y1=bar_sy, line=dict(color='#00ff88' if p_ind_val < 0.05 else '#ffffff', width=2))
                fig_sex_box.add_shape(type='line', x0=0, x1=0, y0=bar_sy - tick_sy, y1=bar_sy, line=dict(color='#00ff88' if p_ind_val < 0.05 else '#ffffff', width=2))
                fig_sex_box.add_shape(type='line', x0=1, x1=1, y0=bar_sy - tick_sy, y1=bar_sy, line=dict(color='#00ff88' if p_ind_val < 0.05 else '#ffffff', width=2))
                fig_sex_box.add_annotation(
                    x=0.5, y=bar_sy + tick_sy * 1.5,
                    text=f"<b>p = {p_ind_val:.4f} ({sig_stars(p_ind_val)}) | d = {cohen_d_sex:.2f}</b>",
                    showarrow=False,
                    font=dict(size=12, color='#00ff88' if p_ind_val < 0.05 else '#ffffff')
                )

                fig_sex_box.update_layout(
                    title=dict(text=f"<b>Distribución por Sexo en Controles Basales (NONE)</b>", font=dict(size=14, color='#bb86fc')),
                    yaxis_title=var_label_none,
                    xaxis_title="Sexo",
                    template='plotly_dark',
                    height=460,
                    showlegend=False,
                    yaxis=dict(range=[min_sy - span_sy * 0.05, bar_sy + span_sy * 0.15])
                )
                st.plotly_chart(fig_sex_box, use_container_width=True)
                st.download_button(
                    "📄 Descargar Datos del Gráfico por Sexo (Boxplot) (.csv)",
                    data=df_sex_eval.to_csv(index=False).encode('utf-8'),
                    file_name=f"datos_grafico_sexo_boxplot_{selected_none_var}.csv",
                    mime="text/csv",
                    key=f"dl_btn_sex_box_{selected_none_var}",
                    use_container_width=True
                )

            with col_sx2:
                # Bar Chart with SEM
                sem_m = s_m / np.sqrt(n_m) if n_m > 0 else 0.0
                sem_h = s_h / np.sqrt(n_h) if n_h > 0 else 0.0

                df_bar_sex = pd.DataFrame([
                    {'Sexo': 'MACHO', 'Media': m_m, 'SEM': sem_m, 'N': n_m},
                    {'Sexo': 'HEMBRA', 'Media': m_h, 'SEM': sem_h, 'N': n_h}
                ])

                fig_bar_sex = px.bar(
                    df_bar_sex,
                    x='Sexo',
                    y='Media',
                    color='Sexo',
                    error_y='SEM',
                    color_discrete_map=SEX_COLORS,
                    title=f"Media ± SEM por Sexo (NONE)",
                    template='plotly_dark'
                )
                fig_bar_sex.update_layout(height=460, yaxis_title=var_label_none)
                st.plotly_chart(fig_bar_sex, use_container_width=True)
                st.download_button(
                    "📄 Descargar Medias ± SEM del Gráfico de Sexo (.csv)",
                    data=df_bar_sex.to_csv(index=False).encode('utf-8'),
                    file_name=f"datos_grafico_sexo_barras_sem_{selected_none_var}.csv",
                    mime="text/csv",
                    key=f"dl_btn_sex_bar_{selected_none_var}",
                    use_container_width=True
                )

            # Table of descriptive metrics by sex
            st.markdown('<div class="table-caption">📋 Estadísticos Descriptivos por Sexo en Controles Basales (NONE)</div>', unsafe_allow_html=True)
            df_desc_sex = pd.DataFrame([
                {
                    "Sexo": "MACHO",
                    "N Sujetos": n_m,
                    "Media": f"{m_m:.4f}",
                    "Desv. Estándar (DE)": f"{s_m:.4f}",
                    "Error Estándar (SEM)": f"{sem_m:.4f}",
                    "Mediana": f"{np.median(vals_macho):.4f}",
                    "IQR": f"{np.percentile(vals_macho, 75) - np.percentile(vals_macho, 25):.4f}",
                    "Mínimo": f"{np.min(vals_macho):.4f}",
                    "Máximo": f"{np.max(vals_macho):.4f}"
                },
                {
                    "Sexo": "HEMBRA",
                    "N Sujetos": n_h,
                    "Media": f"{m_h:.4f}",
                    "Desv. Estándar (DE)": f"{s_h:.4f}",
                    "Error Estándar (SEM)": f"{sem_h:.4f}",
                    "Mediana": f"{np.median(vals_hembra):.4f}",
                    "IQR": f"{np.percentile(vals_hembra, 75) - np.percentile(vals_hembra, 25):.4f}",
                    "Mínimo": f"{np.min(vals_hembra):.4f}",
                    "Máximo": f"{np.max(vals_hembra):.4f}"
                }
            ])
            st.dataframe(df_desc_sex, use_container_width=True)

        # ─── SECCIÓN 3: MATRIZ MULTIVARIABLE CONSOLIDADA DE CONTROLES BASALES ───
        st.divider()
        st.subheader("📊 3. Batería Multivariable Consolidada de Controles Basales (t-Student)")
        st.markdown("Tabla resumen con la batería completa de pruebas **t de Student** (Hemisferios Pareados y Dimorfismo por Sexo) calculadas sobre todas las variables principales del estudio:")

        representative_vars = [
            ("pnn_density_roi", "Densidad PNN Totales (/mm²)"),
            ("pnn_filled_density_roi", "Densidad PNNs Rellenas PV+/PNN+ (/mm²)"),
            ("pnn_hollow_density_roi", "Densidad PNNs Huecas PV-/PNN+ (/mm²)"),
            ("pnn_area_fraction_pct_roi", "Cobertura PNNs en ROI (% Área)"),
            ("pv_density_roi", "Densidad PV+ Totales (/mm²)"),
            ("pv_coloc_density_roi", "Densidad PV+/PNN+ (/mm²)"),
            ("pct_pv_wrapped_roi", "% PV+ Envueltas en PNN"),
            ("wfa_int_density_per_mm2_roi", "WFA Integrada / mm² ROI [PONDERADA]"),
            ("wfa_int_per_mesh_um2_roi", "WFA Integrada / µm² Malla [PONDERADA]"),
            ("total_wfa_int_density_roi", "Suma Total WFA en ROI [NO PONDERADA]"),
            ("mean_wfa_int_density_roi", "WFA Integrada Media / Red [NO PONDERADA]"),
            ("pnn_energy_roi", "Potencia WFA (Lupori Energy)"),
            ("mean_wfa_intensity_roi", "Intensidad Media Anillo WFA"),
            ("mean_pnn_area_roi", "Área Malla PNN (µm²)"),
            ("mean_pv_area_roi", "Área Soma PV+ (µm²)"),
            ("wfa_diffuse_mean_roi", "Fondo Difuso WFA en ROI"),
            ("wfa_signal_to_diffuse_ratio_roi", "Ratio Señal PNN / Fondo Difuso")
        ]

        matrix_rows = []
        for var_k, var_display in representative_vars:
            if var_k not in df_none_all.columns:
                continue

            # 1. Paired Hemispheres
            df_ip = df_none_all[df_none_all['section'] == 'IPSI'][['animal_id', var_k]].groupby('animal_id')[var_k].mean()
            df_ct = df_none_all[df_none_all['section'] == 'CONTRA'][['animal_id', var_k]].groupby('animal_id')[var_k].mean()
            df_pair_k = pd.concat([df_ip.rename('ipsi'), df_ct.rename('contra')], axis=1, join='inner').dropna()

            if len(df_pair_k) >= 2:
                t_k, p_k = ttest_rel(df_pair_k['ipsi'], df_pair_k['contra'])
                m_ip, s_ip = float(np.mean(df_pair_k['ipsi'])), float(np.std(df_pair_k['ipsi'], ddof=1))
                m_ct, s_ct = float(np.mean(df_pair_k['contra'])), float(np.std(df_pair_k['contra'], ddof=1))
                t_pair_str = f"t={t_k:.2f} (p={p_k:.4f}) {sig_stars(p_k)}"
                val_ip_str = f"{m_ip:.2f} ± {s_ip:.2f}"
                val_ct_str = f"{m_ct:.2f} ± {s_ct:.2f}"
            else:
                t_pair_str, val_ip_str, val_ct_str = "N/A", "N/A", "N/A"

            # 2. Sex Comparison (Subject level mean)
            df_subj_k = df_none_all.groupby(['animal_id', 'sex'], as_index=False)[var_k].mean().dropna()
            vm_k = df_subj_k[df_subj_k['sex'] == 'MACHO'][var_k].values
            vh_k = df_subj_k[df_subj_k['sex'] == 'HEMBRA'][var_k].values

            if len(vm_k) >= 2 and len(vh_k) >= 2:
                t_sx_k, p_sx_k = ttest_ind(vm_k, vh_k, equal_var=True)
                mm_k, sm_k = float(np.mean(vm_k)), float(np.std(vm_k, ddof=1))
                mh_k, sh_k = float(np.mean(vh_k)), float(np.std(vh_k, ddof=1))
                t_sex_str = f"t={t_sx_k:.2f} (p={p_sx_k:.4f}) {sig_stars(p_sx_k)}"
                val_m_str = f"{mm_k:.2f} ± {sm_k:.2f}"
                val_h_str = f"{mh_k:.2f} ± {sh_k:.2f}"
            else:
                t_sex_str, val_m_str, val_h_str = "N/A", "N/A", "N/A"

            matrix_rows.append({
                "Variable Biológica": var_display,
                "N Sujetos": len(df_pair_k),
                "IPSI (Media ± DE)": val_ip_str,
                "CONTRA (Media ± DE)": val_ct_str,
                "Prueba t Pareada (IPSI vs CONTRA)": t_pair_str,
                "MACHO (Media ± DE)": val_m_str,
                "HEMBRA (Media ± DE)": val_h_str,
                "Prueba t Sexo (MACHO vs HEMBRA)": t_sex_str
            })

        if matrix_rows:
            df_matrix = pd.DataFrame(matrix_rows)
            st.dataframe(df_matrix, use_container_width=True)
            st.download_button(
                "📥 Descargar Matriz Multivariable de Controles Basales (.csv)",
                data=df_matrix.to_csv(index=False).encode('utf-8'),
                file_name="matriz_controles_basales_none_tstudent.csv",
                mime="text/csv",
                key="dl_matrix_none_tstudent_csv"
            )

# 6. TAB EXPORTACIÓN INTEGRAL (FORMATO PRISM & PARÁMETROS)
with tab_export:
    st.header("📦 Centro de Exportación de Datos Crudos y Parámetros del Experimento")
    st.markdown("""
    <div class="stats-box">
    Exporta la totalidad de los datos procesados del estudio en <b>formatos estandarizados y rotulados</b> listos para su importación directa en <b>GraphPad Prism</b>, R, Python o SPSS, junto con el registro completo de parámetros de corrida del pipeline.
    </div>
    """, unsafe_allow_html=True)

    # All metrics dictionary
    ALL_EXPORT_VARS = {}
    ALL_EXPORT_VARS.update(PV_VARS)
    ALL_EXPORT_VARS.update(PNN_VARS)
    ALL_EXPORT_VARS.update(INTENSITY_VARS)
    ALL_EXPORT_VARS.update(DIFFUSE_VARS)

    # ─── SECCIÓN 1: DESCARGA EN 1 CLIC DEL PAQUETE COMPLETO (ZIP) ───
    st.subheader("🎁 1. Descarga del Paquete Integral Completo (.zip)")
    st.markdown("""
    Descarga en un solo archivo comprimido estructurado:
    * **`01_Master_Dataset_Por_Sujeto_Long.csv`**: Tabla completa promediada por animal con todas las variables numéricas y factores de diseño.
    * **`02_Master_Dataset_Por_Preparado_Corte_Long.csv`**: Tabla con cada corte histológico individual.
    * **`03_Master_Dataset_Unicelular_Completo.csv`**: Base de datos de cada célula, soma PV+ y red PNN segmentada en el experimento.
    * **`04_Matrices_GraphPad_Prism_Column_Format.xlsx`**: Libro de Excel con una hoja por cada métrica clave formateada en columnas de Prism para copiar/pegar directamente.
    * **`05_Matrices_Prism_CSVs/`**: Carpeta con archivos `.csv` individuales por variable para Prism.
    * **`06_Parametros_Corrida_Pipeline.json` y `.txt`**: Registro reproducible con todos los hiperparámetros, resoluciones, umbrales y modelos de IA.
    * **`README_GUIA_IMPORTACION_PRISMA.md`**: Manual detallado de importación y diccionario de variables.
    """)

    zip_data = build_full_export_zip(df_subj, df_img, active_cells_df, ALL_EXPORT_VARS)
    
    col_z1, col_z2 = st.columns([1.5, 2.5])
    with col_z1:
        st.download_button(
            "📦 DESCARGAR PAQUETE COMPLETO (ZIP)",
            data=zip_data,
            file_name=f"pnn_scc_export_completo_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            key="dl_master_zip_btn",
            use_container_width=True
        )
    with col_z2:
        st.info(f"📊 Paquete generado con éxito: **{len(df_subj)}** sujetos, **{len(df_img)}** preparados y **{len(active_cells_df):,}** células registradas.")

    # ─── SECCIÓN 2: GENERADOR INTERACTIVO DE MATRICES FORMATO PRISM ───
    st.divider()
    st.subheader("📊 2. Generador Interactivo de Matrices en Formato GraphPad Prism")
    st.markdown("Selecciona una variable para estructurarla en el formato columnar de GraphPad Prism (donde cada columna es un grupo experimental y cada fila es un animal/réplica biológica):")

    c_sel1, c_sel2, c_sel3 = st.columns([2, 1, 1])
    with c_sel1:
        export_metric = st.selectbox(
            "Variable a Exportar para Prism:",
            list(ALL_EXPORT_VARS.keys()),
            format_func=lambda x: ALL_EXPORT_VARS[x],
            key="prism_metric_select"
        )
    with c_sel2:
        export_level = st.selectbox(
            "Nivel de Agregación:",
            ["Por Sujeto (Promedio Animal)", "Por Preparado / Corte (Imagen)"],
            key="prism_level_select"
        )
    with c_sel3:
        export_format_type = st.selectbox(
            "Tipo de Matriz Prism:",
            ["Column Table (Grupos en Columnas)", "Grouped Table 2-Way (IPSI vs CONTRA)"],
            key="prism_format_select"
        )

    src_for_prism = df_subj if export_level.startswith("Por Sujeto") else df_img
    
    if export_metric not in src_for_prism.columns:
        st.warning(f"La variable '{export_metric}' no está disponible en la tabla de {export_level.lower()}.")
    else:
        if export_format_type.startswith("Column Table"):
            df_prism_preview = generate_prism_column_matrix(src_for_prism, export_metric, level='subject' if export_level.startswith("Por Sujeto") else 'image')
        else:
            df_prism_preview = generate_prism_grouped_2way_matrix(src_for_prism, export_metric)

        st.markdown(f"**Vista Previa de la Matriz para Prism — `{ALL_EXPORT_VARS.get(export_metric, export_metric)}`:**")
        st.dataframe(df_prism_preview, use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                f"📥 Descargar Matriz Prism (.csv) — {export_metric}",
                data=df_prism_preview.to_csv(index=False).encode('utf-8'),
                file_name=f"prism_matrix_{export_metric}_{export_level[:7].strip().lower()}.csv",
                mime="text/csv",
                key=f"dl_prism_csv_{export_metric}",
                use_container_width=True
            )
        with col_dl2:
            # Excel export of single table
            ex_buf = io.BytesIO()
            with pd.ExcelWriter(ex_buf, engine='openpyxl') as wr:
                df_prism_preview.to_excel(wr, sheet_name="Prism_Data", index=False)
            st.download_button(
                f"📥 Descargar Matriz Prism (.xlsx) — {export_metric}",
                data=ex_buf.getvalue(),
                file_name=f"prism_matrix_{export_metric}_{export_level[:7].strip().lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_prism_xlsx_{export_metric}",
                use_container_width=True
            )

    # ─── SECCIÓN 3: TABLAS MAESTRAS INDIVIDUALES (LONG FORMAT) ───
    st.divider()
    st.subheader("📑 3. Tablas Maestras Individuales (Formato Long / Tidy)")
    
    with st.expander("🐭 Ver y Descargar Master Dataset por Sujeto (Animal Level)", expanded=False):
        st.dataframe(df_subj, use_container_width=True)
        st.download_button(
            "📥 Descargar Master Sujeto (.csv)",
            data=df_subj.to_csv(index=False).encode('utf-8'),
            file_name="master_dataset_por_sujeto.csv",
            mime="text/csv",
            key="dl_master_subj_btn"
        )

    with st.expander("🔬 Ver y Descargar Master Dataset por Preparado / Corte (Image Level)", expanded=False):
        st.dataframe(df_img, use_container_width=True)
        st.download_button(
            "📥 Descargar Master Preparado (.csv)",
            data=df_img.to_csv(index=False).encode('utf-8'),
            file_name="master_dataset_por_preparado.csv",
            mime="text/csv",
            key="dl_master_img_btn"
        )

    with st.expander("🧫 Ver y Descargar Master Dataset Unicelular (Células y PNNs Individuales)", expanded=False):
        st.dataframe(active_cells_df.head(200), use_container_width=True)
        st.caption(f"Mostrando las primeras 200 filas de {len(active_cells_df):,} registros celulares totales.")
        st.download_button(
            "📥 Descargar Master Unicelular Completo (.csv)",
            data=active_cells_df.to_csv(index=False).encode('utf-8'),
            file_name="master_dataset_unicelular_completo.csv",
            mime="text/csv",
            key="dl_master_cells_btn"
        )

    # ─── SECCIÓN 4: FICHA TÉCNICA Y PARÁMETROS DE CORRIDA DEL PIPELINE ───
    st.divider()
    st.subheader("⚙️ 4. Ficha Técnica y Registro de Parámetros de Corrida")
    st.markdown("Registro estandarizado de parámetros experimentales, modelos de IA y criterios de procesamiento:")

    params_dict_view = get_pipeline_parameters_dict()
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**🔬 Especificaciones Ópticas y Calibración:**")
        st.json(params_dict_view["optica_y_calibracion_espacial"])
        st.markdown("**🧠 Segmentación PV+ (Cellpose):**")
        st.json(params_dict_view["etapa_1_segmentacion_somas_pv"])
        st.markdown("**⚡ Detección PNN WFA (DoG + IA):**")
        st.json(params_dict_view["etapa_2_deteccion_redes_pnn_wfa"])

    with col_p2:
        st.markdown("**🔬 Colocalización y Subpoblaciones:**")
        st.json(params_dict_view["etapa_3_colocalizacion_y_clasificacion_subpoblaciones"])
        st.markdown("**🌫️ Espacio Difuso Neuropilo WFA:**")
        st.json(params_dict_view["etapa_4_cuantificacion_espacio_difuso_neuropilo"])
        st.markdown("**📐 Formulaciones de Normalización y Ponderación:**")
        st.json(params_dict_view["etapa_5_formulaciones_densidad_y_ponderacion_area"])

    col_dp1, col_dp2 = st.columns(2)
    with col_dp1:
        st.download_button(
            "📥 Descargar Parámetros del Pipeline (.json)",
            data=json.dumps(params_dict_view, indent=4, ensure_ascii=False).encode('utf-8'),
            file_name="parametros_corrida_pipeline.json",
            mime="application/json",
            key="dl_params_json_btn",
            use_container_width=True
        )
    with col_dp2:
        txt_rep = "\n".join([f"{k.upper()}: {json.dumps(v, ensure_ascii=False, indent=2)}" for k, v in params_dict_view.items()])
        st.download_button(
            "📥 Descargar Ficha Técnica (.txt)",
            data=txt_rep.encode('utf-8'),
            file_name="ficha_tecnica_parametros_pipeline.txt",
            mime="text/plain",
            key="dl_params_txt_btn",
            use_container_width=True
        )

