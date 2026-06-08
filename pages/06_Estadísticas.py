import streamlit as st
import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, wilcoxon

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
        background: rgba(30, 33, 48, 0.6);
        border: 1px solid rgba(187, 134, 252, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
    }
    .level-info {
        background: rgba(0, 242, 254, 0.06);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    .sex-header {
        font-size: 1.1rem;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: inline-block;
    }
    .significance-sig {
        color: #00ff88; font-weight: bold; font-size: 1rem;
        background-color: rgba(0,255,136,0.1); padding: 10px;
        border-radius: 8px; border-left: 5px solid #00ff88;
    }
    .significance-nonsig {
        color: #ffaa00; font-weight: bold; font-size: 1rem;
        background-color: rgba(255,170,0,0.1); padding: 10px;
        border-radius: 8px; border-left: 5px solid #ffaa00;
    }
    hr { border: 0; height: 1px;
         background: linear-gradient(to right, transparent, #bb86fc, transparent);
         margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Paso 6: Comparación Estadística")

METRICS_BASE_DIR = "data/processed/metrics"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def load_all_experiment_metrics():
    all_dfs = []
    if not os.path.exists(METRICS_BASE_DIR):
        return pd.DataFrame()
    for root, dirs, files in os.walk(METRICS_BASE_DIR):
        if "test" in root:
            continue
        for f in files:
            if not f.endswith('_nuclei_metrics.csv'):
                continue
            csv_path = os.path.join(root, f)
            rel_path = os.path.relpath(csv_path, METRICS_BASE_DIR)
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                group    = parts[0]
                section  = parts[1]
                filename = parts[2].replace('_nuclei_metrics.csv', '')
            else:
                group = section = "Desconocido"
                filename = f.replace('_nuclei_metrics.csv', '')
            m = re.match(r'(ACF_\d+)', filename)
            animal_id = m.group(1) if m else filename.split('~')[0]
            m2 = re.search(r'~(\d+)$', filename)
            corte_num = int(m2.group(1)) if m2 else 1
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    df['group']      = group
                    df['section']    = section
                    df['image_name'] = filename
                    df['animal_id']  = animal_id
                    df['corte_num']  = corte_num
                    all_dfs.append(df)
            except Exception:
                pass
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

df_raw = load_all_experiment_metrics()

if df_raw.empty:
    st.info("👋 No se encontraron métricas. Ejecuta el pipeline primero.")
    st.stop()

# Backward compat
if 'pv_area_um2' not in df_raw.columns:
    df_raw['pv_area_um2']    = df_raw.apply(lambda r: r['area_um2'] if r['is_pv_plus'] else 0.0, axis=1)
    df_raw['pv_diameter_um'] = df_raw.apply(lambda r: r['diameter_um'] if r['is_pv_plus'] else 0.0, axis=1)
    df_raw['pnn_area_um2']   = df_raw.apply(lambda r: r['area_um2'] if r['is_pnn_plus'] else 0.0, axis=1)
    df_raw['pnn_diameter_um']= df_raw.apply(lambda r: r['diameter_um'] if r['is_pnn_plus'] else 0.0, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# GROUP PARSING — extract sex and condition from group name
# ─────────────────────────────────────────────────────────────────────────────
def parse_group(name):
    """Returns (sex, condition_label, condition_order) from a group folder name.
    Normalizes accents so 'DÍAS'/'DIAS' both match 'DIA'."""
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
    elif '14' in name_norm and 'DIA' in name_norm:
        cond, order = '14 DÍAS', 2
    else:
        cond, order = name, 99

    return sex, cond, order

# Build lookup: group_name → (sex, condition, order)
all_groups = df_raw['group'].dropna().unique().tolist()
group_meta = {}
for g in all_groups:
    sex, cond, order = parse_group(g)
    if sex is not None:
        group_meta[g] = {'sex': sex, 'condition': cond, 'order': order}

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
SKEL_COLS = ['skel_total_length_um','skel_branches_count','skel_mean_thickness_um',
             'skel_max_thickness_um','skel_mean_intensity','skel_neighborhood_wfa_sum',
             'skel_tortuosity_mean','skel_ramification_index']

def _img_summary(group_df):
    total_pv  = int(group_df['is_pv_plus'].sum())
    total_pnn = int(group_df['is_pnn_plus'].sum())
    total_occ = int((group_df['cell_type'] == 'PV+/PNN+').sum())
    total_hol = int((group_df['cell_type'] == 'PV-/PNN+').sum())
    pv_cells  = group_df[group_df['is_pv_plus']  == True]
    pnn_cells = group_df[group_df['is_pnn_plus'] == True]
    def safe_mean(s): return float(s.mean()) if len(s) > 0 and pd.notna(s.mean()) else 0.0
    row = {
        'pv_count': total_pv, 'pnn_count': total_pnn,
        'pnn_count_filled': total_occ, 'pnn_count_hollow': total_hol,
        'pct_pnn_plus':   (total_occ / total_pv  * 100) if total_pv  > 0 else 0.0,
        'pct_pnn_hollow': (total_hol / total_pnn * 100) if total_pnn > 0 else 0.0,
        'mean_pv_area_um2':    safe_mean(pv_cells['pv_area_um2']),
        'mean_pv_diameter_um': safe_mean(pv_cells['pv_diameter_um']),
        'mean_pnn_area_um2':   safe_mean(pnn_cells['pnn_area_um2']),
        'mean_pnn_diameter_um':safe_mean(pnn_cells['pnn_diameter_um']),
        'mean_soma_area_um2':   safe_mean(pv_cells['pv_area_um2']),
        'mean_soma_diameter_um':safe_mean(pv_cells['pv_diameter_um']),
    }
    for col in SKEL_COLS:
        row[f'mean_{col}'] = safe_mean(pnn_cells[col]) if col in pnn_cells.columns else 0.0
    return row

def aggregate_image_level(df):
    keys = ['group','section','image_name','animal_id','corte_num']
    summaries = []
    for vals, grp in df.groupby(keys):
        base = dict(zip(keys, vals))
        base.update(_img_summary(grp))
        summaries.append(base)
    return pd.DataFrame(summaries)

def aggregate_subject_level(df_img):
    """One row per (group, section, animal_id) — preserves hemisphere.
    Used for IPSI vs CONTRA paired comparisons."""
    numeric_cols = [c for c in df_img.columns
                    if c not in ['group','section','image_name','animal_id','corte_num']
                    and pd.api.types.is_numeric_dtype(df_img[c])]
    summaries = []
    for vals, grp in df_img.groupby(['group','section','animal_id']):
        base = dict(zip(['group','section','animal_id'], vals))
        base['n_cortes'] = len(grp)
        for col in numeric_cols:
            base[col] = grp[col].mean()
        summaries.append(base)
    return pd.DataFrame(summaries)

def aggregate_subject_level_between(df_img):
    """One row per (group, animal_id) — collapses IPSI+CONTRA into one value.
    Use this for between-group comparisons so each animal is counted once."""
    numeric_cols = [c for c in df_img.columns
                    if c not in ['group','section','image_name','animal_id','corte_num']
                    and pd.api.types.is_numeric_dtype(df_img[c])]
    summaries = []
    for vals, grp in df_img.groupby(['group','animal_id']):
        base = dict(zip(['group','animal_id'], vals))
        # Record which sections this animal has
        base['sections'] = '+'.join(sorted(grp['section'].dropna().unique().tolist()))
        base['n_sections'] = grp['section'].nunique()
        base['n_cortes']   = len(grp)
        for col in numeric_cols:
            base[col] = grp[col].mean()
        summaries.append(base)
    return pd.DataFrame(summaries)

df_img        = aggregate_image_level(df_raw)
df_subj       = aggregate_subject_level(df_img)          # with section (IPSI/CONTRA mode)
df_subj_btwn  = aggregate_subject_level_between(df_img)  # collapsed (between-group mode)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración del Análisis")

level_type = st.sidebar.radio(
    "Nivel de Análisis:",
    ["Por Célula (distribuciones individuales)",
     "Por Preparado/Corte (imagen)",
     "Por Sujeto (animal, promediando cortes)"],
    key="stats_level_select"
)

comparison_mode = st.sidebar.radio(
    "Modo de Comparación:",
    ["Entre Condiciones (NONE / 3 DÍAS / 14 DÍAS) por Sexo",
     "IPSI vs CONTRA (Hemisferio)"],
    key="stats_compare_mode"
)

CELL_VARS = {
    "pv_area_um2":               "Área del Soma PV+ (µm²)",
    "pv_diameter_um":            "Diámetro del Soma PV+ (µm)",
    "pnn_area_um2":              "Área del Hueco PNN+ (µm²)",
    "pnn_diameter_um":           "Diámetro del Hueco PNN+ (µm)",
    "wfa_sum_intensity":         "Intensidad WFA de la Entidad",
    "skel_total_length_um":      "Longitud Total del Esqueleto (µm) [PNN+]",
    "skel_branches_count":       "Número de Ramas del Esqueleto [PNN+]",
    "skel_mean_thickness_um":    "Espesor Promedio de Red (µm) [PNN+]",
    "skel_max_thickness_um":     "Espesor Máximo de Red (µm) [PNN+]",
    "skel_mean_intensity":       "Intensidad Media sobre Esqueleto [PNN+]",
    "skel_neighborhood_wfa_sum": "Suma Intensidad Vecindad 1.5µm [PNN+]",
    "skel_tortuosity_mean":      "Tortuosidad Promedio [PNN+]",
    "skel_ramification_index":   "Índice de Ramificación [PNN+]",
}

IMG_SUBJ_VARS = {
    "pv_count":                  "N° de Somas PV+ por preparado",
    "pnn_count":                 "N° de Redes PNN+ Totales",
    "pnn_count_filled":          "N° de PNN+ Ocupadas (PV+/PNN+)",
    "pnn_count_hollow":          "N° de PNN+ Huecas (PNN+/PV-)",
    "pct_pnn_plus":              "% de PV+ con Red PNN (Ocupadas)",
    "pct_pnn_hollow":            "% de PNN+ que son Huecas",
    "mean_pv_area_um2":          "Área Promedio Soma PV+ (µm²)",
    "mean_pv_diameter_um":       "Diámetro Promedio Soma PV+ (µm)",
    "mean_pnn_area_um2":         "Área Promedio Hueco PNN+ (µm²)",
    "mean_pnn_diameter_um":      "Diámetro Promedio Hueco PNN+ (µm)",
    "mean_skel_total_length_um":      "Longitud Promedio Esqueleto PNN+ (µm)",
    "mean_skel_branches_count":       "N° Promedio de Ramas PNN+",
    "mean_skel_mean_thickness_um":    "Espesor Promedio Red PNN+ (µm)",
    "mean_skel_mean_intensity":       "Intensidad Promedio Esqueleto PNN+",
    "mean_skel_neighborhood_wfa_sum": "Intensidad Promedio Vecindad PNN+",
    "mean_skel_tortuosity_mean":      "Tortuosidad Promedio PNN+",
    "mean_skel_ramification_index":   "Índice de Ramificación Promedio PNN+",
}

var_options = CELL_VARS if level_type == "Por Célula (distribuciones individuales)" else IMG_SUBJ_VARS

selected_var_key = st.sidebar.selectbox(
    "Variable de Estudio:",
    list(var_options.keys()),
    format_func=lambda x: var_options[x],
    key="stats_var_select"
)

use_bonferroni = st.sidebar.checkbox(
    "Corrección Bonferroni (×3 comparaciones)",
    value=True, key="stats_bonferroni"
)

# ─────────────────────────────────────────────────────────────────────────────
# BASE DATAFRAME SELECTION + ENTITY FILTER
# ─────────────────────────────────────────────────────────────────────────────
if level_type == "Por Célula (distribuciones individuales)":
    df_base = df_raw.copy()
    skel_vars = [k for k in CELL_VARS if k.startswith('skel_')]
    if selected_var_key in ['pv_area_um2', 'pv_diameter_um']:
        df_base = df_base[df_base['is_pv_plus'] == True]
    elif selected_var_key in ['pnn_area_um2', 'pnn_diameter_um'] + skel_vars:
        df_base = df_base[df_base['is_pnn_plus'] == True]
elif level_type == "Por Preparado/Corte (imagen)":
    df_base = df_img.copy()
else:
    # Por Sujeto: use different aggregation depending on comparison mode
    # • Between groups/conditions → IPSI+CONTRA collapsed into one value per animal
    # • IPSI vs CONTRA → keep section split so we can pair hemispheres
    if comparison_mode == "Entre Condiciones (NONE / 3 DÍAS / 14 DÍAS) por Sexo":
        df_base = df_subj_btwn.copy()   # 1 row per animal (IPSI+CONTRA averaged)
    else:
        df_base = df_subj.copy()        # 1 row per (animal, section)

if selected_var_key not in df_base.columns:
    st.error(f"Variable `{selected_var_key}` no disponible en este nivel de análisis.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: p-value → significance stars
# ─────────────────────────────────────────────────────────────────────────────
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
    """Mann-Whitney U, returns (stat, p). Returns nan if insufficient data."""
    a = np.array(a)[~np.isnan(a)]
    b = np.array(b)[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    return stat, p

def add_significance_bar(fig, x0, x1, y_top, label, color='#ffffff', row_offset=0):
    """Adds a bracket with significance label between x0 and x1 on a Plotly figure."""
    bar_y = y_top * (1.05 + row_offset * 0.12)
    tick_h = y_top * 0.015
    # Horizontal line
    fig.add_shape(type='line', x0=x0, x1=x1, y0=bar_y, y1=bar_y,
                  line=dict(color=color, width=1.5))
    # Left tick
    fig.add_shape(type='line', x0=x0, x1=x0, y0=bar_y - tick_h, y1=bar_y,
                  line=dict(color=color, width=1.5))
    # Right tick
    fig.add_shape(type='line', x0=x1, x1=x1, y0=bar_y - tick_h, y1=bar_y,
                  line=dict(color=color, width=1.5))
    # Text
    fig.add_annotation(
        x=(x0 + x1) / 2, y=bar_y + tick_h * 0.5,
        text=label, showarrow=False,
        font=dict(size=13, color=color if label != 'ns' else '#888888'),
        yref='y', xref='x'
    )
    return bar_y

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION ORDER & COLORS
# ─────────────────────────────────────────────────────────────────────────────
COND_ORDER  = ['NONE', '3 DÍAS', '14 DÍAS']
COND_COLORS = {'NONE': '#4facfe', '3 DÍAS': '#bb86fc', '14 DÍAS': '#00ffcc'}
SEX_COLORS  = {'MACHO': '#5bc0de', 'HEMBRA': '#e83e8c'}

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: ENTRE CONDICIONES POR SEXO (main new feature)
# ─────────────────────────────────────────────────────────────────────────────
if comparison_mode == "Entre Condiciones (NONE / 3 DÍAS / 14 DÍAS) por Sexo":

    st.subheader(f"📈 {var_options[selected_var_key]} — Comparación por Condición y Sexo")

    # Add sex/condition columns to df_base
    df_base = df_base.copy()
    df_base['sex']       = df_base['group'].map(lambda g: group_meta.get(g, {}).get('sex'))
    df_base['condition'] = df_base['group'].map(lambda g: group_meta.get(g, {}).get('condition'))
    df_base = df_base[df_base['sex'].notna() & df_base['condition'].notna()]

    sexes = [s for s in ['MACHO', 'HEMBRA'] if s in df_base['sex'].unique()]
    if not sexes:
        st.warning("No se detectaron grupos con sexo (MACHO/HEMBRA) en el nombre de carpeta.")
        st.stop()

    # ── Info banner about N calculation ─────────────────────────────────────
    if level_type == "Por Sujeto (animal, promediando cortes)":
        sections_col_exists = 'sections' in df_base.columns
        n_example = len(df_base[df_base['group'] == df_base['group'].iloc[0]]) if len(df_base) > 0 else 0
        st.markdown("""
        <div class="level-info">
        📌 <b>Nivel: Sujeto/Animal — Comparación entre grupos.</b><br>
        Cada animal contribuye <b>exactamente 1 punto de dato</b>, independientemente de cuántos
        hemisferios (IPSI/CONTRA) o cortes haya. Los valores de IPSI y CONTRA se promedian
        antes de comparar entre condiciones. Esto evita el doble conteo intra-sujeto.
        </div>
        """, unsafe_allow_html=True)
    elif level_type == "Por Preparado/Corte (imagen)":
        st.markdown("""
        <div class="level-info">
        📌 <b>Nivel: Preparado/Corte — Comparación entre grupos.</b><br>
        Cada imagen (corte histológico) es un punto independiente. Un animal con 3 cortes en
        IPSI y 3 en CONTRA aporta 6 puntos. Para evitar pseudoreplicación, considera usar
        el nivel <i>Por Sujeto</i>.
        </div>
        """, unsafe_allow_html=True)

    # ── Build one figure per sex ─────────────────────────────────────────────
    sex_cols = st.columns(len(sexes))

    all_pairwise_results = []  # for the summary table

    for col_idx, sex in enumerate(sexes):
        df_sex = df_base[df_base['sex'] == sex]

        # Gather data per condition (only conditions that exist)
        cond_data = {}
        for cond in COND_ORDER:
            vals = df_sex[df_sex['condition'] == cond][selected_var_key].dropna().values
            if len(vals) > 0:
                cond_data[cond] = vals

        if len(cond_data) < 2:
            with sex_cols[col_idx]:
                st.warning(f"Faltan datos para {sex}.")
            continue

        cond_list = [c for c in COND_ORDER if c in cond_data]

        # ── Pairwise Mann-Whitney U ──────────────────────────────────────────
        pairs = [(cond_list[i], cond_list[j])
                 for i in range(len(cond_list)) for j in range(i+1, len(cond_list))]

        alpha_adj = 0.05 / len(pairs) if use_bonferroni else 0.05

        pair_results = []
        for ca, cb in pairs:
            stat, p = run_mwu(cond_data[ca], cond_data[cb])
            p_adj = p * len(pairs) if use_bonferroni and not np.isnan(p) else p
            stars = sig_stars(p_adj if use_bonferroni else p, alpha=0.05)
            pair_results.append({
                'A': ca, 'B': cb,
                'p_raw': p, 'p_adj': p_adj, 'stars': stars
            })
            all_pairwise_results.append({
                'Sexo': sex, 'Comparación': f"{ca} vs {cb}",
                'N_A': len(cond_data[ca]), 'N_B': len(cond_data[cb]),
                'p (raw)': round(p, 4) if not np.isnan(p) else '-',
                'p (adj Bonf.)': round(p_adj, 4) if not np.isnan(p_adj) else '-',
                'Significancia': stars
            })

        # ── Build box+strip figure ───────────────────────────────────────────
        rows = []
        for cond in cond_list:
            for v in cond_data[cond]:
                rows.append({'Condición': cond, selected_var_key: v})
        df_plot = pd.DataFrame(rows)
        df_plot['Condición'] = pd.Categorical(df_plot['Condición'], categories=cond_list, ordered=True)

        fig = go.Figure()

        for cond in cond_list:
            vals = cond_data[cond]
            color = COND_COLORS.get(cond, '#aaaaaa')
            fig.add_trace(go.Box(
                y=vals, name=cond,
                marker_color=color,
                boxmean='sd',
                line=dict(color=color, width=2),
                fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)',
                boxpoints='all',
                jitter=0.35,
                pointpos=0,
                marker=dict(size=6, opacity=0.7, color=color),
                showlegend=True
            ))

        # Add significance bars
        y_max = max(v.max() for v in cond_data.values() if len(v) > 0)
        x_positions = {cond: i for i, cond in enumerate(cond_list)}

        bar_level = 0
        skipped_pairs = []  # pairs with insufficient data
        for pr in sorted(pair_results, key=lambda x: x['A']):
            label = pr['stars']
            if label == 'N/A':
                skipped_pairs.append(f"{pr['A']} vs {pr['B']} (N insuficiente)")
                continue  # don't draw bar
            x0 = x_positions[pr['A']]
            x1 = x_positions[pr['B']]
            bar_color = '#00ff88' if label not in ('ns', 'N/A') else '#666666'
            add_significance_bar(fig, x0, x1, y_max, label, color=bar_color, row_offset=bar_level)
            bar_level += 1

        fig.update_layout(
            title=dict(
                text=f"<b>{sex}</b>",
                font=dict(size=16, color=SEX_COLORS[sex])
            ),
            yaxis_title=var_options[selected_var_key],
            xaxis=dict(title="Condición", categoryorder='array', categoryarray=cond_list),
            template='plotly_dark',
            height=520,
            margin=dict(t=80, b=40),
            showlegend=False,
            yaxis=dict(range=[0, y_max * 1.35])
        )

        with sex_cols[col_idx]:
            st.plotly_chart(fig, use_container_width=True)
            if skipped_pairs:
                st.caption("⚠️ Sin datos suficientes (N<3) para: " + " · ".join(skipped_pairs))

            # Per-sex summary stats
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.markdown(f"**📊 Resumen — {sex}**")
            for cond in cond_list:
                vals = cond_data[cond]
                st.markdown(
                    f"**{cond}** (N={len(vals)}): "
                    f"media = `{np.mean(vals):.3f}`, "
                    f"DE = `{np.std(vals):.3f}`, "
                    f"mediana = `{np.median(vals):.3f}`"
                )
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Global pairwise table ────────────────────────────────────────────────
    st.subheader("🧪 Tabla de Significancias entre Condiciones")

    bonf_note = f"(α ajustado = {0.05/3:.4f} por 3 comparaciones)" if use_bonferroni else "(sin corrección)"
    st.caption(f"Prueba: Mann-Whitney U · Corrección Bonferroni {bonf_note}")

    if all_pairwise_results:
        df_pairwise = pd.DataFrame(all_pairwise_results)

        def color_stars(val):
            if val in ('***', '**'):
                return 'color: #00ff88; font-weight: bold'
            elif val == '*':
                return 'color: #ffcc00; font-weight: bold'
            elif val == 'ns':
                return 'color: #888888'
            return ''

        st.dataframe(
            df_pairwise.style.map(color_stars, subset=['Significancia']),
            use_container_width=True, hide_index=True
        )

    st.divider()

    # ── PANEL COMBINADO: Todos los individuos sin distinción de sexo ──────────
    st.subheader(f"🌐 Comparación Global (Machos + Hembras combinados)")
    st.caption("Todos los sujetos de ambos sexos agrupados por condición experimental.")

    cond_data_all = {}
    for cond in COND_ORDER:
        vals = df_base[df_base['condition'] == cond][selected_var_key].dropna().values
        if len(vals) > 0:
            cond_data_all[cond] = vals

    cond_list_all = [c for c in COND_ORDER if c in cond_data_all]

    if len(cond_list_all) >= 2:
        # Pairwise tests for combined panel
        pairs_all = [(cond_list_all[i], cond_list_all[j])
                     for i in range(len(cond_list_all)) for j in range(i+1, len(cond_list_all))]
        alpha_adj_all = 0.05 / len(pairs_all) if use_bonferroni else 0.05
        pair_results_all = []
        combined_sig_rows = []
        for ca, cb in pairs_all:
            stat, p = run_mwu(cond_data_all[ca], cond_data_all[cb])
            p_adj = p * len(pairs_all) if use_bonferroni and not np.isnan(p) else p
            stars = sig_stars(p_adj if use_bonferroni else p, alpha=0.05)
            pair_results_all.append({'A': ca, 'B': cb, 'p_raw': p, 'p_adj': p_adj, 'stars': stars})
            combined_sig_rows.append({
                'Comparación': f"{ca} vs {cb}",
                'N_A': len(cond_data_all[ca]), 'N_B': len(cond_data_all[cb]),
                'p (raw)': round(p, 4) if not np.isnan(p) else '-',
                'p (adj Bonf.)': round(p_adj, 4) if not np.isnan(p_adj) else '-',
                'Significancia': stars
            })

        col_comb_plot, col_comb_stats = st.columns([3, 1])
        with col_comb_plot:
            fig_all = go.Figure()
            for cond in cond_list_all:
                vals = cond_data_all[cond]
                color = COND_COLORS.get(cond, '#aaaaaa')
                fig_all.add_trace(go.Box(
                    y=vals, name=cond,
                    marker_color=color,
                    boxmean='sd',
                    line=dict(color=color, width=2),
                    fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)',
                    boxpoints='all',
                    jitter=0.35,
                    pointpos=0,
                    marker=dict(size=7, opacity=0.7, color=color),
                    showlegend=True
                ))
            y_max_all = max(v.max() for v in cond_data_all.values() if len(v) > 0)
            x_pos_all = {cond: i for i, cond in enumerate(cond_list_all)}
            bar_lv = 0
            for pr in pair_results_all:
                if pr['stars'] == 'N/A':
                    continue
                bar_color = '#00ff88' if pr['stars'] not in ('ns', 'N/A') else '#666666'
                add_significance_bar(fig_all, x_pos_all[pr['A']], x_pos_all[pr['B']],
                                     y_max_all, pr['stars'], color=bar_color, row_offset=bar_lv)
                bar_lv += 1

            fig_all.update_layout(
                title=dict(text="<b>Todos los sujetos (M+H)</b>", font=dict(size=15, color='#ffffff')),
                yaxis_title=var_options[selected_var_key],
                xaxis=dict(title="Condición", categoryorder='array', categoryarray=cond_list_all),
                template='plotly_dark',
                height=500,
                margin=dict(t=80, b=40),
                showlegend=True,
                yaxis=dict(range=[0, y_max_all * 1.38])
            )
            st.plotly_chart(fig_all, use_container_width=True)

        with col_comb_stats:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.markdown("**📊 Resumen Global**")
            for cond in cond_list_all:
                vals = cond_data_all[cond]
                st.markdown(
                    f"**{cond}** (N={len(vals)}): "
                    f"media = `{np.mean(vals):.3f}`, "
                    f"DE = `{np.std(vals):.3f}`"
                )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.markdown("**🧪 Significancias (global)**")
            for pr in pair_results_all:
                icon = "✅" if pr['stars'] not in ('ns', '?') else "○"
                st.markdown(f"{icon} **{pr['A']} vs {pr['B']}**: `{pr['stars']}`")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No hay suficientes condiciones con datos para el panel global.")

    st.divider()

    # ── Optional: raw data table ─────────────────────────────────────────────
    with st.expander("📋 Ver tabla de datos completa"):
        show_cols = ['group', 'sex', 'condition', 'animal_id', 'sections', 'n_sections', 'n_cortes', selected_var_key]
        show_cols = [c for c in show_cols if c in df_base.columns]
        st.caption("La columna 'sections' indica los hemisferios incluidos en el promedio de cada animal.")
        st.dataframe(df_base[show_cols].sort_values(['sex','condition','animal_id']),
                     use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: IPSI vs CONTRA (hemisferio) — kept from previous implementation
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.subheader(f"📈 {var_options[selected_var_key]} — IPSI vs CONTRA")

    factor_col = "section"
    if factor_col not in df_base.columns:
        st.error("Columna `section` no encontrada.")
        st.stop()

    # Optional group filter
    available_groups = ['Todos'] + sorted(df_base['group'].dropna().unique().tolist())
    sel_grp = st.sidebar.selectbox("Filtrar por Grupo:", available_groups, key="ipsi_group_filter")
    if sel_grp != 'Todos':
        df_base = df_base[df_base['group'] == sel_grp]

    categories = sorted(df_base[factor_col].dropna().unique())
    if len(categories) < 2:
        st.warning(f"Necesitas datos en al menos 2 secciones. Detectadas: {categories}")
        st.stop()

    if len(categories) > 2:
        selected_cats = st.sidebar.multiselect(
            "Selecciona 2 secciones:", categories, default=categories[:2], key="ipsi_cat_select"
        )
        if len(selected_cats) != 2:
            st.error("Selecciona exactamente 2 secciones.")
            st.stop()
        cat_a, cat_b = selected_cats[0], selected_cats[1]
    else:
        cat_a, cat_b = categories[0], categories[1]

    df_a = df_base[df_base[factor_col] == cat_a]
    df_b = df_base[df_base[factor_col] == cat_b]
    data_a = df_a[selected_var_key].dropna().values
    data_b = df_b[selected_var_key].dropna().values

    if len(data_a) < 2 or len(data_b) < 2:
        st.warning("Datos insuficientes para la comparación hemisférica.")
        st.stop()

    # Try paired by animal_id
    paired_data = None
    if level_type != "Por Célula (distribuciones individuales)" and 'animal_id' in df_base.columns:
        va = df_a[['animal_id', selected_var_key]].rename(columns={selected_var_key: 'val_a'})
        vb = df_b[['animal_id', selected_var_key]].rename(columns={selected_var_key: 'val_b'})
        merged = pd.merge(va, vb, on='animal_id').dropna()
        if len(merged) >= 3:
            paired_data = merged

    col_plot, col_stats = st.columns([2, 1])

    with col_plot:
        if paired_data is not None:
            fig = go.Figure()
            for _, row in paired_data.iterrows():
                fig.add_trace(go.Scatter(
                    x=[cat_a, cat_b], y=[row['val_a'], row['val_b']],
                    mode='lines+markers', name=row['animal_id'],
                    line=dict(color='rgba(187,134,252,0.45)', width=2),
                    marker=dict(size=9), showlegend=True
                ))
            fig.add_trace(go.Scatter(
                x=[cat_a], y=[paired_data['val_a'].mean()],
                error_y=dict(type='data', array=[paired_data['val_a'].std()], visible=True),
                mode='markers', marker=dict(color='#00f2fe', size=14, symbol='diamond'),
                name=f'Media {cat_a}'
            ))
            fig.add_trace(go.Scatter(
                x=[cat_b], y=[paired_data['val_b'].mean()],
                error_y=dict(type='data', array=[paired_data['val_b'].std()], visible=True),
                mode='markers', marker=dict(color='#bb86fc', size=14, symbol='diamond'),
                name=f'Media {cat_b}'
            ))
            fig.update_layout(
                title=f"Pareado IPSI ↔ CONTRA por Sujeto",
                xaxis_title="Hemisferio", yaxis_title=var_options[selected_var_key],
                template='plotly_dark', height=480
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df_plot = df_base[df_base[factor_col].isin([cat_a, cat_b])]
            fig = px.box(df_plot, x=factor_col, y=selected_var_key,
                         color=factor_col, points='all',
                         color_discrete_map={cat_a: '#00f2fe', cat_b: '#bb86fc'},
                         template='plotly_dark')
            fig.update_layout(height=480, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Resumen Descriptivo")
        for data, label, color in [(data_a, cat_a, '🔵'), (data_b, cat_b, '🟣')]:
            st.markdown(f"**{color} {label} (N={len(data)}):**")
            st.markdown(f"- Media ± DE: `{np.mean(data):.3f} ± {np.std(data):.3f}`")
            st.markdown(f"- Mediana [IQR]: `{np.median(data):.3f} [{np.percentile(data,25):.3f} – {np.percentile(data,75):.3f}]`")
        diff_pct = ((np.mean(data_b) - np.mean(data_a)) / np.mean(data_a) * 100) if np.mean(data_a) != 0 else 0
        st.markdown(f"**Δ Medias:** `{diff_pct:+.1f}%`")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Statistical test
    st.subheader("🧪 Análisis Estadístico Inferencial")
    if paired_data is not None and len(paired_data) >= 3:
        try:
            stat, p_val = wilcoxon(paired_data['val_a'].values, paired_data['val_b'].values)
            test_name = "Wilcoxon Signed-Rank (pareado intra-sujeto)"
            st.info(f"🔗 Prueba pareada sobre {len(paired_data)} sujetos con datos en ambos hemisferios.")
        except Exception:
            stat, p_val = mannwhitneyu(data_a, data_b, alternative='two-sided')
            test_name = "Mann-Whitney U (fallback)"
    else:
        stat, p_val = mannwhitneyu(data_a, data_b, alternative='two-sided') if len(data_a) >= 3 and len(data_b) >= 3 else (np.nan, np.nan)
        test_name = "Mann-Whitney U"
        if paired_data is None and level_type != "Por Célula (distribuciones individuales)":
            st.warning("⚠️ No hay sujetos pareados. Se usa Mann-Whitney U no pareada.")

    stars = sig_stars(p_val)
    st.markdown(f"**Prueba:** `{test_name}`")
    st.markdown(f"- Estadístico: `{stat:.4f}` | p-valor: `{p_val:.4f}` | **{stars}**")

    if not np.isnan(p_val):
        css_class = "significance-sig" if p_val < 0.05 else "significance-nonsig"
        icon = "✅" if p_val < 0.05 else "⚠️"
        msg = "DIFERENCIA SIGNIFICATIVA" if p_val < 0.05 else "SIN DIFERENCIA SIGNIFICATIVA"
        st.markdown(f'<div class="{css_class}">{icon} {msg} — {stars} (p={p_val:.4f})</div>',
                    unsafe_allow_html=True)

    st.divider()

    if level_type == "Por Sujeto (animal, promediando cortes)":
        st.subheader("👤 Tabla por Sujeto")
        show_cols = ['group', 'section', 'animal_id', 'n_cortes', selected_var_key]
        show_cols = [c for c in show_cols if c in df_base.columns]
        df_show = df_base[df_base[factor_col].isin([cat_a, cat_b])][show_cols].sort_values(['group','section','animal_id'])
        st.dataframe(df_show, use_container_width=True)

        if paired_data is not None:
            st.subheader("🔗 Pares IPSI ↔ CONTRA")
            ps = paired_data.rename(columns={'animal_id': 'Sujeto', 'val_a': cat_a, 'val_b': cat_b}).copy()
            ps['Δ'] = ps[cat_b] - ps[cat_a]
            ps['Δ %'] = ((ps[cat_b] - ps[cat_a]) / ps[cat_a].replace(0, np.nan) * 100).round(1)
            st.dataframe(ps, use_container_width=True)
    else:
        show_cols = [factor_col, 'group', 'animal_id', selected_var_key]
        show_cols = [c for c in show_cols if c in df_base.columns]
        st.dataframe(df_base[df_base[factor_col].isin([cat_a, cat_b])][show_cols], use_container_width=True)
