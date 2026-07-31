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

st.title("📊 Paso 4: Comparación Estadística")

METRICS_BASE_DIR = "data/processed/metrics"

# Helper to normalize group properties
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
    elif '14' in name_norm and 'DIA' in name_norm:
        cond, order = '14 DÍAS', 2
    else:
        cond, order = name, 99

    return sex, cond, order

# Utility for p-value to stars
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
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    return stat, p

def add_significance_bar(fig, x0, x1, y_top, label, color='#ffffff', row_offset=0):
    bar_y = y_top * (1.05 + row_offset * 0.12)
    tick_h = y_top * 0.015
    fig.add_shape(type='line', x0=x0, x1=x1, y0=bar_y, y1=bar_y,
                  line=dict(color=color, width=1.5))
    fig.add_shape(type='line', x0=x0, x1=x0, y0=bar_y - tick_h, y1=bar_y,
                  line=dict(color=color, width=1.5))
    fig.add_shape(type='line', x0=x1, x1=x1, y0=bar_y - tick_h, y1=bar_y,
                  line=dict(color=color, width=1.5))
    fig.add_annotation(
        x=(x0 + x1) / 2, y=bar_y + tick_h * 0.5,
        text=label, showarrow=False,
        font=dict(size=13, color=color if label != 'ns' else '#888888'),
        yref='y', xref='x'
    )
    return bar_y

COND_ORDER  = ['NONE', '3 DÍAS', '14 DÍAS']
COND_COLORS = {'NONE': '#4facfe', '3 DÍAS': '#bb86fc', '14 DÍAS': '#00ffcc'}
SEX_COLORS  = {'MACHO': '#5bc0de', 'HEMBRA': '#e83e8c'}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD NUCLEI AND DAPI METRICS
# ─────────────────────────────────────────────────────────────────────────────
import pickle

@st.cache_data(ttl=3600)
def load_all_metrics_cached():
    cache_file = os.path.join(METRICS_BASE_DIR, "stats_cache.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                payload = pickle.load(f)
                return payload.get("df_raw_nuclei", pd.DataFrame()), payload.get("df_raw_dapi", pd.DataFrame())
        except Exception:
            pass
            
    def _read_folder(suffix='_nuclei_metrics.csv'):
        all_dfs = []
        if not os.path.exists(METRICS_BASE_DIR):
            return pd.DataFrame()
        for root, dirs, files in os.walk(METRICS_BASE_DIR):
            if "test" in root:
                continue
            for f in files:
                if not f.endswith(suffix):
                    continue
                csv_path = os.path.join(root, f)
                rel_path = os.path.relpath(csv_path, METRICS_BASE_DIR)
                parts = rel_path.split(os.sep)
                if len(parts) >= 3:
                    group    = parts[0]
                    section  = parts[1]
                    filename = parts[2].replace(suffix, '')
                else:
                    group = section = "Desconocido"
                    filename = f.replace(suffix, '')
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
        
    return _read_folder('_nuclei_metrics.csv'), _read_folder('_dapi_metrics.csv')

df_raw_nuclei, df_raw_dapi = load_all_metrics_cached()


if df_raw_nuclei.empty and df_raw_dapi.empty:
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
    numeric_cols = [c for c in df_img.columns
                    if c not in ['group','section','image_name','animal_id','corte_num']
                    and pd.api.types.is_numeric_dtype(df_img[c])]
    summaries = []
    for vals, grp in df_img.groupby(['group','animal_id']):
        base = dict(zip(['group','animal_id'], vals))
        base['sections'] = '+'.join(sorted(grp['section'].dropna().unique().tolist()))
        base['n_sections'] = grp['section'].nunique()
        base['n_cortes']   = len(grp)
        for col in numeric_cols:
            base[col] = grp[col].mean()
        summaries.append(base)
    return pd.DataFrame(summaries)

if not df_raw_nuclei.empty:
    df_img        = aggregate_image_level(df_raw_nuclei)
    df_subj       = aggregate_subject_level(df_img)
    df_subj_btwn  = aggregate_subject_level_between(df_img)
else:
    df_img = pd.DataFrame()
    df_subj = pd.DataFrame()
    df_subj_btwn = pd.DataFrame()

# Sidebar controls
st.sidebar.header("⚙️ Configuración del Análisis")
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

# ─────────────────────────────────────────────────────────────────────────────
# TABULACIÓN POR PASO
# ─────────────────────────────────────────────────────────────────────────────
tab_dapi, tab_pv, tab_pnn, tab_lupori, tab_global_wfa = st.tabs([
    "🧬 Núcleos DAPI (Paso 1)",
    "🧪 Interneuronas PV+ (Paso 2)",
    "🧠 Redes Perineuronales PNN (Paso 3)",
    "⚡ Métricas Lupori (Potencia y Coexpresión 24/07)",
    "🌐 Señal Global Integrada WFA"
])


def run_stats_layout(df_base, var_options, selected_var_key, title_lbl):
    # Set group metadata
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

    # Desglose estricto por Sexo (MACHO vs HEMBRA)
    sexes_in_data = [s for s in ['MACHO', 'HEMBRA'] if s in df_base['sex'].unique()]
    if not sexes_in_data:
        sexes_in_data = df_base['sex'].dropna().unique().tolist()
        
    sex_groups = [(s, df_base[df_base['sex'] == s]) for s in sexes_in_data]
    cols = st.columns(len(sex_groups))

    all_intra_results = []
    all_inter_results = []
    all_cross_results = []

    SECTION_COLORS = {'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'}

    for col_idx, (group_title, df_sub) in enumerate(sex_groups):
        if df_sub.empty:
            continue

        df_sub = df_sub.copy()
        df_sub['condition'] = pd.Categorical(df_sub['condition'], categories=COND_ORDER, ordered=True)
        df_plot = df_sub.sort_values('condition')

        fig = go.Figure()

        present_conds = [c for c in COND_ORDER if c in df_plot['condition'].unique()]
        present_secs = [s for s in ['IPSI', 'CONTRA'] if s in df_plot['section'].unique()]

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
                line=dict(color=color, width=2)
            ))

        fig.update_layout(
            title=dict(text=f"<b>{group_title} — IPSI vs CONTRA por Condición</b>", font=dict(size=16, color=SEX_COLORS.get(group_title, '#bb86fc'))),
            yaxis_title=var_options[selected_var_key],
            xaxis_title="Condición Experimental",
            boxmode='group',
            template='plotly_dark',
            height=480,
            legend=dict(title="Hemisferio", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        with cols[col_idx]:
            st.plotly_chart(fig, use_container_width=True)

        # ───────── CALCULAR ESTADÍSTICAS SIN PROMEDIAR SEXOS NI HEMISFERIOS ─────────
        # 1. INTRA-CONDICIÓN: IPSI vs CONTRA en cada condición (Mismo Sexo)
        for cond in present_conds:
            df_cond = df_sub[df_sub['condition'] == cond]
            val_ipsi = df_cond[df_cond['section'] == 'IPSI'][selected_var_key].dropna().values
            val_contra = df_cond[df_cond['section'] == 'CONTRA'][selected_var_key].dropna().values

            if len(val_ipsi) >= 2 and len(val_contra) >= 2:
                # Prueba pareada si hay animal_id pareado
                if 'animal_id' in df_cond.columns and level_type != "Por Célula (distribuciones individuales)":
                    merged_p = pd.merge(
                        df_cond[df_cond['section'] == 'IPSI'][['animal_id', selected_var_key]].rename(columns={selected_var_key: 'v_ipsi'}),
                        df_cond[df_cond['section'] == 'CONTRA'][['animal_id', selected_var_key]].rename(columns={selected_var_key: 'v_contra'}),
                        on='animal_id'
                    ).dropna()
                    if len(merged_p) >= 3:
                        stat, p = wilcoxon(merged_p['v_ipsi'].values, merged_p['v_contra'].values)
                        test_type = "Wilcoxon Pareado"
                    else:
                        stat, p = run_mwu(val_ipsi, val_contra)
                        test_type = "Mann-Whitney U"
                else:
                    stat, p = run_mwu(val_ipsi, val_contra)
                    test_type = "Mann-Whitney U"

                stars = sig_stars(p)
                all_intra_results.append({
                    'Sexo': group_title,
                    'Condición': cond,
                    'IPSI (Media ± DE)': f"{np.mean(val_ipsi):.2f} ± {np.std(val_ipsi):.2f} (N={len(val_ipsi)})",
                    'CONTRA (Media ± DE)': f"{np.mean(val_contra):.2f} ± {np.std(val_contra):.2f} (N={len(val_contra)})",
                    'Prueba': test_type,
                    'Estadístico (W/U)': f"{stat:.2f}" if not np.isnan(stat) else '-',
                    'p-valor': round(p, 4) if not np.isnan(p) else '-',
                    'Significancia': stars
                })

        # 2. INTER-CONDICIÓN: NONE vs 3 DÍAS vs 14 DÍAS dentro de IPSI y dentro de CONTRA
        for sec in present_secs:
            df_sec = df_sub[df_sub['section'] == sec]
            cond_data = {}
            for c in COND_ORDER:
                vals = df_sec[df_sec['condition'] == c][selected_var_key].dropna().values
                if len(vals) > 0:
                    cond_data[c] = vals

            cond_present = [c for c in COND_ORDER if c in cond_data]
            pairs = [(cond_present[i], cond_present[j]) for i in range(len(cond_present)) for j in range(i+1, len(cond_present))]

            for ca, cb in pairs:
                stat, p = run_mwu(cond_data[ca], cond_data[cb])
                p_adj = p * len(pairs) if use_bonferroni and not np.isnan(p) else p
                stars = sig_stars(p_adj if use_bonferroni else p)
                all_inter_results.append({
                    'Sexo': group_title,
                    'Hemisferio': sec,
                    'Comparación': f"{ca} vs {cb}",
                    'Media A': f"{np.mean(cond_data[ca]):.2f} (N={len(cond_data[ca])})",
                    'Media B': f"{np.mean(cond_data[cb]):.2f} (N={len(cond_data[cb])})",
                    'p-valor (raw)': round(p, 4) if not np.isnan(p) else '-',
                    'p-valor (adj Bonf.)': round(p_adj, 4) if not np.isnan(p_adj) else '-',
                    'Significancia': stars
                })

        # 3. COMPARACIONES CRUZADAS (IPSI de condición A vs CONTRA de condición B)
        cond_sec_pairs = []
        for c in present_conds:
            for s in present_secs:
                vals = df_sub[(df_sub['condition'] == c) & (df_sub['section'] == s)][selected_var_key].dropna().values
                if len(vals) > 0:
                    cond_sec_pairs.append((f"{c} ({s})", vals))

        for i in range(len(cond_sec_pairs)):
            for j in range(i+1, len(cond_sec_pairs)):
                label_a, vals_a = cond_sec_pairs[i]
                label_b, vals_b = cond_sec_pairs[j]
                stat, p = run_mwu(vals_a, vals_b)
                p_adj = p * (len(cond_sec_pairs)*(len(cond_sec_pairs)-1)/2) if use_bonferroni and not np.isnan(p) else p
                stars = sig_stars(p_adj if use_bonferroni else p)
                all_cross_results.append({
                    'Sexo': group_title,
                    'Grupo A': label_a,
                    'Grupo B': label_b,
                    'Media A': f"{np.mean(vals_a):.2f} (N={len(vals_a)})",
                    'Media B': f"{np.mean(vals_b):.2f} (N={len(vals_b)})",
                    'p-valor (raw)': round(p, 4) if not np.isnan(p) else '-',
                    'p-valor (adj Bonf.)': round(p_adj, 4) if not np.isnan(p_adj) else '-',
                    'Significancia': stars
                })

    # Render Tables
    st.markdown("---")
    st.markdown("### 📊 Pruebas de Significancia Estadística (Macho vs Hembra Separados)")

    t_col1, t_col2, t_col3 = st.tabs([
        "🔴 Intra-Grupo: IPSI vs CONTRA (en cada condición)",
        "🔵 Inter-Grupo: Evolución Temporal (por Hemisferio)",
        "🌐 Comparaciones Cruzadas (IPSI de A vs CONTRA de B)"
    ])

    with t_col1:
        st.markdown("#### ⚖️ Comparación Intra-Grupo (IPSI vs CONTRA dentro de cada condición)")
        if all_intra_results:
            df_intra = pd.DataFrame(all_intra_results)
            st.dataframe(df_intra, use_container_width=True, hide_index=True)
        else:
            st.info("No hay suficientes datos para realizar pruebas intra-grupo.")

    with t_col2:
        st.markdown("#### ⏳ Comparación Inter-Grupo entre Condiciones (mismo hemisferio)")
        if all_inter_results:
            df_inter = pd.DataFrame(all_inter_results)
            st.dataframe(df_inter, use_container_width=True, hide_index=True)
        else:
            st.info("No hay suficientes datos para realizar pruebas inter-grupo.")

    with t_col3:
        st.markdown("#### 🌐 Matriz Completa de Comparaciones Cruzadas (Condición x Hemisferio)")
        if all_cross_results:
            df_cross = pd.DataFrame(all_cross_results)
            st.dataframe(df_cross, use_container_width=True, hide_index=True)
        else:
            st.info("No hay suficientes datos para comparaciones cruzadas.")


# 1. TAB DAPI
with tab_dapi:
    if df_raw_dapi.empty:
        st.info("No hay datos de DAPI disponibles.")
    else:
        st.header("Análisis de Núcleos DAPI (Paso 1)")
        ALL_DAPI_VARS = {
            "dapi_count": "N° de núcleos DAPI por corte",
            "mean_dapi_area_um2": "Área Promedio del Núcleo DAPI (µm²)",
            "mean_dapi_diameter_um": "Diámetro Promedio del Núcleo DAPI (µm)",
            "mean_dapi_intensity": "Intensidad DAPI Promedio",
            "area_um2": "Área del Núcleo DAPI (µm²) — por Célula",
            "diameter_um": "Diámetro del Núcleo DAPI (µm) — por Célula",
            "dapi_mean_intensity": "Intensidad Media DAPI — por Célula"
        }
        
        selected_dapi_var = st.selectbox("Variable DAPI:", list(ALL_DAPI_VARS.keys()), format_func=lambda x: ALL_DAPI_VARS[x], key="dapi_var")

        # Aggregate DAPI dynamically based on selected variable
        is_cell_var = selected_dapi_var in ["area_um2", "diameter_um", "dapi_mean_intensity"]
        if is_cell_var and level_type == "Por Célula (distribuciones individuales)":
            df_dapi_base = df_raw_dapi.copy()
        elif level_type == "Por Preparado/Corte (imagen)":
            def _dapi_img(grp):
                return pd.Series({
                    'dapi_count': len(grp),
                    'mean_dapi_area_um2': grp['area_um2'].mean(),
                    'mean_dapi_diameter_um': grp['diameter_um'].mean(),
                    'mean_dapi_intensity': grp['dapi_mean_intensity'].mean(),
                    'area_um2': grp['area_um2'].mean(),
                    'diameter_um': grp['diameter_um'].mean(),
                    'dapi_mean_intensity': grp['dapi_mean_intensity'].mean()
                })
            df_dapi_base = df_raw_dapi.groupby(['group','section','image_name','animal_id','corte_num']).apply(_dapi_img).reset_index()
        else:
            def _dapi_img(grp):
                return pd.Series({
                    'dapi_count': len(grp),
                    'mean_dapi_area_um2': grp['area_um2'].mean(),
                    'mean_dapi_diameter_um': grp['diameter_um'].mean(),
                    'mean_dapi_intensity': grp['dapi_mean_intensity'].mean(),
                    'area_um2': grp['area_um2'].mean(),
                    'diameter_um': grp['diameter_um'].mean(),
                    'dapi_mean_intensity': grp['dapi_mean_intensity'].mean()
                })
            df_dapi_img = df_raw_dapi.groupby(['group','section','image_name','animal_id','corte_num']).apply(_dapi_img).reset_index()
            df_dapi_base = df_dapi_img.groupby(['group', 'section', 'animal_id']).mean(numeric_only=True).reset_index()

        run_stats_layout(df_dapi_base, ALL_DAPI_VARS, selected_dapi_var, "Núcleos DAPI")

# 2. TAB PV+
with tab_pv:
    if df_raw_nuclei.empty:
        st.info("No hay datos de PV+ disponibles.")
    else:
        st.header("Análisis de Interneuronas PV+ (Paso 2)")
        ALL_PV_VARS = {
            "pv_count": "N° de Somas PV+ por corte",
            "mean_pv_area_um2": "Área Promedio Soma PV+ (µm²)",
            "mean_pv_diameter_um": "Diámetro Promedio Soma PV+ (µm)",
            "pv_area_um2": "Área del Soma PV+ (µm²) — por Célula",
            "pv_diameter_um": "Diámetro del Soma PV+ (µm) — por Célula"
        }
        
        selected_pv_var = st.selectbox("Variable PV+:", list(ALL_PV_VARS.keys()), format_func=lambda x: ALL_PV_VARS[x], key="pv_var")

        is_cell_var = selected_pv_var in ["pv_area_um2", "pv_diameter_um"]
        if is_cell_var and level_type == "Por Célula (distribuciones individuales)":
            df_pv_base = df_raw_nuclei[df_raw_nuclei['is_pv_plus'] == True].copy()
        elif level_type == "Por Preparado/Corte (imagen)" or not is_cell_var:
            df_pv_base = df_img.copy()
            if selected_pv_var == "pv_area_um2": df_pv_base['pv_area_um2'] = df_pv_base['mean_pv_area_um2']
            if selected_pv_var == "pv_diameter_um": df_pv_base['pv_diameter_um'] = df_pv_base['mean_pv_diameter_um']
        else:
            df_pv_base = df_subj.copy()
            if selected_pv_var == "pv_area_um2": df_pv_base['pv_area_um2'] = df_pv_base['mean_pv_area_um2']
            if selected_pv_var == "pv_diameter_um": df_pv_base['pv_diameter_um'] = df_pv_base['mean_pv_diameter_um']

        run_stats_layout(df_pv_base, ALL_PV_VARS, selected_pv_var, "Interneuronas PV+")

# 3. TAB PNN
with tab_pnn:
    if df_raw_nuclei.empty:
        st.info("No hay datos de PNN disponibles.")
    else:
        st.header("Análisis de Redes Perineuronales PNN (Paso 3)")
        ALL_PNN_VARS = {
            "pnn_count": "N° de Redes PNN+ Totales",
            "pnn_count_filled": "N° de PNN+ Ocupadas (PV+/PNN+)",
            "pnn_count_hollow": "N° de PNN+ Huecas (PNN+/PV-)",
            "pct_pnn_plus": "% de PV+ con Red PNN (Coexpresión)",
            "pct_pnn_hollow": "% de PNN+ que son Huecas",
            "mean_pnn_area_um2": "Área Promedio de PNN (µm²)",
            "mean_pnn_diameter_um": "Diámetro Promedio de PNN (µm)",
            "mean_score": "Confianza Promedio (PNNscore)",
            "pnn_area_um2": "Área de PNN (µm²) — por Célula",
            "pnn_diameter_um": "Diámetro de PNN (µm) — por Célula",
            "score": "Confianza PNNscore (IA) — por Célula"
        }

        selected_pnn_var = st.selectbox("Variable PNN:", list(ALL_PNN_VARS.keys()), format_func=lambda x: ALL_PNN_VARS[x], key="pnn_var")

        is_cell_var = selected_pnn_var in ["pnn_area_um2", "pnn_diameter_um", "score"]
        if is_cell_var and level_type == "Por Célula (distribuciones individuales)":
            df_pnn_base = df_raw_nuclei[df_raw_nuclei['is_pnn_plus'] == True].copy()
        elif level_type == "Por Preparado/Corte (imagen)" or not is_cell_var:
            df_pnn_base = df_img.copy()
            if selected_pnn_var == "pnn_area_um2": df_pnn_base['pnn_area_um2'] = df_pnn_base['mean_pnn_area_um2']
            if selected_pnn_var == "pnn_diameter_um": df_pnn_base['pnn_diameter_um'] = df_pnn_base['mean_pnn_diameter_um']
            if selected_pnn_var == "score": df_pnn_base['score'] = df_pnn_base['mean_score']
        else:
            df_pnn_base = df_subj.copy()
            if selected_pnn_var == "pnn_area_um2": df_pnn_base['pnn_area_um2'] = df_pnn_base['mean_pnn_area_um2']
            if selected_pnn_var == "pnn_diameter_um": df_pnn_base['pnn_diameter_um'] = df_pnn_base['mean_pnn_diameter_um']
            if selected_pnn_var == "score": df_pnn_base['score'] = df_pnn_base['mean_score']

        run_stats_layout(df_pnn_base, ALL_PNN_VARS, selected_pnn_var, "Redes Perineuronales PNN")

# 4. TAB LUPORI METRICS (POTENCIA, DENSIDAD, ENERGÍA Y COEXPRESIÓN 24/07)
with tab_lupori:
    st.header("⚡ Cuantificación Método Lupori et al. (2023)")
    st.markdown("""
    **Métricas Oficiales:**
    * **Densidad (Density):** $N^\circ \text{de células o PNNs / mm}^2$
    * **Potencia (Energy):** $\text{Densidad} \times \text{Intensidad Promedio Normalizada (0-1)} = \frac{\sum \text{intensity}_i}{\text{Área mm}^2}$
    * **Intensidad Circundante WFA:** Intensidad medida en el anillo de expansión perineuronal ($3\text{--}5\,\mu\text{m}$) alrededor del soma.
    * **Coexpresión:** $\%$ de células PV+ que están rodeadas por una red PNN+.
    """)

    cons_csv = os.path.join(METRICS_BASE_DIR, "consolidated_lupori_metrics.csv")
    if os.path.exists(cons_csv):
        try:
            df_lup = pd.read_csv(cons_csv)
            if 'animal_id' not in df_lup.columns:
                df_lup['animal_id'] = df_lup['filename'].apply(lambda fn: re.match(r'(ACF_\d+)', str(fn)).group(1) if re.match(r'(ACF_\d+)', str(fn)) else str(fn).split('~')[0])
            
            st.success(f"✅ Carga instantánea de la cuantificación Lupori con **{len(df_lup)}** preparados.")
            
            LUPORI_VARS = {
                "pnn_energy": "Potencia PNN (Energy)",
                "coloc_energy": "Potencia Coexpresión PV+/PNN+ (Energy)",
                "pnn_density_mm2": "Densidad de PNN+ (redes / mm²)",
                "pv_density_mm2": "Densidad de PV+ (células / mm²)",
                "coloc_density_mm2": "Densidad de Coexpresión (células PV+/PNN+ / mm²)",
                "pct_pv_surrounded_by_pnn": "% PV+ rodeadas por PNN+ (Coexpresión)",
                "mean_pnn_pericellular_wfa_norm": "Intensidad WFA Circundante (Norm 0-1)",
                "diffuse_wfa_fluorescence": "Fluorescencia Difusa WFA (Norm 0-1)"
            }
            
            sel_lup_var = st.selectbox("Seleccionar Métricas Lupori:", list(LUPORI_VARS.keys()), format_func=lambda x: LUPORI_VARS[x], key="lup_var")
            
            run_stats_layout(df_lup_base, LUPORI_VARS, sel_lup_var, "Métricas Lupori et al.")
            
            st.markdown("### 📋 Tabla Completa Consolidada Lupori (24/07):")
            st.dataframe(df_lup)
        except Exception as e:
            st.error(f"Error al cargar tabla consolidada Lupori: {e}")
    else:
        st.info("⚠️ La cuantificación en lote `consolidated_lupori_metrics.csv` no se encuentra. Ejecuta `uv run python batch_processing.py` para construirla.")

# 5. TAB SEÑAL GLOBAL INTEGRADA WFA
with tab_global_wfa:
    st.header("🌐 Señal Global Integrada de WFA")
    st.markdown("""
    **Medición de Fluorescencia Global Integrada:**
    * **Señal Global Integrada WFA ($\sum \text{Intensidad}$):** Suma total de fluorescencia acumulada del canal WFA a través de todo el tejido del preparado.
    * **Densidad de Señal Integrada por $\text{mm}^2$:** Intensidad WFA integrada dividida entre el área total del preparado en $\text{mm}^2$.
    * **Fluorescencia Difusa/Global WFA:** Intensidad promedio global normalizada ($0\text{--}1$) en el canal WFA.
    """)

    cons_csv = os.path.join(METRICS_BASE_DIR, "consolidated_lupori_metrics.csv")
    if os.path.exists(cons_csv):
        try:
            df_gwfa = pd.read_csv(cons_csv)
            if 'animal_id' not in df_gwfa.columns:
                df_gwfa['animal_id'] = df_gwfa['filename'].apply(lambda fn: re.match(r'(ACF_\d+)', str(fn)).group(1) if re.match(r'(ACF_\d+)', str(fn)) else str(fn).split('~')[0])

            GLOBAL_WFA_VARS = {
                "total_integrated_wfa_signal": "Señal Global Integrada WFA (Suma Total Intensidad)",
                "integrated_wfa_density_mm2": "Densidad de Señal Integrada WFA por mm²",
                "mean_wfa_intensity_raw": "Intensidad Media Global WFA (Raw)",
                "diffuse_wfa_fluorescence": "Fluorescencia Difusa/Global WFA (Norm 0-1)"
            }

            # Filter present columns or compute fallback
            avail_wfa_vars = {k: v for k, v in GLOBAL_WFA_VARS.items() if k in df_gwfa.columns}
            if not avail_wfa_vars:
                avail_wfa_vars = {"diffuse_wfa_fluorescence": "Fluorescencia Difusa/Global WFA (Norm 0-1)"}

            sel_gwfa_var = st.selectbox("Seleccionar Métrica de Señal Global WFA:", list(avail_wfa_vars.keys()), format_func=lambda x: avail_wfa_vars[x], key="gwfa_var")

            if level_type == "Por Sujeto (animal, promediando cortes)":
                df_gwfa_base = df_gwfa.groupby(['group', 'section', 'animal_id']).mean(numeric_only=True).reset_index()
            else:
                df_gwfa_base = df_gwfa.copy()

            run_stats_layout(df_gwfa_base, avail_wfa_vars, sel_gwfa_var, "Señal Global Integrada WFA")

            st.markdown("### 📋 Tabla Consolidada de Señal Global WFA:")
            st.dataframe(df_gwfa)
        except Exception as e:
            st.error(f"Error al cargar métricas de Señal Global WFA: {e}")
    else:
        st.info("⚠️ La cuantificación en lote `consolidated_lupori_metrics.csv` no se encuentra. Ejecuta `uv run python batch_processing.py` para construirla.")




