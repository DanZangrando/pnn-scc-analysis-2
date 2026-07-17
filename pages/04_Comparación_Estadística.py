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
@st.cache_data(ttl=10)
def load_metrics(suffix='_nuclei_metrics.csv'):
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

df_raw_nuclei = load_metrics('_nuclei_metrics.csv')
df_raw_dapi = load_metrics('_dapi_metrics.csv')

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
use_bonferroni = st.sidebar.checkbox(
    "Corrección Bonferroni (×3 comparaciones)",
    value=True, key="stats_bonferroni"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABULACIÓN POR PASO
# ─────────────────────────────────────────────────────────────────────────────
tab_dapi, tab_pv, tab_pnn = st.tabs(["🧬 Núcleos DAPI (Paso 1)", "🧪 Interneuronas PV+ (Paso 2)", "🧠 Redes Perineuronales PNN (Paso 3)"])

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
    df_base = df_base[df_base['sex'].notna() & df_base['condition'].notna()]

    if comparison_mode == "Entre Condiciones (NONE / 3 DÍAS / 14 DÍAS) por Sexo":
        st.subheader(f"📈 {var_options[selected_var_key]} — {title_lbl}")
        
        sexes = [s for s in ['MACHO', 'HEMBRA'] if s in df_base['sex'].unique()]
        if not sexes:
            st.warning("No se detectaron grupos con sexo (MACHO/HEMBRA) en los datos.")
            return

        if level_type == "Por Sujeto (animal, promediando cortes)":
            st.markdown("""
            <div class="level-info">
            📌 <b>Nivel: Sujeto/Animal — Comparación entre grupos.</b><br>
            Cada animal contribuye con 1 punto de dato. Los valores de IPSI y CONTRA se promedian antes de comparar.
            </div>
            """, unsafe_allow_html=True)

        sex_cols = st.columns(len(sexes))
        all_pairwise_results = []

        for col_idx, sex in enumerate(sexes):
            df_sex = df_base[df_base['sex'] == sex]
            cond_data = {}
            for cond in COND_ORDER:
                vals = df_sex[df_sex['condition'] == cond][selected_var_key].dropna().values
                if len(vals) > 0:
                    cond_data[cond] = vals

            if len(cond_data) < 2:
                with sex_cols[col_idx]:
                    st.warning(f"Faltan datos suficientes para {sex}.")
                continue

            cond_list = [c for c in COND_ORDER if c in cond_data]
            pairs = [(cond_list[i], cond_list[j]) for i in range(len(cond_list)) for j in range(i+1, len(cond_list))]
            alpha_adj = 0.05 / len(pairs) if use_bonferroni else 0.05

            pair_results = []
            for ca, cb in pairs:
                stat, p = run_mwu(cond_data[ca], cond_data[cb])
                p_adj = p * len(pairs) if use_bonferroni and not np.isnan(p) else p
                stars = sig_stars(p_adj if use_bonferroni else p, alpha=0.05)
                pair_results.append({'A': ca, 'B': cb, 'p_raw': p, 'p_adj': p_adj, 'stars': stars})
                all_pairwise_results.append({
                    'Sexo': sex, 'Comparación': f"{ca} vs {cb}",
                    'N_A': len(cond_data[ca]), 'N_B': len(cond_data[cb]),
                    'p (raw)': round(p, 4) if not np.isnan(p) else '-',
                    'p (adj Bonf.)': round(p_adj, 4) if not np.isnan(p_adj) else '-',
                    'Significancia': stars
                })

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
                    showlegend=False
                ))

            y_max = max(v.max() for v in cond_data.values() if len(v) > 0)
            x_positions = {cond: i for i, cond in enumerate(cond_list)}
            bar_level = 0
            for pr in pair_results:
                if pr['stars'] == 'N/A': continue
                bar_color = '#00ff88' if pr['stars'] not in ('ns', 'N/A') else '#666666'
                add_significance_bar(fig, x_positions[pr['A']], x_positions[pr['B']], y_max, pr['stars'], color=bar_color, row_offset=bar_level)
                bar_level += 1

            fig.update_layout(
                title=dict(text=f"<b>{sex}</b>", font=dict(size=16, color=SEX_COLORS[sex])),
                yaxis_title=var_options[selected_var_key],
                xaxis=dict(title="Condición"),
                template='plotly_dark',
                height=450,
                yaxis=dict(range=[0, y_max * 1.35])
            )
            with sex_cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                for cond in cond_list:
                    vals = cond_data[cond]
                    st.write(f"**{cond}** (N={len(vals)}): Media=`{np.mean(vals):.2f}` ± `{np.std(vals):.2f}`")
                st.markdown('</div>', unsafe_allow_html=True)

        if all_pairwise_results:
            st.write("### Tabla de Significancias")
            st.dataframe(pd.DataFrame(all_pairwise_results), use_container_width=True, hide_index=True)

    else:
        # IPSI vs CONTRA Mode
        st.subheader(f"📈 {var_options[selected_var_key]} — Hemisferios IPSI vs CONTRA")
        factor_col = "section"
        categories = sorted(df_base[factor_col].dropna().unique())
        if len(categories) < 2:
            st.warning("Se necesitan al menos IPSI y CONTRA en las subcarpetas para comparar.")
            return

        cat_a, cat_b = "IPSI", "CONTRA"
        if "IPSI" not in categories or "CONTRA" not in categories:
            cat_a, cat_b = categories[0], categories[1]

        df_a = df_base[df_base[factor_col] == cat_a]
        df_b = df_base[df_base[factor_col] == cat_b]
        data_a = df_a[selected_var_key].dropna().values
        data_b = df_b[selected_var_key].dropna().values

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
                        marker=dict(size=9)
                    ))
                fig.update_layout(title="Pareado IPSI ↔ CONTRA por Sujeto", template='plotly_dark', height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                df_plot = df_base[df_base[factor_col].isin([cat_a, cat_b])]
                fig = px.box(df_plot, x=factor_col, y=selected_var_key, color=factor_col, points='all', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.write(f"**🔵 {cat_a}** (N={len(data_a)}): Media=`{np.mean(data_a):.2f}` ± `{np.std(data_a):.2f}`")
            st.write(f"**🟣 {cat_b}** (N={len(data_b)}): Media=`{np.mean(data_b):.2f}` ± `{np.std(data_b):.2f}`")
            st.markdown('</div>', unsafe_allow_html=True)

            if paired_data is not None and len(paired_data) >= 3:
                stat, p_val = wilcoxon(paired_data['val_a'].values, paired_data['val_b'].values)
                st.success(f"Prueba Wilcoxon (Pareado): p-val = `{p_val:.4f}` ({sig_stars(p_val)})")
            else:
                stat, p_val = run_mwu(data_a, data_b)
                st.success(f"Prueba Mann-Whitney U: p-val = `{p_val:.4f}` ({sig_stars(p_val)})")

# 1. TAB DAPI
with tab_dapi:
    if df_raw_dapi.empty:
        st.info("No hay datos de DAPI disponibles.")
    else:
        st.header("Análisis de Núcleos DAPI (Paso 1)")
        CELL_DAPI_VARS = {
            "area_um2": "Área del Núcleo DAPI (µm²)",
            "diameter_um": "Diámetro del Núcleo DAPI (µm)",
            "dapi_mean_intensity": "Intensidad Media DAPI"
        }
        IMG_DAPI_VARS = {
            "dapi_count": "N° de núcleos DAPI por corte",
            "mean_dapi_area_um2": "Área Promedio del Núcleo DAPI (µm²)",
            "mean_dapi_diameter_um": "Diámetro Promedio del Núcleo DAPI (µm)",
            "mean_dapi_intensity": "Intensidad DAPI Promedio"
        }
        
        dapi_vars = CELL_DAPI_VARS if level_type == "Por Célula (distribuciones individuales)" else IMG_DAPI_VARS
        selected_dapi_var = st.selectbox("Variable DAPI:", list(dapi_vars.keys()), format_func=lambda x: dapi_vars[x], key="dapi_var")

        # Aggregate DAPI
        if level_type == "Por Célula (distribuciones individuales)":
            df_dapi_base = df_raw_dapi.copy()
        elif level_type == "Por Preparado/Corte (imagen)":
            # Aggregate DAPI by image
            def _dapi_img(grp):
                return pd.Series({
                    'dapi_count': len(grp),
                    'mean_dapi_area_um2': grp['area_um2'].mean(),
                    'mean_dapi_diameter_um': grp['diameter_um'].mean(),
                    'mean_dapi_intensity': grp['dapi_mean_intensity'].mean()
                })
            df_dapi_base = df_raw_dapi.groupby(['group','section','image_name','animal_id','corte_num']).apply(_dapi_img).reset_index()
        else:
            # Por Sujeto
            def _dapi_img(grp):
                return pd.Series({
                    'dapi_count': len(grp),
                    'mean_dapi_area_um2': grp['area_um2'].mean(),
                    'mean_dapi_diameter_um': grp['diameter_um'].mean(),
                    'mean_dapi_intensity': grp['dapi_mean_intensity'].mean()
                })
            df_dapi_img = df_raw_dapi.groupby(['group','section','image_name','animal_id','corte_num']).apply(_dapi_img).reset_index()
            if comparison_mode == "Entre Condiciones (NONE / 3 DÍAS / 14 DÍAS) por Sexo":
                df_dapi_base = df_dapi_img.groupby(['group', 'animal_id']).mean(numeric_only=True).reset_index()
            else:
                df_dapi_base = df_dapi_img.groupby(['group', 'section', 'animal_id']).mean(numeric_only=True).reset_index()

        run_stats_layout(df_dapi_base, dapi_vars, selected_dapi_var, "Núcleos DAPI")

# 2. TAB PV+
with tab_pv:
    if df_raw_nuclei.empty:
        st.info("No hay datos de PV+ disponibles.")
    else:
        st.header("Análisis de Interneuronas PV+ (Paso 2)")
        CELL_PV_VARS = {
            "pv_area_um2": "Área del Soma PV+ (µm²)",
            "pv_diameter_um": "Diámetro del Soma PV+ (µm)"
        }
        IMG_PV_VARS = {
            "pv_count": "N° de Somas PV+ por corte",
            "mean_pv_area_um2": "Área Promedio Soma PV+ (µm²)",
            "mean_pv_diameter_um": "Diámetro Promedio Soma PV+ (µm)"
        }
        
        pv_vars = CELL_PV_VARS if level_type == "Por Célula (distribuciones individuales)" else IMG_PV_VARS
        selected_pv_var = st.selectbox("Variable PV+:", list(pv_vars.keys()), format_func=lambda x: pv_vars[x], key="pv_var")

        # Aggregate PV+ (filter is_pv_plus == True for cell level)
        if level_type == "Por Célula (distribuciones individuales)":
            df_pv_base = df_raw_nuclei[df_raw_nuclei['is_pv_plus'] == True].copy()
        elif level_type == "Por Preparado/Corte (imagen)":
            df_pv_base = df_img.copy()
        else:
            df_pv_base = df_subj_btwn.copy() if comparison_mode == "Entre Condiciones (NONE / 3 DÍAS / 14 DÍAS) por Sexo" else df_subj.copy()

        run_stats_layout(df_pv_base, pv_vars, selected_pv_var, "Interneuronas PV+")

# 3. TAB PNN
with tab_pnn:
    if df_raw_nuclei.empty:
        st.info("No hay datos de PNN disponibles.")
    else:
        st.header("Análisis de Redes Perineuronales PNN (Paso 3)")
        CELL_PNN_VARS = {
            "pnn_area_um2": "Área de PNN (µm²)",
            "pnn_diameter_um": "Diámetro de PNN (µm)",
            "score": "Confianza PNNscore (IA)"
        }
        IMG_PNN_VARS = {
            "pnn_count": "N° de Redes PNN+ Totales",
            "pnn_count_filled": "N° de PNN+ Ocupadas (PV+/PNN+)",
            "pnn_count_hollow": "N° de PNN+ Huecas (PNN+/PV-)",
            "pct_pnn_plus": "% de PV+ con Red PNN (Ocupadas)",
            "pct_pnn_hollow": "% de PNN+ que son Huecas",
            "mean_pnn_area_um2": "Área Promedio de PNN (µm²)",
            "mean_pnn_diameter_um": "Diámetro Promedio de PNN (µm)",
            "mean_score": "Confianza Promedio (PNNscore)"
        }

        pnn_vars = CELL_PNN_VARS if level_type == "Por Célula (distribuciones individuales)" else IMG_PNN_VARS
        selected_pnn_var = st.selectbox("Variable PNN:", list(pnn_vars.keys()), format_func=lambda x: pnn_vars[x], key="pnn_var")

        # Aggregate PNN (filter is_pnn_plus == True for cell level)
        if level_type == "Por Célula (distribuciones individuales)":
            df_pnn_base = df_raw_nuclei[df_raw_nuclei['is_pnn_plus'] == True].copy()
        elif level_type == "Por Preparado/Corte (imagen)":
            df_pnn_base = df_img.copy()
        else:
            df_pnn_base = df_subj_btwn.copy() if comparison_mode == "Entre Condiciones (NONE / 3 DÍAS / 14 DÍAS) por Sexo" else df_subj.copy()

        run_stats_layout(df_pnn_base, pnn_vars, selected_pnn_var, "Redes Perineuronales PNN")
