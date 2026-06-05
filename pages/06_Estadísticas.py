import streamlit as st
import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, wilcoxon, kruskal

# Page configuration
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
    .significance-sig {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.1rem;
        background-color: rgba(0, 255, 136, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #00ff88;
    }
    .significance-nonsig {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.1rem;
        background-color: rgba(255, 170, 0, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #ffaa00;
    }
    hr { border: 0; height: 1px;
         background: linear-gradient(to right, transparent, #bb86fc, transparent);
         margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Paso 6: Comparación Estadística")
st.write("Analiza las métricas de todo el experimento por célula, preparado o sujeto. Compara grupos experimentales o hemisferios (IPSI vs CONTRA).")

METRICS_BASE_DIR = "data/processed/metrics"

# ---------------------------------------------------------------------------
# DATA LOADING — with correct animal_id extraction
# ---------------------------------------------------------------------------
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
                group   = parts[0]
                section = parts[1]
                filename = parts[2].replace('_nuclei_metrics.csv', '')
            else:
                group = "Desconocido"
                section = "Desconocido"
                filename = f.replace('_nuclei_metrics.csv', '')

            # Correct animal_id: "ACF_49" from "ACF_49~1"
            m = re.match(r'(ACF_\d+)', filename)
            animal_id = m.group(1) if m else filename.split('~')[0]
            # Corte number: 1, 2, 3...
            m_corte = re.search(r'~(\d+)$', filename)
            corte_num = int(m_corte.group(1)) if m_corte else 1

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

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

df_raw = load_all_experiment_metrics()

if df_raw.empty:
    st.info("👋 No se encontraron archivos de métricas. Ejecuta el pipeline en los pasos anteriores.")
    st.stop()

# Generate split metrics if missing (backward compat)
if 'pv_area_um2' not in df_raw.columns:
    df_raw['pv_area_um2']    = df_raw.apply(lambda r: r['area_um2'] if r['is_pv_plus'] else 0.0, axis=1)
    df_raw['pv_diameter_um'] = df_raw.apply(lambda r: r['diameter_um'] if r['is_pv_plus'] else 0.0, axis=1)
    df_raw['pnn_area_um2']   = df_raw.apply(lambda r: r['area_um2'] if r['is_pnn_plus'] else 0.0, axis=1)
    df_raw['pnn_diameter_um']= df_raw.apply(lambda r: r['diameter_um'] if r['is_pnn_plus'] else 0.0, axis=1)

# ---------------------------------------------------------------------------
# AGGREGATION HELPERS
# ---------------------------------------------------------------------------
SKEL_COLS = ['skel_total_length_um','skel_branches_count','skel_mean_thickness_um',
             'skel_max_thickness_um','skel_mean_intensity','skel_neighborhood_wfa_sum',
             'skel_tortuosity_mean','skel_ramification_index']

def _img_summary(group_df):
    """One row per (group, section, image_name, animal_id, corte_num)."""
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
        'mean_soma_area_um2':   safe_mean(pv_cells['pv_area_um2']),    # compat
        'mean_soma_diameter_um':safe_mean(pv_cells['pv_diameter_um']), # compat
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
    """Average all cortes of the same animal+section into one row per (group, section, animal_id)."""
    numeric_cols = [c for c in df_img.columns
                    if c not in ['group','section','image_name','animal_id','corte_num']
                    and pd.api.types.is_numeric_dtype(df_img[c])]
    keys = ['group','section','animal_id']
    summaries = []
    for vals, grp in df_img.groupby(keys):
        base = dict(zip(keys, vals))
        base['n_cortes'] = len(grp)
        for col in numeric_cols:
            base[col] = grp[col].mean()
        summaries.append(base)
    return pd.DataFrame(summaries)

df_img  = aggregate_image_level(df_raw)
df_subj = aggregate_subject_level(df_img)

# ---------------------------------------------------------------------------
# SIDEBAR CONFIG
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Análisis")

level_type = st.sidebar.radio(
    "Nivel de Análisis:",
    ["Por Célula (distribuciones individuales)",
     "Por Preparado/Corte (imagen)",
     "Por Sujeto (animal, promediando cortes)"],
    key="stats_level_select"
)

comparison_factor = st.sidebar.selectbox(
    "Factor de Comparación:",
    ["section (IPSI vs CONTRA)", "group (Grupos Experimentales)"],
    key="stats_factor_select"
)

# Variable options
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

if level_type == "Por Célula (distribuciones individuales)":
    var_options = CELL_VARS
elif level_type == "Por Preparado/Corte (imagen)":
    var_options = IMG_SUBJ_VARS
else:
    var_options = IMG_SUBJ_VARS

selected_var_key = st.sidebar.selectbox(
    "Variable de Estudio:",
    list(var_options.keys()),
    format_func=lambda x: var_options[x],
    key="stats_var_select"
)

# ---------------------------------------------------------------------------
# CHOOSE BASE DATAFRAME & APPLY FILTERS
# ---------------------------------------------------------------------------
if level_type == "Por Célula (distribuciones individuales)":
    df_analysis = df_raw.copy()
    # Auto-filter to relevant entity type
    skel_vars = [k for k in CELL_VARS if k.startswith('skel_')]
    if selected_var_key in ['pv_area_um2', 'pv_diameter_um']:
        df_analysis = df_analysis[df_analysis['is_pv_plus'] == True]
    elif selected_var_key in ['pnn_area_um2', 'pnn_diameter_um'] + skel_vars:
        df_analysis = df_analysis[df_analysis['is_pnn_plus'] == True]
elif level_type == "Por Preparado/Corte (imagen)":
    df_analysis = df_img.copy()
else:
    df_analysis = df_subj.copy()

# ---------------------------------------------------------------------------
# FACTOR & CATEGORIES
# ---------------------------------------------------------------------------
factor_col = "section" if "section" in comparison_factor else "group"

if factor_col not in df_analysis.columns:
    st.error(f"Columna `{factor_col}` no encontrada en los datos.")
    st.stop()

categories = sorted(df_analysis[factor_col].dropna().unique())

if len(categories) < 2:
    st.warning(f"Se necesitan al menos 2 categorías en `{factor_col}`. Detectadas: {categories}")
    st.info("Asegúrate de haber procesado preparados en carpetas distintas (IPSI/CONTRA, o distintos grupos).")
    st.stop()

if len(categories) > 2:
    selected_cats = st.sidebar.multiselect(
        "Selecciona exactamente 2 categorías:",
        categories, default=categories[:2], key="stats_cat_select"
    )
    if len(selected_cats) != 2:
        st.error("Selecciona exactamente 2 categorías.")
        st.stop()
    cat_a, cat_b = selected_cats[0], selected_cats[1]
else:
    cat_a, cat_b = categories[0], categories[1]

df_a = df_analysis[df_analysis[factor_col] == cat_a]
df_b = df_analysis[df_analysis[factor_col] == cat_b]

if selected_var_key not in df_a.columns:
    st.error(f"La variable `{selected_var_key}` no está disponible en el nivel de análisis seleccionado. Cambia a otro nivel o variable.")
    st.stop()

data_a = df_a[selected_var_key].dropna().values
data_b = df_b[selected_var_key].dropna().values

if len(data_a) < 2 or len(data_b) < 2:
    st.warning(f"Datos insuficientes: '{cat_a}' → {len(data_a)}, '{cat_b}' → {len(data_b)} observaciones.")
    st.stop()

# ---------------------------------------------------------------------------
# CONTEXT INFO BANNER
# ---------------------------------------------------------------------------
is_paired_possible = (factor_col == "section") and (level_type != "Por Célula (distribuciones individuales)")

if level_type == "Por Célula (distribuciones individuales)":
    info_text = (f"📌 <b>Nivel: Célula individual.</b> Cada punto es una célula/red segmentada. "
                 f"N total: {len(df_a)} ({cat_a}) vs {len(df_b)} ({cat_b}).")
elif level_type == "Por Preparado/Corte (imagen)":
    info_text = (f"📌 <b>Nivel: Preparado/Corte.</b> Cada punto es el promedio de una imagen (corte histológico). "
                 f"Un animal con 3 cortes aporta 3 puntos. "
                 f"N: {len(df_a)} ({cat_a}) vs {len(df_b)} ({cat_b}).")
else:
    info_text = (f"📌 <b>Nivel: Sujeto/Animal.</b> Los cortes de cada animal se promedian → 1 valor por animal. "
                 f"N: {len(df_a)} sujetos ({cat_a}) vs {len(df_b)} sujetos ({cat_b}). "
                 f"{'Comparación intra-sujeto (pareada) posible.' if is_paired_possible else ''}")

st.markdown(f'<div class="level-info">{info_text}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAIRING LOGIC
# ---------------------------------------------------------------------------
# For IPSI vs CONTRA at image or subject level → try to pair by animal_id
paired_data = None
if is_paired_possible and 'animal_id' in df_analysis.columns:
    vals_a = df_a[['animal_id', selected_var_key]].rename(columns={selected_var_key: 'val_a'})
    vals_b = df_b[['animal_id', selected_var_key]].rename(columns={selected_var_key: 'val_b'})
    merged = pd.merge(vals_a, vals_b, on='animal_id').dropna()
    if len(merged) >= 3:
        paired_data = merged

# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------
st.subheader(f"📈 {var_options[selected_var_key]}: {cat_a} vs {cat_b}")

col_plot, col_stats = st.columns([2, 1])

with col_plot:
    if paired_data is not None:
        # Paired line plot (IPSI ↔ CONTRA per animal)
        fig = go.Figure()
        for _, row in paired_data.iterrows():
            fig.add_trace(go.Scatter(
                x=[cat_a, cat_b],
                y=[row['val_a'], row['val_b']],
                mode='lines+markers',
                name=row['animal_id'],
                line=dict(color='rgba(187,134,252,0.45)', width=2),
                marker=dict(size=9, color=['#00f2fe', '#bb86fc']),
                showlegend=True
            ))
        # Add mean ± SD error bars on top
        fig.add_trace(go.Scatter(
            x=[cat_a], y=[paired_data['val_a'].mean()],
            error_y=dict(type='data', array=[paired_data['val_a'].std()], visible=True),
            mode='markers', marker=dict(color='#00f2fe', size=14, symbol='diamond'),
            name=f'Media {cat_a}', showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[cat_b], y=[paired_data['val_b'].mean()],
            error_y=dict(type='data', array=[paired_data['val_b'].std()], visible=True),
            mode='markers', marker=dict(color='#bb86fc', size=14, symbol='diamond'),
            name=f'Media {cat_b}', showlegend=True
        ))
        fig.update_layout(
            title=f"Comparación Pareada por Sujeto: {cat_a} ↔ {cat_b}",
            xaxis_title="Hemisferio", yaxis_title=var_options[selected_var_key],
            template='plotly_dark', height=480,
            legend=dict(font=dict(size=10))
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Box + strip plot (unpaired)
        df_plot = df_analysis[df_analysis[factor_col].isin([cat_a, cat_b])].copy()
        fig = px.box(
            df_plot, x=factor_col, y=selected_var_key,
            color=factor_col, points="all",
            color_discrete_map={cat_a: '#00f2fe', cat_b: '#bb86fc'},
            labels={factor_col: "Categoría", selected_var_key: var_options[selected_var_key]},
            template='plotly_dark'
        )
        fig.update_layout(height=480, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Resumen Descriptivo")

    def desc(data, label, color):
        mean_v, std_v = np.mean(data), np.std(data)
        med_v = np.median(data)
        q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
        st.markdown(f"**{color} {label} (N={len(data)}):**")
        st.markdown(f"- Media ± DE: `{mean_v:.3f} ± {std_v:.3f}`")
        st.markdown(f"- Mediana [IQR]: `{med_v:.3f} [{q1:.3f} – {q3:.3f}]`")

    desc(data_a, cat_a, "🔵")
    desc(data_b, cat_b, "🟣")

    diff_pct = ((np.mean(data_b) - np.mean(data_a)) / np.mean(data_a) * 100) if np.mean(data_a) != 0 else 0
    st.markdown(f"**Δ Medias:** `{diff_pct:+.1f}%` en {cat_b} respecto a {cat_a}")

    if paired_data is not None:
        st.markdown(f"**Pares identificados:** `{len(paired_data)}` sujetos con datos en ambas secciones")

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# STATISTICAL TEST
# ---------------------------------------------------------------------------
st.subheader("🧪 Análisis Estadístico Inferencial")

stat, p_val = np.nan, np.nan
stat_name, test_func_name = "", ""

if paired_data is not None and len(paired_data) >= 3:
    # Wilcoxon pareado intra-sujeto
    try:
        stat, p_val = wilcoxon(paired_data['val_a'].values, paired_data['val_b'].values)
        stat_name = "Wilcoxon W"
        test_func_name = "Wilcoxon Signed-Rank (pareado intra-sujeto IPSI vs CONTRA)"
        st.info(f"🔗 **Prueba pareada** aplicada sobre {len(paired_data)} sujetos con datos en ambos hemisferios.")
    except Exception as e:
        st.error(f"Error en Wilcoxon: {e}. Se usará Mann-Whitney U.")
        stat, p_val = mannwhitneyu(data_a, data_b, alternative='two-sided')
        stat_name = "Mann-Whitney U"
        test_func_name = "Mann-Whitney U (fallback no pareado)"
else:
    # Mann-Whitney U no pareado
    if len(data_a) >= 3 and len(data_b) >= 3:
        stat, p_val = mannwhitneyu(data_a, data_b, alternative='two-sided')
        stat_name = "Mann-Whitney U"
        if is_paired_possible and paired_data is None:
            test_func_name = "Mann-Whitney U (no pareado — no se encontraron sujetos con datos en ambas secciones)"
            st.warning("⚠️ No se encontraron animales con datos en ambas secciones para parear. Se usó Mann-Whitney U no pareada.")
        else:
            test_func_name = "Mann-Whitney U (muestras independientes)"
    else:
        st.warning(f"Datos insuficientes para test estadístico (se requieren ≥3 por grupo).")
        st.stop()

col_test1, col_test2 = st.columns([1, 1])
with col_test1:
    st.markdown(f"**Prueba:** `{test_func_name}`")
    st.markdown(f"- Estadístico `{stat_name}`: `{stat:.4f}`")
    st.markdown(f"- p-valor: `{p_val:.4f}`")

    if not np.isnan(p_val):
        if p_val < 0.001:
            sig_str = "p < 0.001 ***"
        elif p_val < 0.01:
            sig_str = f"p = {p_val:.4f} **"
        elif p_val < 0.05:
            sig_str = f"p = {p_val:.4f} *"
        else:
            sig_str = f"p = {p_val:.4f} (ns)"

        if p_val < 0.05:
            st.markdown(f"""
                <div class="significance-sig">
                    ✅ DIFERENCIA SIGNIFICATIVA — {sig_str}<br>
                    <span style="font-size:0.9rem; font-weight:normal; color:#ffffff;">
                        '{var_options[selected_var_key]}' difiere significativamente entre '{cat_a}' y '{cat_b}'.
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="significance-nonsig">
                    ⚠️ SIN DIFERENCIA SIGNIFICATIVA — {sig_str}<br>
                    <span style="font-size:0.9rem; font-weight:normal; color:#ffffff;">
                        No hay evidencia suficiente para distinguir '{cat_a}' de '{cat_b}'.
                    </span>
                </div>""", unsafe_allow_html=True)

with col_test2:
    # Effect size (rank-biserial correlation for Mann-Whitney, or r for Wilcoxon)
    n_a, n_b = len(data_a), len(data_b)
    if stat_name == "Mann-Whitney U" and not np.isnan(stat):
        r_rb = 1 - (2 * stat) / (n_a * n_b) if n_a * n_b > 0 else np.nan
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.markdown("### 📐 Tamaño del Efecto")
        st.markdown(f"**Correlación rank-biserial r:** `{r_rb:.3f}`")
        if not np.isnan(r_rb):
            r_abs = abs(r_rb)
            if r_abs < 0.1:   effect_str = "Efecto trivial (< 0.1)"
            elif r_abs < 0.3: effect_str = "Efecto pequeño (0.1–0.3)"
            elif r_abs < 0.5: effect_str = "Efecto mediano (0.3–0.5)"
            else:             effect_str = "Efecto grande (> 0.5)"
            st.markdown(f"**Interpretación:** {effect_str}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif stat_name == "Wilcoxon W" and not np.isnan(stat):
        n_pairs = len(paired_data)
        r_w = stat / (n_pairs * (n_pairs + 1) / 2) if n_pairs > 0 else np.nan
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.markdown("### 📐 Tamaño del Efecto")
        st.markdown(f"**r de Wilcoxon:** `{r_w:.3f}` (normalizado sobre N={n_pairs} pares)")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# SUBJECT-LEVEL TABLE (only when relevant)
# ---------------------------------------------------------------------------
if level_type == "Por Sujeto (animal, promediando cortes)":
    st.subheader("👤 Tabla de Datos por Sujeto")
    display_cols = ['group', 'section', 'animal_id', 'n_cortes', selected_var_key]
    display_cols = [c for c in display_cols if c in df_analysis.columns]
    df_show = df_analysis[df_analysis[factor_col].isin([cat_a, cat_b])][display_cols].sort_values(['group','section','animal_id'])
    df_show = df_show.rename(columns={
        'group': 'Grupo', 'section': 'Sección', 'animal_id': 'Sujeto',
        'n_cortes': 'N Cortes', selected_var_key: var_options[selected_var_key]
    })
    st.dataframe(df_show, use_container_width=True)

    if paired_data is not None:
        st.subheader("🔗 Pares IPSI ↔ CONTRA por Sujeto")
        paired_show = paired_data.rename(columns={
            'animal_id': 'Sujeto', 'val_a': cat_a, 'val_b': cat_b
        }).copy()
        paired_show['Δ (CONTRA−IPSI)'] = paired_show[cat_b] - paired_show[cat_a]
        paired_show['Δ %'] = ((paired_show[cat_b] - paired_show[cat_a]) / paired_show[cat_a].replace(0, np.nan) * 100).round(1)
        st.dataframe(paired_show, use_container_width=True)
elif level_type == "Por Preparado/Corte (imagen)":
    st.subheader("📋 Tabla de Preparados")
    display_cols = ['group','section','animal_id','corte_num','image_name', selected_var_key]
    display_cols = [c for c in display_cols if c in df_analysis.columns]
    df_show = df_analysis[df_analysis[factor_col].isin([cat_a, cat_b])][display_cols].sort_values(['group','section','animal_id','corte_num'])
    st.dataframe(df_show, use_container_width=True)
else:
    st.subheader("📋 Tabla de Datos")
    display_cols = [factor_col, 'image_name', selected_var_key]
    display_cols = [c for c in display_cols if c in df_analysis.columns]
    st.dataframe(df_analysis[df_analysis[factor_col].isin([cat_a, cat_b])][display_cols], use_container_width=True)
