import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from itertools import combinations

st.set_page_config(page_title="Estadísticas Globales", layout="wide")

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #bb86fc; }
    div[data-testid="stMetricLabel"] { color: #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Página 4: Estadísticas Globales y Comparativa")

METRICS_BASE_DIR = "data/processed/metrics"

if not os.path.exists(METRICS_BASE_DIR):
    st.error(f"No se encontró `{METRICS_BASE_DIR}`. Procesa imágenes en lote primero.")
    st.stop()

groups = sorted([d for d in os.listdir(METRICS_BASE_DIR) if os.path.isdir(os.path.join(METRICS_BASE_DIR, d))])
if not groups:
    st.warning("No hay grupos procesados detectados.")
    st.stop()

# ── Sidebar ──────────────────────────────────
st.sidebar.header("📊 Filtros de Análisis")
selected_groups = st.sidebar.multiselect("Grupos a comparar:", groups, default=groups)

if not selected_groups:
    st.warning("Selecciona al menos un grupo.")
    st.stop()

# ── Load data ────────────────────────────────
all_summaries = []
all_cell_data = []

for group in selected_groups:
    group_dir = os.path.join(METRICS_BASE_DIR, group)
    if not os.path.isdir(group_dir):
        continue
        
    sections = [d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))]
    for section in sections:
        section_dir = os.path.join(group_dir, section)
        for f in os.listdir(section_dir):
            if not f.endswith('_summary.json'):
                continue
                
            base_fname = f.replace('_summary.json', '')
            # Extract individual ID (e.g. ACF_86~1 -> ACF_86)
            indiv_id = base_fname.split('~')[0] if '~' in base_fname else base_fname
            
            with open(os.path.join(section_dir, f)) as jf:
                s = json.load(jf)
                s['group'] = group
                s['section'] = section
                s['individual_id'] = indiv_id
                s['filename'] = base_fname
                all_summaries.append(s)

            csv_path = os.path.join(section_dir, f.replace('_summary.json', '_nuclei_metrics.csv'))
            if os.path.exists(csv_path):
                df_c = pd.read_csv(csv_path)
                df_c['group'] = group
                df_c['section'] = section
                df_c['individual_id'] = indiv_id
                df_c['filename'] = base_fname
                all_cell_data.append(df_c)

if not all_summaries:
    st.error("No se encontraron archivos de resumen para los grupos seleccionados.")
    st.stop()

df_sum = pd.DataFrame(all_summaries)
df_cells = pd.concat(all_cell_data, ignore_index=True) if all_cell_data else pd.DataFrame()

# ── Derived metrics ───────────────────────────
df_sum['pct_pv_in_dapi'] = (df_sum['total_pv_segmentation'] / df_sum['total_dapi'].replace(0, np.nan)) * 100
df_sum['pct_pnn_in_pv'] = (df_sum['pnn_plus'] / df_sum['total_pv_segmentation'].replace(0, np.nan)) * 100
df_sum['pv_count'] = df_sum['total_pv_segmentation']
df_sum['pnn_count'] = df_sum['pnn_plus']

# ── Individual Aggregation ────────────────────
df_indiv = df_sum.groupby(['group', 'individual_id', 'section']).agg(
    pv_count=('pv_count', 'mean'),
    pnn_count=('pnn_count', 'mean'),
    pct_pnn_in_pv=('pct_pnn_in_pv', 'mean'),
    total_dapi=('total_dapi', 'mean'),
    N_imagenes=('filename', 'count')
).reset_index()

if not df_cells.empty and 'wfa_sum_intensity' in df_cells.columns:
    df_wfa = df_cells.groupby(['group', 'individual_id', 'section'])['wfa_sum_intensity'].mean().reset_index()
    df_wfa = df_wfa.rename(columns={'wfa_sum_intensity': 'wfa_mean_intensity'})
    df_indiv = df_indiv.merge(df_wfa, on=['group', 'individual_id', 'section'], how='left')
else:
    df_indiv['wfa_mean_intensity'] = np.nan

# ── Top KPI row ───────────────────────────────
st.subheader("🔢 Resumen Global (Enfoque en PV+)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Individuos", df_indiv['individual_id'].nunique())
k2.metric("Total Células PV+", f"{df_sum['total_pv_segmentation'].sum():,}")
k3.metric("Total PV+/PNN+", f"{df_sum['pnn_plus'].sum():,}")
avg_pct = (df_sum['pnn_plus'].sum() / max(1, df_sum['total_pv_segmentation'].sum())) * 100
k4.metric("Proporción Global PV+ con PNN+", f"{avg_pct:.1f}%")

st.divider()

# ── Plotting logic ───────────────────────────

def create_paired_box_plot(df, y_col, title, y_label):
    fig = px.box(
        df, x="group", y=y_col, color="section",
        points="all", hover_data=["individual_id", "N_imagenes"],
        title=title, labels={"group": "Grupo", y_col: y_label, "section": "Sección"},
        template="plotly_dark",
        color_discrete_map={"IPSI": "#ff7f0e", "CONTRA": "#1f77b4"} # Distinct colors
    )
    
    # Add paired lines between IPSI and CONTRA for the same individual within the same group
    for grp in df['group'].unique():
        grp_df = df[df['group'] == grp]
        for indiv in grp_df['individual_id'].unique():
            indiv_df = grp_df[grp_df['individual_id'] == indiv]
            if len(indiv_df['section'].unique()) > 1: # Has both IPSI and CONTRA
                # Draw a line connecting the points
                fig.add_trace(go.Scatter(
                    x=[grp, grp],
                    y=[indiv_df[indiv_df['section'] == 'IPSI'][y_col].mean(), 
                       indiv_df[indiv_df['section'] == 'CONTRA'][y_col].mean()],
                    mode='lines',
                    line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
    fig.update_layout(
        boxmode='group',
        height=400,
        margin=dict(l=30, r=10, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# ── Row 1 ─────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        create_paired_box_plot(df_indiv, 'pv_count', 'Conteo de Células PV+ por Individuo', 'Conteo PV+ (Media)'),
        width="stretch"
    )
with c2:
    st.plotly_chart(
        create_paired_box_plot(df_indiv, 'pnn_count', 'Conteo de PV+/PNN+ por Individuo', 'Conteo PV+/PNN+ (Media)'),
        width="stretch"
    )

# ── Row 2 ─────────────────────────────────────
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(
        create_paired_box_plot(df_indiv, 'pct_pnn_in_pv', 'Proporción de PV+ que presentan PNN+ (%)', '% PNN+ en PV+'),
        width="stretch"
    )
with c4:
    if not df_cells.empty and 'wfa_sum_intensity' in df_cells.columns:
        fig_violin = px.violin(
            df_cells, x="group", y="wfa_sum_intensity", color="section",
            box=True, title="Distribución de Intensidad WFA (Anillo Exterior PV+)",
            labels={"group": "Grupo", "wfa_sum_intensity": "Intensidad Suma WFA", "section": "Sección"},
            template="plotly_dark",
            color_discrete_map={"IPSI": "#ff7f0e", "CONTRA": "#1f77b4"}
        )
        fig_violin.update_layout(
            violinmode='group',
            height=400,
            margin=dict(l=30, r=10, t=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_violin, width="stretch")
    else:
        st.info("No hay datos celulares disponibles para el violin plot.")

st.divider()

st.divider()

# ── Tablas Estadísticas Formales ───────────────
st.subheader("🔬 Análisis Estadístico")
tab1, tab2, tab3 = st.tabs([
    "1. Comparación Entre Grupos", 
    "2. Comparación de Deltas (Efecto Tratamiento)", 
    "3. IPSI vs CONTRA (Interna)"
])

with tab1:
    st.markdown("**Test U de Mann-Whitney (Datos Independientes)**")
    if len(df_indiv['group'].unique()) < 2:
        st.info("Se necesitan al menos 2 grupos para comparar.")
    else:
        res_indep = []
        groups_list = list(df_indiv['group'].unique())
        sections_list = df_indiv['section'].unique()
        
        metrics = ['pv_count', 'pnn_count', 'pct_pnn_in_pv', 'wfa_mean_intensity']
        metric_names = ['Conteo PV+', 'Conteo PNN+', '% PNN+ en PV+', 'Intensidad WFA Media']
        
        for sec in sections_list:
            for g1, g2 in combinations(groups_list, 2):
                df_g1 = df_indiv[(df_indiv['group'] == g1) & (df_indiv['section'] == sec)]
                df_g2 = df_indiv[(df_indiv['group'] == g2) & (df_indiv['section'] == sec)]
                
                if len(df_g1) < 3 or len(df_g2) < 3:
                    continue
                    
                for met, mname in zip(metrics, metric_names):
                    vals1 = pd.to_numeric(df_g1[met], errors='coerce').dropna()
                    vals2 = pd.to_numeric(df_g2[met], errors='coerce').dropna()
                    
                    try:
                        stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    except Exception:
                        stat, pval = np.nan, np.nan
                        
                    res_indep.append({
                        'Sección': sec,
                        'Comparación': f"{g1} vs {g2}",
                        'Métrica': mname,
                        f'N ({g1})': len(vals1),
                        f'N ({g2})': len(vals2),
                        f'Media ({g1})': vals1.mean(),
                        f'Media ({g2})': vals2.mean(),
                        'p-valor': pval,
                        'Significativo (α=0.05)': 'Sí' if pval < 0.05 else 'No'
                    })
        
        if res_indep:
            df_res_indep = pd.DataFrame(res_indep)
            st.dataframe(df_res_indep, width="stretch")
        else:
            st.info("No hay suficientes datos (N>=3) por grupo/sección para aplicar el test de Mann-Whitney.")

with tab2:
    st.markdown("**Test U de Mann-Whitney sobre el Delta (IPSI - CONTRA)**")
    st.info("Compara si el cambio (asimetría) inducido en un grupo es significativamente distinto al de otro grupo.")
    if len(df_indiv['group'].unique()) < 2:
        st.info("Se necesitan al menos 2 grupos para comparar.")
    else:
        # Calcular deltas por individuo
        delta_list = []
        for grp in df_indiv['group'].unique():
            grp_df = df_indiv[df_indiv['group'] == grp]
            pivot_df = grp_df.pivot(index='individual_id', columns='section', values=['pv_count', 'pnn_count', 'pct_pnn_in_pv', 'wfa_mean_intensity'])
            for indiv in pivot_df.index:
                wfa_delta = pivot_df.loc[indiv, ('wfa_mean_intensity', 'IPSI')] - pivot_df.loc[indiv, ('wfa_mean_intensity', 'CONTRA')] if 'wfa_mean_intensity' in pivot_df.columns.get_level_values(0) and not pd.isna(pivot_df.loc[indiv, ('wfa_mean_intensity', 'IPSI')]) and not pd.isna(pivot_df.loc[indiv, ('wfa_mean_intensity', 'CONTRA')]) else np.nan
                delta_list.append({
                    'group': grp,
                    'individual_id': indiv,
                    'delta_pv': pivot_df.loc[indiv, ('pv_count', 'IPSI')] - pivot_df.loc[indiv, ('pv_count', 'CONTRA')],
                    'delta_pnn': pivot_df.loc[indiv, ('pnn_count', 'IPSI')] - pivot_df.loc[indiv, ('pnn_count', 'CONTRA')],
                    'delta_pct': pivot_df.loc[indiv, ('pct_pnn_in_pv', 'IPSI')] - pivot_df.loc[indiv, ('pct_pnn_in_pv', 'CONTRA')],
                    'delta_wfa': wfa_delta
                })
        
        if delta_list:
            df_deltas = pd.DataFrame(delta_list)
            res_deltas = []
            groups_list = list(df_deltas['group'].unique())
            
            metrics = ['delta_pv', 'delta_pnn', 'delta_pct', 'delta_wfa']
            metric_names = ['Δ Conteo PV+', 'Δ Conteo PNN+', 'Δ % PNN+ en PV+', 'Δ Int. WFA Media']
            
            for g1, g2 in combinations(groups_list, 2):
                df_g1 = df_deltas[df_deltas['group'] == g1]
                df_g2 = df_deltas[df_deltas['group'] == g2]
                
                if len(df_g1) < 3 or len(df_g2) < 3:
                    continue
                    
                for met, mname in zip(metrics, metric_names):
                    vals1 = pd.to_numeric(df_g1[met], errors='coerce').dropna()
                    vals2 = pd.to_numeric(df_g2[met], errors='coerce').dropna()
                    
                    try:
                        stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    except Exception:
                        stat, pval = np.nan, np.nan
                        
                    res_deltas.append({
                        'Comparación': f"{g1} vs {g2}",
                        'Métrica': mname,
                        f'N ({g1})': len(vals1),
                        f'N ({g2})': len(vals2),
                        f'Media Δ ({g1})': vals1.mean(),
                        f'Media Δ ({g2})': vals2.mean(),
                        'p-valor': pval,
                        'Significativo (α=0.05)': 'Sí' if pval < 0.05 else 'No'
                    })
                    
            if res_deltas:
                st.dataframe(pd.DataFrame(res_deltas), width="stretch")
            else:
                st.info("No hay suficientes pares IPSI/CONTRA para comparar deltas entre grupos.")
        else:
            st.info("No se pudieron calcular los deltas.")

with tab3:
    st.markdown("**Test de Rangos con Signo de Wilcoxon (Datos Pareados)**")
    res_pareados = []
    for grp in df_indiv['group'].unique():
        grp_df = df_indiv[df_indiv['group'] == grp]
        
        pivot_df = grp_df.pivot(index='individual_id', columns='section', values=['pv_count', 'pnn_count', 'pct_pnn_in_pv', 'wfa_mean_intensity'])
        
        if len(pivot_df) < 3:
            st.warning(f"Grupo {grp}: No hay suficientes pares IPSI/CONTRA (N={len(pivot_df)}) para un test robusto.")
            continue
            
        metrics = ['pv_count', 'pnn_count', 'pct_pnn_in_pv', 'wfa_mean_intensity']
        metric_names = ['Conteo PV+', 'Conteo PNN+', '% PNN+ en PV+', 'Intensidad WFA Media']
        
        for met, mname in zip(metrics, metric_names):
            ipsi_vals = pd.to_numeric(pivot_df[(met, 'IPSI')], errors='coerce')
            contra_vals = pd.to_numeric(pivot_df[(met, 'CONTRA')], errors='coerce')
            valid_idx = ipsi_vals.notna() & contra_vals.notna()
            ipsi_vals = ipsi_vals[valid_idx]
            contra_vals = contra_vals[valid_idx]
            
            try:
                if len(ipsi_vals) >= 3:
                    stat, pval = stats.wilcoxon(ipsi_vals, contra_vals)
                else:
                    stat, pval = np.nan, np.nan
            except Exception:
                stat, pval = np.nan, np.nan
                
            res_pareados.append({
                'Grupo': grp,
                'Métrica': mname,
                'N Pareados': len(pivot_df),
                'Media IPSI': ipsi_vals.mean(),
                'Media CONTRA': contra_vals.mean(),
                'p-valor': pval,
                'Significativo (α=0.05)': 'Sí' if pval < 0.05 else 'No'
            })
    
    if res_pareados:
        df_res_pareados = pd.DataFrame(res_pareados)
        st.dataframe(df_res_pareados, width="stretch")
    else:
        st.info("No se generaron resultados estadísticos pareados.")

st.divider()

# ── Aggregated summary table ───────────────────
st.subheader("📋 Tabla de Resumen por Grupo y Sección")
agg = df_indiv.groupby(['group', 'section']).agg(
    N_individuos=('individual_id', 'nunique'),
    Media_Imagenes_por_Indiv=('N_imagenes', 'mean'),
    PV_media=('pv_count', 'mean'),
    PNN_media=('pnn_count', 'mean'),
    pct_PNN_en_PV_media=('pct_pnn_in_pv', 'mean'),
    Intensidad_WFA_media=('wfa_mean_intensity', 'mean'),
    DAPI_media_referencia=('total_dapi', 'mean'),
).round(2).reset_index()

st.dataframe(agg, width="stretch")

st.divider()

# ── Raw data expander ──────────────────────────
with st.expander("🗂️ Ver datos crudos por imagen"):
    st.dataframe(df_sum, width="stretch")

# ── Download ──────────────────────────────────
csv_bytes = df_sum.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Resumen CSV",
    data=csv_bytes,
    file_name="resumen_estadistico_ipsi_contra.csv",
    mime="text/csv",
)
