import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
df_sum['pct_pv'] = (df_sum['total_pv_segmentation'] / df_sum['total_dapi'].replace(0, np.nan)) * 100
df_sum['pct_pnn'] = (df_sum['pnn_plus'] / df_sum['total_dapi'].replace(0, np.nan)) * 100
df_sum['pct_coloc'] = (df_sum['dapi_pv_coloc'] / df_sum['total_pv_segmentation'].replace(0, np.nan)) * 100

# ── Top KPI row ───────────────────────────────
st.subheader("🔢 Resumen Global")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Imágenes", len(df_sum))
k2.metric("Total Núcleos (DAPI)", f"{df_sum['total_dapi'].sum():,}")
k3.metric("Total PNN+", f"{df_sum['pnn_plus'].sum():,}")
k4.metric("Total Colocalización PV/PNN", f"{df_sum['dapi_pv_coloc'].sum():,}")

st.divider()

# ── Plotting logic ───────────────────────────

def create_paired_box_plot(df, y_col, title, y_label):
    fig = px.box(
        df, x="group", y=y_col, color="section",
        points="all", hover_data=["individual_id", "filename"],
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
        create_paired_box_plot(df_sum, 'pct_pv', 'Densidad de Células PV+ (% DAPI)', '% PV+'),
        use_container_width=True
    )
with c2:
    st.plotly_chart(
        create_paired_box_plot(df_sum, 'pct_pnn', 'Densidad de Redes PNN+ (% DAPI)', '% PNN+'),
        use_container_width=True
    )

# ── Row 2 ─────────────────────────────────────
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(
        create_paired_box_plot(df_sum, 'pct_coloc', 'Colocalización PV+ / PNN+ (% de PV)', '% Coloc'),
        use_container_width=True
    )
with c4:
    if not df_cells.empty and 'wfa_sum_intensity' in df_cells.columns:
        fig_violin = px.violin(
            df_cells, x="group", y="wfa_sum_intensity", color="section",
            box=True, title="Distribución de Intensidad WFA (todas las células)",
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
        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.info("No hay datos celulares disponibles para el violin plot.")

st.divider()

# ── Aggregated summary table ───────────────────
st.subheader("📋 Tabla de Resumen por Grupo y Sección")
agg = df_sum.groupby(['group', 'section']).agg(
    N_imagenes=('total_dapi', 'count'),
    N_individuos=('individual_id', 'nunique'),
    DAPI_media=('total_dapi', 'mean'),
    pct_PV_media=('pct_pv', 'mean'),
    pct_PNN_media=('pct_pnn', 'mean'),
    pct_Coloc_media=('pct_coloc', 'mean'),
).round(2).reset_index()

st.dataframe(agg, use_container_width=True)

st.divider()

# ── Raw data expander ──────────────────────────
with st.expander("🗂️ Ver datos crudos por imagen"):
    st.dataframe(df_sum, use_container_width=True)

# ── Download ──────────────────────────────────
csv_bytes = df_sum.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Resumen CSV",
    data=csv_bytes,
    file_name="resumen_estadistico_ipsi_contra.csv",
    mime="text/csv",
)
