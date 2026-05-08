import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    st.error(f"No se encontró `{METRICS_BASE_DIR}`. Procesa imágenes primero.")
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

# Colour palette (one per group, consistent across all charts)
PALETTE = px.colors.qualitative.Bold
group_colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted(selected_groups))}

# ── Load data ────────────────────────────────
all_summaries = []
all_cell_data = []

for group in selected_groups:
    group_dir = os.path.join(METRICS_BASE_DIR, group)
    if not os.path.isdir(group_dir):
        continue
    for f in os.listdir(group_dir):
        if not f.endswith('_summary.json'):
            continue
        with open(os.path.join(group_dir, f)) as jf:
            s = json.load(jf)
            s['group'] = group
            s['filename'] = f.replace('_summary.json', '')
            all_summaries.append(s)

        csv_path = os.path.join(group_dir, f.replace('_summary.json', '_nuclei_metrics.csv'))
        if os.path.exists(csv_path):
            df_c = pd.read_csv(csv_path)
            df_c['group'] = group
            df_c['filename'] = f.replace('_summary.json', '')
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

# ── Helper: build Plotly box+strip ───────────
def box_strip(df, y_col, title, y_label, color_map):
    fig = go.Figure()
    for grp in sorted(df['group'].unique()):
        sub = df[df['group'] == grp][y_col].dropna()
        color = color_map.get(grp, '#888')
        fig.add_trace(go.Box(
            y=sub,
            name=grp,
            marker_color=color,
            boxmean=True,
            line_width=1.5,
            boxpoints='all',
            jitter=0.4,
            pointpos=0,
            marker=dict(size=7, opacity=0.75, line=dict(color='white', width=0.5)),
            showlegend=False,
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#bb86fc')),
        yaxis_title=y_label,
        xaxis_title="",
        template='plotly_dark',
        height=380,
        margin=dict(l=30, r=10, t=50, b=80),
        xaxis=dict(tickangle=-30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ── Helper: violin ────────────────────────────
def violin_plot(df, y_col, title, y_label, color_map):
    fig = go.Figure()
    for grp in sorted(df['group'].unique()):
        sub = df[df['group'] == grp][y_col].dropna()
        if sub.empty:
            continue
        color = color_map.get(grp, '#888')
        fig.add_trace(go.Violin(
            y=sub, name=grp,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color.replace(')', ',0.35)').replace('rgb', 'rgba') if color.startswith('rgb') else color,
            line_color=color,
            points='all',
            jitter=0.3,
            marker=dict(size=4, opacity=0.6),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#bb86fc')),
        yaxis_title=y_label,
        xaxis_title="",
        template='plotly_dark',
        height=380,
        margin=dict(l=30, r=10, t=50, b=80),
        xaxis=dict(tickangle=-30),
        showlegend=False,
        violingap=0.1,
        violingroupgap=0.05,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ── Row 1 ─────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        box_strip(df_sum, 'pct_pv', 'Densidad de Células PV+ (% DAPI)', '% PV+', group_colors),
        use_container_width=True
    )
with c2:
    st.plotly_chart(
        box_strip(df_sum, 'pct_pnn', 'Densidad de Redes PNN+ (% DAPI)', '% PNN+', group_colors),
        use_container_width=True
    )

# ── Row 2 ─────────────────────────────────────
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(
        box_strip(df_sum, 'pct_coloc', 'Colocalización PV+ / PNN+ (% de PV envueltas)', '% Coloc', group_colors),
        use_container_width=True
    )
with c4:
    if not df_cells.empty and 'wfa_sum_intensity' in df_cells.columns:
        # Show all cells regardless of PNN status so all groups appear
        st.plotly_chart(
            violin_plot(df_cells, 'wfa_sum_intensity',
                        'Distribución de Intensidad WFA (todas las células)',
                        'WFA Sum Intensity', group_colors),
            use_container_width=True
        )
    else:
        st.info("No hay datos celulares disponibles para el violin plot.")

st.divider()

# ── Aggregated summary table ───────────────────
st.subheader("📋 Tabla de Resumen por Grupo")
agg = df_sum.groupby('group').agg(
    N_imagenes=('total_dapi', 'count'),
    DAPI_media=('total_dapi', 'mean'),
    DAPI_std=('total_dapi', 'std'),
    pct_PV_media=('pct_pv', 'mean'),
    pct_PV_std=('pct_pv', 'std'),
    pct_PNN_media=('pct_pnn', 'mean'),
    pct_PNN_std=('pct_pnn', 'std'),
    pct_Coloc_media=('pct_coloc', 'mean'),
    pct_Coloc_std=('pct_coloc', 'std'),
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
    file_name="resumen_estadistico.csv",
    mime="text/csv",
)
