import os
import sys
import io
import json
import zipfile
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def get_pipeline_parameters_dict(pixel_size_um=0.8913, standard_roi_area_mm2=0.348015):
    """
    Retorna un diccionario estructurado y exhaustivo con todos los parámetros
    y configuraciones de cada etapa del pipeline de procesamiento.
    """
    return {
        "metadata_general": {
            "proyecto": "PNN-SCC-ANALYSIS (Corteza Somatosensorial)",
            "fecha_exportacion": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "organismo": "Rattus norvegicus / Mus musculus (Modelo Experimental SCC)",
            "region_anatomica": "Corteza Somatosensorial Primaria (S1 / SCC)",
            "canales_fluorescencia": {
                "Canal 0 (DAPI / AGR488)": "Contramarcación Nuclear (405 / 488 nm)",
                "Canal 1 (PV 546)": "Interneuronas Parvalbúmina-positivas (PV546 nm)",
                "Canal 2 (WFA 647)": "Redes Perineuronales - Wisteria floribunda agglutinin (647 nm)"
            }
        },
        "optica_y_calibracion_espacial": {
            "magnificacion_objetivo": "10x",
            "resolucion_sensor_px": "2048 x 2048",
            "escala_espacial_um_por_px": pixel_size_um,
            "superficie_campo_completo_global_um2": round((2048 * pixel_size_um) ** 2, 2),
            "superficie_campo_completo_global_mm2": round(((2048 * pixel_size_um) ** 2) / 1e6, 4),
            "area_roi_estandar_corteza_mm2": standard_roi_area_mm2,
            "area_roi_estandar_corteza_um2": round(standard_roi_area_mm2 * 1e6, 2)
        },
        "etapa_1_segmentacion_somas_pv": {
            "metodo": "Cellpose Deep Learning (Preentrenado cyto2 / nuclei + adaptativo)",
            "diametro_promedio_estimado_px": 17.0,
            "diametro_promedio_estimado_um": round(17.0 * pixel_size_um, 2),
            "flow_threshold": 0.40,
            "cellprob_threshold": 0.00,
            "min_size_px": 15,
            "preprocesamiento_pv": "Normalización por contraste dinámico / Percentil robusto",
            "salida": "Máscara de Somas PV (m_pv.tif) y métricas morfométricas individuales"
        },
        "etapa_2_deteccion_redes_pnn_wfa": {
            "arquitectura": "Doble Etapa de Deep Learning (Localizador Convolucional + Clasificador de Parches ResNet)",
            "preprocesamiento_wfa": "Ninguno (Raw) — Señal nativa en rango dinámico completo sin filtros espaciales destructivos",
            "etapa_2a_pnn_loc": {
                "modelo": "Red Neuronal Convolucional de Localización de Centroides (model_loc)",
                "tamano_mosaico_px": 512,
                "solapamiento_mosaicos_px": 64,
                "supresion_no_maximos_nms": 0.30,
                "pnn_loc_threshold": "Umbral de sensibilidad inicial para detectar coordenadas candidatas (Y, X)"
            },
            "etapa_2b_pnn_score": {
                "modelo": "Red Neuronal Profunda de Clasificación de Parches (model_score / ResNet)",
                "tamano_parche_evaluado_px": "64 x 64",
                "pnn_score_threshold": "0.50 (Probabilidad mínima requerida para confirmar una red PNN auténtica)",
                "score_salida": "PNNscore continuo entre 0.0 y 1.0 (Confianza de la IA)"
            },
            "etapa_2c_delineacion_malla_pericelular": {
                "radio_disco_perasomatico_um": 20.0,
                "radio_disco_perasomatico_px": round(20.0 / pixel_size_um, 1),
                "anillo_pericelular": "Corona externa (20 µm) menos erosión de soma (2.0 µm)",
                "delineacion_morfometrica": "Umbralización adaptativa local (Otsu sobre el disco perasomático) para aislar la malla real sobre el fondo local circundante",
                "metricas_resultantes": "Área de malla (µm²), diámetro (µm), densidad integrada y WFA pericelular"
            }
        },
        "etapa_3_colocalizacion_y_clasificacion_subpoblaciones": {
            "criterio_colocalizacion": "Contención espacial del soma PV dentro de la corona/malla PNN (>50% superposición o distancia < 5.0 µm)",
            "clases_celulares": [
                "PV+/PNN+ (Interneurona PV envuelta en Red Perineuronal / PNN Rellena)",
                "PV+/PNN- (Interneurona PV aislada / Libre de PNN)",
                "PV-/PNN+ (Red PNN Hueca que rodea soma no-PV, ej. piramidal u otro)",
                "PV-/PNN- (Núcleos / células no marcadas en el campo)"
            ],
            "mascara_validacion": "m_pv_pnn_coloc (Somas PV+ con PNN confirmados para auditoría visual en Napari)"
        },
        "etapa_4_cuantificacion_espacio_difuso_neuropilo": {
            "definicion": "Intensidad media de fluorescencia en el parénquima libre de somas y condensaciones de WFA (m_wfa = 0)",
            "ambitos_medicion": [
                "Dentro del ROI cortical estándar",
                "Fuera del ROI (tejido circundante)",
                "Global (campo completo del preparado)"
            ],
            "ratio_contraste": "Intensidad Media de Red PNN / Intensidad Fondo Difuso ROI"
        },
        "etapa_5_formulaciones_densidad_y_ponderacion_area": {
            "densidad_numerica_mm2": "N_componentes / Area_ROI (mm²)",
            "densidad_integrada_por_mm2_roi": "Sum(wfa_sum_intensity) / Area_ROI (mm²)",
            "densidad_integrada_por_um2_malla": "Sum(wfa_sum_intensity) / Sum(pnn_area_um2)",
            "cobertura_pnn_pct_roi": "(Sum(pnn_area_um2) / (Area_ROI_mm2 * 1e6)) * 100",
            "potencia_wfa_lupori_energy": "Densidad_PNN (redes/mm²) * Intensidad_Media_Red",
            "potencia_wfa_ponderada_cobertura": "(Cobertura_PNN_pct / 100) * Intensidad_Media_Red",
            "porcentaje_pv_envueltas": "(N_PV_PNN_coloc / N_PV_total) * 100",
            "porcentaje_pnn_huecas": "(N_PNN_huecas / N_PNN_total) * 100"
        }
    }

def format_prism_group_name(condition, sex, section):
    """Genera un rótulo unificado y estandarizado para GraphPad Prism."""
    c_str = str(condition).strip().replace(" ", "_").upper()
    s_str = str(sex).strip().upper()
    sec_str = str(section).strip().upper()
    return f"{c_str}_{s_str}_{sec_str}"

def generate_prism_column_matrix(df_source, metric_col, level='subject'):
    """
    Transforma un DataFrame en formato Long a una matriz ancha en Formato Columnar de GraphPad Prism.
    Cada columna es un grupo experimental (ej. NONE_MACHO_IPSI) y las filas son réplicas/sujetos individuales.
    """
    if df_source.empty or metric_col not in df_source.columns:
        return pd.DataFrame()

    df = df_source.copy()
    
    # Ensure grouping columns
    if 'condition' not in df.columns or 'sex' not in df.columns or 'section' not in df.columns:
        return pd.DataFrame()

    # Drop NaNs for the metric
    df_clean = df.dropna(subset=[metric_col]).copy()
    if df_clean.empty:
        return pd.DataFrame()

    # Aggregate if multiple entries per animal (e.g. if slice level)
    if level == 'subject' and 'animal_id' in df_clean.columns:
        df_clean = df_clean.groupby(['animal_id', 'condition', 'sex', 'section'], as_index=False)[metric_col].mean()

    # Create prism group label
    df_clean['prism_group'] = df_clean.apply(
        lambda r: format_prism_group_name(r['condition'], r['sex'], r['section']),
        axis=1
    )

    # Pivot to columns
    groups = df_clean['prism_group'].unique().tolist()
    # Sort groups logically: NONE first, then ACF 3D, ACF 14D
    def sort_key(g):
        prio = 0
        if "NONE" in g: prio = 10
        elif "3" in g: prio = 20
        elif "14" in g: prio = 30
        else: prio = 40
        sex_p = 1 if "MACHO" in g else 2
        sec_p = 1 if "IPSI" in g else 2
        return (prio, sex_p, sec_p, g)

    sorted_groups = sorted(groups, key=sort_key)
    
    # Collect lists of values per group
    cols_dict = {}
    max_len = 0
    for g in sorted_groups:
        vals = df_clean[df_clean['prism_group'] == g][metric_col].values.tolist()
        cols_dict[g] = vals
        if len(vals) > max_len:
            max_len = len(vals)

    # Pad with NaN to form rectangular DataFrame
    for g in sorted_groups:
        while len(cols_dict[g]) < max_len:
            cols_dict[g].append(np.nan)

    prism_df = pd.DataFrame(cols_dict)
    return prism_df

def generate_prism_grouped_2way_matrix(df_source, metric_col):
    """
    Genera una tabla Prism 2-Way Grouped:
    Filas: Grupos experimentales (Condición + Sexo)
    Columnas: Hemisferios (IPSI vs CONTRA con réplicas)
    """
    if df_source.empty or metric_col not in df_source.columns:
        return pd.DataFrame()

    df = df_source.copy().dropna(subset=[metric_col])
    if 'animal_id' in df.columns:
        df = df.groupby(['animal_id', 'condition', 'sex', 'section'], as_index=False)[metric_col].mean()

    df['row_group'] = df['condition'].astype(str) + " (" + df['sex'].astype(str) + ")"
    
    # Split IPSI and CONTRA
    ipsi_df = df[df['section'] == 'IPSI']
    contra_df = df[df['section'] == 'CONTRA']

    row_groups = df['row_group'].unique().tolist()
    
    out_rows = []
    for rg in row_groups:
        v_ip = ipsi_df[ipsi_df['row_group'] == rg][metric_col].values.tolist()
        v_ct = contra_df[contra_df['row_group'] == rg][metric_col].values.tolist()
        max_reps = max(len(v_ip), len(v_ct), 1)
        for i in range(max_reps):
            out_rows.append({
                "Grupo_Experimental": rg,
                "Replica_Num": i + 1,
                "IPSI": v_ip[i] if i < len(v_ip) else np.nan,
                "CONTRA": v_ct[i] if i < len(v_ct) else np.nan
            })

    return pd.DataFrame(out_rows)

def build_full_export_zip(df_subj, df_img, df_cells, all_vars_dict, pixel_size_um=0.8913, standard_roi_area_mm2=0.348015):
    """
    Genera un archivo ZIP en memoria con todos los datos crudos organizados,
    matrices GraphPad Prism (CSV y Excel multi-hoja) y la ficha de parámetros.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Master Dataset por Sujeto
        if not df_subj.empty:
            zf.writestr("01_Master_Dataset_Por_Sujeto_Long.csv", df_subj.to_csv(index=False).encode('utf-8'))

        # 2. Master Dataset por Preparado / Corte
        if not df_img.empty:
            zf.writestr("02_Master_Dataset_Por_Preparado_Corte_Long.csv", df_img.to_csv(index=False).encode('utf-8'))

        # 3. Master Dataset Unicelular (todas las células/PNNs segmentadas)
        if not df_cells.empty:
            zf.writestr("03_Master_Dataset_Unicelular_Completo.csv", df_cells.to_csv(index=False).encode('utf-8'))

        # 4. Excel Multi-Hoja Formato Prism Columnar
        key_metrics_for_prism = [
            ("pnn_density_roi", "PNN_Densidad_ROI"),
            ("pv_density_roi", "PV_Densidad_ROI"),
            ("pv_coloc_density_roi", "PV_PNN_Envueltas_Dens_ROI"),
            ("pct_pv_wrapped_roi", "Pct_PV_Envueltas_ROI"),
            ("pnn_filled_density_roi", "PNN_Rellenas_Dens_ROI"),
            ("pnn_hollow_density_roi", "PNN_Huecas_Dens_ROI"),
            ("pct_pnn_hollow_roi", "Pct_PNN_Huecas_ROI"),
            ("wfa_int_density_per_mm2_roi", "WFA_IntDens_Por_mm2_ROI"),
            ("total_wfa_int_density_roi", "WFA_Total_Sum_ROI"),
            ("wfa_int_per_mesh_um2_roi", "WFA_IntDens_Por_um2_Malla"),
            ("pnn_area_fraction_pct_roi", "PNN_Cobertura_Area_Pct_ROI"),
            ("pnn_energy_roi", "PNN_Lupori_Energy_ROI"),
            ("pnn_energy_area_weighted_roi", "PNN_Energy_Ponderada_ROI"),
            ("mean_wfa_intensity_roi", "WFA_Intensidad_Media_ROI"),
            ("mean_pnn_area_roi", "PNN_Area_Malla_um2_ROI"),
            ("mean_pv_area_roi", "PV_Area_Soma_um2_ROI"),
            ("wfa_diffuse_mean_roi", "WFA_Fondo_Difuso_ROI"),
            ("wfa_signal_to_diffuse_ratio_roi", "Ratio_Senal_Fondo_Difuso_ROI")
        ]

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Add Master sheets
            if not df_subj.empty:
                df_subj.to_excel(writer, sheet_name="Master_Sujeto", index=False)
            if not df_img.empty:
                df_img.to_excel(writer, sheet_name="Master_Preparado", index=False)

            # Add each Prism matrix
            for var_col, sheet_name in key_metrics_for_prism:
                if var_col in df_subj.columns:
                    p_mat = generate_prism_column_matrix(df_subj, var_col, level='subject')
                    if not p_mat.empty:
                        # Truncate sheet name to 31 chars (Excel limit)
                        s_name = sheet_name[:31]
                        p_mat.to_excel(writer, sheet_name=s_name, index=False)

        zf.writestr("04_Matrices_GraphPad_Prism_Column_Format.xlsx", excel_buffer.getvalue())

        # 5. Directorio de CSVs individuales para GraphPad Prism
        for var_col, sheet_name in key_metrics_for_prism:
            if var_col in df_subj.columns:
                p_mat = generate_prism_column_matrix(df_subj, var_col, level='subject')
                if not p_mat.empty:
                    zf.writestr(f"05_Matrices_Prism_CSVs/{sheet_name}.csv", p_mat.to_csv(index=False).encode('utf-8'))

        # 6. Ficha de Parámetros del Pipeline en JSON y TXT
        params_dict = get_pipeline_parameters_dict(pixel_size_um, standard_roi_area_mm2)
        params_json_str = json.dumps(params_dict, indent=4, ensure_ascii=False)
        zf.writestr("06_Parametros_Corrida_Pipeline.json", params_json_str.encode('utf-8'))

        # Formato texto plano legible
        txt_lines = [
            "================================================================================",
            "FICHA TÉCNICA Y REGISTRO DE PARÁMETROS DE CORRIDA DEL EXPERIMENTO",
            "Proyecto: Cuantificación y Análisis Morfométrico de Redes Perineuronales (WFA) y PV+",
            f"Fecha de Generación: {params_dict['metadata_general']['fecha_exportacion']}",
            "================================================================================\n",
            "1. ESPECIFICACIONES ÓPTICAS Y CALIBRACIÓN:",
            f"   - Magnificación Objetivo: {params_dict['optica_y_calibracion_espacial']['magnificacion_objetivo']}",
            f"   - Escala Espacial: {params_dict['optica_y_calibracion_espacial']['escala_espacial_um_por_px']} µm/px",
            f"   - Dimensión Campo Completo: {params_dict['optica_y_calibracion_espacial']['resolucion_sensor_px']} px ({params_dict['optica_y_calibracion_espacial']['superficie_campo_completo_global_mm2']} mm²)",
            f"   - Área ROI Estándar de Corteza: {params_dict['optica_y_calibracion_espacial']['area_roi_estandar_corteza_mm2']} mm² ({params_dict['optica_y_calibracion_espacial']['area_roi_estandar_corteza_um2']} µm²)\n",
            "2. SEGMENTACIÓN DE SOMAS PV+ (CELLPOSE):",
            f"   - Modelo: {params_dict['etapa_1_segmentacion_somas_pv']['metodo']}",
            f"   - Diámetro Estimado: {params_dict['etapa_1_segmentacion_somas_pv']['diametro_promedio_estimado_px']} px ({params_dict['etapa_1_segmentacion_somas_pv']['diametro_promedio_estimado_um']} µm)",
            f"   - Flow Threshold: {params_dict['etapa_1_segmentacion_somas_pv']['flow_threshold']}",
            f"   - Cellprob Threshold: {params_dict['etapa_1_segmentacion_somas_pv']['cellprob_threshold']}\n",
            "3. DETECCIÓN DE REDES PNN (WFA) — ARQUITECTURA DE DOS ETAPAS:",
            f"   - Preprocesamiento WFA: {params_dict['etapa_2_deteccion_redes_pnn_wfa']['preprocesamiento_wfa']}",
            f"   - Etapa A (PNN Loc): {params_dict['etapa_2_deteccion_redes_pnn_wfa']['etapa_2a_pnn_loc']['modelo']} (Mosaicos: {params_dict['etapa_2_deteccion_redes_pnn_wfa']['etapa_2a_pnn_loc']['tamano_mosaico_px']} px, NMS: {params_dict['etapa_2_deteccion_redes_pnn_wfa']['etapa_2a_pnn_loc']['supresion_no_maximos_nms']})",
            f"   - Etapa B (PNN Score): {params_dict['etapa_2_deteccion_redes_pnn_wfa']['etapa_2b_pnn_score']['modelo']} (Parche: {params_dict['etapa_2_deteccion_redes_pnn_wfa']['etapa_2b_pnn_score']['tamano_parche_evaluado_px']}, Umbral Confirmación: {params_dict['etapa_2_deteccion_redes_pnn_wfa']['etapa_2b_pnn_score']['pnn_score_threshold']})",
            f"   - Delineación de Malla: {params_dict['etapa_2_deteccion_redes_pnn_wfa']['etapa_2c_delineacion_malla_pericelular']['delineacion_morfometrica']}\n",
            "4. COLOCALIZACIÓN Y CLASIFICACIÓN CELULAR:",
            f"   - Criterio: {params_dict['etapa_3_colocalizacion_y_clasificacion_subpoblaciones']['criterio_colocalizacion']}",
            "   - Clases: PV+/PNN+ (Rellena), PV+/PNN- (Libre), PV-/PNN+ (Hueca), PV-/PNN- (No marcada)\n",
            "5. FORMULACIONES DE NORMALIZACIÓN Y PONDERACIÓN:",
            f"   - Densidad Numérica: {params_dict['etapa_5_formulaciones_densidad_y_ponderacion_area']['densidad_numerica_mm2']}",
            f"   - Densidad Integrada por mm² de ROI: {params_dict['etapa_5_formulaciones_densidad_y_ponderacion_area']['densidad_integrada_por_mm2_roi']}",
            f"   - Densidad Integrada por µm² de Malla: {params_dict['etapa_5_formulaciones_densidad_y_ponderacion_area']['densidad_integrada_por_um2_malla']}",
            f"   - Cobertura PNN (%): {params_dict['etapa_5_formulaciones_densidad_y_ponderacion_area']['cobertura_pnn_pct_roi']}",
            f"   - Potencia WFA (Lupori Energy): {params_dict['etapa_5_formulaciones_densidad_y_ponderacion_area']['potencia_wfa_lupori_energy']}",
            "================================================================================"
        ]
        zf.writestr("06_Parametros_Corrida_Pipeline.txt", "\n".join(txt_lines).encode('utf-8'))

        # 7. Guía README para importar en GraphPad Prism
        readme_md = """# Guía de Uso e Importación de Datos en GraphPad Prism

Este paquete contiene la exportación integral de datos crudos procesados y parámetros de corrida del pipeline de análisis de Redes Perineuronales (PNN) e Interneuronas PV+ en corteza somatosensorial.

---

## 📁 Contenido del Paquete

1. **`01_Master_Dataset_Por_Sujeto_Long.csv`**:
   - Tabla consolidada promediada por animal (Sujeto).
   - Variables de diseño: `animal_id`, `condition`, `sex`, `section`, `group`.
   - Todas las métricas de densidad, morfometría, intensidades ponderadas y ratios.

2. **`02_Master_Dataset_Por_Preparado_Corte_Long.csv`**:
   - Tabla con cada corte histológico individual (`image_name`, `corte_num`).

3. **`03_Master_Dataset_Unicelular_Completo.csv`**:
   - Registro individual de cada célula, soma PV+ y red PNN segmentada en el experimento.
   - Incluye centroides espaciales `(centroid_y, centroid_x)`, área en µm², diámetros, intensidades WFA pericelulares y de neuropilo, pertenencia a ROI y subpoblación celular (`cell_type`).

4. **`04_Matrices_GraphPad_Prism_Column_Format.xlsx`**:
   - Libro de Excel con una pestaña por variable principal estructurada en **Formato Columnar de Prism**.
   - Cada columna corresponde a un grupo experimental (`NONE_MACHO_IPSI`, `ACF_3D_MACHO_IPSI`, etc.).
   - Las filas contienen los valores individuales de cada réplica biológica (animal).

5. **`05_Matrices_Prism_CSVs/`**:
   - Archivos CSV individuales por variable para importar directamente en GraphPad Prism sin necesidad de manipular hojas de cálculo.

6. **`06_Parametros_Corrida_Pipeline.json` y `.txt`**:
   - Registro reproducible con todos los hiperparámetros ópticos, umbrales y modelos utilizados.

---

## 🚀 Cómo Importar en GraphPad Prism

### Opción A: Copiar y Pegar Directo (Formato Columnar - Column Table)
1. En GraphPad Prism, crea un nuevo archivo y selecciona **Column Table**.
2. Abre `04_Matrices_GraphPad_Prism_Column_Format.xlsx` o el CSV correspondiente de la carpeta `05_Matrices_Prism_CSVs/`.
3. Selecciona todas las columnas y filas con datos y pégalas en la tabla de Prism (`Ctrl + V`).
4. Haz clic en **Analyze** para ejecutar ANOVA de 1 Vía, Test de Student o comparaciones múltiples (Tukey, Bonferroni, Sidak).

### Opción B: Tabla Agrupada (Grouped Table - 2-Way / 3-Way ANOVA)
1. En GraphPad Prism, selecciona **Grouped Table**.
2. Define las columnas como **IPSI** y **CONTRA** (o **MACHO** y **HEMBRA**) con réplicas en subcolumnas.
3. Define las filas con las condiciones experimentales (**NONE**, **ACF 3 DÍAS**, **ACF 14 DÍAS**).
"""
        zf.writestr("README_GUIA_IMPORTACION_PRISMA.md", readme_md.encode('utf-8'))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def build_all_graphics_and_data_zip(df_data_base, level_name="sujeto"):
    """
    Genera un paquete ZIP exhaustivo y estructurado que contiene:
    1. Todas las combinaciones de agrupaciones y cohortes (Global, Solo NONE, Solo ACF, Filtros de Sexo y Hemisferio).
    2. Para cada combinación:
       - Gráfico PNG de alta resolución (scale=2).
       - Archivo CSV con los datos exactos con el MISMO NOMBRE que el gráfico.
       - Archivo HTML interactivo autónomo de Plotly.
    3. Archivo CSV con toda la información RAW del experimento en la raíz.
    4. Resúmenes cuantitativos y guía de estructura.
    """
    zip_buffer = io.BytesIO()

    # Pre-clean dataframe
    df = df_data_base.copy()
    if 'cond_order' not in df.columns and 'condition' in df.columns:
        df['cond_order'] = df['condition'].map(lambda c: 0 if c == 'NONE' else (1 if '3' in str(c) else (2 if '14' in str(c) else 3)))
        df = df.sort_values(by=['cond_order', 'sex', 'section'])

    # Helper function to generate and write figure + csv + html into zip
    def _export_plot_and_data_to_zip(zf, folder_path, file_basename, fig, df_plot_data):
        # 1. Write CSV with exact data
        csv_bytes = df_plot_data.to_csv(index=False).encode('utf-8')
        zf.writestr(f"{folder_path}/{file_basename}.csv", csv_bytes)

        # 2. Write PNG image
        try:
            png_bytes = fig.to_image(format='png', scale=2)
            zf.writestr(f"{folder_path}/{file_basename}.png", png_bytes)
        except Exception:
            pass

        # 3. Write HTML interactive
        html_bytes = fig.to_html(include_plotlyjs='cdn').encode('utf-8')
        zf.writestr(f"{folder_path}/{file_basename}.html", html_bytes)

    # Helper for bar chart
    def _build_bar(df_sub, metric_col, title_txt, y_lbl, grp_mode):
        if df_sub.empty:
            return go.Figure()
        
        if grp_mode == "pool_condicion":
            agg = df_sub.groupby(['condition'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            fig = px.bar(
                agg, x='condition', y='mean_val', color='condition', error_y='sem_val',
                color_discrete_map={'NONE': '#00ff88', 'ACF 3 DÍAS': '#ff3366', 'ACF 14 DÍAS': '#ffbb00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Condición", showlegend=False)
            return fig, agg

        elif grp_mode == "por_hemisferio":
            agg = df_sub.groupby(['condition', 'section'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            fig = px.bar(
                agg, x='condition', y='mean_val', color='section', barmode='group', error_y='sem_val',
                color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Condición", legend_title="Hemisferio")
            return fig, agg

        elif grp_mode == "por_sexo":
            agg = df_sub.groupby(['condition', 'sex'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            fig = px.bar(
                agg, x='condition', y='mean_val', color='sex', barmode='group', error_y='sem_val',
                color_discrete_map={'MACHO': '#00f2fe', 'HEMBRA': '#ff7b00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Condición", legend_title="Sexo")
            return fig, agg

        else:
            # Desglose Completo
            agg = df_sub.groupby(['condition', 'section', 'sex'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            agg['group_lbl'] = agg['condition'] + " (" + agg['sex'] + ")"
            fig = px.bar(
                agg, x='group_lbl', y='mean_val', color='section', barmode='group', error_y='sem_val',
                color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Grupo Experimental", legend_title="Hemisferio")
            return fig, agg

    # Helper for boxplot
    def _build_box(df_sub, metric_col, title_txt, y_lbl, grp_mode):
        if df_sub.empty:
            return go.Figure()
        fig = go.Figure()

        if grp_mode == "pool_condicion":
            for cond_name, col_cond in [('NONE', '#00ff88'), ('ACF 3 DÍAS', '#ff3366'), ('ACF 14 DÍAS', '#ffbb00')]:
                df_c = df_sub[df_sub['condition'] == cond_name]
                if not df_c.empty:
                    fig.add_trace(go.Box(
                        x=df_c['condition'], y=df_c[metric_col], name=cond_name, marker_color=col_cond,
                        boxmean='sd', boxpoints='all', jitter=0.3, marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, title=title_txt, yaxis_title=y_lbl, template='plotly_dark', showlegend=False)
            return fig, df_sub

        elif grp_mode == "por_hemisferio":
            for sec_name, col_sec in [('IPSI', '#00f2fe'), ('CONTRA', '#ff7b00')]:
                df_s = df_sub[df_sub['section'] == sec_name]
                if not df_s.empty:
                    fig.add_trace(go.Box(
                        x=df_s['condition'], y=df_s[metric_col], name=sec_name, marker_color=col_sec,
                        boxmean='sd', boxpoints='all', jitter=0.3, pointpos=-0.2 if sec_name == 'IPSI' else 0.2,
                        marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_lbl, template='plotly_dark')
            return fig, df_sub

        elif grp_mode == "por_sexo":
            for sx_name, col_sx in [('MACHO', '#00f2fe'), ('HEMBRA', '#ff7b00')]:
                df_sx = df_sub[df_sub['sex'] == sx_name]
                if not df_sx.empty:
                    fig.add_trace(go.Box(
                        x=df_sx['condition'], y=df_sx[metric_col], name=sx_name, marker_color=col_sx,
                        boxmean='sd', boxpoints='all', jitter=0.3, pointpos=-0.2 if sx_name == 'MACHO' else 0.2,
                        marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_lbl, template='plotly_dark')
            return fig, df_sub

        else:
            # Desglose Completo
            for sec_name, col_sec in [('IPSI', '#00f2fe'), ('CONTRA', '#ff7b00')]:
                df_s = df_sub[df_sub['section'] == sec_name]
                if not df_s.empty:
                    fig.add_trace(go.Box(
                        x=df_s['condition'] + " - " + df_s['sex'], y=df_s[metric_col], name=sec_name, marker_color=col_sec,
                        boxmean='sd', boxpoints='all', jitter=0.3, pointpos=-0.2 if sec_name == 'IPSI' else 0.2,
                        marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_lbl, template='plotly_dark')
            return fig, df_sub

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. ROOT RAW DATA CSV & EXCEL
        raw_cols_all = [c for c in [
            'animal_id', 'image_name', 'group', 'condition', 'sex', 'section',
            'pnn_density_roi', 'wfa_int_density_per_mm2_roi', 'wfa_int_per_mesh_um2_roi',
            'pnn_area_fraction_pct_roi', 'total_wfa_int_density_roi', 'mean_wfa_int_density_roi',
            'mean_wfa_intensity_roi', 'mean_pnn_area_roi', 'mean_pnn_diameter_roi',
            'pv_density_roi', 'pnn_filled_density_roi', 'pnn_hollow_density_roi', 'pv_isolated_density_roi'
        ] if c in df.columns]
        
        df_raw_master = df[raw_cols_all].copy()
        zf.writestr("00_RAW_DATA_COMPLETA_MASTER.csv", df_raw_master.to_csv(index=False).encode('utf-8'))

        # Matrices Prism
        df_prism_mm2 = generate_prism_column_matrix(df, 'wfa_int_density_per_mm2_roi', level=level_name)
        df_prism_raw = generate_prism_column_matrix(df, 'total_wfa_int_density_roi', level=level_name)
        df_prism_mesh = generate_prism_column_matrix(df, 'wfa_int_per_mesh_um2_roi', level=level_name)
        df_prism_2way = generate_prism_grouped_2way_matrix(df, 'wfa_int_density_per_mm2_roi')

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as wr:
            df_raw_master.to_excel(wr, sheet_name="Raw_Data_Completa", index=False)
            df_prism_mm2.to_excel(wr, sheet_name="Prism_Ponderada_mm2_ROI", index=False)
            df_prism_raw.to_excel(wr, sheet_name="Prism_NoPonderada_SumaRaw", index=False)
            df_prism_mesh.to_excel(wr, sheet_name="Prism_Ponderada_Malla_um2", index=False)
            df_prism_2way.to_excel(wr, sheet_name="Prism_2Way_IPSI_vs_CONTRA", index=False)
        zf.writestr("00_MATRICES_GRAPHPAD_PRISM_Y_RAW_DATA.xlsx", excel_buf.getvalue())

        # Parameters
        params_dict = get_pipeline_parameters_dict()
        zf.writestr("00_PARAMETROS_RUN_EXPERIMENTO.json", json.dumps(params_dict, indent=2, ensure_ascii=False).encode('utf-8'))

        # 2. DEFINITION OF SCOPES AND SUB-CONFIGURATIONS
        scope_configs = [
            # A. Global Comparison
            ("01_Comparacion_Global_NONE_vs_ACF", df),
            # B. Solo NONE
            ("02_Solo_Controles_Basales_NONE", df[df['condition'] == 'NONE']),
            # C. Solo ACF
            ("03_Solo_Tratamientos_Lesion_ACF", df[df['condition'].isin(['ACF 3 DÍAS', 'ACF 14 DÍAS'])]),
            # D. Filtros por Sexo
            ("04_Filtro_Solo_MACHOS", df[df['sex'] == 'MACHO']),
            ("05_Filtro_Solo_HEMBRAS", df[df['sex'] == 'HEMBRA']),
            # E. Filtros por Hemisferio
            ("06_Filtro_Solo_IPSI_Lesionado", df[df['section'] == 'IPSI']),
            ("07_Filtro_Solo_CONTRA_Intacto", df[df['section'] == 'CONTRA'])
        ]

        grouping_modes = [
            ("01_Desglose_Completo_Condicion_Sexo_Hemisferio", "desglose_completo"),
            ("02_Pool_Condicion_Global", "pool_condicion"),
            ("03_Por_Condicion_y_Hemisferio", "por_hemisferio"),
            ("04_Por_Condicion_y_Sexo", "por_sexo")
        ]

        metrics_info = [
            (
                "grafico_wfa_pnn_ponderada_mm2_roi",
                "wfa_int_density_per_mm2_roi",
                "Densidad Integrada de Redes PNN por mm² ROI (Media ± SEM)",
                "Densidad Integrada de PNNs / mm² ROI"
            ),
            (
                "grafico_wfa_pnn_no_ponderada_suma_neta",
                "total_wfa_int_density_roi",
                "Suma Total de Fluorescencia en Redes PNN en ROI (Media ± SEM)",
                "Suma de Fluorescencia en PNNs"
            ),
            (
                "grafico_wfa_pnn_ponderada_malla_um2",
                "wfa_int_per_mesh_um2_roi",
                "Densidad Integrada por µm² de Malla PNN (Concentración Intrínseca)",
                "Densidad Integrada de PNNs / µm² Malla"
            ),
            (
                "grafico_wfa_pnn_promedio_por_red",
                "mean_wfa_int_density_roi",
                "Densidad Integrada Promedio por Red PNN Individual (Media ± SEM)",
                "Fluorescencia Media por Red PNN"
            ),
            (
                "grafico_pnn_boxplot_distribucion_individual",
                "wfa_int_density_per_mm2_roi",
                "Distribución de Réplicas: WFA en PNNs / mm² ROI",
                "Densidad Integrada de PNNs / mm² ROI"
            )
        ]

        # 3. GENERATE ALL COMBINATIONS
        for scope_folder, df_scope in scope_configs:
            if df_scope.empty:
                continue

            for grp_folder, grp_mode in grouping_modes:
                dest_path = f"{scope_folder}/{grp_folder}"

                for file_base, metric_col, title_txt, y_lbl in metrics_info:
                    if metric_col not in df_scope.columns:
                        continue

                    if "boxplot" in file_base:
                        fig, df_pdata = _build_box(df_scope, metric_col, f"{title_txt} [{scope_folder[3:].replace('_', ' ')}]", y_lbl, grp_mode)
                    else:
                        fig, df_pdata = _build_bar(df_scope, metric_col, f"{title_txt} [{scope_folder[3:].replace('_', ' ')}]", y_lbl, grp_mode)

                    _export_plot_and_data_to_zip(zf, dest_path, file_base, fig, df_pdata)

        # 4. README GUIDE
        readme_text = f"""# PAQUETE COMPLETO DE GRÁFICOS Y DATOS CRUDOS
Fecha de Exportación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Nivel de Análisis: {level_name.upper()}
Número Total de Observaciones: {len(df)}

Este archivo ZIP contiene la totalidad de gráficos del experimento exportados en TODAS las combinaciones de agrupación y filtros posibles, acompañados de sus archivos CSV con los datos exactos que componen cada figura.

ESTRUCTURA DE CARPETAS:
================================================================================
1. 00_RAW_DATA_COMPLETA_MASTER.csv
   - Base de datos cruda completa con todos los sujetos, cortes y métricas.

2. 00_MATRICES_GRAPHPAD_PRISM_Y_RAW_DATA.xlsx
   - Libro de Excel multi-hoja estructurado para GraphPad Prism.

3. 01_Comparacion_Global_NONE_vs_ACF/
   - Controles basales (NONE) frente a grupos con lesión excitotóxica (ACF 3D y 14D).
   - Subcarpetas por modo de agrupación (Desglose Completo, Pool por Condición, Por Hemisferio, Por Sexo).

4. 02_Solo_Controles_Basales_NONE/
   - Gráficos y datos exclusivos de animales Control Basal (NONE).

5. 03_Solo_Tratamientos_Lesion_ACF/
   - Gráficos y datos exclusivos de los tratamientos de lesión temporal (ACF 3D vs ACF 14D).

6. 04_Filtro_Solo_MACHOS/ y 05_Filtro_Solo_HEMBRAS/
   - Evaluaciones filtradas por sexo.

7. 06_Filtro_Solo_IPSI_Lesionado/ y 07_Filtro_Solo_CONTRA_Intacto/
   - Evaluaciones filtradas por hemisferio.

FORMATOS POR GRÁFICA:
================================================================================
Cada gráfico cuenta con tres archivos sincronizados con el MISMO NOMBRE:
- [nombre_grafico].png  -> Imagen estática en alta resolución (renderizada a 300 DPI con escala x2).
- [nombre_grafico].csv  -> Datos numéricos exactos a partir de los cuales se construyó la gráfica.
- [nombre_grafico].html -> Gráfico interactivo autónomo de Plotly (con zoom, hover y rotación).
"""
        zf.writestr("README_ESTRUCTURA_Y_METRICAS.txt", readme_text.encode('utf-8'))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_all_graphics_and_data_to_directory(df_data_base, target_directory=None, level_name="sujeto"):
    """
    Genera directamente en el disco una carpeta con toda la estructura de subcarpetas ordenadas:
    - 01_Comparacion_Global_NONE_vs_ACF/
    - 02_Solo_Controles_Basales_NONE/
    - 03_Solo_Tratamientos_Lesion_ACF/
    - 04_Filtro_Solo_MACHOS/
    - 05_Filtro_Solo_HEMBRAS/
    - 06_Filtro_Solo_IPSI_Lesionado/
    - 07_Filtro_Solo_CONTRA_Intacto/
    Cada una conteniendo sus subcarpetas por agrupación y los archivos .png y .csv emparejados.
    """
    if target_directory is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_directory = os.path.join(os.getcwd(), "exports", f"graficas_y_datos_completos_{timestamp_str}")

    os.makedirs(target_directory, exist_ok=True)

    df = df_data_base.copy()
    if 'cond_order' not in df.columns and 'condition' in df.columns:
        df['cond_order'] = df['condition'].map(lambda c: 0 if c == 'NONE' else (1 if '3' in str(c) else (2 if '14' in str(c) else 3)))
        df = df.sort_values(by=['cond_order', 'sex', 'section'])

    # Helper for bar chart
    def _build_bar(df_sub, metric_col, title_txt, y_lbl, grp_mode):
        if df_sub.empty:
            return go.Figure(), pd.DataFrame()
        
        if grp_mode == "pool_condicion":
            agg = df_sub.groupby(['condition'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            fig = px.bar(
                agg, x='condition', y='mean_val', color='condition', error_y='sem_val',
                color_discrete_map={'NONE': '#00ff88', 'ACF 3 DÍAS': '#ff3366', 'ACF 14 DÍAS': '#ffbb00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Condición", showlegend=False)
            return fig, agg

        elif grp_mode == "por_hemisferio":
            agg = df_sub.groupby(['condition', 'section'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            fig = px.bar(
                agg, x='condition', y='mean_val', color='section', barmode='group', error_y='sem_val',
                color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Condición", legend_title="Hemisferio")
            return fig, agg

        elif grp_mode == "por_sexo":
            agg = df_sub.groupby(['condition', 'sex'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            fig = px.bar(
                agg, x='condition', y='mean_val', color='sex', barmode='group', error_y='sem_val',
                color_discrete_map={'MACHO': '#00f2fe', 'HEMBRA': '#ff7b00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Condición", legend_title="Sexo")
            return fig, agg

        else:
            # Desglose Completo
            agg = df_sub.groupby(['condition', 'section', 'sex'], as_index=False).agg(
                mean_val=(metric_col, 'mean'), std_val=(metric_col, 'std'), n=(metric_col, 'count')
            )
            agg['sem_val'] = agg.apply(lambda r: r['std_val']/np.sqrt(r['n']) if r['n']>1 else 0.0, axis=1)
            agg['group_lbl'] = agg['condition'] + " (" + agg['sex'] + ")"
            fig = px.bar(
                agg, x='group_lbl', y='mean_val', color='section', barmode='group', error_y='sem_val',
                color_discrete_map={'IPSI': '#00f2fe', 'CONTRA': '#ff7b00'},
                title=title_txt, template='plotly_dark'
            )
            fig.update_layout(height=480, yaxis_title=y_lbl, xaxis_title="Grupo Experimental", legend_title="Hemisferio")
            return fig, agg

    # Helper for boxplot
    def _build_box(df_sub, metric_col, title_txt, y_lbl, grp_mode):
        if df_sub.empty:
            return go.Figure(), pd.DataFrame()
        fig = go.Figure()

        if grp_mode == "pool_condicion":
            for cond_name, col_cond in [('NONE', '#00ff88'), ('ACF 3 DÍAS', '#ff3366'), ('ACF 14 DÍAS', '#ffbb00')]:
                df_c = df_sub[df_sub['condition'] == cond_name]
                if not df_c.empty:
                    fig.add_trace(go.Box(
                        x=df_c['condition'], y=df_c[metric_col], name=cond_name, marker_color=col_cond,
                        boxmean='sd', boxpoints='all', jitter=0.3, marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, title=title_txt, yaxis_title=y_lbl, template='plotly_dark', showlegend=False)
            return fig, df_sub

        elif grp_mode == "por_hemisferio":
            for sec_name, col_sec in [('IPSI', '#00f2fe'), ('CONTRA', '#ff7b00')]:
                df_s = df_sub[df_sub['section'] == sec_name]
                if not df_s.empty:
                    fig.add_trace(go.Box(
                        x=df_s['condition'], y=df_s[metric_col], name=sec_name, marker_color=col_sec,
                        boxmean='sd', boxpoints='all', jitter=0.3, pointpos=-0.2 if sec_name == 'IPSI' else 0.2,
                        marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_lbl, template='plotly_dark')
            return fig, df_sub

        elif grp_mode == "por_sexo":
            for sx_name, col_sx in [('MACHO', '#00f2fe'), ('HEMBRA', '#ff7b00')]:
                df_sx = df_sub[df_sub['sex'] == sx_name]
                if not df_sx.empty:
                    fig.add_trace(go.Box(
                        x=df_sx['condition'], y=df_sx[metric_col], name=sx_name, marker_color=col_sx,
                        boxmean='sd', boxpoints='all', jitter=0.3, pointpos=-0.2 if sx_name == 'MACHO' else 0.2,
                        marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_lbl, template='plotly_dark')
            return fig, df_sub

        else:
            # Desglose Completo
            for sec_name, col_sec in [('IPSI', '#00f2fe'), ('CONTRA', '#ff7b00')]:
                df_s = df_sub[df_sub['section'] == sec_name]
                if not df_s.empty:
                    fig.add_trace(go.Box(
                        x=df_s['condition'] + " - " + df_s['sex'], y=df_s[metric_col], name=sec_name, marker_color=col_sec,
                        boxmean='sd', boxpoints='all', jitter=0.3, pointpos=-0.2 if sec_name == 'IPSI' else 0.2,
                        marker=dict(size=7), line=dict(width=2)
                    ))
            fig.update_layout(height=480, boxmode='group', title=title_txt, yaxis_title=y_lbl, template='plotly_dark')
            return fig, df_sub

    # 1. ROOT MASTER FILES
    raw_cols_all = [c for c in [
        'animal_id', 'image_name', 'group', 'condition', 'sex', 'section',
        'pnn_density_roi', 'wfa_int_density_per_mm2_roi', 'wfa_int_per_mesh_um2_roi',
        'pnn_area_fraction_pct_roi', 'total_wfa_int_density_roi', 'mean_wfa_int_density_roi',
        'mean_wfa_intensity_roi', 'mean_pnn_area_roi', 'mean_pnn_diameter_roi',
        'pv_density_roi', 'pnn_filled_density_roi', 'pnn_hollow_density_roi', 'pv_isolated_density_roi'
    ] if c in df.columns]

    df_raw_master = df[raw_cols_all].copy()
    df_raw_master.to_csv(os.path.join(target_directory, "00_RAW_DATA_COMPLETA_MASTER.csv"), index=False)

    df_prism_mm2 = generate_prism_column_matrix(df, 'wfa_int_density_per_mm2_roi', level=level_name)
    df_prism_raw = generate_prism_column_matrix(df, 'total_wfa_int_density_roi', level=level_name)
    df_prism_mesh = generate_prism_column_matrix(df, 'wfa_int_per_mesh_um2_roi', level=level_name)
    df_prism_2way = generate_prism_grouped_2way_matrix(df, 'wfa_int_density_per_mm2_roi')

    with pd.ExcelWriter(os.path.join(target_directory, "00_MATRICES_GRAPHPAD_PRISM_Y_RAW_DATA.xlsx"), engine='openpyxl') as wr:
        df_raw_master.to_excel(wr, sheet_name="Raw_Data_Completa", index=False)
        df_prism_mm2.to_excel(wr, sheet_name="Prism_Ponderada_mm2_ROI", index=False)
        df_prism_raw.to_excel(wr, sheet_name="Prism_NoPonderada_SumaRaw", index=False)
        df_prism_mesh.to_excel(wr, sheet_name="Prism_Ponderada_Malla_um2", index=False)
        df_prism_2way.to_excel(wr, sheet_name="Prism_2Way_IPSI_vs_CONTRA", index=False)

    params_dict = get_pipeline_parameters_dict()
    with open(os.path.join(target_directory, "00_PARAMETROS_RUN_EXPERIMENTO.json"), "w", encoding="utf-8") as f:
        json.dump(params_dict, f, indent=2, ensure_ascii=False)

    # 2. DEFINITION OF SCOPES AND SUB-CONFIGURATIONS
    scope_configs = [
        ("01_Comparacion_Global_NONE_vs_ACF", df),
        ("02_Solo_Controles_Basales_NONE", df[df['condition'] == 'NONE']),
        ("03_Solo_Tratamientos_Lesion_ACF", df[df['condition'].isin(['ACF 3 DÍAS', 'ACF 14 DÍAS'])]),
        ("04_Filtro_Solo_MACHOS", df[df['sex'] == 'MACHO']),
        ("05_Filtro_Solo_HEMBRAS", df[df['sex'] == 'HEMBRA']),
        ("06_Filtro_Solo_IPSI_Lesionado", df[df['section'] == 'IPSI']),
        ("07_Filtro_Solo_CONTRA_Intacto", df[df['section'] == 'CONTRA'])
    ]

    grouping_modes = [
        ("01_Desglose_Completo_Condicion_Sexo_Hemisferio", "desglose_completo"),
        ("02_Pool_Condicion_Global", "pool_condicion"),
        ("03_Por_Condicion_y_Hemisferio", "por_hemisferio"),
        ("04_Por_Condicion_y_Sexo", "por_sexo")
    ]

    metrics_info = [
        (
            "grafico_wfa_pnn_ponderada_mm2_roi",
            "wfa_int_density_per_mm2_roi",
            "Densidad Integrada de Redes PNN por mm² ROI (Media ± SEM)",
            "Densidad Integrada de PNNs / mm² ROI"
        ),
        (
            "grafico_wfa_pnn_no_ponderada_suma_neta",
            "total_wfa_int_density_roi",
            "Suma Total de Fluorescencia en Redes PNN en ROI (Media ± SEM)",
            "Suma de Fluorescencia en PNNs"
        ),
        (
            "grafico_wfa_pnn_ponderada_malla_um2",
            "wfa_int_per_mesh_um2_roi",
            "Densidad Integrada por µm² de Malla PNN (Concentración Intrínseca)",
            "Densidad Integrada de PNNs / µm² Malla"
        ),
        (
            "grafico_wfa_pnn_promedio_por_red",
            "mean_wfa_int_density_roi",
            "Densidad Integrada Promedio por Red PNN Individual (Media ± SEM)",
            "Fluorescencia Media por Red PNN"
        ),
        (
            "grafico_pnn_boxplot_distribucion_individual",
            "wfa_int_density_per_mm2_roi",
            "Distribución de Réplicas: WFA en PNNs / mm² ROI",
            "Densidad Integrada de PNNs / mm² ROI"
        )
    ]

    # 3. GENERATE ALL SUBDIRECTORIES AND WRITE FILES
    for scope_folder, df_scope in scope_configs:
        if df_scope.empty:
            continue

        for grp_folder, grp_mode in grouping_modes:
            full_sub_dir = os.path.join(target_directory, scope_folder, grp_folder)
            os.makedirs(full_sub_dir, exist_ok=True)

            for file_base, metric_col, title_txt, y_lbl in metrics_info:
                if metric_col not in df_scope.columns:
                    continue

                if "boxplot" in file_base:
                    fig, df_pdata = _build_box(df_scope, metric_col, f"{title_txt} [{scope_folder[3:].replace('_', ' ')}]", y_lbl, grp_mode)
                else:
                    fig, df_pdata = _build_bar(df_scope, metric_col, f"{title_txt} [{scope_folder[3:].replace('_', ' ')}]", y_lbl, grp_mode)

                # Write CSV
                df_pdata.to_csv(os.path.join(full_sub_dir, f"{file_base}.csv"), index=False)

                # Write PNG
                try:
                    png_data = fig.to_image(format='png', scale=2)
                    with open(os.path.join(full_sub_dir, f"{file_base}.png"), "wb") as pf:
                        pf.write(png_data)
                except Exception:
                    pass

                # Write HTML
                fig.write_html(os.path.join(full_sub_dir, f"{file_base}.html"), include_plotlyjs='cdn')

    # 4. WRITE README
    readme_text = f"""# ESTRUCTURA DE CARPETAS Y DATOS CRUDOS EXPORTADOS
Fecha de Creación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ruta en Disco: {target_directory}
Nivel: {level_name.upper()}

Esta carpeta contiene la totalidad de gráficos y datos del experimento organizados en subcarpetas temáticas y agrupaciones:

📁 01_Comparacion_Global_NONE_vs_ACF/
   ├── 📁 01_Desglose_Completo_Condicion_Sexo_Hemisferio/
   ├── 📁 02_Pool_Condicion_Global/
   ├── 📁 03_Por_Condicion_y_Hemisferio/
   └── 📁 04_Por_Condicion_y_Sexo/

📁 02_Solo_Controles_Basales_NONE/
   ├── 📁 01_Desglose_Completo_Condicion_Sexo_Hemisferio/
   ├── 📁 02_Pool_Condicion_Global/
   ├── 📁 03_Por_Condicion_y_Hemisferio/
   └── 📁 04_Por_Condicion_y_Sexo/

📁 03_Solo_Tratamientos_Lesion_ACF/
   ├── 📁 01_Desglose_Completo_Condicion_Sexo_Hemisferio/
   ├── 📁 02_Pool_Condicion_Global/
   ├── 📁 03_Por_Condicion_y_Hemisferio/
   └── 📁 04_Por_Condicion_y_Sexo/

📁 04_Filtro_Solo_MACHOS/
📁 05_Filtro_Solo_HEMBRAS/
📁 06_Filtro_Solo_IPSI_Lesionado/
📁 07_Filtro_Solo_CONTRA_Intacto/

Cada subcarpeta contiene pares de archivos con nombres IDÉNTICOS:
- [nombre_grafico].png (Imagen de alta resolución)
- [nombre_grafico].csv (Datos de la gráfica)
- [nombre_grafico].html (Gráfica interactiva)
"""
    with open(os.path.join(target_directory, "README_ESTRUCTURA_Y_METRICAS.txt"), "w", encoding="utf-8") as rf:
        rf.write(readme_text)

    return target_directory


