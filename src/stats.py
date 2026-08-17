import os
import re
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, wilcoxon

import sys
sys.path.append(os.path.abspath("src"))
from image_io import extract_animal_id
from roi import load_rois, get_roi_json_path, points_in_rois, get_point_region_assignment

SEX_COLORS = {'MACHO': '#5bc0de', 'HEMBRA': '#e83e8c'}

def ensure_roi_annotations(df, metrics_base_dir):
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
            roi_json_path = os.path.join(metrics_base_dir, group, section, f"{img_name}_rois.json")
            if not os.path.exists(roi_json_path):
                roi_json_path = get_roi_json_path(metrics_base_dir, img_name)
            roi_cache[cache_key] = load_rois(roi_json_path)
            
        regs_dict = roi_cache[cache_key]
        in_roi = bool(points_in_rois([[cy, cx]], regs_dict, target_region="ALL")[0])
        reg_assigned = get_point_region_assignment([[cy, cx]], regs_dict)[0]
        
        is_roi_list.append(in_roi)
        roi_region_list.append(reg_assigned)
        
    df['is_in_roi'] = is_roi_list
    df['roi_region'] = roi_region_list
    return df

def load_all_metrics(metrics_base_dir):
    cache_file = os.path.join(metrics_base_dir, "stats_cache.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                payload = pickle.load(f)
                df_nuclei = payload.get("df_raw_nuclei", pd.DataFrame())
                if not df_nuclei.empty and 'image_name' in df_nuclei.columns:
                    df_nuclei['animal_id'] = df_nuclei['image_name'].apply(extract_animal_id)
                df_nuclei = ensure_roi_annotations(df_nuclei, metrics_base_dir)
                return df_nuclei
        except Exception:
            pass
            
    all_dfs = []
    if not os.path.exists(metrics_base_dir):
        return pd.DataFrame()
    for root, dirs, files in os.walk(metrics_base_dir):
        if "test" in root:
            continue
        for f in files:
            if not f.endswith('_nuclei_metrics.csv'):
                continue
            csv_path = os.path.join(root, f)
            rel_path = os.path.relpath(csv_path, metrics_base_dir)
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                group    = parts[0]
                section  = parts[1]
                filename = parts[2].replace('_nuclei_metrics.csv', '')
            else:
                group = section = "Desconocido"
                filename = f.replace('_nuclei_metrics.csv', '')
            animal_id = extract_animal_id(filename)
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
    res = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return ensure_roi_annotations(res, metrics_base_dir)

def img_summary(group_df):
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
        'mean_score': safe_mean(pnn_cells['score']) if 'score' in pnn_cells.columns else 0.0
    }
    return row

def aggregate_image_level(df):
    keys = ['group','section','image_name','animal_id','corte_num']
    summaries = []
    for vals, grp in df.groupby(keys):
        base = dict(zip(keys, vals))
        base.update(img_summary(grp))
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

def get_p_asterisks(p_val):
    if p_val is None or np.isnan(p_val): return ""
    if p_val < 0.001: return "***"
    if p_val < 0.01:  return "**"
    if p_val < 0.05:  return "*"
    return "ns"
