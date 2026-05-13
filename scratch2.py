import os
import json
import pandas as pd
import numpy as np
import scipy.stats as stats

METRICS_BASE_DIR = "data/processed/metrics"

groups = sorted([d for d in os.listdir(METRICS_BASE_DIR) if os.path.isdir(os.path.join(METRICS_BASE_DIR, d))])
all_summaries = []
all_cell_data = []

for group in groups:
    group_dir = os.path.join(METRICS_BASE_DIR, group)
    sections = [d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))]
    for section in sections:
        section_dir = os.path.join(group_dir, section)
        for f in os.listdir(section_dir):
            if not f.endswith('_summary.json'):
                continue
            base_fname = f.replace('_summary.json', '')
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

df_sum = pd.DataFrame(all_summaries)
df_cells = pd.concat(all_cell_data, ignore_index=True) if all_cell_data else pd.DataFrame()

df_indiv = df_sum.groupby(['group', 'individual_id', 'section']).agg(
    pv_count=('total_pv_segmentation', 'mean'),
    pnn_count=('pnn_plus', 'mean'),
).reset_index()

if not df_cells.empty and 'wfa_sum_intensity' in df_cells.columns:
    df_wfa = df_cells.groupby(['group', 'individual_id', 'section'])['wfa_sum_intensity'].mean().reset_index()
    df_wfa = df_wfa.rename(columns={'wfa_sum_intensity': 'wfa_mean_intensity'})
    df_indiv = df_indiv.merge(df_wfa, on=['group', 'individual_id', 'section'], how='left')

g1 = "ACF 14 DÍAS HEMBRA"
g2 = "ACF 14 DÍAS MACHO"
sec = "CONTRA"

df_g1 = df_indiv[(df_indiv['group'] == g1) & (df_indiv['section'] == sec)]
df_g2 = df_indiv[(df_indiv['group'] == g2) & (df_indiv['section'] == sec)]

vals1 = df_g1['wfa_mean_intensity'].dropna()
vals2 = df_g2['wfa_mean_intensity'].dropna()

print("vals1:")
print(vals1)
print("vals2:")
print(vals2)

try:
    stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
    print("mannwhitneyu SUCCESS:", pval)
except Exception as e:
    print("mannwhitneyu EXCEPTION:", str(e))

