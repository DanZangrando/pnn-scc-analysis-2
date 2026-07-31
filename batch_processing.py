import os
import json
import glob
import re
import torch
import pandas as pd
import numpy as np

from cellpose import models
import sys

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/counting_perineuronal_nets"))

from pipeline import run_pipeline_on_file, load_channels_tif

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelos en dispositivo: {device}")

    # Cellpose DAPI & PV
    model_dapi = models.CellposeModel(gpu=torch.cuda.is_available())
    model_pv = models.CellposeModel(gpu=torch.cuda.is_available())

    # PNNloc
    from models.FasterRCNN import FasterRCNNWrapper
    model_loc = FasterRCNNWrapper(in_channels=1, out_channels=1, model_pretrained=False)
    ckpt_loc = "data/models/pnn_v2_fasterrcnn_640/best.pth"
    if os.path.exists(ckpt_loc):
        checkpoint_loc = torch.load(ckpt_loc, map_location=device)
        model_loc.load_state_dict(checkpoint_loc['model'])
    model_loc.to(device).eval()

    # PNNscore
    from models.ConvNet import ConvNet
    model_score = ConvNet(in_channels=1, num_classes=1)
    ckpt_score = "data/models/pnn_v2_scoring_rank_learning/best.pth"
    if os.path.exists(ckpt_score):
        checkpoint_score = torch.load(ckpt_score, map_location=device)
        model_score.load_state_dict(checkpoint_score['model'])
    model_score.to(device).eval()

    return model_dapi, model_pv, model_loc, model_score, device

def main():
    RAW_BASE = "data/raw"
    if not os.path.exists(RAW_BASE) or not any(os.path.isdir(os.path.join(RAW_BASE, d)) for d in os.listdir(RAW_BASE) if not d.startswith('.')):
        RAW_BASE = "data/processed/mips"

    SEGM_BASE = "data/processed/segmented"
    METRICS_BASE = "data/processed/metrics"
    CONFIG_PATH = "experiment_config.json"

    calib_data = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            calib_data = json.load(f)

    px_size = float(calib_data.get("pixel_size_um", 1.0))

    model_dapi, model_pv, model_loc, model_score, device = load_models()

    groups = sorted([d for d in os.listdir(RAW_BASE) if os.path.isdir(os.path.join(RAW_BASE, d)) and not d.startswith('.')])
    print(f"Grupos encontrados: {groups}")

    consolidated_summaries = []

    for group in groups:
        group_dir = os.path.join(RAW_BASE, group)
        sections = sorted([s for s in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, s)) and not s.startswith('.')])
        if not sections:
            # Maybe tif files are directly under group
            sections = ["."]

        for sec in sections:
            sec_dir = os.path.join(group_dir, sec) if sec != "." else group_dir
            tif_files = sorted([os.path.join(sec_dir, f) for f in os.listdir(sec_dir) if f.lower().endswith('.tif')])

            out_segm_dir = os.path.join(SEGM_BASE, group, sec if sec != "." else "")
            out_metrics_dir = os.path.join(METRICS_BASE, group, sec if sec != "." else "")
            os.makedirs(out_segm_dir, exist_ok=True)
            os.makedirs(out_metrics_dir, exist_ok=True)

            for tif_path in tif_files:
                fname = os.path.basename(tif_path)
                print(f"\n--- Procesando: [{group}] / [{sec}] / {fname} ---")
                try:
                    summary = run_pipeline_on_file(
                        tif_path=tif_path,
                        out_segm_dir=out_segm_dir,
                        out_metrics_dir=out_metrics_dir,
                        model_dapi=model_dapi,
                        model_pv_obj=model_pv,
                        model_loc=model_loc,
                        model_score=model_score,
                        device=device,
                        filter_type="Ninguno",
                        diameter=int(calib_data.get("dapi_diameter", 30)),
                        flow_threshold=float(calib_data.get("dapi_flow_threshold", 0.4)),
                        cellprob_threshold=float(calib_data.get("dapi_cellprob_threshold", 0.0)),
                        pv_filter_type="Ninguno",
                        pv_diameter=int(calib_data.get("pv_diameter", 30)),
                        pv_flow_threshold=float(calib_data.get("pv_flow_threshold", 0.4)),
                        pv_cellprob_threshold=float(calib_data.get("pv_cellprob_threshold", 0.0)),
                        loc_threshold=float(calib_data.get("lupori_loc_threshold", 0.20)),
                        score_threshold=float(calib_data.get("lupori_score_threshold", 0.30)),
                        tile_size=int(calib_data.get("lupori_tile_size", 1024)),
                        tile_overlap=int(calib_data.get("lupori_tile_overlap", 32)),
                        px_size=px_size,
                        do_pv_segmentation=True,
                        calib_data=calib_data
                    )

                    m = re.match(r'(ACF_\d+)', fname)
                    animal_id = m.group(1) if m else fname.split('~')[0]
                    m2 = re.search(r'~(\d+)\.', fname)
                    corte_num = int(m2.group(1)) if m2 else 1

                    summary["group"] = group
                    summary["section"] = sec
                    summary["filename"] = fname
                    summary["animal_id"] = animal_id
                    summary["corte_num"] = corte_num
                    consolidated_summaries.append(summary)

                except Exception as e:
                    print(f"Error procesando {tif_path}: {e}")
                    import traceback
                    traceback.print_exc()

    if consolidated_summaries:
        df_cons = pd.DataFrame(consolidated_summaries)
        output_csv = os.path.join(METRICS_BASE, "consolidated_lupori_metrics.csv")
        df_cons.to_csv(output_csv, index=False)
        print(f"\n✅ Procesamiento completo! Tabla consolidada guardada en: {output_csv}")
        
        # Build ultra-fast stats cache for Page 05
        precompute_stats_cache(METRICS_BASE, df_cons)
    else:
        print("\n⚠️ No se procesaron archivos.")

def precompute_stats_cache(metrics_dir, df_cons):
    import pickle
    print("\n⚡ Pre-calculando caché optimizada para la Página 05 (Estadística)...")
    
    all_nuclei = []
    all_dapi = []
    
    for root, dirs, files in os.walk(metrics_dir):
        if "test" in root:
            continue
        for f in files:
            csv_path = os.path.join(root, f)
            rel_path = os.path.relpath(csv_path, metrics_dir)
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                group = parts[0]
                sec = parts[1]
                fn = parts[2]
            else:
                group = sec = "Desconocido"
                fn = f
                
            if f.endswith("_nuclei_metrics.csv"):
                base_fn = fn.replace("_nuclei_metrics.csv", "")
                m = re.match(r'(ACF_\d+)', base_fn)
                aid = m.group(1) if m else base_fn.split('~')[0]
                m2 = re.search(r'~(\d+)$', base_fn)
                c_num = int(m2.group(1)) if m2 else 1
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        df['group'] = group
                        df['section'] = sec
                        df['image_name'] = base_fn
                        df['animal_id'] = aid
                        df['corte_num'] = c_num
                        all_nuclei.append(df)
                except Exception:
                    pass
            elif f.endswith("_dapi_metrics.csv"):
                base_fn = fn.replace("_dapi_metrics.csv", "")
                m = re.match(r'(ACF_\d+)', base_fn)
                aid = m.group(1) if m else base_fn.split('~')[0]
                m2 = re.search(r'~(\d+)$', base_fn)
                c_num = int(m2.group(1)) if m2 else 1
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        df['group'] = group
                        df['section'] = sec
                        df['image_name'] = base_fn
                        df['animal_id'] = aid
                        df['corte_num'] = c_num
                        all_dapi.append(df)
                except Exception:
                    pass
                    
    df_raw_nuclei = pd.concat(all_nuclei, ignore_index=True) if all_nuclei else pd.DataFrame()
    df_raw_dapi = pd.concat(all_dapi, ignore_index=True) if all_dapi else pd.DataFrame()
    
    cache_path = os.path.join(metrics_dir, "stats_cache.pkl")
    cache_payload = {
        "df_raw_nuclei": df_raw_nuclei,
        "df_raw_dapi": df_raw_dapi,
        "df_cons": df_cons
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_payload, f)
    print(f"⚡ Caché creada exitosamente en: {cache_path}")

if __name__ == "__main__":
    main()

