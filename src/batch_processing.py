import os
import sys
import json
import re
import pandas as pd

sys.path.append(os.path.abspath("src"))
from image_io import extract_animal_id, get_or_create_mip
from ai_models import load_models
from pipeline_runner import run_pipeline_on_file

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

    px_size = float(calib_data.get("pixel_size_um", 0.8913))
    model_pv, model_loc, model_score, device = load_models()

    groups = sorted([d for d in os.listdir(RAW_BASE) if os.path.isdir(os.path.join(RAW_BASE, d)) and not d.startswith('.')])
    print(f"Grupos encontrados: {groups}")

    consolidated_summaries = []

    for group in groups:
        group_dir = os.path.join(RAW_BASE, group)
        sections = sorted([s for s in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, s)) and not s.startswith('.')])
        if not sections:
            sections = ["."]

        for sec in sections:
            sec_dir = os.path.join(group_dir, sec) if sec != "." else group_dir
            tif_files = sorted([os.path.join(sec_dir, f) for f in os.listdir(sec_dir) if f.lower().endswith(('.tif', '.czi'))])

            out_segm_dir = os.path.join(SEGM_BASE, group, sec if sec != "." else "")
            out_metrics_dir = os.path.join(METRICS_BASE, group, sec if sec != "." else "")
            os.makedirs(out_segm_dir, exist_ok=True)
            os.makedirs(out_metrics_dir, exist_ok=True)

            for raw_path in tif_files:
                fname = os.path.basename(raw_path)
                print(f"\n--- Procesando: [{group}] / [{sec}] / {fname} ---")
                try:
                    mip_path = get_or_create_mip(raw_path, px_size)
                    summary = run_pipeline_on_file(
                        tif_path=mip_path,
                        out_segm_dir=out_segm_dir,
                        out_metrics_dir=out_metrics_dir,
                        model_pv_obj=model_pv,
                        model_loc=model_loc,
                        model_score=model_score,
                        device=device,
                        pv_filter_type=calib_data.get("pv_cellpose_filter_type", "Ninguno"),
                        pv_diameter=float(calib_data.get("pv_cellpose_diameter", 30.0)),
                        pv_flow_threshold=float(calib_data.get("pv_cellpose_flow_threshold", 0.4)),
                        pv_cellprob_threshold=float(calib_data.get("pv_cellpose_cellprob_threshold", 0.0)),
                        loc_threshold=float(calib_data.get("lupori_loc_threshold", 0.15)),
                        score_threshold=float(calib_data.get("lupori_score_threshold", 0.50)),
                        tile_size=int(calib_data.get("lupori_tile_size", 640)),
                        tile_overlap=int(calib_data.get("lupori_tile_overlap", 64)),
                        px_size=px_size,
                        do_pv_segmentation=calib_data.get("do_pv_segmentation", True),
                        calib_data=calib_data
                    )

                    animal_id = extract_animal_id(fname)
                    summary["group"] = group
                    summary["section"] = sec
                    summary["filename"] = fname
                    summary["animal_id"] = animal_id
                    consolidated_summaries.append(summary)

                except Exception as e:
                    print(f"Error procesando {raw_path}: {e}")

    if consolidated_summaries:
        df_cons = pd.DataFrame(consolidated_summaries)
        output_csv = os.path.join(METRICS_BASE, "consolidated_lupori_metrics.csv")
        df_cons.to_csv(output_csv, index=False)
        print(f"\n✅ Procesamiento batch completo. Tabla guardada en: {output_csv}")

if __name__ == "__main__":
    main()
