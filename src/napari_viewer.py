import os
import sys
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
import napari

sys.path.append(os.path.abspath("src"))
from roi import load_rois, save_rois, get_roi_json_path
from image_io import load_channels_tif

def main():
    if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
        print("\n" + "*"*80)
        print("ERROR: No se detectó un servidor gráfico (variable DISPLAY no configurada).")
        print("*"*80 + "\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Visor Napari Avanzado para PNN SSC Analysis")
    parser.add_argument("--path", required=True, help="Ruta al archivo TIF o CZI")
    parser.add_argument("--pixel_size", type=float, default=1.0, help="Tamaño de pixel en micras")
    parser.add_argument("--step", type=str, default="all", choices=["dapi", "pv", "wfa", "pnn", "all"], help="Paso del análisis")
    parser.add_argument("--edit-roi", action="store_true", help="Habilitar modo de edición de ROIs multicapa")
    args = parser.parse_args()
    
    path = args.path
    if not os.path.exists(path):
        print(f"Error: El archivo {path} no existe.")
        return
        
    print(f"Cargando imagen en Napari: {path} (Paso: {args.step})")
    
    scale = (args.pixel_size, args.pixel_size)
    viewer = napari.Viewer(title=f"PNN SSC Analysis — {os.path.basename(path)} (Paso: {args.step})")
    
    is_masks_file = "_masks.tif" in path
    
    raw_path = path
    if is_masks_file:
        raw_path = path.replace("data/processed/segmented", "data/processed/mips").replace("_masks.tif", ".tif")
        if not os.path.exists(raw_path):
            raw_path = path.replace("data/processed/segmented", "data/processed/mips").replace("_masks.tif", ".TIF")

    wfa_raw = None
    if os.path.exists(raw_path):
        try:
            (pv_raw, wfa_raw, dapi_raw, agr_raw) = load_channels_tif(raw_path)
            viewer.add_image(dapi_raw, name="01 - DAPI (Núcleos)", colormap="blue", scale=scale, blending="additive", visible=False)
            viewer.add_image(wfa_raw, name="02 - WFA (Red Perineuronal)", colormap="red", scale=scale, blending="additive", visible=True)
            viewer.add_image(pv_raw, name="03 - PV (Parvalbúmina)", colormap="green", scale=scale, blending="additive", visible=False)
        except Exception as e:
            print(f"Advertencia al cargar canales biológicos: {e}")

    # ─── CARGA DE MAPA DE CALOR DE POTENCIA (LUPORI ENERGY MAP) ───
    segmented_dir = os.path.dirname(path) if is_masks_file else os.path.dirname(path).replace("data/processed/mips", "data/processed/segmented").replace("data/raw", "data/processed/segmented")
    base_file_name = os.path.basename(path).replace("_masks.tif", "").replace(".tif", "").replace(".czi", "")
    
    heatmap_png_path = os.path.join(segmented_dir, f"{base_file_name}_power_heatmap.png")
    if os.path.exists(heatmap_png_path):
        try:
            heatmap_bgr = cv2.imread(heatmap_png_path)
            if heatmap_bgr is not None:
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                viewer.add_image(heatmap_rgb, name="🔥 Mapa de Calor de Potencia (Lupori Energy)", scale=scale, opacity=0.65, visible=False)
        except Exception as e:
            print(f"Advertencia al cargar Mapa de Calor de Potencia: {e}")

    # ─── CARGA DE MÁSCARAS SEGMENTADAS ───
    masks_tif_path = path if is_masks_file else os.path.join(segmented_dir, f"{base_file_name}_masks.tif")
    
    if os.path.exists(masks_tif_path):
        try:
            stk_masks = tiff.imread(masks_tif_path)
            
            # Channel 1: PV Mask
            if len(stk_masks.shape) >= 3 and stk_masks.shape[0] >= 2:
                m_pv = stk_masks[1]
                if np.max(m_pv) > 0:
                    viewer.add_labels(m_pv, name="🧪 Interneuronas PV+ (Cellpose)", scale=scale, opacity=0.45, visible=False)
            
            # Channel 2: Total PNN Mask
            m_wfa = None
            if len(stk_masks.shape) >= 3 and stk_masks.shape[0] >= 3:
                m_wfa = stk_masks[2]
                if np.max(m_wfa) > 0:
                    viewer.add_labels(m_wfa, name="🧠 PNNs Totales (IA Detectadas)", scale=scale, opacity=0.5, visible=False)
            
            # Channel 3: Pericellular Ring Mask (4µm)
            if len(stk_masks.shape) >= 3 and stk_masks.shape[0] >= 4:
                m_ring = stk_masks[3]
                if np.max(m_ring) > 0:
                    viewer.add_labels(m_ring, name="⭕ Anillos Pericelulares 4µm (Muestreo Potencia)", scale=scale, opacity=0.6, visible=False)

            # ─── CLASIFICACIÓN PNN PV+ VS PNN PV- (MÉTRICAS CSV) ───
            metrics_dir = segmented_dir.replace("data/processed/segmented", "data/processed/metrics")
            csv_path = os.path.join(metrics_dir, f"{base_file_name}_nuclei_metrics.csv")
            
            if os.path.exists(csv_path) and m_wfa is not None:
                try:
                    df_metrics = pd.read_csv(csv_path)
                    
                    df_coloc = df_metrics[df_metrics['cell_type'] == "PV+/PNN+"]
                    df_hollow = df_metrics[df_metrics['cell_type'] == "PV-/PNN+"]
                    
                    # Create green mask for PV+/PNN+
                    coloc_labels = set(df_coloc['label'].astype(int))
                    m_pnn_coloc = np.zeros_like(m_wfa, dtype=np.uint16)
                    for lbl in coloc_labels:
                        m_pnn_coloc[m_wfa == lbl] = lbl
                    if np.max(m_pnn_coloc) > 0:
                        viewer.add_labels(m_pnn_coloc, name="🟢 PNNs PV+ (Coinmunomarcadas PV+/PNN+)", scale=scale, opacity=0.65, visible=False)

                    # Create red mask for PV-/PNN+
                    hollow_labels = set(df_hollow['label'].astype(int))
                    m_pnn_hollow = np.zeros_like(m_wfa, dtype=np.uint16)
                    for lbl in hollow_labels:
                        m_pnn_hollow[m_wfa == lbl] = lbl
                    if np.max(m_pnn_hollow) > 0:
                        viewer.add_labels(m_pnn_hollow, name="🔴 PNNs PV- (Huecas PV-/PNN+)", scale=scale, opacity=0.65, visible=False)

                    # Add Centroid Points
                    if not df_coloc.empty:
                        pts_coloc = df_coloc[['centroid_y', 'centroid_x']].values
                        viewer.add_points(pts_coloc, name="📍 Centroides PNN (PV+/PNN+)", face_color="lime", edge_color="white", size=14, scale=scale, visible=False)

                    if not df_hollow.empty:
                        pts_hollow = df_hollow[['centroid_y', 'centroid_x']].values
                        viewer.add_points(pts_hollow, name="📍 Centroides PNN (PV-/PNN+)", face_color="magenta", edge_color="white", size=14, scale=scale, visible=False)

                except Exception as e_csv:
                    print(f"Advertencia al clasificar PNNs desde CSV: {e_csv}")

        except Exception as e:
            print(f"Advertencia al cargar máscaras TIF: {e}")

    # ─── ROIS MULTICAPA (A, B, C) ───
    base_name = os.path.basename(path)
    metrics_dir = segmented_dir.replace("data/processed/segmented", "data/processed/metrics")
    roi_json_path = get_roi_json_path(metrics_dir, base_name)
    regions_loaded = load_rois(roi_json_path)

    layer_a = viewer.add_shapes(
        regions_loaded.get("A", []),
        shape_type='polygon',
        name="🅰️ ROI - Región A",
        edge_color='cyan',
        edge_width=3,
        face_color=[0.0, 0.95, 1.0, 0.25],
        scale=scale,
        opacity=0.8
    )

    layer_b = viewer.add_shapes(
        regions_loaded.get("B", []),
        shape_type='polygon',
        name="🅱️ ROI - Región B",
        edge_color='magenta',
        edge_width=3,
        face_color=[1.0, 0.0, 1.0, 0.25],
        scale=scale,
        opacity=0.8
    )

    layer_c = viewer.add_shapes(
        regions_loaded.get("C", []),
        shape_type='polygon',
        name="🅒 ROI - Región C",
        edge_color='yellow',
        edge_width=3,
        face_color=[1.0, 1.0, 0.0, 0.25],
        scale=scale,
        opacity=0.8
    )

    if args.edit_roi:
        try:
            viewer.layers.selection.active = layer_a
            layer_a.mode = 'add_polygon' if len(regions_loaded.get("A", [])) == 0 else 'select'
        except Exception:
            pass

    def _get_current_regions_dict():
        return {
            "A": [shape for shape in layer_a.data if len(shape) >= 3],
            "B": [shape for shape in layer_b.data if len(shape) >= 3],
            "C": [shape for shape in layer_c.data if len(shape) >= 3]
        }

    @viewer.bind_key('Control-s', overwrite=True)
    def save_rois_shortcut(v):
        regs = _get_current_regions_dict()
        w_shape = wfa_raw.shape if wfa_raw is not None else (1000, 1000)
        save_rois(roi_json_path, regs, image_shape=w_shape, pixel_size_um=args.pixel_size)
        n_a, n_b, n_c = len(regs['A']), len(regs['B']), len(regs['C'])
        v.status = f"✅ ROIs guardadas — Región A: {n_a}, Región B: {n_b}, Región C: {n_c}"

    napari.run()

    try:
        regs = _get_current_regions_dict()
        w_shape = wfa_raw.shape if wfa_raw is not None else (1000, 1000)
        save_rois(roi_json_path, regs, image_shape=w_shape, pixel_size_um=args.pixel_size)
        print(f"✅ ROIs guardadas al cerrar Napari en: {roi_json_path}")
    except Exception as e:
        print(f"Advertencia al guardar ROIs al salir: {e}")

if __name__ == "__main__":
    main()
