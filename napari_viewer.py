import argparse
import os
import sys
import numpy as np
import tifffile as tiff
import napari

def main():
    if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
        print("\n" + "*"*80)
        print("ERROR: No se detectó un servidor gráfico (la variable DISPLAY no está seteada).")
        print("Si estás usando WSL (Windows Subsystem for Linux) desde Windows, asegúrate de")
        print("que WSLg esté activo o que un servidor X (como VcXsrv o Xming) esté corriendo.")
        print("*"*80 + "\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Visor Napari para PNN SSC Analysis")
    parser.add_argument("--path", required=True, help="Ruta al archivo TIF (Original o Segmentado)")
    parser.add_argument("--pixel_size", type=float, default=1.0, help="Tamaño de pixel en micras")
    parser.add_argument("--step", type=str, default="all", choices=["dapi", "pv", "wfa", "pnn", "all"], help="Paso del análisis para filtrar máscaras y canales")
    args = parser.parse_args()
    
    path = args.path
    if not os.path.exists(path):
        print(f"Error: El archivo {path} no existe.")
        return
        
    print(f"Cargando imagen: {path} (Paso: {args.step})")
    img = tiff.imread(path)
    
    # Leer metadata de ejes si está disponible
    with tiff.TiffFile(path) as tif:
        axes = tif.series[0].axes
        
    # Si viene con eje Z, hacemos MIP para visualización en 2D
    if 'Z' in axes and len(img.shape) >= 4:
        z_idx = axes.index('Z')
        img = np.max(img, axis=z_idx)
        axes = axes.replace('Z', '')
        
    if axes == 'YXC':
        img = np.transpose(img, (2, 0, 1))
        
    # Verificar shape de canales
    num_channels = img.shape[0] if len(img.shape) > 2 else 1
    print(f"Estructura cargada: {img.shape} con {num_channels} canales.")
    
    # Escala para calibración en micras (Y, X)
    scale = (args.pixel_size, args.pixel_size)
    
    # Inicializar el visor de Napari
    viewer = napari.Viewer(title=f"PNN SSC Analysis — {os.path.basename(path)} (Paso: {args.step})")
    
    # Intentar Dual-Loading si es un archivo de máscaras
    is_masks_file = "_masks.tif" in path
    raw_loaded = False
    
    if is_masks_file:
        # Encontrar la imagen MIP o raw original (priorizando el MIP procesado 2D)
        raw_path = path.replace("data/processed/segmented", "data/processed/mips").replace("_masks.tif", ".tif")
        if not os.path.exists(raw_path):
            raw_path = path.replace("data/processed/segmented", "data/processed/mips").replace("_masks.tif", ".TIF")
            
        if not os.path.exists(raw_path):
            # Fallback to raw if MIP is missing
            raw_path = path.replace("data/processed/segmented", "data/raw").replace("_masks.tif", ".tif")
            if not os.path.exists(raw_path):
                raw_path_alt = path.replace("data/processed/segmented", "data/raw").replace("_masks.tif", ".TIF")
                if os.path.exists(raw_path_alt):
                    raw_path = raw_path_alt
                
        if os.path.exists(raw_path):
            print(f"Detectado archivo de máscaras. Cargando canales biológicos de: {raw_path}")
            try:
                raw_img = tiff.imread(raw_path)
                with tiff.TiffFile(raw_path) as tif:
                    raw_axes = tif.series[0].axes
                if 'Z' in raw_axes and len(raw_img.shape) >= 4:
                    z_idx = raw_axes.index('Z')
                    raw_img = np.max(raw_img, axis=z_idx)
                    raw_axes = raw_axes.replace('Z', '')
                if raw_axes == 'YXC':
                    raw_img = np.transpose(raw_img, (2, 0, 1))
                
                # Canales de referencia biológica dependientes del paso
                num_raw_ch = raw_img.shape[0] if len(raw_img.shape) > 2 else 1
                
                if args.step in ["dapi", "all"]:
                    if num_raw_ch >= 2:
                        viewer.add_image(
                            raw_img[1] if len(raw_img.shape) > 2 else raw_img,
                            name="01 - DAPI (Núcleos)",
                            colormap="blue",
                            scale=scale,
                            blending="additive",
                            visible=True
                        )
                    elif num_raw_ch >= 1:
                        viewer.add_image(
                            raw_img[0] if len(raw_img.shape) > 2 else raw_img,
                            name="01 - DAPI (Núcleos)",
                            colormap="blue",
                            scale=scale,
                            blending="additive",
                            visible=True
                        )
                        
                if args.step in ["wfa", "pnn", "all"]:
                    if num_raw_ch >= 3:
                        viewer.add_image(
                            raw_img[2],
                            name="02 - WFA (Red Perineuronal)",
                            colormap="green",
                            scale=scale,
                            blending="additive",
                            visible=True
                        )
                        
                if args.step in ["pv", "all"]:
                    if num_raw_ch >= 4:
                        viewer.add_image(
                            raw_img[3],
                            name="03 - PV (Parvalbúmina)",
                            colormap="gray",
                            scale=scale,
                            blending="additive",
                            visible=True
                        )
                
                # Cargar el mapa de probabilidad de PNNloc (heatmap) si existe (solo en paso WFA/PNN o ALL)
                if args.step in ["wfa", "pnn", "all"]:
                    prob_map_path = path.replace("_masks.tif", "_prob_map.tif")
                    if os.path.exists(prob_map_path):
                        print(f"Cargando mapa de probabilidad PNNloc: {prob_map_path}")
                        prob_img = tiff.imread(prob_map_path)
                        viewer.add_image(
                            prob_img,
                            name="04 - Mapa de Calor (PNNloc Probability)",
                            colormap="inferno",
                            scale=scale,
                            blending="additive",
                            visible=True if args.step in ["wfa", "pnn"] else False
                        )
                raw_loaded = True
            except Exception as e:
                print(f"Error cargando imagen raw original: {e}")
        else:
            print(f"Advertencia: No se encontró la imagen raw correspondiente en: {raw_path}")
            
    # Si no es archivo de máscaras o falló el dual load de la raw, cargar del archivo directamente
    if not raw_loaded:
        if args.step in ["dapi", "all"]:
            if num_channels >= 2:
                viewer.add_image(
                    img[1] if len(img.shape) > 2 else img,
                    name="01 - DAPI (Núcleos)",
                    colormap="blue",
                    scale=scale,
                    blending="additive",
                    visible=True
                )
            elif num_channels >= 1:
                viewer.add_image(
                    img[0] if len(img.shape) > 2 else img,
                    name="01 - DAPI (Núcleos)",
                    colormap="blue",
                    scale=scale,
                    blending="additive",
                    visible=True
                )
        if args.step in ["wfa", "pnn", "all"]:
            if num_channels >= 3:
                viewer.add_image(
                    img[2],
                    name="02 - WFA (PNN)",
                    colormap="green",
                    scale=scale,
                    blending="additive",
                    visible=True
                )
        if args.step in ["pv", "all"]:
            if num_channels >= 4:
                viewer.add_image(
                    img[3],
                    name="03 - PV (Parvalbúmina)",
                    colormap="gray",
                    scale=scale,
                    blending="additive",
                    visible=True
                )
            
    # Agregar las capas de etiquetas (masks)
    if is_masks_file:
        dapi_mask = np.zeros_like(img[0])
        pv_mask = np.zeros_like(img[0])
        wfa_mask = np.zeros_like(img[0])

        if num_channels == 4:
            dapi_mask = img[0]
            pv_mask = img[1]
            wfa_mask = img[2]
        elif num_channels >= 5:
            dapi_mask = img[0]
            pv_mask = img[1]
            wfa_mask = img[4]
        else:
            if num_channels >= 1: dapi_mask = img[0]
            if num_channels >= 2: pv_mask = img[1]
            if num_channels >= 4: wfa_mask = img[3]

        if args.step == "dapi":
            viewer.add_labels(
                dapi_mask.astype(np.uint16),
                name="Máscara DAPI (núcleos segmentados)",
                scale=scale,
                visible=True
            )
        elif args.step == "pv":
            viewer.add_labels(
                pv_mask.astype(np.uint16),
                name="Máscara PV (somas de interneuronas PV+)",
                scale=scale,
                visible=True
            )
        elif args.step in ["wfa", "pnn"]:
            viewer.add_labels(
                wfa_mask.astype(np.uint16),
                name="Máscara PNN (Redes Perineuronales)",
                scale=scale,
                visible=True
            )
        elif args.step == "all":
            # Dividir dinámicamente la máscara de WFA en Ocupadas (con PV+) y Huecas (sin PV+)
            wfa_labels = np.unique(wfa_mask)
            wfa_labels = wfa_labels[wfa_labels > 0]
            
            wfa_huecas = np.zeros_like(wfa_mask, dtype=np.uint16)
            wfa_ocupadas = np.zeros_like(wfa_mask, dtype=np.uint16)
            
            for wfa_lbl in wfa_labels:
                submask = (wfa_mask == wfa_lbl)
                pv_sub = pv_mask[submask]
                overlapping_pv = np.unique(pv_sub)
                overlapping_pv = overlapping_pv[overlapping_pv > 0]
                
                if len(overlapping_pv) > 0:
                    wfa_ocupadas[submask] = wfa_lbl
                else:
                    wfa_huecas[submask] = wfa_lbl

            viewer.add_labels(
                dapi_mask.astype(np.uint16),
                name="05 - Máscara DAPI (núcleos segmentados)",
                scale=scale,
                visible=False
            )
            viewer.add_labels(
                pv_mask.astype(np.uint16),
                name="06 - Máscara PV (somas de interneuronas PV+)",
                scale=scale,
                visible=True
            )
            viewer.add_labels(
                wfa_ocupadas.astype(np.uint16),
                name="07 - PNN+ Ocupadas → hueco WFA con soma PV+ dentro",
                scale=scale,
                visible=True
            )
            viewer.add_labels(
                wfa_huecas.astype(np.uint16),
                name="08 - PNN+ Huecas → hueco WFA sin soma PV+ (PV-/PNN+)",
                scale=scale,
                visible=True
            )
    else:
        # Pilas heredadas
        if num_channels >= 4:
            if args.step in ["dapi", "all"]:
                viewer.add_labels(
                    img[0].astype(np.uint16),
                    name="Máscara DAPI",
                    scale=scale,
                    visible=True if args.step == "dapi" else False
                )
            if args.step in ["pv", "all"]:
                viewer.add_labels(
                    img[1].astype(np.uint16),
                    name="Máscara PV",
                    scale=scale,
                    visible=True
                )
            if args.step in ["wfa", "pnn", "all"]:
                viewer.add_labels(
                    img[3].astype(np.uint16) if num_channels >= 5 else img[2].astype(np.uint16),
                    name="Máscara WFA/PNN",
                    scale=scale,
                    visible=True
                )
            
    # Lanzar la interfaz de Napari
    napari.run()

if __name__ == "__main__":
    main()
