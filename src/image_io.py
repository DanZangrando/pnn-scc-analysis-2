import os
import re
import numpy as np
import tifffile as tiff

def extract_animal_id(filename):
    clean = os.path.basename(str(filename))
    for suf in ['_nuclei_metrics.csv', '_dapi_metrics.csv', '_masks.tif', '_prob_map.tif', '_segmented.tif', '_rois.json', '_summary.json', '.TIF', '.tif', '.czi', '.CZI']:
        clean = clean.replace(suf, '')
    m = re.match(r'^(ACF_[A-Za-z0-9]+)', clean)
    if m:
        return m.group(1)
    return clean.split('~')[0].split('_')[0]

def extract_czi_pixel_size(czi_path):
    try:
        import czifile
        import xml.etree.ElementTree as ET
        with czifile.CziFile(czi_path) as czi:
            xml_str = czi.metadata()
            root = ET.fromstring(xml_str)
            for dist in root.findall('.//Distance'):
                if dist.get('Id') == 'X':
                    val_elem = dist.find('Value')
                    if val_elem is not None:
                        return float(val_elem.text) * 1e6
    except Exception as e:
        print(f"Advertencia extrayendo resolución CZI: {e}")
    return 0.8913

def get_or_create_mip(raw_path, px_size=0.8913, force_recreate=False):
    if "data/processed/mips" in raw_path:
        return raw_path
        
    mip_path = raw_path.replace("data/raw", "data/processed/mips")
    for ext in ['.czi', '.CZI', '.TIF', '.tif']:
        if mip_path.endswith(ext):
            mip_path = mip_path[:-len(ext)] + ".tif"
            break

    os.makedirs(os.path.dirname(mip_path), exist_ok=True)
    
    if not os.path.exists(mip_path) or force_recreate:
        print(f"Generando MIP proyectado (4 canales) para {raw_path} -> {mip_path}")
        if raw_path.endswith('.czi') or raw_path.endswith('.CZI'):
            import czifile
            px_size_extracted = extract_czi_pixel_size(raw_path)
            if px_size_extracted > 0:
                px_size = px_size_extracted
            with czifile.CziFile(raw_path) as czi:
                img = czi.asarray()
                img = np.squeeze(img)
                # Raw squeezed shape is (4, Z, Y, X) -> max intensity projection along Z (axis 1)
                if len(img.shape) == 4:
                    img = np.max(img, axis=1)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = np.transpose(img, (2, 0, 1))
        else:
            img = tiff.imread(raw_path)
            with tiff.TiffFile(raw_path) as tif:
                axes = tif.series[0].axes
            if 'Z' in axes and len(img.shape) >= 4:
                z_idx = axes.index('Z')
                img = np.max(img, axis=z_idx)
                axes = axes.replace('Z', '')
            if axes == 'YXC':
                img = np.transpose(img, (2, 0, 1))

        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        num_ch = img.shape[0]
        if num_ch < 4:
            padding = np.zeros((4 - num_ch, img.shape[1], img.shape[2]), dtype=img.dtype)
            img = np.concatenate([img, padding], axis=0)
        elif num_ch > 4:
            img = img[:4]
            
        tiff.imwrite(mip_path, img.astype(np.uint16), imagej=True,
                     metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX'})
                     
    return mip_path

def load_channels_tif(path):
    if "data/raw" in path:
        path = get_or_create_mip(path)
        
    img = tiff.imread(path)
    
    with tiff.TiffFile(path) as tif:
        axes = tif.series[0].axes
        
    if 'Z' in axes and len(img.shape) >= 4:
        z_idx = axes.index('Z')
        img = np.max(img, axis=z_idx)
        axes = axes.replace('Z', '')
        
    if axes == 'YXC':
        img = np.transpose(img, (2, 0, 1))
        
    # Standard Channel Mapping for CZI dataset:
    # 0 = AGR (AF488 - Auto-fluorescence / Green)
    # 1 = DAPI (DAPI405 - Nuclei / Blue)
    # 2 = WFA (AF647 - Red Perineuronal Nets / Red)
    # 3 = PV (AF546 - Parvalbumin Interneurons / Green)
    agr  = img[0, :, :] if img.shape[0] >= 1 else np.zeros_like(img[0])
    dapi = img[1, :, :] if img.shape[0] >= 2 else img[0, :, :]
    wfa  = img[2, :, :] if img.shape[0] >= 3 else np.zeros_like(img[0])
    pv   = img[3, :, :] if img.shape[0] >= 4 else np.zeros_like(img[0])
    
    return (pv, wfa, dapi, agr)
