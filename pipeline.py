import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage import exposure, draw
import pandas as pd

def load_channels_tif(path):
    img = tiff.imread(path)
    
    with tiff.TiffFile(path) as tif:
        axes = tif.series[0].axes
        
    if 'Z' in axes and len(img.shape) >= 4:
        z_idx = axes.index('Z')
        img = np.max(img, axis=z_idx)
        axes = axes.replace('Z', '')
        
    if axes == 'YXC':
        img = np.transpose(img, (2, 0, 1))
        
    # Expected (C, Y, X)
    agr = img[0, :, :] if img.shape[0] >= 1 else np.zeros_like(img[0])
    dapi = img[1, :, :] if img.shape[0] >= 2 else img[0, :, :]
    wfa = img[2, :, :] if img.shape[0] >= 3 else np.zeros_like(img[0])
    pv = img[3, :, :] if img.shape[0] >= 4 else np.zeros_like(img[0])
    
    return (pv, wfa, dapi, agr)

def run_pipeline_on_file(tif_path, out_segm_dir, out_metrics_dir,
                         model_dapi, model_pv_obj,
                         filter_type, diameter, flow_threshold, cellprob_threshold,
                         pv_filter_type, pv_diameter, pv_flow_threshold, pv_cellprob_threshold,
                         pnn_radius_um, pnn_threshold, pnn_exclusion_dist_um,
                         px_size, do_pv_segmentation, calib_data):
    fname = os.path.basename(tif_path)
    (p_raw, w_raw, d_raw, a_raw) = load_channels_tif(tif_path)

    # DAPI preprocessing
    in_dapi = d_raw.copy()
    if filter_type == "Otsu Global":
        t = threshold_otsu(in_dapi)
        in_dapi[in_dapi < t] = 0
    elif filter_type == "CLAHE (Adaptativo Local)":
        clahe = exposure.equalize_adapthist(in_dapi, clip_limit=0.03)
        in_dapi = (clahe * 65535).astype(np.uint16)

    m_dapi, _, _ = model_dapi.eval(in_dapi, diameter=diameter, channels=[0, 0],
                                    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

    # PV preprocessing
    m_pv = np.zeros_like(m_dapi)
    if do_pv_segmentation and model_pv_obj is not None:
        in_pv = p_raw.copy()
        if pv_filter_type == "Otsu Global":
            t = threshold_otsu(in_pv)
            in_pv[in_pv < t] = 0
        elif pv_filter_type == "CLAHE (Adaptativo Local)":
            clahe = exposure.equalize_adapthist(in_pv, clip_limit=0.03)
            in_pv = (clahe * 65535).astype(np.uint16)
        m_pv, _, _ = model_pv_obj.eval(in_pv, diameter=pv_diameter, channels=[0, 0],
                                        flow_threshold=pv_flow_threshold, cellprob_threshold=pv_cellprob_threshold)

    # PNN analysis
    p_batch = regionprops(m_dapi, intensity_image=w_raw)
    r_batch = []
    for pb in p_batch:
        cr = pb.centroid
        rd, cd = draw.disk(cr, pnn_radius_um / px_size, shape=w_raw.shape)
        wfa_s = np.sum(w_raw[rd, cd])
        r_batch.append({
            'label': pb.label,
            'centroid_y': cr[0],
            'centroid_x': cr[1],
            'area_um2': pb.area * (px_size ** 2),
            'diameter_um': pb.equivalent_diameter_area * px_size,
            'dapi_mean_intensity': pb.intensity_mean,
            'wfa_sum_intensity': wfa_s,
            'is_pnn_plus': wfa_s > pnn_threshold,
            'is_pv_plus': m_pv[int(cr[0]), int(cr[1])] > 0
        })

    # NMS
    pnn_cands = [i for i, r in enumerate(r_batch) if r['is_pnn_plus']]
    if len(pnn_cands) > 1 and pnn_exclusion_dist_um > 0:
        sorted_idx = sorted(pnn_cands, key=lambda i: r_batch[i]['wfa_sum_intensity'], reverse=True)
        kept = []
        for si in sorted_idx:
            cy, cx = r_batch[si]['centroid_y'], r_batch[si]['centroid_x']
            if not any(np.sqrt((cy - k[0])**2 + (cx - k[1])**2) * px_size < pnn_exclusion_dist_um for k in kept):
                kept.append((cy, cx))
            else:
                r_batch[si]['is_pnn_plus'] = False

    df_b = pd.DataFrame(r_batch)
    df_b.to_csv(os.path.join(out_metrics_dir, fname.replace('.TIF', '_nuclei_metrics.csv').replace('.tif', '_nuclei_metrics.csv')), index=False)

    # TIFF output
    orig_b = tiff.imread(tif_path)
    with tiff.TiffFile(tif_path) as tif:
        axes = tif.series[0].axes
    if 'Z' in axes and len(orig_b.shape) >= 4:
        orig_b = np.max(orig_b, axis=axes.index('Z'))
        axes = axes.replace('Z', '')
    if axes == 'YXC':
        orig_b = np.transpose(orig_b, (2, 0, 1))
        
    m_pnn_b = np.zeros_like(m_dapi, dtype=np.uint16)
    pnn_lbls = df_b[df_b['is_pnn_plus']]['label'].values if not df_b.empty else []
    if len(pnn_lbls) > 0:
        lut = np.zeros(int(np.max(m_dapi)) + 1, dtype=np.uint16)
        for lb in pnn_lbls:
            lut[int(lb)] = int(lb)
        m_pnn_b = lut[m_dapi.astype(int)]

    stk = np.concatenate([orig_b,
                          np.expand_dims(m_dapi.astype(np.uint16), 0),
                          np.expand_dims(m_pv.astype(np.uint16), 0),
                          np.expand_dims(m_pnn_b, 0)], axis=0)
    ch_names = calib_data.get('channels', ['AGR', 'DAPI', 'WFA', 'PV'])
    tiff.imwrite(os.path.join(out_segm_dir, fname.replace('.TIF', '_segmented.tif').replace('.tif', '_segmented.tif')),
                 stk, imagej=True,
                 metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                           'Labels': ch_names + ['DAPI_Mask', 'PV_Mask', 'PNN_Mask']})

    # Summary JSON
    summary = {
        "total_dapi": len(df_b),
        "total_pv_segmentation": int(np.max(m_pv)),
        "pnn_plus": int(df_b['is_pnn_plus'].sum()) if not df_b.empty else 0,
        "pnn_minus": int((~df_b['is_pnn_plus']).sum()) if not df_b.empty else 0,
        "dapi_pv_coloc": int(df_b['is_pv_plus'].sum()) if not df_b.empty else 0,
        "pixel_size": px_size
    }
    with open(os.path.join(out_metrics_dir, fname.replace('.TIF', '_summary.json').replace('.tif', '_summary.json')), 'w') as fs:
        json.dump(summary, fs, indent=4)

    return summary
