import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.path as mpath
from skimage.draw import polygon as skimage_polygon

REGIONS = ["A", "B", "C"]

def get_roi_json_path(metrics_dir, image_name):
    clean = os.path.basename(str(image_name))
    for suf in ['_nuclei_metrics.csv', '_dapi_metrics.csv', '_masks.tif', '_prob_map.tif', '_segmented.tif', '_rois.json', '.TIF', '.tif', '.czi', '.CZI']:
        clean = clean.replace(suf, '')
    return os.path.join(metrics_dir, f"{clean}_rois.json")

def load_rois(roi_json_path):
    default_regions = {"A": [], "B": [], "C": []}
    if not os.path.exists(roi_json_path):
        return default_regions
    try:
        with open(roi_json_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "regions" in data and isinstance(data["regions"], dict):
                    res = default_regions.copy()
                    for r in REGIONS:
                        res[r] = data["regions"].get(r, [])
                    return res
                elif "polygons" in data and isinstance(data["polygons"], list):
                    res = default_regions.copy()
                    res["A"] = data["polygons"]
                    return res
    except Exception as e:
        print(f"Error cargando ROIs de {roi_json_path}: {e}")
    return default_regions

def save_rois(roi_json_path, regions_dict, image_shape=None, pixel_size_um=1.0):
    os.makedirs(os.path.dirname(roi_json_path), exist_ok=True)
    
    clean_regions = {"A": [], "B": [], "C": []}
    regions_metadata = {}
    total_rois = 0
    total_area_um2 = 0.0

    for reg in REGIONS:
        polys = regions_dict.get(reg, [])
        valid_polys = []
        reg_area_um2 = 0.0
        
        for idx, poly in enumerate(polys):
            poly_arr = np.array(poly)
            if len(poly_arr) >= 3:
                y = poly_arr[:, 0]
                x = poly_arr[:, 1]
                area_px = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                area_um2 = float(area_px * (pixel_size_um ** 2))
                reg_area_um2 += area_um2
                valid_polys.append(poly_arr.tolist())

        clean_regions[reg] = valid_polys
        total_rois += len(valid_polys)
        total_area_um2 += reg_area_um2
        regions_metadata[reg] = {
            "n_rois": len(valid_polys),
            "area_um2": reg_area_um2,
            "area_mm2": reg_area_um2 / 1e6
        }

    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pixel_size_um": pixel_size_um,
        "image_shape": list(image_shape) if image_shape is not None else None,
        "n_rois_total": total_rois,
        "total_roi_area_um2": total_area_um2,
        "total_roi_area_mm2": total_area_um2 / 1e6,
        "regions": clean_regions,
        "regions_metadata": regions_metadata
    }
    
    with open(roi_json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Guardadas {total_rois} ROIs en: {roi_json_path}")
    return payload

def points_in_rois(coords_yx, regions_dict, target_region="ALL"):
    if coords_yx is None or len(coords_yx) == 0:
        return np.zeros(len(coords_yx) if coords_yx is not None else 0, dtype=bool)
    
    coords_yx = np.array(coords_yx)
    inside_mask = np.zeros(len(coords_yx), dtype=bool)
    points_xy = coords_yx[:, [1, 0]]
    
    if target_region == "ALL":
        target_regs = REGIONS
    elif target_region in REGIONS:
        target_regs = [target_region]
    else:
        target_regs = REGIONS

    for reg in target_regs:
        polygons = regions_dict.get(reg, []) if isinstance(regions_dict, dict) else []
        for poly in polygons:
            poly_arr = np.array(poly)
            if len(poly_arr) >= 3:
                poly_xy = poly_arr[:, [1, 0]]
                path = mpath.Path(poly_xy)
                contained = path.contains_points(points_xy)
                inside_mask = inside_mask | contained
                
    return inside_mask

def get_point_region_assignment(coords_yx, regions_dict):
    if coords_yx is None or len(coords_yx) == 0:
        return ["NONE"] * (len(coords_yx) if coords_yx is not None else 0)
        
    coords_yx = np.array(coords_yx)
    points_xy = coords_yx[:, [1, 0]]
    assignments = ["NONE"] * len(coords_yx)
    
    for reg in REGIONS:
        polygons = regions_dict.get(reg, []) if isinstance(regions_dict, dict) else []
        for poly in polygons:
            poly_arr = np.array(poly)
            if len(poly_arr) >= 3:
                poly_xy = poly_arr[:, [1, 0]]
                path = mpath.Path(poly_xy)
                contained = path.contains_points(points_xy)
                for idx, is_inc in enumerate(contained):
                    if is_inc:
                        assignments[idx] = reg
    return assignments

def create_roi_mask(image_shape_2d, regions_dict, target_region="ALL"):
    mask = np.zeros(image_shape_2d, dtype=bool)
    if not regions_dict:
        return mask
        
    if target_region == "ALL":
        target_regs = REGIONS
    elif target_region in REGIONS:
        target_regs = [target_region]
    else:
        target_regs = REGIONS

    for reg in target_regs:
        polygons = regions_dict.get(reg, []) if isinstance(regions_dict, dict) else []
        for poly in polygons:
            poly_arr = np.array(poly)
            if len(poly_arr) >= 3:
                rr, cc = skimage_polygon(poly_arr[:, 0], poly_arr[:, 1], shape=image_shape_2d)
                mask[rr, cc] = True
            
    return mask

def compute_roi_area_mm2(image_shape_2d, regions_dict, pixel_size_um=1.0, target_region="ALL"):
    mask = create_roi_mask(image_shape_2d, regions_dict, target_region=target_region)
    total_px = np.sum(mask)
    return float(total_px * (pixel_size_um ** 2) / 1e6)
