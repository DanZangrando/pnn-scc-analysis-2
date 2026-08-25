import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.path as mpath
from skimage.draw import polygon as skimage_polygon

REGIONS = ["ROI"]

# Standard Template Polygon (Coordinates based on reference sample ACF_43_SSC_C_1)
STANDARD_TEMPLATE_POLYGON = [
    [305.6388854980469, 1323.8775634765625],
    [1168.636474609375, 1141.2020263671875],
    [1272.5921630859375, 1618.8048095703125],
    [411.86297607421875, 1816.47802734375]
]

# Exact reference area with px_size = 0.8913 µm
STANDARD_ROI_AREA_UM2 = 348015.0625
STANDARD_ROI_AREA_MM2 = 0.3480150625

def get_roi_json_path(metrics_dir, image_name):
    clean = os.path.basename(str(image_name))
    for suf in ['_nuclei_metrics.csv', '_dapi_metrics.csv', '_masks.tif', '_prob_map.tif', '_segmented.tif', '_rois.json', '.TIF', '.tif', '.czi', '.CZI']:
        clean = clean.replace(suf, '')
    return os.path.join(metrics_dir, f"{clean}_rois.json")

def load_rois(roi_json_path):
    """Returns a dict {'ROI': list_of_polygons}."""
    default_regions = {"ROI": []}
    if not os.path.exists(roi_json_path):
        return default_regions
    try:
        with open(roi_json_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "regions" in data and isinstance(data["regions"], dict):
                    # Priority: 'ROI' -> 'B' -> 'A'
                    if "ROI" in data["regions"] and len(data["regions"]["ROI"]) > 0:
                        return {"ROI": data["regions"]["ROI"]}
                    elif "B" in data["regions"] and len(data["regions"]["B"]) > 0:
                        return {"ROI": data["regions"]["B"]}
                    elif "A" in data["regions"] and len(data["regions"]["A"]) > 0:
                        return {"ROI": data["regions"]["A"]}
                elif "polygons" in data and isinstance(data["polygons"], list):
                    return {"ROI": data["polygons"]}
    except Exception as e:
        print(f"Error cargando ROIs de {roi_json_path}: {e}")
    return default_regions

def load_roi_details(roi_json_path):
    """Returns (regions_dict, is_adjusted, area_mm2, updated_at)."""
    regions = load_rois(roi_json_path)
    is_adjusted = False
    area_mm2 = 0.0
    updated_at = ""
    
    if os.path.exists(roi_json_path):
        try:
            with open(roi_json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    is_adjusted = bool(data.get("is_adjusted", False) or data.get("status") == "adjusted")
                    area_mm2 = float(data.get("total_roi_area_mm2", 0.0))
                    updated_at = str(data.get("updated_at", ""))
                    if area_mm2 == 0.0 and len(regions.get("ROI", [])) > 0:
                        poly = np.array(regions["ROI"][0])
                        if len(poly) >= 3:
                            y, x = poly[:, 0], poly[:, 1]
                            a_px = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            area_mm2 = float((a_px * (0.8913 ** 2)) / 1e6)
        except Exception:
            pass
            
    return regions, is_adjusted, area_mm2, updated_at

def save_rois(roi_json_path, regions_dict, image_shape=None, pixel_size_um=0.8913, is_adjusted=True):
    os.makedirs(os.path.dirname(roi_json_path), exist_ok=True)
    
    # Extract ROI polygons
    polys = regions_dict.get("ROI", [])
    if not polys and "B" in regions_dict:
        polys = regions_dict.get("B", [])
    elif not polys and "A" in regions_dict:
        polys = regions_dict.get("A", [])

    valid_polys = []
    total_area_um2 = 0.0

    for poly in polys:
        poly_arr = np.array(poly)
        if len(poly_arr) >= 3:
            y = poly_arr[:, 0]
            x = poly_arr[:, 1]
            area_px = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            area_um2 = float(area_px * (pixel_size_um ** 2))
            total_area_um2 += area_um2
            valid_polys.append(poly_arr.tolist())

    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pixel_size_um": pixel_size_um,
        "image_shape": list(image_shape) if image_shape is not None else None,
        "is_adjusted": bool(is_adjusted),
        "status": "adjusted" if is_adjusted else "template",
        "n_rois_total": len(valid_polys),
        "total_roi_area_um2": total_area_um2,
        "total_roi_area_mm2": total_area_um2 / 1e6,
        "regions": {
            "ROI": valid_polys,
            "B": valid_polys  # Kept for backward compatibility
        },
        "regions_metadata": {
            "ROI": {
                "n_rois": len(valid_polys),
                "area_um2": total_area_um2,
                "area_mm2": total_area_um2 / 1e6,
                "status": "adjusted" if is_adjusted else "template"
            },
            "B": {
                "n_rois": len(valid_polys),
                "area_um2": total_area_um2,
                "area_mm2": total_area_um2 / 1e6
            }
        }
    }
    
    with open(roi_json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return payload

def points_in_rois(coords_yx, regions_dict, target_region="ALL"):
    if coords_yx is None or len(coords_yx) == 0:
        return np.zeros(len(coords_yx) if coords_yx is not None else 0, dtype=bool)
    
    coords_yx = np.array(coords_yx)
    inside_mask = np.zeros(len(coords_yx), dtype=bool)
    points_xy = coords_yx[:, [1, 0]]
    
    polygons = regions_dict.get("ROI", [])
    if not polygons and "B" in regions_dict:
        polygons = regions_dict.get("B", [])
    elif not polygons and "A" in regions_dict:
        polygons = regions_dict.get("A", [])

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
        
    inside = points_in_rois(coords_yx, regions_dict)
    return ["ROI" if inc else "NONE" for inc in inside]

def create_roi_mask(image_shape_2d, regions_dict, target_region="ALL"):
    mask = np.zeros(image_shape_2d, dtype=bool)
    if not regions_dict:
        return mask
        
    polygons = regions_dict.get("ROI", [])
    if not polygons and "B" in regions_dict:
        polygons = regions_dict.get("B", [])
    elif not polygons and "A" in regions_dict:
        polygons = regions_dict.get("A", [])

    for poly in polygons:
        poly_arr = np.array(poly)
        if len(poly_arr) >= 3:
            rr, cc = skimage_polygon(poly_arr[:, 0], poly_arr[:, 1], shape=image_shape_2d)
            mask[rr, cc] = True
            
    return mask

def compute_roi_area_mm2(image_shape_2d, regions_dict, pixel_size_um=0.8913, target_region="ALL"):
    mask = create_roi_mask(image_shape_2d, regions_dict, target_region=target_region)
    total_px = np.sum(mask)
    return float(total_px * (pixel_size_um ** 2) / 1e6)

def propagate_template_rois_all(all_tasks, template_poly=None, force=False, pixel_size_um=0.8913):
    """
    Propagates the standardized template ROI to all images in the experiment.
    If force is False, skips images where the user already adjusted the ROI.
    """
    if template_poly is None:
        template_poly = STANDARD_TEMPLATE_POLYGON

    created_count = 0
    skipped_adjusted_count = 0
    
    for group, section, filename, raw_path in all_tasks:
        base_fn = os.path.splitext(filename)[0]
        roi_json_path = os.path.join("data/processed/metrics", group, section, f"{base_fn}_rois.json")
        
        should_write = True
        if os.path.exists(roi_json_path) and not force:
            _, is_adj, _, _ = load_roi_details(roi_json_path)
            if is_adj:
                should_write = False
                skipped_adjusted_count += 1
                
        if should_write:
            regs = {"ROI": [template_poly]}
            save_rois(roi_json_path, regs, image_shape=(2048, 2048), pixel_size_um=pixel_size_um, is_adjusted=False)
            created_count += 1
            
    return created_count, skipped_adjusted_count

def check_all_rois_consistency(all_tasks, ref_area_mm2=STANDARD_ROI_AREA_MM2, tolerance_pct=1.0):
    """
    Checks all ROIs in the dataset and reports areas, adjustment status, and consistency.
    """
    records = []
    for group, section, filename, raw_path in all_tasks:
        base_fn = os.path.splitext(filename)[0]
        roi_json_path = os.path.join("data/processed/metrics", group, section, f"{base_fn}_rois.json")
        
        has_file = os.path.exists(roi_json_path)
        regs, is_adj, area_mm2, updated_at = load_roi_details(roi_json_path) if has_file else ({"ROI": []}, False, 0.0, "")
        
        n_polys = len(regs.get("ROI", []))
        diff_pct = abs(area_mm2 - ref_area_mm2) / ref_area_mm2 * 100.0 if area_mm2 > 0 else 100.0
        is_consistent = (diff_pct <= tolerance_pct) and (n_polys == 1)
        
        status_label = "✅ Ajustado" if is_adj else ("⏳ Plantilla" if has_file and n_polys > 0 else "❌ Sin ROI")
        
        records.append({
            "group": group,
            "section": section,
            "filename": filename,
            "has_roi": (n_polys > 0),
            "is_adjusted": is_adj,
            "status": status_label,
            "n_polys": n_polys,
            "area_mm2": area_mm2,
            "area_um2": area_mm2 * 1e6,
            "diff_pct": diff_pct,
            "is_consistent": is_consistent,
            "updated_at": updated_at
        })
        
    df_report = pd.DataFrame(records)
    total_imgs = len(df_report)
    adjusted_imgs = int(df_report['is_adjusted'].sum())
    template_imgs = int(((df_report['has_roi'] == True) & (df_report['is_adjusted'] == False)).sum())
    missing_imgs = int((df_report['has_roi'] == False).sum())
    consistent_imgs = int(df_report['is_consistent'].sum())
    
    summary = {
        "total_imgs": total_imgs,
        "adjusted_imgs": adjusted_imgs,
        "template_imgs": template_imgs,
        "missing_imgs": missing_imgs,
        "consistent_imgs": consistent_imgs,
        "pct_consistent": (consistent_imgs / total_imgs * 100.0) if total_imgs > 0 else 0.0,
        "pct_adjusted": (adjusted_imgs / total_imgs * 100.0) if total_imgs > 0 else 0.0,
        "ref_area_mm2": ref_area_mm2
    }
    return df_report, summary
