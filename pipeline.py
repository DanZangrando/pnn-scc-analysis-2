import os
import json
import numpy as np
import cv2
import tifffile as tiff
from cellpose import models
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage import exposure, draw
from skimage.morphology import skeletonize, disk, binary_dilation, binary_closing, remove_small_objects
from skan import Skeleton, summarize
import pandas as pd
import scipy.ndimage as ndi
from skimage.measure import label as cc_label

def get_or_create_mip(raw_path, px_size=1.0):
    if "data/processed/mips" in raw_path:
        return raw_path
        
    # Construct MIP path: replace 'data/raw' with 'data/processed/mips'
    mip_path = raw_path.replace("data/raw", "data/processed/mips")
    os.makedirs(os.path.dirname(mip_path), exist_ok=True)
    
    if not os.path.exists(mip_path):
        print(f"Creando MIP para {raw_path} -> {mip_path}")
        img = tiff.imread(raw_path)
        with tiff.TiffFile(raw_path) as tif:
            axes = tif.series[0].axes
            
        if 'Z' in axes and len(img.shape) >= 4:
            z_idx = axes.index('Z')
            img = np.max(img, axis=z_idx)
            axes = axes.replace('Z', '')
            
        if axes == 'YXC':
            img = np.transpose(img, (2, 0, 1))
            
        # Ensure it has exactly 4 channels (pad or slice)
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
    # If loading from raw directory, redirect to cached MIP
    if "data/raw" in path:
        px_size = 1.0
        try:
            if os.path.exists("experiment_config.json"):
                with open("experiment_config.json", 'r') as f:
                    cfg = json.load(f)
                    px_size = cfg.get("pixel_size_um", 1.0)
        except Exception:
            pass
        path = get_or_create_mip(path, px_size)
        
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

def connect_nearby_fragments(binary_mask, radius_um, px_size):
    radius_px = int(radius_um / px_size)
    if radius_px < 1:
        return binary_mask
    selem = disk(radius_px)
    dilated = binary_dilation(binary_mask, selem)
    closed = binary_closing(dilated, selem)
    return closed

def prune_skeleton(skeleton_binary, pruning_min_voxels):
    from scipy.ndimage import convolve
    if pruning_min_voxels <= 0 or not np.any(skeleton_binary):
        return skeleton_binary
        
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    pruned_skel = skeleton_binary.copy()
    max_passes = 30    
    for _pass in range(max_passes):
        neighbors = convolve(pruned_skel.astype(int), kernel, mode='constant', cval=0)
        
        # Endpoints: 1 neighbor. Isolated: 0.
        endpoints_coords = np.argwhere(pruned_skel & (neighbors <= 1))
        
        if len(endpoints_coords) == 0:
            break
            
        pixels_to_remove = set()
        
        for ep_y, ep_x in endpoints_coords:
            branch_path = [(ep_y, ep_x)]
            current_y, current_x = ep_y, ep_x
            
            is_short_branch = True
            for step in range(pruning_min_voxels - 1):
                active_neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = current_y + dy, current_x + dx
                        if 0 <= ny < pruned_skel.shape[0] and 0 <= nx < pruned_skel.shape[1]:
                            if pruned_skel[ny, nx] and (ny, nx) not in branch_path:
                                active_neighbors.append((ny, nx))
                
                if len(active_neighbors) > 1:
                    break
                elif len(active_neighbors) == 1:
                    current_y, current_x = active_neighbors[0]
                    branch_path.append((current_y, current_x))
                else:
                    break
            else:
                is_short_branch = False
                
            if is_short_branch:
                for py, px in branch_path:
                    pixels_to_remove.add((py, px))
                    
        if not pixels_to_remove:
            break
            
        for py, px in pixels_to_remove:
            pruned_skel[py, px] = False
            
    pruned_skel = remove_small_objects(pruned_skel, min_size=pruning_min_voxels + 1, connectivity=2)
    return skeletonize(pruned_skel)

def filter_skeleton_by_nucleus_connectivity(skeleton, nuclear_mask):
    if not np.any(skeleton):
        return skeleton
    
    labeled_skeleton, n_components = cc_label(skeleton, return_num=True, connectivity=2)
    
    if n_components <= 1:
        return skeleton
    
    nuclear_overlap = labeled_skeleton * nuclear_mask
    components_touching = np.unique(nuclear_overlap)
    components_touching = components_touching[components_touching > 0]
    
    if len(components_touching) == 0:
        nucleus_coords = np.argwhere(nuclear_mask)
        if len(nucleus_coords) == 0:
            return skeleton
        
        nucleus_centroid = nucleus_coords.mean(axis=0)
        min_dist = np.inf
        closest_component = 1
        
        for comp_id in range(1, n_components + 1):
            comp_coords = np.argwhere(labeled_skeleton == comp_id)
            if len(comp_coords) == 0:
                continue
            
            dists = np.linalg.norm(comp_coords - nucleus_centroid, axis=1)
            min_comp_dist = np.min(dists)
            
            if min_comp_dist < min_dist:
                min_dist = min_comp_dist
                closest_component = comp_id
        
        return (labeled_skeleton == closest_component)
    
    return np.isin(labeled_skeleton, components_touching)

def analyze_cell_skeleton(skeleton_labels, label, px_size):
    ys, xs = np.where(skeleton_labels == label)
    if len(ys) <= 2:
        return 0.0, 0, 0.0, 0, 0, 1.0, 0.0
    
    min_y, max_y = np.min(ys), np.max(ys)
    min_x, max_x = np.min(xs), np.max(xs)
    
    min_y = max(0, min_y - 2)
    max_y = min(skeleton_labels.shape[0], max_y + 3)
    min_x = max(0, min_x - 2)
    max_x = min(skeleton_labels.shape[1], max_x + 3)
    
    crop = (skeleton_labels[min_y:max_y, min_x:max_x] == label)
    
    try:
        sk_obj = Skeleton(crop, spacing=px_size)
        summary = summarize(sk_obj)
        if not summary.empty:
            total_length = float(summary['branch-distance'].sum())
            n_branches = int(len(summary))
            avg_branch_len = float(summary['branch-distance'].mean())
            
            # Additional topological metrics
            degrees = sk_obj.degrees
            n_endpoints = int(np.sum(degrees == 1))
            n_junctions = int(np.sum(degrees > 2))
            
            # Tortuosity
            tortuosity = summary['branch-distance'] / summary['euclidean-distance'].replace(0, np.nan)
            tortuosity_mean = float(tortuosity.mean()) if not tortuosity.isna().all() else 1.0
            
            # Ramification index
            ramification_index = float(n_branches / max(n_junctions, 1))
            
            return total_length, n_branches, avg_branch_len, n_endpoints, n_junctions, tortuosity_mean, ramification_index
    except Exception:
        pass
    return 0.0, 0, 0.0, 0, 0, 1.0, 0.0

def analyze_skeleton_thickness_intensity(w_raw, skeleton_labels, label, wfa_edt, px_size):
    ys, xs = np.where(skeleton_labels == label)
    if len(ys) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # 1. Local Thickness (diameter)
    local_radii = wfa_edt[ys, xs]
    local_diameters_um = local_radii * 2.0 * px_size
    mean_thickness = float(np.mean(local_diameters_um))
    max_thickness = float(np.max(local_diameters_um))
    
    # 2. Local WFA Intensity along skeleton
    local_intensities = w_raw[ys, xs]
    mean_intensity = float(np.mean(local_intensities))
    
    # 3. Sum of WFA Intensity in a 1.5 um neighborhood around the skeleton
    dil_px = max(1, int(1.5 / px_size))
    cell_skel_mask = (skeleton_labels == label)
    
    min_y, max_y = np.min(ys), np.max(ys)
    min_x, max_x = np.min(xs), np.max(xs)
    
    min_y = max(0, min_y - dil_px - 1)
    max_y = min(skeleton_labels.shape[0], max_y + dil_px + 2)
    min_x = max(0, min_x - dil_px - 1)
    max_x = min(skeleton_labels.shape[1], max_x + dil_px + 2)
    
    local_skel = cell_skel_mask[min_y:max_y, min_x:max_x]
    selem = disk(dil_px)
    local_dilated = binary_dilation(local_skel, selem)
    
    local_wfa = w_raw[min_y:max_y, min_x:max_x]
    neighborhood_sum = float(np.sum(local_wfa[local_dilated]))
    
    return mean_thickness, max_thickness, mean_intensity, neighborhood_sum

def run_pipeline_on_file(tif_path, out_segm_dir, out_metrics_dir,
                         model_dapi, model_pv_obj,
                         filter_type, diameter, flow_threshold, cellprob_threshold,
                         pv_filter_type, pv_diameter, pv_flow_threshold, pv_cellprob_threshold,
                         pv_expansion_dist_um, pnn_threshold, pnn_exclusion_dist_um,
                         px_size, do_pv_segmentation, calib_data):
    fname = os.path.basename(tif_path)
    base_name, _ = os.path.splitext(fname)
    (p_raw, w_raw, d_raw, a_raw) = load_channels_tif(tif_path)

    # DAPI preprocessing (keep for overlay visualization only)
    in_dapi = d_raw.copy()
    if filter_type == "Otsu Global":
        t = threshold_otsu(in_dapi)
        in_dapi[in_dapi < t] = 0
    elif filter_type == "CLAHE (Adaptativo Local)":
        clahe = exposure.equalize_adapthist(in_dapi, clip_limit=0.03)
        in_dapi = (clahe * 65535).astype(np.uint16)

    m_dapi, _, _ = model_dapi.eval(in_dapi, diameter=diameter, 
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
        m_pv, _, _ = model_pv_obj.eval(in_pv, diameter=pv_diameter, 
                                        flow_threshold=pv_flow_threshold, cellprob_threshold=pv_cellprob_threshold)

    # WFA Cellpose preprocessing (segmenting PNN somas/holes)
    m_wfa_cellpose = np.zeros_like(m_dapi)
    do_wfa_cellpose = calib_data.get('do_wfa_cellpose', True)
    if do_wfa_cellpose and model_pv_obj is not None:
        in_wfa_cp = w_raw.copy()
        wfa_cp_filter = calib_data.get('wfa_cellpose_filter_type', 'Ninguno')
        if wfa_cp_filter == "Otsu Global":
            t = threshold_otsu(in_wfa_cp)
            in_wfa_cp[in_wfa_cp < t] = 0
        elif wfa_cp_filter == "CLAHE (Adaptativo Local)":
            clahe = exposure.equalize_adapthist(in_wfa_cp, clip_limit=0.03)
            in_wfa_cp = (clahe * 65535).astype(np.uint16)
        wfa_cp_diam = float(calib_data.get('wfa_cellpose_diameter', 30.0))
        wfa_cp_flow = float(calib_data.get('wfa_cellpose_flow_threshold', 0.4))
        wfa_cp_prob = float(calib_data.get('wfa_cellpose_cellprob_threshold', 0.0))
        m_wfa_cellpose, _, _ = model_pv_obj.eval(in_wfa_cp, diameter=wfa_cp_diam,
                                                 flow_threshold=wfa_cp_flow, cellprob_threshold=wfa_cp_prob)

    # Somas regionprops
    pv_props = regionprops(m_pv)
    wfa_props = regionprops(m_wfa_cellpose)

    # Fetch parameters
    pnn_wfa_threshold_method = calib_data.get('pnn_wfa_threshold_method', 'Automático (Otsu)')
    pnn_wfa_manual_threshold = float(calib_data.get('pnn_wfa_manual_threshold', 10000.0))
    max_pnn_distance_um = float(calib_data.get('max_pnn_distance_um', 20.0))
    
    pnn_connect_fragments = calib_data.get('pnn_connect_fragments', False)
    pnn_connection_radius_um = float(calib_data.get('pnn_connection_radius_um', 1.0))
    pnn_pruning_min_voxels = int(calib_data.get('pnn_pruning_min_voxels', 0))
    pnn_filter_by_nucleus = calib_data.get('pnn_filter_by_nucleus', False)
    pnn_gaussian_sigma = float(calib_data.get('pnn_gaussian_sigma', 1.0))

    # Global WFA Preprocessing (Smoothing)
    w_proc = w_raw.copy()
    if pnn_gaussian_sigma > 0:
        w_proc = ndi.gaussian_filter(w_raw.astype(np.float32), sigma=pnn_gaussian_sigma)

    # Global WFA Binarization
    if pnn_wfa_threshold_method == "Automático (Otsu)":
        try:
            t_wfa = threshold_otsu(w_proc[w_proc > 0])
        except Exception:
            t_wfa = 1000.0
    else:
        t_wfa = pnn_wfa_manual_threshold
        
    pnn_binary = w_proc > t_wfa
    
    if pnn_connect_fragments:
        pnn_binary = connect_nearby_fragments(pnn_binary, pnn_connection_radius_um, px_size)
        
    # Erode the WFA Cellpose mask so we preserve the outer ring boundary for skeletonization
    from skimage.morphology import binary_erosion
    
    # Erode by 2.0 um to preserve the ring/boundary of WFA Cellpose masks
    erode_px = max(1, int(2.0 / px_size))
    selem = disk(erode_px)
    
    m_wfa_eroded = np.zeros_like(m_wfa_cellpose)
    for wfa_prop in wfa_props:
        wfa_lbl = wfa_prop.label
        submask = (m_wfa_cellpose == wfa_lbl)
        eroded_sub = binary_erosion(submask, selem)
        m_wfa_eroded[eroded_sub] = wfa_lbl
        
    pnn_binary_for_skeleton = pnn_binary & (~(m_wfa_eroded > 0))
    wfa_skeleton = skeletonize(pnn_binary_for_skeleton)
    
    if pnn_pruning_min_voxels > 0:
        wfa_skeleton = prune_skeleton(wfa_skeleton, pnn_pruning_min_voxels)

    # Calculate WFA distance transform once globally for local thickness
    wfa_edt = ndi.distance_transform_edt(pnn_binary)

    # Voronoi Partitioning of Skeleton based on PV somas
    # Voronoi Partitioning of Skeleton based on WFA Cellpose somas (hollow & filled PNNs)
    skeleton_labels = np.zeros_like(m_wfa_cellpose, dtype=np.uint16)
    max_dist_px = max_pnn_distance_um / px_size
    
    if np.max(m_wfa_cellpose) > 0:
        distances, indices = ndi.distance_transform_edt(m_wfa_cellpose == 0, return_indices=True)
        nearest_labels = m_wfa_cellpose[indices[0], indices[1]]
        valid_mask = (wfa_skeleton > 0) & (distances <= max_dist_px)
        skeleton_labels[valid_mask] = nearest_labels[valid_mask]


    # Pre-map WFA somas to PV somas to find overlaps based on largest surface area
    wfa_to_pv = {}
    pv_to_wfa = {}
    for wfa_prop in wfa_props:
        wfa_label = wfa_prop.label
        wfa_mask = (m_wfa_cellpose == wfa_label)
        
        # Get all PV labels overlapping with this WFA mask
        pv_in_wfa = m_pv[wfa_mask]
        unique_pv, counts = np.unique(pv_in_wfa, return_counts=True)
        
        # Filter out background (0)
        valid_idx = unique_pv > 0
        unique_pv = unique_pv[valid_idx]
        counts = counts[valid_idx]
        
        if len(unique_pv) > 0:
            # Find the PV label with the maximum overlap area (count of pixels)
            best_idx = np.argmax(counts)
            best_pv_lbl = unique_pv[best_idx]
            
            wfa_to_pv[wfa_label] = best_pv_lbl
            pv_to_wfa[best_pv_lbl] = wfa_label

    r_batch = []
    # Keep track of matched PV labels
    matched_pv_labels = set(wfa_to_pv.values())

    # 1. Process all WFA PNN+ somas (both PV+ and hollow PV-)
    for wfa_prop in wfa_props:
        wfa_label = wfa_prop.label
        wfa_mask = (m_wfa_cellpose == wfa_label)
        
        # PNN soma properties
        w_cy, w_cx = wfa_prop.centroid
        w_area = wfa_prop.area * (px_size ** 2)
        w_diam = wfa_prop.equivalent_diameter_area * px_size
        
        # Check if matched to any PV+ soma
        pv_label = wfa_to_pv.get(wfa_label, None)
        
        if pv_label is not None:
            # PV+/PNN+ Cell
            pv_prop = next((p for p in pv_props if p.label == pv_label), None)
            if pv_prop is not None:
                cy, cx = pv_prop.centroid
                pv_area = pv_prop.area * (px_size ** 2)
                pv_diameter = pv_prop.equivalent_diameter_area * px_size
            else:
                cy, cx = w_cy, w_cx
                pv_area = 0.0
                pv_diameter = 0.0
            cell_type = "PV+/PNN+"
            is_pv_plus = True
        else:
            # PV-/PNN+ (Hollow PNN)
            cy, cx = w_cy, w_cx
            pv_area = 0.0
            pv_diameter = 0.0
            cell_type = "PV-/PNN+"
            is_pv_plus = False

        # Calculate WFA sum intensity inside the soma/hole mask itself
        wfa_s = float(np.sum(w_raw[wfa_mask]))

        # Apply Cell connectivity filter to skeleton if active
        if pnn_filter_by_nucleus:
            cell_skel = (skeleton_labels == wfa_label)
            if np.any(cell_skel):
                filtered_skel = filter_skeleton_by_nucleus_connectivity(cell_skel, wfa_mask)
                skeleton_labels[skeleton_labels == wfa_label] = 0
                skeleton_labels[filtered_skel] = wfa_label

        # Skeleton metrics via Skan helper
        skel_length, skel_branches, skel_avg_branch, skel_endpoints, skel_junctions, skel_tortuosity, skel_ramification = analyze_cell_skeleton(skeleton_labels, wfa_label, px_size)
        
        # Thickness and intensity along skeleton
        skel_mean_thick, skel_max_thick, skel_mean_int, skel_neighborhood_sum = analyze_skeleton_thickness_intensity(
            w_raw, skeleton_labels, wfa_label, wfa_edt, px_size
        )

        r_batch.append({
            'label': wfa_label,
            'centroid_y': cy,
            'centroid_x': cx,
            'area_um2': pv_area if is_pv_plus else w_area,
            'diameter_um': pv_diameter if is_pv_plus else w_diam,
            'wfa_sum_intensity': wfa_s,
            'is_pnn_plus': True,
            'is_pv_plus': is_pv_plus,
            'pv_label': pv_label if is_pv_plus else -1,
            'skel_total_length_um': skel_length,
            'skel_branches_count': skel_branches,
            'skel_avg_branch_len_um': skel_avg_branch,
            'skel_endpoints_count': skel_endpoints,
            'skel_junctions_count': skel_junctions,
            'skel_tortuosity_mean': skel_tortuosity,
            'skel_ramification_index': skel_ramification,
            'skel_mean_thickness_um': skel_mean_thick,
            'skel_max_thickness_um': skel_max_thick,
            'skel_mean_intensity': skel_mean_int,
            'skel_neighborhood_wfa_sum': skel_neighborhood_sum,
            'cell_type': cell_type,
            'pv_area_um2': pv_area,
            'pv_diameter_um': pv_diameter,
            'pnn_area_um2': w_area,
            'pnn_diameter_um': w_diam
        })

    # 2. Process all PV+ cells without PNN (PV+/PNN-)
    max_wfa_label = int(np.max(m_wfa_cellpose)) if np.max(m_wfa_cellpose) > 0 else 0
    for pvp in pv_props:
        pv_label = pvp.label
        if pv_label in matched_pv_labels:
            continue
            
        pv_mask = (m_pv == pv_label)
        cy, cx = pvp.centroid
        pv_area = pvp.area * (px_size ** 2)
        pv_diameter = pvp.equivalent_diameter_area * px_size
        
        # Calculate WFA sum intensity inside the PV soma itself
        wfa_s = float(np.sum(w_raw[pv_mask]))

        # Unique label ID to avoid collisions
        unique_label = max_wfa_label + pv_label
        
        r_batch.append({
            'label': unique_label,
            'centroid_y': cy,
            'centroid_x': cx,
            'area_um2': pv_area,
            'diameter_um': pv_diameter,
            'wfa_sum_intensity': wfa_s,
            'is_pnn_plus': False,
            'is_pv_plus': True,
            'pv_label': pv_label,
            'skel_total_length_um': 0.0,
            'skel_branches_count': 0,
            'skel_avg_branch_len_um': 0.0,
            'skel_endpoints_count': 0,
            'skel_junctions_count': 0,
            'skel_tortuosity_mean': 1.0,
            'skel_ramification_index': 0.0,
            'skel_mean_thickness_um': 0.0,
            'skel_max_thickness_um': 0.0,
            'skel_mean_intensity': 0.0,
            'skel_neighborhood_wfa_sum': 0.0,
            'cell_type': "PV+/PNN-",
            'pv_area_um2': pv_area,
            'pv_diameter_um': pv_diameter,
            'pnn_area_um2': 0.0,
            'pnn_diameter_um': 0.0
        })

    # Keep all segmented cells without NMS exclusion
    r_batch = [r for r in r_batch if r['cell_type'] != "PV-/PNN-"]

    df_b = pd.DataFrame(r_batch)
    if df_b.empty:
        df_b = pd.DataFrame(columns=[
            'label', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um', 
            'wfa_sum_intensity', 'is_pnn_plus', 'is_pv_plus', 'pv_label',
            'skel_total_length_um', 'skel_branches_count', 'skel_avg_branch_len_um',
            'skel_endpoints_count', 'skel_junctions_count', 'skel_tortuosity_mean', 'skel_ramification_index',
            'skel_mean_thickness_um', 'skel_max_thickness_um', 'skel_mean_intensity', 'skel_neighborhood_wfa_sum',
            'cell_type', 'pv_area_um2', 'pv_diameter_um', 'pnn_area_um2', 'pnn_diameter_um'
        ])
        
    csv_name = f"{base_name}_nuclei_metrics.csv"
    df_b.to_csv(os.path.join(out_metrics_dir, csv_name), index=False)

    # TIFF output - only save the 4 masks to save space
    stk = np.stack([m_dapi.astype(np.uint16),
                    m_pv.astype(np.uint16),
                    skeleton_labels.astype(np.uint16),
                    m_wfa_cellpose.astype(np.uint16)], axis=0)
                          
    segm_name = f"{base_name}_masks.tif"
    tiff.imwrite(os.path.join(out_segm_dir, segm_name),
                 stk, imagej=True,
                 metadata={'spacing': px_size, 'unit': 'um', 'Axes': 'CYX',
                           'Labels': ['DAPI_Mask', 'PV_Mask', 'PNN_Skeleton_Mask', 'WFA_Cellpose_Mask']})

    # DAPI-centric metrics calculation - aligned with Cellpose spatial mapping (no arbitrary threshold/diameter)
    pnn_radius_um = float(calib_data.get('pnn_radius_um', 20.0))
    dapi_props = regionprops(m_dapi, intensity_image=w_raw)
    dapi_batch = []
    
    for db in dapi_props:
        cy, cx = db.centroid
        
        # Check colocalization: centroid inside PV mask or WFA Cellpose mask
        is_pv_coloc = bool(m_pv[int(cy), int(cx)] > 0)
        is_pnn = bool(m_wfa_cellpose[int(cy), int(cx)] > 0)
        
        # Calculate WFA sum in a disk for historical reference only
        r_px = pnn_radius_um / px_size if pnn_radius_um > 0 else 20.0 / px_size
        rd, cd = draw.disk((cy, cx), r_px, shape=w_raw.shape)
        wfa_sum = float(np.sum(w_raw[rd, cd]))
        
        dapi_batch.append({
            'label': db.label,
            'centroid_y': cy,
            'centroid_x': cx,
            'area_um2': db.area * (px_size ** 2),
            'diameter_um': db.equivalent_diameter_area * px_size,
            'dapi_mean_intensity': float(db.intensity_mean),
            'wfa_sum_intensity': wfa_sum,
            'is_pnn_plus': is_pnn,
            'is_pv_plus': is_pv_coloc
        })
        
    df_dapi = pd.DataFrame(dapi_batch)
    if df_dapi.empty:
        df_dapi = pd.DataFrame(columns=[
            'label', 'centroid_y', 'centroid_x', 'area_um2', 'diameter_um',
            'dapi_mean_intensity', 'wfa_sum_intensity', 'is_pnn_plus', 'is_pv_plus'
        ])
    
    dapi_csv_name = f"{base_name}_dapi_metrics.csv"
    df_dapi.to_csv(os.path.join(out_metrics_dir, dapi_csv_name), index=False)

    # Summary JSON calculations
    total_pv_segmentation = int(np.max(m_pv)) if np.max(m_pv) > 0 else 0
    pv_pnn_plus = int(sum(1 for r in r_batch if r['cell_type'] == "PV+/PNN+"))
    pv_pnn_minus = int(sum(1 for r in r_batch if r['cell_type'] == "PV+/PNN-"))
    hollow_pnn_plus = int(sum(1 for r in r_batch if r['cell_type'] == "PV-/PNN+"))
    total_pnn_plus = int(sum(1 for r in r_batch if r['is_pnn_plus']))
    
    summary = {
        "total_dapi": int(np.max(m_dapi)) if np.max(m_dapi) > 0 else 0,
        "total_pv_segmentation": total_pv_segmentation,
        "pnn_plus": total_pnn_plus,            # For backward compatibility
        "pnn_minus": pv_pnn_minus,              # For backward compatibility
        "dapi_pv_coloc": pv_pnn_plus,           # For backward compatibility
        "pv_pnn_plus": pv_pnn_plus,
        "pv_pnn_minus": pv_pnn_minus,
        "hollow_pnn_plus": hollow_pnn_plus,
        "total_pnn_plus": total_pnn_plus,
        "pixel_size": px_size
    }
    
    json_name = f"{base_name}_summary.json"
    with open(os.path.join(out_metrics_dir, json_name), 'w') as fs:
        json.dump(summary, fs, indent=4)

    return summary
