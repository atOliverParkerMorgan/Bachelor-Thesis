import cv2
import numpy as np
from .seg_common import mask_from_cluster_ids
import improutils as iu

def segment_crust(sorted_labels, log_mask, max_bark_thickness_px=45):
    """
    K-means bark segmentation: isolates the large outer ring 
    by combining morphological cleaning and a spatial edge constraint.
    """
    # 1. Get the raw mask from K-means (contains the inner ring noise)
    raw_crust_mask = mask_from_cluster_ids(sorted_labels, cluster_ids={1}, valid_mask=log_mask)
    
    # 2. Morphological opening to remove small inner ring blobs
    k_size = max_bark_thickness_px if max_bark_thickness_px % 2 != 0 else max_bark_thickness_px + 1
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    inner_log = cv2.erode(log_mask, edge_kernel, iterations=1)
    
    # The valid edge zone is the full log_mask minus the shrunken inner_log
    edge_zone = cv2.bitwise_xor(log_mask, inner_log)
    
    # 4. Final output: Keep only thick structures that fall within the edge zone
    final_crust = cv2.bitwise_and(raw_crust_mask, edge_zone)
    return final_crust

def mask_from_cluster_ids(sorted_labels, cluster_ids, valid_mask=None):
    """Build a binary mask from selected sorted K-means cluster ids."""
    cluster_ids = list(cluster_ids)
    mask = np.isin(sorted_labels, cluster_ids).astype(np.uint8) * 255

    if valid_mask is not None:
        valid_mask_bin = np.where(valid_mask > 0, 255, 0).astype(np.uint8)
        mask = cv2.bitwise_and(mask, valid_mask_bin)

    return mask