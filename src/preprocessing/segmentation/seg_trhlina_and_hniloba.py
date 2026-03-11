import cv2
import numpy as np
import improutils as iu
from .seg_common import kmeans_brightness_labels, mask_from_cluster_ids, to_binary
from skimage.segmentation import morphological_chan_vese


TRHLINA_MIN_COMPONENT_AREA = 20
TRHLINA_MAX_CIRCULARITY = 0.58
TRHLINA_MIN_ASPECT_RATIO = 2.0
TRHLINA_MAX_EXTENT = 0.60
TRHLINA_MAX_OUTER_RING_OVERLAP = 0.85
TRHLINA_MAX_COMPONENT_AREA_RATIO = 0.20

TRHLINA_MIN_MAJOR_TO_MINOR_RATIO = 1.25
TRHLINA_CENTERLINE_MAX_MINOR_FACTOR = 0.9


def segment_trhlina(img, k=5):
    """Segment crack-like dark structures on a wood log image."""
    # Emphasize narrow dark structures while suppressing broad wood texture.
    gray = iu.to_gray(img)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # Promote crack pixels before clustering.
    enhanced = cv2.addWeighted(cv2.bitwise_not(gray), 0.5, blackhat, 0.5, 0)
    
    # Denoise while preserving crack edges.
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=120, sigmaSpace=75)
    
    # Run K-means using a 3-channel representation expected by shared helpers.
    filtered_3c = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    k_labels, _k_centers = kmeans_brightness_labels(filtered_3c, k=k)
    initial_mask = mask_from_cluster_ids(k_labels, [k - 1, k - 2])
    
    # Bridge tiny discontinuities in thin crack lines.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_init_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Chan-Vese expects normalized image and binary initialization.
    gray_float = gray.astype(np.float32) / 255.0
    init_level_set = (clean_init_mask > 0).astype(np.float32)
    
    snake_mask_float = morphological_chan_vese(
        gray_float,
        num_iter=25,
        init_level_set=init_level_set,
        smoothing=0,
    )
    
    return (snake_mask_float * 255).astype(np.uint8)


def refine_trhlina_mask(raw_trhlina_mask, log_mask, outer_ring, background_mask):
    """Remove unlikely crack components using contour shape and position filters."""
    candidate = cv2.bitwise_and(to_binary(raw_trhlina_mask), to_binary(log_mask))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    log_area = max(1, int(cv2.countNonZero(to_binary(log_mask))))

    filtered = np.zeros_like(candidate)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < TRHLINA_MIN_COMPONENT_AREA:
            continue
        if (area / float(log_area)) > TRHLINA_MAX_COMPONENT_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        extent = float(area) / float(w * h)

        component = np.zeros_like(candidate)
        cv2.drawContours(component, [contour], -1, 255, thickness=-1)
        comp_pixels = int(cv2.countNonZero(component))
        if comp_pixels == 0:
            continue

        outer_overlap = int(cv2.countNonZero(cv2.bitwise_and(component, outer_ring))) / float(comp_pixels)

        elongated = aspect_ratio >= TRHLINA_MIN_ASPECT_RATIO and circularity <= TRHLINA_MAX_CIRCULARITY
        thin_irregular = circularity <= 0.32 and extent <= TRHLINA_MAX_EXTENT
        not_outer_round_noise = outer_overlap <= TRHLINA_MAX_OUTER_RING_OVERLAP

        if (elongated or thin_irregular) and not_outer_round_noise:
            cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, close_kernel)

    # Separate the final mask to trhlina and hniloba based on major axis direction
    # relative to log center.
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    trhlina_mask = np.zeros_like(filtered)
    hniloba_mask = np.zeros_like(filtered)

    log_binary = to_binary(log_mask)
    moments = cv2.moments(log_binary)
    if moments["m00"] > 0:
        log_center = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
    else:
        h, w = log_binary.shape[:2]
        log_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        comp_center = np.array(rect[0], dtype=np.float32)

        # If there is no clear major axis, keep it as hniloba.
        aspect_ratio_min_over_max = float(iu.aspect_ratio(contour))
        has_major_axis = aspect_ratio_min_over_max <= (1.0 / TRHLINA_MIN_MAJOR_TO_MINOR_RATIO)
        if not has_major_axis:
            cv2.drawContours(hniloba_mask, [contour], -1, 255, thickness=-1)
            continue

        box = cv2.boxPoints(rect).astype(np.float32)
        longest_edge = None
        longest_len = 0.0
        for i in range(4):
            edge = box[(i + 1) % 4] - box[i]
            edge_len = float(np.linalg.norm(edge))
            if edge_len > longest_len:
                longest_len = edge_len
                longest_edge = edge

        if longest_edge is None or longest_len <= 1e-6:
            cv2.drawContours(hniloba_mask, [contour], -1, 255, thickness=-1)
            continue

        axis_unit = longest_edge / longest_len
        to_center = log_center - comp_center

        # Trhlina usually follows a radial direction, so the line through the major
        # axis should pass near the log center.
        perp_dist_to_axis = abs(float(axis_unit[0] * to_center[1] - axis_unit[1] * to_center[0]))
        minor_axis_len = max(1.0, float(longest_len * aspect_ratio_min_over_max))
        max_allowed_perp_dist = TRHLINA_CENTERLINE_MAX_MINOR_FACTOR * minor_axis_len

        if perp_dist_to_axis <= max_allowed_perp_dist:
            cv2.drawContours(trhlina_mask, [contour], -1, 255, thickness=-1)
        else:
            cv2.drawContours(hniloba_mask, [contour], -1, 255, thickness=-1)
    
    background_mask_dilated = cv2.dilate(
        to_binary(background_mask),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )

    trhlina_mask = cv2.bitwise_and(trhlina_mask, cv2.bitwise_not(background_mask_dilated))

    return trhlina_mask, hniloba_mask