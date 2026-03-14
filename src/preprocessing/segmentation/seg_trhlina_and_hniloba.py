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
    
    # Chan-Vese expects normalized image and binary initialization.
    gray_float = gray.astype(np.float32) / 255.0
    init_level_set = (initial_mask > 0).astype(np.float32)
    
    snake_mask_float = morphological_chan_vese(
        gray_float,
        num_iter=9,
        init_level_set=init_level_set,
        smoothing=0,
    )
    
    return (snake_mask_float * 255).astype(np.uint8)


def refine_trhlina_mask(raw_trhlina_mask, log_mask, outer_ring, background_mask):
    # Restrict to log interior, exclude background.
    masked = cv2.bitwise_and(raw_trhlina_mask, log_mask)
    masked = cv2.bitwise_and(masked, cv2.bitwise_not(background_mask))
    masked = cv2.bitwise_and(masked, cv2.bitwise_not(outer_ring))
    trhlina_mask = np.zeros_like(masked)
    hniloba_mask = np.zeros_like(masked)

    log_area = int(np.count_nonzero(log_mask))

    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TRHLINA_MIN_COMPONENT_AREA:
            continue

        # Draw filled component for overlap queries.
        comp_mask = np.zeros_like(masked)
        cv2.drawContours(comp_mask, [cnt], -1, 255, cv2.FILLED)

        # Fraction of this component that falls inside the outer ring.
        overlap_px = int(np.count_nonzero(cv2.bitwise_and(comp_mask, outer_ring)))
        outer_overlap = overlap_px / area if area > 0 else 0.0

        # Perimeter-based circularity (1 = perfect circle, lower = elongated).
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

        # Bounding-rect aspect ratio and extent.
        _x, _y, w, h = cv2.boundingRect(cnt)
        rect_min = min(w, h)
        aspect_ratio = max(w, h) / rect_min if rect_min > 0 else 1.0
        extent = area / (w * h) if w * h > 0 else 0.0

        # Ellipse major-to-minor ratio (more stable than bounding rect for skewed shapes).
        if len(cnt) >= 5:
            _center, (axis_a, axis_b), _angle = cv2.fitEllipse(cnt)
            ax_min = min(axis_a, axis_b)
            major_to_minor = max(axis_a, axis_b) / ax_min if ax_min > 0 else 1.0
        else:
            major_to_minor = aspect_ratio

        # Large dark blobs that fill much of the log are rot, not cracks.
        area_ratio = area / log_area if log_area > 0 else 0.0

        # Trhlina (crack): elongated, low circularity, not dominated by outer bark.
        is_trhlina = (
            circularity <= TRHLINA_MAX_CIRCULARITY
            and aspect_ratio >= TRHLINA_MIN_ASPECT_RATIO
            and extent <= TRHLINA_MAX_EXTENT
            and major_to_minor >= TRHLINA_MIN_MAJOR_TO_MINOR_RATIO
            and outer_overlap <= TRHLINA_MAX_OUTER_RING_OVERLAP
            and area_ratio <= TRHLINA_MAX_COMPONENT_AREA_RATIO
        )

        if is_trhlina:
            cv2.drawContours(trhlina_mask, [cnt], -1, 255, cv2.FILLED)
        else:
            cv2.drawContours(hniloba_mask, [cnt], -1, 255, cv2.FILLED)

    return trhlina_mask, hniloba_mask