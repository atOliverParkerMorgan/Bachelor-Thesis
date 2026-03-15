import cv2
import numpy as np
import improutils as iu

from .seg_common import to_binary

# --- Tuning constants ---
TRHLINA_SCHARR_THRESHOLD = 50
TRHLINA_MIN_AREA = 8

# Radius (pixels) within which a dark component is considered near a suk → hniloba
HNILOBA_SUK_PROXIMITY_PX = 30


def segment_trhlina(log_img: np.ndarray, background_mask: np.ndarray) -> np.ndarray:

    gray = iu.to_gray(log_img)
    neutral_gray = gray.copy()
    background_mask_dilated = cv2.dilate(background_mask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32, 32)), iterations=1)

    neutral_gray[background_mask_dilated == 255] = 125

    # Black-hat highlights dark features smaller than the kernel
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
    blackhat = cv2.morphologyEx(neutral_gray, cv2.MORPH_BLACKHAT, kernel_bh)

    # Emphasise dark regions before gradient computation
    enhanced = cv2.addWeighted(cv2.bitwise_not(neutral_gray), 0.5, blackhat, 0.5, 0)

    # Smooth while preserving crack edges
    filtered = cv2.bilateralFilter(enhanced, d=7, sigmaColor=110, sigmaSpace=20)

    # Scharr gradient magnitude
    scharr_x = cv2.Scharr(filtered, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(filtered, cv2.CV_64F, 0, 1)
    scharr_mag = cv2.magnitude(scharr_x, scharr_y)
    scharr_mag = cv2.normalize(scharr_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, thresh = cv2.threshold(scharr_mag, TRHLINA_SCHARR_THRESHOLD, 255, cv2.THRESH_BINARY)
    filled, _, _ = iu.find_contours(thresh, min_area=TRHLINA_MIN_AREA, fill=True)

    return filled


def split_by_suk_proximity(
    candidate: np.ndarray,
    suk_mask: np.ndarray,
    proximity_px: int = HNILOBA_SUK_PROXIMITY_PX,
) -> tuple[np.ndarray, np.ndarray]:

    binary_candidate = to_binary(candidate)
    binary_suk = to_binary(suk_mask)

    if cv2.countNonZero(binary_suk) == 0:
        # No knots detected — everything is classified as a crack
        return binary_candidate, np.zeros_like(binary_candidate)

    # Dilate suk region to define the proximity zone
    kernel_size = 2 * proximity_px + 1
    proximity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    suk_zone = cv2.dilate(binary_suk, proximity_kernel, iterations=1)

    n_labels, label_img, _, _ = cv2.connectedComponentsWithStats(binary_candidate, connectivity=8)

    trhlina = np.zeros_like(binary_candidate)
    hniloba = np.zeros_like(binary_candidate)
    for label_id in range(1, n_labels):
        component = (label_img == label_id).astype(np.uint8) * 255
        near_suk = cv2.countNonZero(cv2.bitwise_and(component, suk_zone)) > 0
        if near_suk:
            hniloba = cv2.bitwise_or(hniloba, component)
        else:
            trhlina = cv2.bitwise_or(trhlina, component)

    return trhlina, hniloba


def refine_trhlina_mask(
    raw_mask: np.ndarray,
    log_mask: np.ndarray,
    outer_ring: np.ndarray | None,
    background_mask: np.ndarray,
    suk_mask: np.ndarray | None = None,
    proximity_px: int = HNILOBA_SUK_PROXIMITY_PX,
) -> tuple[np.ndarray, np.ndarray]:

    candidate = cv2.bitwise_and(to_binary(raw_mask), to_binary(log_mask))

    # Remove crack detections that fall on the outer bark ring
    if outer_ring is not None:
        outer_guard = cv2.dilate(
            to_binary(outer_ring),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(outer_guard))

    if suk_mask is not None:
        return split_by_suk_proximity(candidate, suk_mask, proximity_px)

    return candidate, np.zeros_like(candidate)

