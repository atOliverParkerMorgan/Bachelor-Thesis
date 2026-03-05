import cv2
import numpy as np
import improutils as iu


def _passes_crack_aspect_ratio(contour, max_aspect_ratio):
    ratio = iu.aspect_ratio(contour)
    return ratio <= max_aspect_ratio or ratio >= (1.0 / max(max_aspect_ratio, 1e-6))


def segment_trhlina_and_hniloba(
    inner_log_img,
    knot_mask,
    crack_min_area,
    crack_max_aspect_ratio,
    trhlina_hniloba_roundness_boundary,
    crack_threshold=90,
    hniloba_min_area=0,
    restrict_hniloba_to_knot=True,
):
    """Split edge contours into cracks/trhlina and decay/hniloba by roundness."""
    gray = iu.to_gray(inner_log_img)

    scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    scharr_mag = cv2.magnitude(scharr_x, scharr_y)
    scharr_norm = cv2.normalize(
        scharr_mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    crack_candidates = iu.segmentation_one_threshold(scharr_norm, int(crack_threshold))
    _, _, contours = iu.find_contours(
        crack_candidates,
        min_area=min(crack_min_area, hniloba_min_area),
        fill=False,
    )

    trhlina_mask = np.zeros_like(crack_candidates)
    hniloba_mask = np.zeros_like(crack_candidates)
    trhlina_contours = []
    hniloba_contours = []

    boundary = float(trhlina_hniloba_roundness_boundary)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        roundness = iu.roundness(contour)

        if roundness <= boundary:
            if contour_area >= crack_min_area and _passes_crack_aspect_ratio(
                contour, crack_max_aspect_ratio
            ):
                trhlina_contours.append(contour)
        else:
            if contour_area >= hniloba_min_area:
                hniloba_contours.append(contour)

    if trhlina_contours:
        cv2.drawContours(trhlina_mask, trhlina_contours, -1, 255, thickness=cv2.FILLED)

    if hniloba_contours:
        cv2.drawContours(hniloba_mask, hniloba_contours, -1, 255, thickness=cv2.FILLED)

    if restrict_hniloba_to_knot and knot_mask is not None:
        hniloba_mask = cv2.bitwise_and(hniloba_mask, knot_mask)

    return trhlina_mask, hniloba_mask
