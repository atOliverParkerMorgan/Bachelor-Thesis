import cv2
import numpy as np
import improutils as iu


def radial_dilate(mask, kernel_size, center=None):
    """Dilate a mask radially outwards (away from center)."""
    if kernel_size <= 1:
        return mask

    h, w = mask.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)

    max_radius = np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
    polar_img = cv2.warpPolar(mask, (w, h), center, max_radius, cv2.WARP_POLAR_LINEAR)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    anchor_point = (kernel_size - 1, 0)
    polar_dilated = cv2.dilate(
        polar_img,
        dilation_kernel,
        anchor=anchor_point,
        iterations=1,
    )

    result = cv2.warpPolar(
        polar_dilated,
        (w, h),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    return result


def keep_mask_near_log_edge(mask, log_mask, max_edge_distance_ratio, max_edge_distance_px):
    """Keep mask pixels only within a limited inward distance from log boundary."""
    log_bin = (log_mask > 0).astype(np.uint8)
    if not np.any(log_bin):
        return np.zeros_like(mask)

    distance_to_edge = cv2.distanceTransform(log_bin, cv2.DIST_L2, 5)
    max_distance_inside_log = float(distance_to_edge.max())
    if max_distance_inside_log <= 0:
        return np.zeros_like(mask)

    ratio = float(np.clip(max_edge_distance_ratio, 0.0, 1.0))
    allowed_distance = ratio * max_distance_inside_log
    if max_edge_distance_px and max_edge_distance_px > 0:
        allowed_distance = min(allowed_distance, float(max_edge_distance_px))

    if allowed_distance <= 0:
        return np.zeros_like(mask)

    near_edge_region = (distance_to_edge <= allowed_distance) & (log_bin == 1)
    filtered = np.zeros_like(mask)
    filtered[(mask > 0) & near_edge_region] = 255
    return filtered


def segment_crust(
    img,
    log_mask,
    alpha,
    beta,
    wood_thresh,
    wood_close_kernel_size,
    crust_connect_kernel_size,
    sobel_kernel_size,
    crust_add_kernel_size,
    crust_max_edge_distance_ratio,
    crust_max_edge_distance_px,
):
    """Isolate outer bark (kura) using threshold and edge-driven extension."""
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=log_mask)

    _, thresh_wood = cv2.threshold(gray_masked, wood_thresh, 255, cv2.THRESH_BINARY)

    close_kernel = np.ones((wood_close_kernel_size, wood_close_kernel_size), np.uint8)
    wood_closed = cv2.morphologyEx(thresh_wood, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    solid_wood_mask = np.zeros_like(thresh_wood)
    contours, _ = cv2.findContours(wood_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid_wood_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

    crust_mask = cv2.bitwise_and(log_mask, cv2.bitwise_not(solid_wood_mask))

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (crust_connect_kernel_size, crust_connect_kernel_size),
    )
    crust_mask = cv2.morphologyEx(crust_mask, cv2.MORPH_CLOSE, connect_kernel, iterations=1)
    final_crust_mask = cv2.bitwise_and(crust_mask, log_mask)
    crust_mask_from_edges = np.zeros_like(final_crust_mask)

    if sobel_kernel_size and sobel_kernel_size > 1:
        inner_log_mask = cv2.subtract(log_mask, crust_mask)
        inner_log_img = cv2.bitwise_and(img, img, mask=inner_log_mask)

        gray_inner = iu.to_gray(inner_log_img)
        sobel_x = cv2.Sobel(gray_inner, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        sobel_y = cv2.Sobel(gray_inner, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

        gradient_norm = cv2.normalize(
            gradient_magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        edge_mask = iu.segmentation_one_threshold(gradient_norm, 150)
        edge_contours_mask, _, _ = iu.find_contours(edge_mask, min_area=0, fill=False)

        if crust_add_kernel_size and crust_add_kernel_size > 1:
            moments = cv2.moments(log_mask)
            if moments["m00"] != 0:
                center = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]),
                )
            else:
                center = None

            crust_mask_from_edges = radial_dilate(edge_contours_mask, crust_add_kernel_size, center)
        else:
            crust_mask_from_edges = edge_contours_mask

    final_crust_mask = cv2.bitwise_or(final_crust_mask, crust_mask_from_edges)
    final_crust_mask = keep_mask_near_log_edge(
        final_crust_mask,
        log_mask,
        crust_max_edge_distance_ratio,
        crust_max_edge_distance_px,
    )
    return final_crust_mask
