import cv2
import numpy as np


def _to_binary(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def get_solid_log_mask(log_mask, close_kernel_size=0):
    """Create a solid log mask using only the convex hull of the largest contour."""
    # Normalize to a binary mask before morphology/contour extraction.
    binary_mask = _to_binary(log_mask)

    if close_kernel_size and close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(log_mask, dtype=np.uint8)

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    solid_mask = np.zeros_like(log_mask, dtype=np.uint8)
    cv2.drawContours(solid_mask, [hull], -1, 255, thickness=cv2.FILLED)
    return solid_mask


def segment_background(log_mask, crust_mask=None, close_kernel_size=0, return_log=False):
    """Generate background and (optionally) log masks as exact complements.

    ``log_segment`` is always computed as the binary inverse of ``background_mask``.
    ``crust_mask`` is accepted for API compatibility but intentionally not applied,
    so the log/background complement relation is preserved.
    """
    solid_log = get_solid_log_mask(log_mask, close_kernel_size)
    log_segment = _to_binary(solid_log)
    background_mask = cv2.bitwise_not(log_segment)

    if return_log:
        return background_mask, log_segment

    return background_mask