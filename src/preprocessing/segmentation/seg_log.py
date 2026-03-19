import cv2
import numpy as np
from .seg_common import to_binary


def get_solid_log_mask(log_mask):
    """Create a solid log mask using only the convex hull of the largest contour."""
    # Normalize to a binary mask before morphology/contour extraction.
    binary_mask = to_binary(log_mask)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(log_mask, dtype=np.uint8)

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)

    solid_mask = np.zeros_like(log_mask, dtype=np.uint8)
    cv2.drawContours(solid_mask, [hull], -1, 255, thickness=cv2.FILLED)
    return solid_mask


def extract_log_mask(img, min_area, close_kernel_size):
    """Find the overall mask of the entire log slice."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if close_kernel_size and close_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = contours
    if min_area and min_area > 0:
        contours = [
            contour for contour in all_contours if cv2.contourArea(contour) >= min_area
        ]
        

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        log_mask = np.zeros_like(bw)
        cv2.drawContours(log_mask, [largest_contour], -1, 255, cv2.FILLED)
        return log_mask

    return None