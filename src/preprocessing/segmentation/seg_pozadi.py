import cv2
import numpy as np
import iu

def get_solid_log_mask(log_mask, close_kernel_size=0):
    """Create a solid log mask using only the convex hull of the largest contour."""
    # Normalize to a binary mask before morphology/contour extraction.
    binary_mask = np.where(log_mask > 0, 255, 0).astype(np.uint8)

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
    print(f"Convex hull area: {cv2.contourArea(hull)}")
    iu.plot_images(solid_mask)
    return solid_mask


def segment_background(log_mask, crust_mask=None, close_kernel_size=0):
    """Generate the background mask from the inverse of the log convex hull."""
    solid_log = get_solid_log_mask(log_mask, close_kernel_size)
    background_mask = cv2.bitwise_not(solid_log)

    if crust_mask is not None:
        background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(crust_mask))

    return background_mask