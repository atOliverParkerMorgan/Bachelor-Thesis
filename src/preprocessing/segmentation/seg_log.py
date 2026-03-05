import cv2
import numpy as np


def extract_log_mask(img, min_area, close_kernel_size):
    """Find the overall mask of the entire log slice."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    if not contours and all_contours:
        contours = all_contours

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        log_mask = np.zeros_like(bw)
        cv2.drawContours(log_mask, [largest_contour], -1, 255, cv2.FILLED)
        return log_mask

    raise ValueError("No contours found in thresholded log mask.")
