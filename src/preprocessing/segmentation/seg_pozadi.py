import cv2
import numpy as np


def clean_background_mask(mask, close_kernel_size):
    """Bridge background gaps while preserving the central log hole."""
    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    inverted = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    clean_inverted = np.zeros_like(mask)
    cv2.drawContours(clean_inverted, [largest], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_not(clean_inverted)


def segment_background(log_mask, crust_mask, bg_close_kernel_size):
    """Generate background (pozadi) mask from log and bark masks."""
    background_mask = clean_background_mask(cv2.bitwise_not(log_mask), bg_close_kernel_size)
    background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(log_mask))
    if crust_mask is not None:
        background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(crust_mask))
    return background_mask
