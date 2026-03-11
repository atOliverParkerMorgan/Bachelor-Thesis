import cv2
import numpy as np
from .seg_log import extract_log_mask, get_solid_log_mask
from .seg_common import to_binary


def segment_background(img):
    """Generate background and (optionally) log masks as exact complements.

    ``log_segment`` is always computed as the binary inverse of ``background_mask``.
    ``crust_mask`` is accepted for API compatibility but intentionally not applied,
    so the log/background complement relation is preserved.
    """
    log_mask = extract_log_mask(img, min_area=0, close_kernel_size=5)

    solid_log = get_solid_log_mask(log_mask)
    log_mask = to_binary(solid_log)
    background_mask = cv2.bitwise_not(log_mask)
    return background_mask