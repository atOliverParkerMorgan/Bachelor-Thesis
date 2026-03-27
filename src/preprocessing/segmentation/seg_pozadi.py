import cv2
import numpy as np
from .seg_log import extract_log_mask, get_solid_log_mask
from .seg_common import to_binary


def segment_background_and_inner_log(img):
    """Generate background and (optionally) log masks as exact complements.

    ``log_segment`` is always computed as the binary inverse of ``background_mask``.
    ``crust_mask`` is accepted for API compatibility but intentionally not applied,
    so the log/background complement relation is preserved.
    """
    inner_log_mask = extract_log_mask(img, min_area=128_000, close_kernel_size=5)
    if inner_log_mask is None:
        return None, None

    solid_log = get_solid_log_mask(inner_log_mask)
    background_mask = cv2.bitwise_not(to_binary(solid_log))
    return background_mask, inner_log_mask