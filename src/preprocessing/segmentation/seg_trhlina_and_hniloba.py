import cv2
import numpy as np
import improutils as iu

try:
    from .seg_common import fourier_bandpass_filter
except ImportError:
    from seg_common import fourier_bandpass_filter  # type: ignore


def _passes_crack_aspect_ratio(contour, max_aspect_ratio):
    ratio = iu.aspect_ratio(contour)
    return ratio <= max_aspect_ratio or ratio >= (1.0 / max(max_aspect_ratio, 1e-6))


def segment_trhlina_and_hniloba(dark_log_img, fft_min_freq=0, fft_max_freq=600):

    filtered_gray = fourier_bandpass_filter(dark_log_img, fft_min_freq, fft_max_freq)
    return filtered_gray, filtered_gray