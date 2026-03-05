import cv2
import improutils as iu


def segment_suk(img, log_mask, intensity_threshold, min_area, gauss_kernel_size=5):
    """Detect bright knots ('suky') within the log body."""
    gray = iu.to_gray(img)
    kernel_size = gauss_kernel_size | 1
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    bright_mask = iu.segmentation_one_threshold(blurred, intensity_threshold)

    suk_inside_log = cv2.bitwise_and(bright_mask, log_mask)
    clean_mask, _, _ = iu.find_contours(
        suk_inside_log,
        min_area=min_area,
        fill=True,
        external=True,
    )
    return clean_mask
