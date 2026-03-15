import cv2
import numpy as np
import improutils as iu


def to_binary(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def apply_clahe(gray_img, clip_limit=2.0, tile_grid=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_img)

def kmeans_brightness_labels(image, k=4, attempts=3, seed=42):
    """Run deterministic K-means and return labels sorted by cluster brightness.

    Label meaning for ``k=4`` after sorting:
    0 darkest, 1 second darkest, 2 second brightest, 3 brightest.
    """
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    cv2.setRNGSeed(int(seed))
    _, labels, centers = cv2.kmeans(
        pixels,
        int(k),
        None,
        criteria,
        int(attempts),
        cv2.KMEANS_PP_CENTERS,
    )

    brightness = np.sum(centers, axis=1)
    sort_indices = np.argsort(brightness)
    sorted_centers = centers[sort_indices]

    rank_map = np.zeros(int(k), dtype=np.int32)
    for new_id, old_id in enumerate(sort_indices):
        rank_map[old_id] = new_id

    sorted_labels = rank_map[labels.flatten()].reshape(image.shape[:2])
    return sorted_labels, sorted_centers


def mask_from_cluster_ids(sorted_labels, cluster_ids, valid_mask=None):
    """Build a binary mask from selected sorted K-means cluster ids."""
    if np.isscalar(cluster_ids):
        cluster_ids = [int(cluster_ids)]
    else:
        cluster_ids = list(cluster_ids)
    mask = np.isin(sorted_labels, cluster_ids).astype(np.uint8) * 255

    if valid_mask is not None:
        valid_mask_bin = np.where(valid_mask > 0, 255, 0).astype(np.uint8)
        mask = cv2.bitwise_and(mask, valid_mask_bin)

    return mask


def fourier_bandpass_filter(image, min_freq, max_freq):
    """Apply circular band-pass filtering in frequency domain and return grayscale output."""
    gray = iu.to_gray(image)

    if max_freq <= 0:
        return gray

    min_freq = max(0.0, float(min_freq))
    max_freq = float(max_freq)
    if max_freq <= min_freq:
        return gray

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    max_supported_freq = float(np.hypot(rows / 2.0, cols / 2.0))
    max_freq = min(max_freq, max_supported_freq)

    y_coords, x_coords = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x_coords - ccol) ** 2 + (y_coords - crow) ** 2)

    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[(dist_from_center >= min_freq) & (dist_from_center <= max_freq)] = 1.0

    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)

    filtered = np.fft.ifft2(f_ishift)
    filtered = np.abs(filtered)
    return cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def segment_using_superpixels_and_kmeans(
    img,
    k=4,
    attempts=1,
    seed=42,
    kmeans_labels=None,
    region_size=20,
    ruler=50.0,
):
    if kmeans_labels is None:
        kmeans_labels = [0]

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    slic = cv2.ximgproc.createSuperpixelSLIC(
        blurred,
        algorithm=cv2.ximgproc.SLIC,
        region_size=int(region_size),
        ruler=float(ruler),
    )
    slic.iterate(10)

    labels = slic.getLabels()
    n_sp = slic.getNumberOfSuperpixels()

    # Vectorized per-superpixel mean (avoids a slow Python loop over each superpixel).
    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels, minlength=n_sp).astype(np.float64)
    avg_img = np.zeros_like(img)
    for c in range(3):
        channel_sums = np.bincount(flat_labels, weights=img[:, :, c].ravel().astype(np.float64), minlength=n_sp)
        avg_img[:, :, c] = (channel_sums / np.maximum(counts, 1)).astype(np.uint8)[labels]

    k_labels, _ = kmeans_brightness_labels(avg_img, k=k, attempts=attempts, seed=seed)

    bark_mask = mask_from_cluster_ids(k_labels, kmeans_labels)

    return bark_mask

def has_outliers(img, intensity_thresh=(222, 250), min_area=64, plot=False):
    gray_img = iu.to_gray(img)
    seg_mask = iu.segmentation_two_thresholds(gray_img, intensity_thresh[0], intensity_thresh[1])
    contours_drawn, count, _ = iu.find_contours(seg_mask, min_area=min_area)
    if plot:
        iu.plot_images(seg_mask, contours_drawn, titles=["Segmentation Mask", "Contours"])
    return count > 0