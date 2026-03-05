import cv2
import numpy as np
import improutils as iu


def normalize_dataset_intensity(image, gamma=1.2):
    """Normalize brightness across datasets using histogram equalization + gamma correction."""
    gray = iu.to_gray(image)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    equalized = cv2.equalizeHist(gray_norm)

    gamma = max(0.1, float(gamma))
    if abs(gamma - 1.0) < 1e-6:
        return equalized

    lut = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(equalized, lut)


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
