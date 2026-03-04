import argparse
import configparser
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm
import improutils as iu
# --- Core Segmentation Logic ---

def extract_log_mask(img, min_area, close_kernel_size):
    """Finds the overall mask of the entire log slice."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if close_kernel_size and close_kernel_size > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if min_area and min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        log_mask = np.zeros_like(bw)
        cv2.drawContours(log_mask, [largest_contour], -1, 255, cv2.FILLED)
        return log_mask
    raise ValueError("No contours found matching the area criteria.")


def radial_dilate(mask, kernel_size, center=None):
    """
    Dilates a mask radially outwards (away from center) by a fixed pixel amount.
    """
    if kernel_size <= 1:
        return mask

    h, w = mask.shape[:2]
    
    # Determine Center (default to image center if not provided)
    if center is None:
        center = (w / 2.0, h / 2.0)
    
    # Calculate maximum radius to ensure we cover the corners
    max_radius = np.sqrt((w / 2.0)**2 + (h / 2.0)**2)
    
    # Warp to Polar Coordinates
    polar_img = cv2.warpPolar(
        mask, 
        (w, h), 
        center, 
        max_radius, 
        cv2.WARP_POLAR_LINEAR
    )
    
    # Shape is (1, kernel_size) to affect only the Radius (columns)
    # We use an asymmetric anchor to force growth in only one direction
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    
    # Anchor at (-1, -1) is center. 
    # Anchor at (0, 0) dilates to the right (outwards) because it looks at 'future' pixels.
    # Actually, to "smear" a pixel at r=10 to r=15, we want dst[15] to see src[10].
    # This requires the anchor to be at the far right of the kernel.
    anchor_point = (kernel_size - 1, 0) 
    
    polar_dilated = cv2.dilate(
        polar_img, 
        dilation_kernel, 
        anchor=anchor_point, 
        iterations=1
    )
    
    # Warp back to Cartesian
    result = cv2.warpPolar(
        polar_dilated, 
        (w, h), 
        center, 
        max_radius, 
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    )
    
    # Clean up interpolation artifacts (optional but recommended for binary masks)
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    
    return result


def keep_mask_near_log_edge(
    mask,
    log_mask,
    max_edge_distance_ratio,
    max_edge_distance_px,
):
    """Keep mask pixels only within a limited inward distance from log boundary."""
    log_bin = (log_mask > 0).astype(np.uint8)
    if not np.any(log_bin):
        return np.zeros_like(mask)

    distance_to_edge = cv2.distanceTransform(log_bin, cv2.DIST_L2, 5)
    max_distance_inside_log = float(distance_to_edge.max())
    if max_distance_inside_log <= 0:
        return np.zeros_like(mask)

    ratio = float(np.clip(max_edge_distance_ratio, 0.0, 1.0))
    allowed_distance = ratio * max_distance_inside_log
    if max_edge_distance_px and max_edge_distance_px > 0:
        allowed_distance = min(allowed_distance, float(max_edge_distance_px))

    if allowed_distance <= 0:
        return np.zeros_like(mask)

    near_edge_region = (distance_to_edge <= allowed_distance) & (log_bin == 1)
    filtered = np.zeros_like(mask)
    filtered[(mask > 0) & near_edge_region] = 255
    return filtered


def segment_crust(
    img,
    log_mask,
    alpha,
    beta,
    wood_thresh,
    wood_close_kernel_size,
    crust_connect_kernel_size,
    sobel_kernel_size,
    crust_add_kernel_size,
    crust_max_edge_distance_ratio,
    crust_max_edge_distance_px,
):
    """Isolates outer bark (kura) using a basic threshold-driven pipeline."""
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=log_mask)

    _, thresh_wood = cv2.threshold(gray_masked, wood_thresh, 255, cv2.THRESH_BINARY)

    close_kernel = np.ones((wood_close_kernel_size, wood_close_kernel_size), np.uint8)
    wood_closed = cv2.morphologyEx(
        thresh_wood, cv2.MORPH_CLOSE, close_kernel, iterations=2
    )

    solid_wood_mask = np.zeros_like(thresh_wood)
    contours, _ = cv2.findContours(
        wood_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(
            solid_wood_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED
        )

    crust_mask = cv2.bitwise_and(log_mask, cv2.bitwise_not(solid_wood_mask))

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (crust_connect_kernel_size, crust_connect_kernel_size)
    )
    crust_mask = cv2.morphologyEx(
        crust_mask, cv2.MORPH_CLOSE, connect_kernel, iterations=1
    )
    final_crust_mask = cv2.bitwise_and(crust_mask, log_mask)
    crust_mask_from_edges = np.zeros_like(final_crust_mask)
    
    if sobel_kernel_size and sobel_kernel_size > 1:
        inner_log_mask = cv2.subtract(log_mask, crust_mask)
        inner_log_img = cv2.bitwise_and(img, img, mask=inner_log_mask)

        gray = iu.to_gray(inner_log_img)
        
        # Compute gradients using Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        
        # Extract strong edge mask (log outline)
        gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        edge_mask = iu.segmentation_one_threshold(gradient_norm, 150)
        edge_contours_mask, _, _ = iu.find_contours(edge_mask, min_area=0, fill=False)

        # Add edge-derived crust extension to bark mask
        if crust_add_kernel_size and crust_add_kernel_size > 1:
            M = cv2.moments(log_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
            else:
                center = None # Falls back to image center

            # Use the new radial dilation
            crust_mask_from_edges = radial_dilate(
                edge_contours_mask, 
                crust_add_kernel_size,
                center
            )
        else:
            crust_mask_from_edges = edge_contours_mask
        
    final_crust_mask = cv2.bitwise_or(final_crust_mask, crust_mask_from_edges)

    final_crust_mask = keep_mask_near_log_edge(
        final_crust_mask,
        log_mask,
        crust_max_edge_distance_ratio,
        crust_max_edge_distance_px,
    )


    return final_crust_mask


def segment_suk(img, log_mask, intensity_threshold, min_area, gauss_kernel_size=5):
    """Detects bright knots ('suky') within the log body."""
    gray = iu.to_gray(img)
    kernel_size = gauss_kernel_size | 1  # Ensure odd
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    bright_mask = iu.segmentation_one_threshold(blurred, intensity_threshold)

    suk_inside_log = cv2.bitwise_and(bright_mask, log_mask)
    clean_mask, _, _ = iu.find_contours(
        suk_inside_log, min_area=min_area, fill=True, external=True
    )
    return clean_mask


def segment_decay_in_knots(
    log, knots_mask, min_area, lower_threshold=70, upper_threshold=255
):
    """Detects decay/rot within existing knot regions."""
    log_knots = iu.apply_mask(iu.negative(log), knots_mask)
    log_knots_gray = iu.to_gray(log_knots)
    decay_mask = iu.segmentation_two_thresholds(
        log_knots_gray, lower=lower_threshold, higher=upper_threshold
    )
    clean_mask, _, _ = iu.find_contours(
        decay_mask, min_area=min_area, fill=True, external=True
    )
    return clean_mask


def clean_background_mask(mask, close_kernel_size):
    """Bridges background gaps while preserving the central log hole."""
    if close_kernel_size > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    inv = cv2.bitwise_not(mask)
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    lrg = max(cnts, key=cv2.contourArea)
    clean_inv = np.zeros_like(mask)
    cv2.drawContours(clean_inv, [lrg], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_not(clean_inv)


# --- Contour Descriptors ---


def aspect_ratio(contour):
    """Calculate aspect ratio of bounding rectangle (width/height)."""
    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return 0.0
    return float(w) / h


def roundness(contour):
    """Calculate roundness (4*pi*area/perimeter^2). Circle = 1.0."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def solidity(contour):
    """Calculate solidity (area/convex_hull_area)."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area) / hull_area


def extent(contour):
    """Calculate extent (contour_area/bounding_rect_area)."""
    area = cv2.contourArea(contour)
    _, _, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    if rect_area == 0:
        return 0.0
    return float(area) / rect_area


def segment_cracks(
    inner_log_img,
    crust_mask,
    crack_threshold,
    edge_exclude_kernel,
    crust_exclude_kernel,
    min_crack_area,
    max_aspect_ratio,
    max_roundness,
    sobel_kernel_size,
    gauss_kernel_size,
):
    """Detect cracks in the inner log using gradient analysis."""
    gray = iu.to_gray(inner_log_img)
    
    # Compute gradients using Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Extract strong edge mask (log outline)
    gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edge_mask = iu.segmentation_one_threshold(gradient_norm, 150)
    edge_contours_mask, _, _ = iu.find_contours(edge_mask, min_area=0, fill=True)
    edge_exclusion_mask = cv2.dilate(
        edge_contours_mask, 
        np.ones((edge_exclude_kernel, edge_exclude_kernel), np.uint8), 
        iterations=1
    )

    crust_exclusion_mask = crust_mask
    if crust_exclude_kernel and crust_exclude_kernel > 1:
        crust_exclusion_mask = cv2.dilate(
            crust_exclusion_mask,
            np.ones((crust_exclude_kernel, crust_exclude_kernel), np.uint8),
            iterations=1,
        )

    exclusion_mask = cv2.bitwise_or(edge_exclusion_mask, crust_exclusion_mask)
    
    # Remove strong edges to focus on internal cracks
    gradient_magnitude[exclusion_mask == 255] = 0
    
    # Smooth the gradient
    kernel_size = gauss_kernel_size | 1  # Ensure odd
    gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (kernel_size, kernel_size), 0)
    
    # Normalize and threshold for crack detection
    gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    crack_candidates = iu.segmentation_one_threshold(gradient_norm, crack_threshold)
    
    # Filter contours by geometric descriptors
    _, _, contours = iu.find_contours(crack_candidates, min_area=min_crack_area, fill=False)
    
    crack_mask = np.zeros_like(crack_candidates)
    for cnt in contours:
        ar = aspect_ratio(cnt)
        rnd = roundness(cnt)
        
        # Cracks are elongated (low aspect ratio) and not circular (low roundness)
        if ar < max_aspect_ratio and rnd < max_roundness:
            cv2.drawContours(crack_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    
    return crack_mask


# --- Processing Pipeline ---


def build_masks(img, config, requested_masks=None):
    """Generates all semantic masks for a single frame.
    
    Args:
        img: Input image
        config: Configuration dict
        requested_masks: Set of mask names to generate. If None, generates all.
                        Valid names: 'pozadi', 'kura', 'suk', 'hniloba', 'trhlina'
    
    Returns:
        Dict mapping mask names to their arrays (or None if not requested)
    """
    if requested_masks is None:
        requested_masks = {'pozadi', 'kura', 'suk', 'hniloba', 'trhlina'}
    
    # Always need log_mask as base
    log_mask = extract_log_mask(
        img.copy(), config["min_log_area"], config["log_close_kernel_size"]
    )
    log_masked = iu.apply_mask(img, log_mask)
    
    results = {}
    
    # Generate bark (kura) if needed by itself or as dependency
    need_bark = 'kura' in requested_masks or 'suk' in requested_masks or \
                'hniloba' in requested_masks or 'trhlina' in requested_masks
    
    if need_bark:
        crust_mask = segment_crust(
            log_masked,
            log_mask,
            config["crust_alpha"],
            config["crust_beta"],
            config["crust_wood_thresh"],
            config["crust_wood_close_kernel_size"],
            config["crust_connect_kernel_size"],
            config["crust_sobel_kernel_size"],
            config["crust_add_kernel_size"],
            config["crust_max_edge_distance_ratio"],
            config["crust_max_edge_distance_px"],
        )
        inner_log_mask = cv2.subtract(log_mask, crust_mask)
        if 'kura' in requested_masks:
            results['kura'] = crust_mask
    else:
        crust_mask = None
        inner_log_mask = None
    
    # Generate knot (suk) if needed by itself or as dependency
    need_knot = 'suk' in requested_masks or 'hniloba' in requested_masks
    
    if need_knot and inner_log_mask is not None:
        knot_mask = segment_suk(
            log_masked, log_mask, config["suk_intensity_threshold"], 
            min_area=config["suk_min_area"],
            gauss_kernel_size=config["suk_gauss_kernel_size"]
        )
        knot_mask = cv2.bitwise_and(knot_mask, inner_log_mask)
        if 'suk' in requested_masks:
            results['suk'] = knot_mask
    else:
        knot_mask = None
    
    # Generate decay (hniloba)
    if 'hniloba' in requested_masks and knot_mask is not None:
        decay_mask = segment_decay_in_knots(
            log_masked,
            knot_mask,
            min_area=config["hniloba_min_area"],
            lower_threshold=config["hniloba_lower_threshold"],
            upper_threshold=config["hniloba_upper_threshold"],
        )
        results['hniloba'] = decay_mask
    
    # Generate cracks (trhlina)
    if 'trhlina' in requested_masks and inner_log_mask is not None:
        inner_log_img = iu.apply_mask(log_masked, inner_log_mask)
        crack_mask = segment_cracks(
            inner_log_img,
            crust_mask,
            config["crack_threshold"],
            config["crack_edge_exclude_kernel"],
            config["crack_crust_exclude_kernel"],
            config["crack_min_area"],
            config["crack_max_aspect_ratio"],
            config["crack_max_roundness"],
            config["crack_sobel_kernel_size"],
            config["crack_gauss_kernel_size"],
        )
        results['trhlina'] = crack_mask
    
    # Generate background (pozadi)
    if 'pozadi' in requested_masks:
        background_mask = clean_background_mask(
            cv2.bitwise_not(log_mask), config["bg_close_kernel_size"]
        )
        background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(log_mask))
        background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(crust_mask)) if crust_mask is not None else background_mask
        results['pozadi'] = background_mask
    
    return results


DEFAULT_CONFIG = {
    "min_log_area": 127_000,
    "log_close_kernel_size": 0,
    "crust_alpha": 1.45,
    "crust_beta": -50,
    "crust_wood_thresh": 220,
    "crust_wood_close_kernel_size": 7,
    "crust_connect_kernel_size": 5,
    "crust_sobel_kernel_size": 7,
    "crust_add_kernel_size": 8,
    "crust_max_edge_distance_ratio": 0.22,
    "crust_max_edge_distance_px": 0,
    "bg_close_kernel_size": 21,
    "suk_intensity_threshold": 220,
    "suk_min_area": 250,
    "suk_gauss_kernel_size": 5,
    "hniloba_min_area": 0,
    "hniloba_lower_threshold": 70,
    "hniloba_upper_threshold": 255,
    "crack_threshold": 109,
    "crack_edge_exclude_kernel": 16,
    "crack_crust_exclude_kernel": 7,
    "crack_crust_add_kernel_size": 8,
    "crack_min_area": 8,
    "crack_max_aspect_ratio": 0.5,
    "crack_max_roundness": 0.9,
    "crack_sobel_kernel_size": 7,
    "crack_gauss_kernel_size": 7,
}

LEGACY_CONFIG_KEYS = {
    "crack_edge_dilate_kernel": "crack_edge_exclude_kernel",
}

DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"


def load_config(path, defaults):
    if path is None:
        return defaults.copy()

    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"Config not found: {path}")

    section = parser["segmentation"] if "segmentation" in parser else parser["DEFAULT"]
    config = defaults.copy()
    for key, raw_value in section.items():
        key = LEGACY_CONFIG_KEYS.get(key, key)
        if key not in defaults:
            raise KeyError(f"Unknown config key: {key}")

        if isinstance(defaults[key], int):
            config[key] = int(raw_value)
        elif isinstance(defaults[key], float):
            config[key] = float(raw_value)
        else:
            config[key] = raw_value

    return config


def resolve_config_path(config_arg, input_path):
    if config_arg is None:
        return None

    path = Path(config_arg)
    if path.exists() and path.is_dir():
        if input_path is None:
            raise ValueError("Config directory requires --input to select a file")
        candidate = path / f"{Path(input_path).name}.config"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Config not found: {candidate}")

    if path.exists():
        return path

    if path.suffix:
        candidate = DEFAULT_CONFIG_DIR / path.name
        if candidate.exists():
            return candidate
    else:
        candidate = DEFAULT_CONFIG_DIR / f"{path.name}.config"
        if candidate.exists():
            return candidate
        candidate = DEFAULT_CONFIG_DIR / path.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Config not found: {config_arg}")


def build_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, help="Path to .config file")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--masks", "-m",
        nargs="+",
        choices=["pozadi", "kura", "suk", "hniloba", "trhlina", "all"],
        default=["all"],
        help="Masks to generate (default: all)"
    )
    parser.add_argument("--min-log-area", type=int, default=defaults["min_log_area"])
    parser.add_argument(
        "--log-close-kernel-size",
        type=int,
        default=defaults["log_close_kernel_size"],
    )
    parser.add_argument(
        "--crust-alpha", type=float, default=defaults["crust_alpha"]
    )
    parser.add_argument(
        "--crust-beta", type=int, default=defaults["crust_beta"]
    )
    parser.add_argument(
        "--crust-wood-thresh", type=int, default=defaults["crust_wood_thresh"]
    )
    parser.add_argument(
        "--crust-wood-close-kernel-size",
        type=int,
        default=defaults["crust_wood_close_kernel_size"],
    )
    parser.add_argument(
        "--crust-connect-kernel-size",
        type=int,
        default=defaults["crust_connect_kernel_size"],
    )
    parser.add_argument(
        "--crust-sobel-kernel-size",
        type=int,
        default=defaults["crust_sobel_kernel_size"],
    )
    parser.add_argument(
        "--crust-add-kernel-size",
        type=int,
        default=defaults["crust_add_kernel_size"],
    )
    parser.add_argument(
        "--crust-max-edge-distance-ratio",
        type=float,
        default=defaults["crust_max_edge_distance_ratio"],
        help="Max inward bark distance as a ratio of log radius-like thickness (0-1).",
    )
    parser.add_argument(
        "--crust-max-edge-distance-px",
        type=int,
        default=defaults["crust_max_edge_distance_px"],
        help="Optional absolute cap (px) for crust inward distance; 0 disables cap.",
    )
    parser.add_argument(
        "--bg-close-kernel-size", type=int, default=defaults["bg_close_kernel_size"]
    )
    parser.add_argument(
        "--suk-intensity-threshold",
        type=int,
        default=defaults["suk_intensity_threshold"],
    )
    parser.add_argument(
        "--suk-min-area",
        type=int,
        default=defaults["suk_min_area"],
    )
    parser.add_argument(
        "--suk-gauss-kernel-size",
        type=int,
        default=defaults["suk_gauss_kernel_size"],
    )
    parser.add_argument(
        "--hniloba-min-area",
        type=int,
        default=defaults["hniloba_min_area"],
    )
    parser.add_argument(
        "--hniloba-lower-threshold",
        type=int,
        default=defaults["hniloba_lower_threshold"],
    )
    parser.add_argument(
        "--hniloba-upper-threshold",
        type=int,
        default=defaults["hniloba_upper_threshold"],
    )
    parser.add_argument(
        "--crack-threshold",
        type=int,
        default=defaults["crack_threshold"],
    )
    parser.add_argument(
        "--crack-edge-exclude-kernel",
        type=int,
        default=defaults["crack_edge_exclude_kernel"],
    )
    parser.add_argument(
        "--crack-crust-exclude-kernel",
        type=int,
        default=defaults["crack_crust_exclude_kernel"],
    )
    parser.add_argument(
        "--crack-crust-add-kernel-size",
        type=int,
        default=defaults["crack_crust_add_kernel_size"],
    )
    parser.add_argument(
        "--crack-edge-dilate-kernel",
        type=int,
        dest="crack_edge_exclude_kernel",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--crack-min-area",
        type=int,
        default=defaults["crack_min_area"],
    )
    parser.add_argument(
        "--crack-max-aspect-ratio",
        type=float,
        default=defaults["crack_max_aspect_ratio"],
    )
    parser.add_argument(
        "--crack-max-roundness",
        type=float,
        default=defaults["crack_max_roundness"],
    )
    parser.add_argument(
        "--crack-sobel-kernel-size",
        type=int,
        default=defaults["crack_sobel_kernel_size"],
    )
    parser.add_argument(
        "--crack-gauss-kernel-size",
        type=int,
        default=defaults["crack_gauss_kernel_size"],
    )
    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "-c", type=Path)
    pre_parser.add_argument("--input", "-i", type=Path)
    pre_args, _ = pre_parser.parse_known_args()

    try:
        config_path = resolve_config_path(pre_args.config, pre_args.input)
        defaults = load_config(config_path, DEFAULT_CONFIG)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    parser = build_parser(defaults)
    args = parser.parse_args()

    files = sorted(
        [
            f
            for f in args.input.rglob("*")
            if f.suffix.lower() in [".png", ".jpg", ".tif", ".bmp"]
        ]
    )
    config = vars(args)
    config["bg_close_kernel_size"] |= 1
    config["log_close_kernel_size"] |= 1
    config["suk_gauss_kernel_size"] |= 1
    config["crust_wood_close_kernel_size"] |= 1
    config["crust_connect_kernel_size"] |= 1
    config["crust_add_kernel_size"] = max(1, config["crust_add_kernel_size"])
    config["crust_sobel_kernel_size"] |= 1
    config["crack_edge_exclude_kernel"] |= 1
    config["crack_crust_exclude_kernel"] |= 1
    config["crack_crust_add_kernel_size"] = max(1, config["crack_crust_add_kernel_size"])
    config["crack_sobel_kernel_size"] |= 1
    config["crack_gauss_kernel_size"] |= 1
    
    # Determine which masks to generate
    if "all" in args.masks:
        requested_masks = {'pozadi', 'kura', 'suk', 'hniloba', 'trhlina'}
    else:
        requested_masks = set(args.masks)
    
    print(f"Processing {args.output} masks: {', '.join(sorted(requested_masks))}")

    for f_path in tqdm(files, desc="Processing"):
        try:
            img = iu.load_image(str(f_path))
            mask_dict = build_masks(img, config, requested_masks)
        except Exception:
            continue

        if not mask_dict:
            continue

        rel = Path(str(f_path.relative_to(args.input)).replace("subset", ""))
        
        # Save generated masks
        for mask_name, mask in mask_dict.items():
            mask_path = args.output / "masks" / mask_name / rel.with_suffix(".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)

        img_path = args.output / "images" / rel
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), img)


if __name__ == "__main__":
    main()
