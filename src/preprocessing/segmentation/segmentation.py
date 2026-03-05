import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
import improutils as iu

try:
    from .seg_common import fourier_bandpass_filter, normalize_dataset_intensity
    from .seg_hniloba import segment_decay_in_knots
    from .seg_trhlina import segment_cracks
    from .seg_log import extract_log_mask
    from .seg_kura import segment_crust
    from .seg_suk import segment_suk
    from .seg_pozadi import segment_background
except ImportError:
    from seg_common import fourier_bandpass_filter, normalize_dataset_intensity
    from seg_hniloba import segment_decay_in_knots
    from seg_trhlina import segment_cracks
    from seg_log import extract_log_mask
    from seg_kura import segment_crust
    from seg_suk import segment_suk
    from seg_pozadi import segment_background


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

    normalized_gray = normalize_dataset_intensity(
        log_masked,
        gamma=config["gamma_correction"],
    )
    normalized_log = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
    normalized_log = iu.apply_mask(normalized_log, log_mask)
    
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
            normalized_log, log_mask, config["suk_intensity_threshold"],
            min_area=config["suk_min_area"],
            gauss_kernel_size=config["suk_gauss_kernel_size"]
        )
        knot_mask = cv2.bitwise_and(knot_mask, inner_log_mask)
        if 'suk' in requested_masks:
            results['suk'] = knot_mask
    else:
        knot_mask = None
    
    # Prepare optional frequency-domain preprocessing for decay/crack segmentation
    freq_filtered_log = None
    needs_frequency_filter = (
        ('hniloba' in requested_masks or 'trhlina' in requested_masks)
        and inner_log_mask is not None
        and config["fft_max_freq"] > 0
    )
    if needs_frequency_filter:
        filtered_gray = fourier_bandpass_filter(
            normalized_log,
            config["fft_min_freq"],
            config["fft_max_freq"],
        )
        freq_filtered_log = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)
        freq_filtered_log = iu.apply_mask(freq_filtered_log, log_mask)

    # Generate decay (hniloba)
    if 'hniloba' in requested_masks and knot_mask is not None:
        hniloba_input = freq_filtered_log if freq_filtered_log is not None else normalized_log
        decay_mask = segment_decay_in_knots(
            hniloba_input,
            knot_mask,
            min_area=config["hniloba_min_area"],
            lower_threshold=config["hniloba_lower_threshold"],
            upper_threshold=config["hniloba_upper_threshold"],
        )
        results['hniloba'] = decay_mask
    
    # Generate cracks (trhlina)
    if 'trhlina' in requested_masks and inner_log_mask is not None:
        crack_source = freq_filtered_log if freq_filtered_log is not None else normalized_log
        inner_log_img = iu.apply_mask(crack_source, inner_log_mask)
        crack_mask = segment_cracks(
            inner_log_img,
            crust_mask,
            config["crack_threshold"],
            config["crack_edge_exclude_kernel"],
            config["crack_crust_exclude_kernel"],
            config["crack_min_area"],
            config["crack_max_aspect_ratio"],
            config["crack_max_roundness"],
            config["crack_gauss_kernel_size"],
        )
        results['trhlina'] = crack_mask
    
    # Generate background (pozadi)
    if 'pozadi' in requested_masks:
        background_mask = segment_background(
            log_mask,
            crust_mask,
            config["bg_close_kernel_size"],
        )
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
    "gamma_correction": 1.2,
    "fft_min_freq": 25,
    "fft_max_freq": 20,
    "crack_threshold": 109,
    "crack_edge_exclude_kernel": 16,
    "crack_crust_exclude_kernel": 7,
    "crack_min_area": 8,
    "crack_max_aspect_ratio": 0.5,
    "crack_max_roundness": 0.9,
    "crack_gauss_kernel_size": 7,
}


def build_parser(defaults):
    parser = argparse.ArgumentParser()
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
        "--gamma-correction",
        type=float,
        default=defaults["gamma_correction"],
        help="Gamma correction used after histogram equalization for cross-dataset brightness normalization.",
    )
    parser.add_argument(
        "--fft-min-freq",
        type=int,
        default=defaults["fft_min_freq"],
        help="Minimum passed frequency radius for Fourier band-pass (0 keeps all low frequencies).",
    )
    parser.add_argument(
        "--fft-max-freq",
        type=int,
        default=defaults["fft_max_freq"],
        help="Maximum passed frequency radius for Fourier band-pass (<=0 disables filtering).",
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
        "--crack-gauss-kernel-size",
        type=int,
        default=defaults["crack_gauss_kernel_size"],
    )
    return parser


def main():
    defaults = DEFAULT_CONFIG.copy()
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
    config["crack_gauss_kernel_size"] |= 1
    config["fft_min_freq"] = max(0, int(config["fft_min_freq"]))
    config["fft_max_freq"] = max(0, int(config["fft_max_freq"]))
    config["gamma_correction"] = max(0.1, float(config["gamma_correction"]))
    
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
