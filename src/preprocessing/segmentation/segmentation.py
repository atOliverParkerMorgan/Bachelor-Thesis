import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm
import improutils as iu
from .seg_kura import segment_crust
from .seg_suk import segment_suk
from .seg_pozadi import segment_background
from .seg_trhlina_and_hniloba import segment_trhlina


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT_HINT = Path(__file__).resolve().parents[3]
for import_path in [str(REPO_ROOT_HINT), str(MODULE_DIR)]:
    if import_path not in sys.path:
        sys.path.insert(0, import_path)




MASK_NAMES = ["pozadi", "kura", "suk", "hniloba", "trhlina"]

CRACK_MIN_AREA = 10
CRACK_MAX_ASPECT_RATIO = 0.5
TRHLINA_HNILOBA_ROUNDNESS_BOUNDARY = 0.15
CRACK_THRESHOLD = 90
HNILOBA_MIN_AREA = 0
RESTRICT_HNILOBA_TO_SUK = True
BG_CLOSE_KERNEL_SIZE = 21
FFT_MIN_FREQ = 15
FFT_MAX_FREQ = 200

OUTER_RING_WIDTH_RATIO = 0.03
OUTER_RING_WIDTH_MIN_PX = 6
OUTER_RING_WIDTH_MAX_PX = 30
CRUST_BAND_WIDTH_MULTIPLIER = 2.2

TRHLINA_MIN_COMPONENT_AREA = 20
TRHLINA_MAX_CIRCULARITY = 0.58
TRHLINA_MIN_ASPECT_RATIO = 2.0
TRHLINA_MAX_EXTENT = 0.60
TRHLINA_MAX_OUTER_RING_OVERLAP = 0.85
TRHLINA_MAX_COMPONENT_AREA_RATIO = 0.20

KURA_MIN_COMPONENT_AREA = 80
KURA_MIN_BOUNDARY_OVERLAP_RATIO = 0.01
KURA_MIN_BOUNDARY_OVERLAP_PIXELS = 40


def _to_binary_u8(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _log_distance_transform(log_mask):
    return cv2.distanceTransform(_to_binary_u8(log_mask), cv2.DIST_L2, 3)


def _outer_geometry_from_log(log_mask):
    h, w = log_mask.shape[:2]
    ring_width = int(np.clip(min(h, w) * OUTER_RING_WIDTH_RATIO, OUTER_RING_WIDTH_MIN_PX, OUTER_RING_WIDTH_MAX_PX))
    dist = _log_distance_transform(log_mask)
    outer_ring = np.where((dist > 0) & (dist <= ring_width), 255, 0).astype(np.uint8)
    crust_band = np.where((dist > 0) & (dist <= ring_width * CRUST_BAND_WIDTH_MULTIPLIER), 255, 0).astype(np.uint8)
    return outer_ring, crust_band


def _filter_trhlina_mask(raw_trhlina_mask, log_mask, outer_ring):
    candidate = cv2.bitwise_and(_to_binary_u8(raw_trhlina_mask), _to_binary_u8(log_mask))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    log_area = max(1, int(cv2.countNonZero(_to_binary_u8(log_mask))))

    filtered = np.zeros_like(candidate)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < TRHLINA_MIN_COMPONENT_AREA:
            continue
        if (area / float(log_area)) > TRHLINA_MAX_COMPONENT_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = float(max(w, h)) / float(max(1, min(w, h)))
        extent = float(area) / float(w * h)

        component = np.zeros_like(candidate)
        cv2.drawContours(component, [contour], -1, 255, thickness=-1)
        comp_pixels = int(cv2.countNonZero(component))
        if comp_pixels == 0:
            continue

        outer_overlap = int(cv2.countNonZero(cv2.bitwise_and(component, outer_ring))) / float(comp_pixels)

        elongated = aspect_ratio >= TRHLINA_MIN_ASPECT_RATIO and circularity <= TRHLINA_MAX_CIRCULARITY
        thin_irregular = circularity <= 0.32 and extent <= TRHLINA_MAX_EXTENT
        not_outer_round_noise = outer_overlap <= TRHLINA_MAX_OUTER_RING_OVERLAP

        if (elongated or thin_irregular) and not_outer_round_noise:
            cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, close_kernel)
    return filtered


def _refine_kura_outer_crust(raw_kura_mask, log_mask, crust_band, outer_ring, trhlina_mask=None):
    kura_candidate = cv2.bitwise_and(_to_binary_u8(raw_kura_mask), _to_binary_u8(log_mask))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(kura_candidate, connectivity=8)

    outer_ring_dilated = cv2.dilate(
        _to_binary_u8(outer_ring),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    kept = np.zeros_like(kura_candidate)
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < KURA_MIN_COMPONENT_AREA:
            continue

        component = np.where(labels == label, 255, 0).astype(np.uint8)
        boundary_overlap = int(cv2.countNonZero(cv2.bitwise_and(component, outer_ring_dilated)))
        overlap_ratio = boundary_overlap / float(area)

        if boundary_overlap >= KURA_MIN_BOUNDARY_OVERLAP_PIXELS or overlap_ratio >= KURA_MIN_BOUNDARY_OVERLAP_RATIO:
            kept = cv2.bitwise_or(kept, component)

    kept = cv2.bitwise_and(kept, _to_binary_u8(crust_band))

    if trhlina_mask is not None:
        # Suppress crack noise only in the inner crust zone so the outer bark contour stays intact.
        outer_guard = cv2.dilate(
            _to_binary_u8(outer_ring),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        inner_crust_zone = cv2.bitwise_and(_to_binary_u8(crust_band), cv2.bitwise_not(outer_guard))
        crack_core = cv2.erode(
            _to_binary_u8(trhlina_mask),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        crack_suppression = cv2.bitwise_and(crack_core, inner_crust_zone)
        kept = cv2.bitwise_and(kept, cv2.bitwise_not(crack_suppression))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, close_kernel)
    kept = cv2.bitwise_and(kept, _to_binary_u8(crust_band))
    return kept


def build_masks(img, requested_masks=None):
    """Generate requested masks with a fixed, minimal segmentation pipeline."""
    if requested_masks is None:
        requested_masks = set(MASK_NAMES)
    
    background_mask = segment_background(img)
    log_mask = cv2.bitwise_not(background_mask)
    
    log_img = iu.apply_mask(img, log_mask)

    results = {}
    
    need_kura_or_dark = ("kura" in requested_masks or "trhlina" in requested_masks or "hniloba" in requested_masks)
    if need_kura_or_dark:
        raw_kura_mask = segment_crust(log_img)
    else:
        raw_kura_mask = None

    need_dark_components = ("trhlina" in requested_masks or "hniloba" in requested_masks)
    if need_dark_components:
        raw_trhlina_mask = segment_trhlina(log_img)
        raw_trhlina_mask = cv2.bitwise_and(raw_trhlina_mask, log_mask)
        outer_ring, crust_band = _outer_geometry_from_log(log_mask)
        trhlina_mask = _filter_trhlina_mask(raw_trhlina_mask, log_mask, outer_ring)
    else:
        trhlina_mask = None
        if raw_kura_mask is not None:
            outer_ring, crust_band = _outer_geometry_from_log(log_mask)
        else:
            outer_ring, crust_band = None, None

    if raw_kura_mask is not None:
        kura_mask = _refine_kura_outer_crust(
            raw_kura_mask,
            log_mask,
            crust_band,
            outer_ring,
            trhlina_mask=trhlina_mask,
        )
    else:
        kura_mask = None

    if "kura" in requested_masks and kura_mask is not None:
        results["kura"] = kura_mask

    need_suk = "suk" in requested_masks or "hniloba" in requested_masks
    if need_suk:
        suk_mask = segment_suk(log_img)
        if "suk" in requested_masks:
            results["suk"] = suk_mask
    else:
        suk_mask = None

    if ("trhlina" in requested_masks or "hniloba" in requested_masks):
        if "trhlina" in requested_masks:
            results["trhlina"] = trhlina_mask
        if "hniloba" in requested_masks:
            results["hniloba"] = trhlina_mask # TODO
    elif "hniloba" in requested_masks:
        results["hniloba"] = np.zeros_like(log_mask, dtype=np.uint8)

    if "pozadi" in requested_masks:
        results["pozadi"] = background_mask

    return results


def find_repo_root(start):
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root (missing pyproject.toml/src)")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", "-t", required=True, help="Tree folder name (for example: dub5)")
    parser.add_argument(
        "--masks", "-m",
        nargs="+",
        choices=[*MASK_NAMES, "all"],
        default=["all"],
        help="Masks to generate (default: all)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    input_dir = repo_root / "src" / "png" / args.tree
    output_dir = repo_root / "src" / "output" / args.tree

    if not input_dir.exists():
        raise FileNotFoundError(f"Input tree folder does not exist: {input_dir}")

    files = sorted(
        [
            f
            for f in input_dir.rglob("*")
            if f.suffix.lower() in [".png", ".jpg", ".tif", ".bmp"]
        ]
    )

    # Determine which masks to generate
    if "all" in args.masks:
        requested_masks = set(MASK_NAMES)
    else:
        requested_masks = set(args.masks)

    print(f"Processing tree {args.tree} masks: {', '.join(sorted(requested_masks))}")

    for f_path in tqdm(files, desc="Processing"):
        try:
            img = iu.load_image(str(f_path))
            mask_dict = build_masks(img, requested_masks)
        except Exception as exc:
            tqdm.write(f"[WARN] Failed to process {f_path}: {exc}")
            continue

        if not mask_dict:
            continue

        rel = Path(str(f_path.relative_to(input_dir)).replace("subset", ""))

        # Save generated masks
        for mask_name, mask in mask_dict.items():
            mask_path = output_dir / "masks" / mask_name / rel.with_suffix(".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)

        img_path = output_dir / "images" / rel
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), img)


if __name__ == "__main__":
    main()
