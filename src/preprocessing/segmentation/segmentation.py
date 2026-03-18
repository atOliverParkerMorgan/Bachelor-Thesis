import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import improutils as iu

from .seg_kura import segment_crust, refine_kura_outer_crust
from .seg_suk import segment_suk
from .seg_pozadi import segment_background_and_inner_log
from .seg_trhlina_and_hniloba import segment_trhlina, refine_trhlina_mask
from .seg_common import apply_clahe

MASK_NAMES = ["pozadi", "kura", "suk", "hniloba", "trhlina"]

OUTER_RING_WIDTH_RATIO = 0.03
OUTER_RING_WIDTH_MIN_PX = 6
OUTER_RING_WIDTH_MAX_PX = 30
CRUST_BAND_WIDTH_MULTIPLIER = 2.2

def _to_binary_u8(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)

def _log_distance_transform(log_mask):
    return cv2.distanceTransform(_to_binary_u8(log_mask), cv2.DIST_L2, 3)

def _outer_geometry_from_log(log_mask):
    """Build geometry masks used by crack/bark refinement."""
    h, w = log_mask.shape[:2]
    ring_width = int(np.clip(min(h, w) * OUTER_RING_WIDTH_RATIO, OUTER_RING_WIDTH_MIN_PX, OUTER_RING_WIDTH_MAX_PX))
    dist = _log_distance_transform(log_mask)
    outer_ring = np.where((dist > 0) & (dist <= ring_width), 255, 0).astype(np.uint8)
    crust_band = np.where((dist > 0) & (dist <= ring_width * CRUST_BAND_WIDTH_MULTIPLIER), 255, 0).astype(np.uint8)
    return outer_ring, crust_band


def build_masks(img, requested_masks=None):
    """Generate requested segmentation masks for a single log cross-section image."""
    if requested_masks is None:
        requested_masks = set(MASK_NAMES)

    background_mask, inner_log_mask = segment_background_and_inner_log(img)
    log_mask = cv2.bitwise_not(background_mask)
    log_img = iu.apply_mask(img, log_mask)

    log_img_clahe = apply_clahe(log_img)
    outer_ring, crust_band = _outer_geometry_from_log(log_mask)

    results = {}

    # Suk 
    need_suk = any(m in requested_masks for m in ("suk", "hniloba", "trhlina"))
    suk_mask = segment_suk(log_img_clahe) if need_suk else None
    if "suk" in requested_masks and suk_mask is not None:
        results["suk"] = suk_mask

    # Praskliny a hniloba
    trhlina_mask = None
    hniloba_mask = None
    dark_combined = None


    if "trhlina" in requested_masks or "hniloba" in requested_masks:
        raw_dark_mask = segment_trhlina(log_img, background_mask)
        raw_dark_mask = cv2.bitwise_and(raw_dark_mask, log_mask)
        trhlina_mask, hniloba_mask = refine_trhlina_mask(
            raw_dark_mask, log_mask, outer_ring, background_mask, suk_mask=suk_mask
        )
        
        dark_combined = cv2.bitwise_or(trhlina_mask, hniloba_mask)
        if "trhlina" in requested_masks:
            results["trhlina"] = trhlina_mask
        if "hniloba" in requested_masks:
            results["hniloba"] = hniloba_mask

    # kura
    kura_mask = None
    if "kura" in requested_masks or "pozadi" in requested_masks:
        raw_kura_mask = segment_crust(log_img_clahe)
        kura_mask = refine_kura_outer_crust(
            raw_kura_mask,
            log_mask,
            crust_band,
            outer_ring,
            trhlina_and_hniloba_mask=dark_combined,
        )
        if "kura" in requested_masks:
            results["kura"] = kura_mask

    # pozadi
    if "pozadi" in requested_masks:
        foreground = kura_mask if kura_mask is not None else np.zeros_like(log_mask)
        for extra_mask in (trhlina_mask, hniloba_mask):
            if extra_mask is not None:
                foreground = cv2.bitwise_or(foreground, extra_mask)
        results["pozadi"] = cv2.bitwise_and(
            iu.negative(inner_log_mask), cv2.bitwise_not(foreground)
        )

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

    if not files:
        raise FileNotFoundError(f"No image files found in: {input_dir}")

    if "all" in args.masks:
        requested_masks = set(MASK_NAMES)
    else:
        requested_masks = set(args.masks)

    print(f"Processing tree {args.tree} masks: {', '.join(sorted(requested_masks))}")

    is_tty = sys.stdout.isatty()
    with tqdm(
        total=len(files),
        desc="Processing",
        unit="slice",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=False,
        ascii=True,
    ) as progress:
        for index, f_path in enumerate(files, start=1):
            progress.set_postfix_str(f_path.name)
            try:
                img = iu.load_image(str(f_path))
                mask_dict = build_masks(img, requested_masks)

            except Exception as exc:
                tqdm.write(f"[WARN] Failed to process {f_path}: {exc}")
                mask_dict = None

            if mask_dict:
                rel = Path(str(f_path.relative_to(input_dir)).replace("subset", ""))

                for mask_name, mask in mask_dict.items():
                    mask_path = output_dir / "masks" / mask_name / rel.with_suffix(".png")
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), mask)

                img_path = output_dir / "images" / rel
                img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_path), img)

            progress.update(1)
            if not is_tty and (index % 25 == 0 or index == len(files)):
                print(f"Processed {index}/{len(files)} slices", flush=True)


if __name__ == "__main__":
    main()