import argparse
import sys
import unicodedata
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

MASK_NAMES = [
    "pozadi",
    "zdrave_drevo",
    "suk",
    "hniloba",
    "kura",
    "trhlina",
    "poskozeni_hmyzem",
]

MASK_NAME_ALIASES = {
    "pozadi": "pozadi",
    "zdrave drevo": "zdrave_drevo",
    "suk": "suk",
    "hniloba": "hniloba",
    "kura": "kura",
    "trhlina": "trhlina",
    "trhilina": "trhlina",
    "poskozeni hmyzem": "poskozeni_hmyzem",
    "all": "all",
}

OUTER_RING_WIDTH_RATIO = 0.03
OUTER_RING_WIDTH_MIN_PX = 6
OUTER_RING_WIDTH_MAX_PX = 30
CRUST_BAND_WIDTH_MULTIPLIER = 2.2


def _normalize_label_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_only.lower().replace("_", " ").strip()


def _normalize_requested_masks(raw_masks: list[str]) -> set[str]:
    normalized: set[str] = set()
    for raw_name in raw_masks:
        key = _normalize_label_token(raw_name)
        mapped = MASK_NAME_ALIASES.get(key)
        if mapped is None:
            supported = ", ".join(sorted(MASK_NAMES))
            raise ValueError(f"Unsupported mask name '{raw_name}'. Supported mask names: {supported} or all")
        normalized.add(mapped)
    return normalized

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
    
    # No Log to segment
    if background_mask is None or inner_log_mask is None:
        return None

    log_mask = cv2.bitwise_not(background_mask)
    log_img = iu.apply_mask(img, log_mask)

    log_img_clahe = apply_clahe(log_img)
    outer_ring, crust_band = _outer_geometry_from_log(log_mask)

    results = {}

    # Suk 
    need_suk = any(m in requested_masks for m in ("suk", "hniloba", "trhlina", "zdrave_drevo"))
    suk_mask = segment_suk(log_img_clahe) if need_suk else None
    if "suk" in requested_masks and suk_mask is not None:
        results["suk"] = suk_mask

    # Praskliny a hniloba
    trhlina_mask = None
    hniloba_mask = None
    dark_combined = None


    if "trhlina" in requested_masks or "hniloba" in requested_masks or "zdrave_drevo" in requested_masks:
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
    if "kura" in requested_masks or "zdrave_drevo" in requested_masks:
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

    # Explicit empty-air/background mask outside the log.
    if "pozadi" in requested_masks:
        results["pozadi"] = background_mask

    # Placeholder: this classical pipeline does not currently detect insect damage.
    poskozeni_hmyzem_mask = np.zeros_like(log_mask)
    if "poskozeni_hmyzem" in requested_masks:
        results["poskozeni_hmyzem"] = poskozeni_hmyzem_mask

    # Healthy wood is everything inside the log that is not any other class mask.
    if "zdrave_drevo" in requested_masks:
        occupied = np.zeros_like(log_mask)
        for class_mask in (kura_mask, suk_mask, hniloba_mask, trhlina_mask, poskozeni_hmyzem_mask):
            if class_mask is not None:
                occupied = cv2.bitwise_or(occupied, class_mask)

        zdrave_drevo_mask = cv2.bitwise_and(log_mask, cv2.bitwise_not(occupied))
        results["zdrave_drevo"] = zdrave_drevo_mask

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
        default=["all"],
        help="Masks to generate (default: all). Accepts legacy and Czech display names.",
    )
    
    parser.add_argument(
        "--clean-logs-only",
        action="store_true",
        help="Only remove or report invalid logs (no segmentation performed)",
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

    if args.clean_logs_only:
        print(f"Cleaning invalid logs in tree {args.tree} (no segmentation)...")
        invalid_files = []
        with tqdm(
            total=len(files),
            desc="Checking",
            unit="slice",
            dynamic_ncols=True,
            file=sys.stdout,
            disable=False,
            ascii=True,
        ) as progress:
            for f_path in files:
                try:
                    img = iu.load_image(str(f_path))
                    mask_dict = build_masks(img)
                    if mask_dict is None:
                        tqdm.write(f"[CLEAN] No log detected in {f_path}, removing.")
                        f_path.unlink()
                        invalid_files.append(f_path)
                except Exception as exc:
                    tqdm.write(f"[CLEAN] Failed to process {f_path}: {exc}, removing.")
                    f_path.unlink()
                    invalid_files.append(f_path)
                progress.update(1)
        print(f"Removed {len(invalid_files)} invalid log files.")
        return

    normalized_masks = _normalize_requested_masks(args.masks)

    if "all" in normalized_masks:
        requested_masks = set(MASK_NAMES)
    else:
        requested_masks = normalized_masks

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
                if mask_dict is None:
                    tqdm.write(f"[WARN] No log detected in {f_path}, skipping masks.")
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