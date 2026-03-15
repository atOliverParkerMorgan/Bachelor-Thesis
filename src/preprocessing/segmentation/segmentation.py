import argparse
import json
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


def compute_dataset_normalization(files: list[Path], max_sample_slices: int = 160) -> tuple[float, float]:
    """Estimate global Mean and Standard Deviation from evenly sampled slices across the dataset."""
    if not files:
        return 128.0, 64.0 # Fallback safe values

    sample_count = min(max_sample_slices, len(files))
    sample_indices = np.linspace(0, len(files) - 1, num=sample_count, dtype=int)

    samples = []
    for idx in sample_indices:
        f_path = files[int(idx)]
        try:
            img = iu.load_image(str(f_path))
            gray = iu.to_gray(img) 
            gray = apply_clahe(gray)
            samples.append(gray[::4, ::4].ravel()) 
        except Exception:
            pass
            
    if not samples:
        return 128.0, 64.0
        
    all_pixels = np.concatenate(samples)
    
    # Calculate global mean and standard deviation
    global_mean = float(np.mean(all_pixels))
    global_std = float(np.std(all_pixels))
    
    return global_mean, global_std


def build_normalization_lut(global_mean: float, global_std: float, target_mean: float = 128.0, target_std: float = 64.0) -> np.ndarray:
    """Build a fast Lookup Table to shift dataset variance to our target standard."""
    if global_std < 1e-3:
        global_std = 1.0 # Avoid division by zero
        
    # Create an array of all possible pixel values [0-255]
    indices = np.arange(256, dtype=np.float32)
    
    # Apply standard distribution matching formula: Z-score -> scale -> shift
    normalized = (indices - global_mean) * (target_std / global_std) + target_mean
    
    # Clip to valid 8-bit range and convert
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_dataset_normalization(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply precomputed intensity normalization LUT to grayscale image."""
    return cv2.LUT(img, lut)


def normalization_cache_key(files: list[Path]) -> dict[str, object]:
    return {
        "count": len(files),
        "first": files[0].name if files else "",
        "last": files[-1].name if files else "",
        "max_mtime": max((path.stat().st_mtime for path in files), default=0.0),
    }


def load_cached_normalization(cache_path: Path, key: dict[str, object]) -> tuple[float, float] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if payload.get("key") != key:
            return None
        global_mean = float(payload.get("mean", 128.0))
        global_std = float(payload.get("std", 64.0))
        return global_mean, global_std
    except Exception:
        return None


def save_cached_normalization(cache_path: Path, key: dict[str, object], global_mean: float, global_std: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": key, "mean": global_mean, "std": global_std}
    with cache_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle)

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


def build_masks(img: np.ndarray, requested_masks: set[str] | None = None) -> dict[str, np.ndarray]:
    """Generate requested segmentation masks for a single log cross-section image."""
    if requested_masks is None:
        requested_masks = set(MASK_NAMES)

    # --- Step 1: Log / background geometry (always needed) ---
    background_mask, inner_log_mask = segment_background_and_inner_log(img)
    log_mask = cv2.bitwise_not(background_mask)
    log_img = iu.apply_mask(img, log_mask)
    outer_ring, crust_band = _outer_geometry_from_log(log_mask)

    results: dict[str, np.ndarray] = {}

    # --- Step 2: Suk (knots) — computed early; needed for hniloba/trhlina split ---
    need_suk = any(m in requested_masks for m in ("suk", "hniloba", "trhlina"))
    suk_mask = segment_suk(log_img) if need_suk else None
    if "suk" in requested_masks and suk_mask is not None:
        results["suk"] = suk_mask

    # --- Step 3: Dark features (cracks + rot), split by suk proximity ---
    trhlina_mask: np.ndarray | None = None
    hniloba_mask: np.ndarray | None = None
    dark_combined: np.ndarray | None = None
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

    # --- Step 4: Bark (kura) ---
    kura_mask: np.ndarray | None = None
    if "kura" in requested_masks or "pozadi" in requested_masks:
        raw_kura_mask = segment_crust(log_img)
        kura_mask = refine_kura_outer_crust(
            raw_kura_mask,
            log_mask,
            crust_band,
            outer_ring,
            trhlina_and_hniloba_mask=dark_combined,
        )
        if "kura" in requested_masks:
            results["kura"] = kura_mask

    # --- Step 5: Background (pozadi) ---
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
    parser.add_argument(
        "--normalization-sample-slices",
        type=int,
        default=160,
        help="How many slices to sample when estimating global normalization (default: 160)",
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

    if "all" in args.masks:
        requested_masks = set(MASK_NAMES)
    else:
        requested_masks = set(args.masks)

    print(f"Processing tree {args.tree} masks: {', '.join(sorted(requested_masks))}")

    cache_path = output_dir / ".normalization_cache.json"
    cache_key = normalization_cache_key(files)
    cached = load_cached_normalization(cache_path, cache_key)

    if cached is not None:
        global_mean, global_std = cached
        print(f"Using cached dataset normalization for {len(files)} slices")
    else:
        sample_slices = max(1, int(args.normalization_sample_slices))
        print(
            f"Computing dataset normalization from {min(sample_slices, len(files))}/"
            f"{len(files)} sampled slices..."
        )
        global_mean, global_std = compute_dataset_normalization(files, max_sample_slices=sample_slices)
        save_cached_normalization(cache_path, cache_key, global_mean, global_std)

    print(f"  Dataset Original Global Mean: {global_mean:.1f}, Global StdDev: {global_std:.1f}")
    
    # We target a middle-gray mean (128) and a moderate spread (64)
    normalization_lut = build_normalization_lut(global_mean, global_std, target_mean=128.0, target_std=64.0)

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
                # Ensure grayscale before processing
                img = iu.to_gray(img)

                # Local contrast normalization (CLAHE)
                img = apply_clahe(img, clip_limit=2.0)

                # Global intensity normalization to match dataset statistics
                img = apply_dataset_normalization(img, normalization_lut)

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