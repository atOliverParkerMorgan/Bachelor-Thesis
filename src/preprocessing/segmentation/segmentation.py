import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm
import improutils as iu
from .seg_common import kmeans_brightness_labels, mask_from_cluster_ids
from .seg_log import extract_log_mask
from .seg_kura import segment_crust
from .seg_suk import segment_suk
from .seg_pozadi import segment_background
from .seg_trhlina_and_hniloba import segment_trhlina_and_hniloba


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


def build_masks(img, requested_masks=None):
    """Generate requested masks with a fixed, minimal segmentation pipeline."""
    if requested_masks is None:
        requested_masks = set(MASK_NAMES)
    
    log_mask = extract_log_mask(img.copy(), min_area=0, close_kernel_size=5)

    # Keep log/background as exact complements by deriving both from seg_pozadi.
    background_mask, log_segment = segment_background(
        log_mask,
        close_kernel_size=BG_CLOSE_KERNEL_SIZE,
        return_log=True,
    )
    log_mask = log_segment
    sorted_labels, _ = kmeans_brightness_labels(img, k=3)

    results = {}

    need_kura_or_dark = (
        "kura" in requested_masks
        or "trhlina" in requested_masks
        or "hniloba" in requested_masks
    )
    if need_kura_or_dark:
        kura_mask = segment_crust(sorted_labels, log_mask)
        # Two darkest clusters inside log are used as crack/decay candidates.
        dark_inside_log = mask_from_cluster_ids(
            sorted_labels,
            cluster_ids={0},
            valid_mask=log_mask,
        )
    else:
        kura_mask = None
        dark_inside_log = None

    if "kura" in requested_masks and kura_mask is not None:
        results["kura"] = kura_mask

    need_suk = "suk" in requested_masks or "hniloba" in requested_masks
    if need_suk:
        suk_mask = segment_suk(img, log_mask, intensity_threshold=220, min_area=250, gauss_kernel_size=5)
        if "suk" in requested_masks:
            results["suk"] = suk_mask
    else:
        suk_mask = None

    if ("trhlina" in requested_masks or "hniloba" in requested_masks) and dark_inside_log is not None:
        # Use only the darkest K-means region as the crack/decay source image.
        dark_log_img = iu.apply_mask(img, dark_inside_log)
        trhlina_mask, hniloba_mask = segment_trhlina_and_hniloba(dark_log_img)
        if "trhlina" in requested_masks:
            results["trhlina"] = trhlina_mask
        if "hniloba" in requested_masks:
            results["hniloba"] = hniloba_mask
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
