from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import Compose

from src.custom_model.dataset import WoodDefectDataset
from src.custom_model.transforms import get_train_transforms


def _parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be 3 comma-separated ints")
    try:
        parsed = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("patch size values must be ints") from exc
    if any(v <= 0 for v in parsed):
        raise argparse.ArgumentTypeError("patch size values must be > 0")
    return parsed


def _unwrap_transform_output(output):
    if isinstance(output, list):
        if not output:
            raise ValueError("Transform produced an empty list.")
        return output[0]
    return output


def _clone_patch_dict(data: dict) -> dict:
    cloned = {}
    for key, value in data.items():
        if hasattr(value, "clone"):
            cloned[key] = value.clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _to_numpy_3d(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 4:
        return arr[0]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 3D or 4D tensor/array, got shape {arr.shape}")


def _get_affine(sample_dict: dict) -> np.ndarray:
    image = sample_dict.get("image")
    if image is not None and hasattr(image, "meta"):
        affine = image.meta.get("affine")
        if affine is not None:
            return np.asarray(affine, dtype=np.float64)
    return np.eye(4, dtype=np.float64)


def _save_nifti(array_3d: np.ndarray, affine: np.ndarray, out_path: Path, is_label: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_label:
        arr = np.asarray(array_3d, dtype=np.uint8)
    else:
        arr = np.asarray(array_3d, dtype=np.float32)
    img = nib.Nifti1Image(arr, affine)
    nib.save(img, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export augmented image/label NIfTI patches using custom_model train transforms."
    )
    parser.add_argument("--image-dir", required=True, help="Directory with image .nii.gz volumes")
    parser.add_argument("--label-dir", required=True, help="Directory with label .nii.gz volumes")
    parser.add_argument("--out-dir", default="./output/augmented_dataset", help="Output root folder")
    parser.add_argument("--patch-size", type=_parse_patch_size, default=(128, 384, 128))
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--rare-label-idx", type=int, default=6)
    parser.add_argument("--rare-oversample", type=int, default=8)
    parser.add_argument("--num-aug-per-case", type=int, default=4)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit number of source cases (0 = all cases)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    args = parser.parse_args()

    if args.num_aug_per_case <= 0:
        raise ValueError("--num-aug-per-case must be >= 1")

    if args.max_cases < 0:
        raise ValueError("--max-cases must be >= 0")

    if args.seed is not None:
        np.random.seed(args.seed)

    dataset = WoodDefectDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        rare_label_idx=args.rare_label_idx,
        oversample_factor=0,
    )
    source_cases = dataset.base_samples
    if args.max_cases > 0:
        source_cases = source_cases[: args.max_cases]
    if not source_cases:
        raise ValueError("No source cases found.")

    train_transform = get_train_transforms(
        patch_size=args.patch_size,
        num_samples=1,
        num_classes=args.num_classes,
        rare_label_idx=args.rare_label_idx,
        rare_class_oversample=args.rare_oversample,
    )
    train_ops = list(train_transform.transforms)
    if len(train_ops) < 7:
        raise RuntimeError("Unexpected train transform layout: expected at least 7 operations.")

    # Keep exactly the same augmentation behavior as training while avoiding
    # repeated full-volume preprocessing work and memory spikes.
    pre_transform = Compose(train_ops[:5])
    crop_transform = train_ops[5]
    post_crop_aug = Compose(train_ops[6:])

    out_root = Path(args.out_dir)
    images_out = out_root / "imagesTr"
    labels_out = out_root / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(source_cases)} case(s) to: {out_root}")
    saved_count = 0

    for case_idx, sample in enumerate(source_cases):
        case_id = sample.get("case_id", f"case_{case_idx}")
        print(f"[{case_idx + 1}/{len(source_cases)}] {case_id}")

        preprocessed = _unwrap_transform_output(pre_transform(sample))
        for aug_idx in range(args.num_aug_per_case):
            cropped = _unwrap_transform_output(crop_transform(preprocessed))
            augmented = _unwrap_transform_output(post_crop_aug(_clone_patch_dict(cropped)))

            image_arr = _to_numpy_3d(augmented["image"])
            label_arr = _to_numpy_3d(augmented["label"])
            affine = _get_affine(augmented)

            base_name = f"{case_id}_aug{aug_idx:03d}"
            image_path = images_out / f"{base_name}_0000.nii.gz"
            label_path = labels_out / f"{base_name}.nii.gz"

            _save_nifti(image_arr, affine, image_path, is_label=False)
            _save_nifti(label_arr, affine, label_path, is_label=True)

            saved_count += 1

    print(f"Done. Saved {saved_count} augmented sample pair(s).")
    print(f"Images: {images_out}")
    print(f"Labels: {labels_out}")


if __name__ == "__main__":
    main()
