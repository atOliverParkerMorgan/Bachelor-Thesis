from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path

import numpy as np
from monai.transforms import Compose
from PIL import Image, ImageDraw

from src.custom_model.dataset import WoodDefectDataset
from src.custom_model.transforms import get_train_transforms


_LABEL_COLORS = np.array(
    [
        [0, 0, 0],
        [30, 144, 255],
        [255, 165, 0],
        [220, 20, 60],
        [0, 200, 83],
        [156, 39, 176],
        [255, 235, 59],
    ],
    dtype=np.uint8,
)


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


def _to_numpy_3d(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        return value[0]
    if value.ndim == 3:
        return value
    raise ValueError(f"Expected 3D or 4D tensor/array, got shape {value.shape}")


def _slice_along_axis(volume: np.ndarray, axis: int) -> np.ndarray:
    center = volume.shape[axis] // 2
    if axis == 0:
        slc = volume[center, :, :]
    elif axis == 1:
        slc = volume[:, center, :]
    else:
        slc = volume[:, :, center]
    return np.ascontiguousarray(slc)


def _scale_to_uint8(image_2d: np.ndarray) -> np.ndarray:
    arr = image_2d.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - arr_min) / (arr_max - arr_min)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _label_to_rgb(label_2d: np.ndarray, num_classes: int) -> np.ndarray:
    label = label_2d.astype(np.int64)
    palette = _LABEL_COLORS
    if num_classes > len(_LABEL_COLORS):
        rng = np.random.default_rng(7)
        extra = rng.integers(0, 256, size=(num_classes - len(_LABEL_COLORS), 3), dtype=np.uint8)
        palette = np.vstack([_LABEL_COLORS, extra])
    label = np.clip(label, 0, len(palette) - 1)
    return palette[label]


def _resize_nn(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image)
    resized = pil.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
    return np.asarray(resized)


def _build_axis_row(
    before_img: np.ndarray,
    before_lbl: np.ndarray,
    after_img: np.ndarray,
    after_lbl: np.ndarray,
    axis: int,
    num_classes: int,
) -> np.ndarray:
    bi = _slice_along_axis(before_img, axis)
    bl = _slice_along_axis(before_lbl, axis)
    ai = _slice_along_axis(after_img, axis)
    al = _slice_along_axis(after_lbl, axis)

    target_h = max(bi.shape[0], ai.shape[0])
    target_w = max(bi.shape[1], ai.shape[1])

    bi_u8 = _resize_nn(_scale_to_uint8(bi), (target_h, target_w))
    ai_u8 = _resize_nn(_scale_to_uint8(ai), (target_h, target_w))
    bl_rgb = _resize_nn(_label_to_rgb(bl, num_classes), (target_h, target_w))
    al_rgb = _resize_nn(_label_to_rgb(al, num_classes), (target_h, target_w))

    bi_rgb = np.repeat(bi_u8[..., None], 3, axis=2)
    ai_rgb = np.repeat(ai_u8[..., None], 3, axis=2)

    spacer = np.full((target_h, 8, 3), 24, dtype=np.uint8)
    return np.concatenate([bi_rgb, spacer, bl_rgb, spacer, ai_rgb, spacer, al_rgb], axis=1)


def _add_header(image: np.ndarray, text: str) -> np.ndarray:
    header_h = 30
    canvas = np.full((image.shape[0] + header_h, image.shape[1], 3), 10, dtype=np.uint8)
    canvas[header_h:, :, :] = image
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((8, 8), text, fill=(255, 255, 255))
    return np.asarray(pil)


def _pad_row_width(row: np.ndarray, target_width: int, fill: int = 24) -> np.ndarray:
    if row.shape[1] == target_width:
        return row
    if row.shape[1] > target_width:
        return row[:, :target_width, :]
    pad = np.full((row.shape[0], target_width - row.shape[1], 3), fill, dtype=np.uint8)
    return np.concatenate([row, pad], axis=1)


def _unwrap_transform_output(output):
    if isinstance(output, list):
        if not output:
            raise ValueError("Transform produced an empty list.")
        return output[0]
    return output


def _clone_patch_dict(data: dict) -> dict:
    """Clone patch tensors so random augmentations don't mutate the preview source."""
    cloned = {}
    for key, value in data.items():
        if hasattr(value, "clone"):
            cloned[key] = value.clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save before/after visualization for custom_model train augmentations."
    )
    parser.add_argument("--image-dir", required=True, help="Directory with image .nii.gz volumes")
    parser.add_argument("--label-dir", required=True, help="Directory with label .nii.gz volumes")
    parser.add_argument("--out-dir", default="./output/augmentation_preview", help="Output folder")
    parser.add_argument("--case-index", type=int, default=0, help="Dataset index to preview")
    parser.add_argument("--num-aug", type=int, default=3, help="How many random augmentations")
    parser.add_argument("--patch-size", type=_parse_patch_size, default=(128, 384, 128))
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--rare-label-idx", type=int, default=6)
    parser.add_argument("--rare-oversample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    args = parser.parse_args()

    if args.num_aug <= 0:
        raise ValueError("--num-aug must be >= 1")

    if args.seed is not None:
        np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = WoodDefectDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        rare_label_idx=args.rare_label_idx,
        oversample_factor=0,
    )
    if not dataset.base_samples:
        raise ValueError("Dataset is empty.")

    if args.case_index < 0 or args.case_index >= len(dataset.base_samples):
        raise IndexError(f"--case-index must be in [0, {len(dataset.base_samples) - 1}]")

    sample = dataset.base_samples[args.case_index]
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

    # Run full-volume deterministic preprocessing only once to avoid repeated
    # memory spikes from loading/orientation/intensity scaling.
    pre_transform = Compose(train_ops[:5])
    crop_transform = train_ops[5]
    post_crop_aug = Compose(train_ops[6:])

    preprocessed = _unwrap_transform_output(pre_transform(sample))

    case_id = sample.get("case_id", f"case_{args.case_index}")
    print(f"Previewing case: {case_id}")
    print(f"Saving outputs to: {out_dir}")

    for i in range(args.num_aug):
        cropped = _unwrap_transform_output(crop_transform(preprocessed))
        before_img = _to_numpy_3d(cropped["image"])
        before_lbl = _to_numpy_3d(cropped["label"])

        after = _unwrap_transform_output(post_crop_aug(_clone_patch_dict(cropped)))
        after_img = _to_numpy_3d(after["image"])
        after_lbl = _to_numpy_3d(after["label"])

        rows = []
        axis_names = ["sagittal", "coronal", "axial"]
        for axis, axis_name in enumerate(axis_names):
            row = _build_axis_row(
                before_img=before_img,
                before_lbl=before_lbl,
                after_img=after_img,
                after_lbl=after_lbl,
                axis=axis,
                num_classes=args.num_classes,
            )
            rows.append(_add_header(row, f"{axis_name} | before image/label | after image/label"))

        target_width = max(r.shape[1] for r in rows)
        rows = [_pad_row_width(r, target_width) for r in rows]

        spacer_h = np.full((10, target_width, 3), 24, dtype=np.uint8)
        canvas = rows[0]
        for row in rows[1:]:
            canvas = np.concatenate([canvas, spacer_h, row], axis=0)

        out_path = out_dir / f"{case_id}_aug_{i:02d}.png"
        Image.fromarray(canvas).save(out_path)
        print(f"Saved: {out_path}")

        # Promptly free per-iteration arrays before the next random sample.
        del cropped, after, before_img, before_lbl, after_img, after_lbl, rows, canvas
        gc.collect()


if __name__ == "__main__":
    main()
