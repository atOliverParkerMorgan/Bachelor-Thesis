#!/usr/bin/env python3
"""Prepare Dataset001 into nnU-Net v2 raw dataset format."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image

BACKGROUND_ALIASES = {"background", "pozadi"}


@dataclass(frozen=True)
class LabelEntry:
    original_name: str
    safe_name: str
    color: Tuple[int, int, int]


def make_safe_label_name(name: str) -> str:
    """Normalize labels to ASCII-friendly names for dataset.json keys."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    safe = "_".join(ascii_name.strip().lower().split())
    safe = safe.replace("-", "_")
    return safe or "class"


def parse_labelmap(labelmap_path: Path) -> List[LabelEntry]:
    labels: List[LabelEntry] = []
    with labelmap_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            rgb_text = parts[1].strip()
            rgb_values = tuple(int(value.strip()) for value in rgb_text.split(","))
            if len(rgb_values) != 3:
                raise ValueError(f"Invalid RGB mapping in {labelmap_path}: {line}")
            labels.append(LabelEntry(name, make_safe_label_name(name), rgb_values))
    if not labels:
        raise ValueError(f"No labels found in {labelmap_path}")
    return labels


def parse_slice_list(series_dir: Path, series_name: str) -> List[str]:
    list_path = series_dir / "ImageSets" / "Segmentation" / f"{series_name}.txt"
    if not list_path.exists():
        return sorted(path.stem for path in (series_dir / "SegmentationObject").glob("*.png"))

    items: List[str] = []
    with list_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            item = raw_line.strip()
            if item:
                items.append(item)
    if not items:
        raise ValueError(f"Empty slice list: {list_path}")
    return items


def load_spacing(geometry_root: Path, series_name: str) -> Tuple[float, float, float]:
    geom_path = geometry_root / series_name / "geometry.json"
    if not geom_path.exists():
        return (1.0, 1.0, 1.0)

    with geom_path.open("r", encoding="utf-8") as file_handle:
        geometry = json.load(file_handle)
    spacing = geometry.get("spacing")
    if not isinstance(spacing, list) or len(spacing) != 3:
        return (1.0, 1.0, 1.0)
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def as_gray_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.uint8)


def as_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def convert_mask_rgb_to_ids(
    mask_rgb: np.ndarray,
    color_to_id: Dict[Tuple[int, int, int], int],
    source_path: Path,
) -> np.ndarray:
    mask_ids = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    flat_rgb = mask_rgb.reshape(-1, 3)
    unique_colors = np.unique(flat_rgb, axis=0)

    for color_arr in unique_colors:
        color = tuple(int(value) for value in color_arr.tolist())
        if color not in color_to_id:
            raise ValueError(f"Unknown label color {color} in {source_path}")
        class_id = color_to_id[color]
        mask_ids[np.all(mask_rgb == color_arr, axis=2)] = class_id

    return mask_ids


def write_nifti(volume_zyx: np.ndarray, out_path: Path, spacing_xyz: Tuple[float, float, float]) -> None:
    image = sitk.GetImageFromArray(volume_zyx)
    image.SetSpacing(spacing_xyz)
    sitk.WriteImage(image, str(out_path), useCompression=True)


def list_series_dirs(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("dub")
        and (path / "labelmap.txt").exists()
        and (path / "SegmentationObject").exists()
        and (path / "SegmentationClass").exists()
    )


def list_candidate_roots(source_root: Path) -> List[Path]:
    project_root = Path(__file__).resolve().parents[3]
    raw_candidates = [
        project_root / "src" / "nn_UNet" / "datasets"
    ]

    seen: set[Path] = set()
    candidates: List[Path] = []
    for candidate in raw_candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)

    return candidates


def discover_series_dirs_from_roots(source_root: Path) -> tuple[Path, List[Path], tempfile.TemporaryDirectory[str] | None]:
    candidates = list_candidate_roots(source_root)

    # 1) Try already-extracted folder layouts.
    for candidate in candidates:
        if not candidate.exists():
            continue
        series_dirs = list_series_dirs(candidate)
        if series_dirs:
            return candidate, series_dirs, None

    # 2) Try zip-based layout (for example src/nn_UNet/datasets/dub1_seg.zip).
    zip_roots = [candidate for candidate in candidates if candidate.exists()]
    for zip_root in zip_roots:
        zip_files = sorted(path for path in zip_root.glob("*.zip") if path.name.startswith("dub"))
        if not zip_files:
            continue

        temp_dir = tempfile.TemporaryDirectory(prefix="nnunet_prepare_")
        temp_root = Path(temp_dir.name)

        for zip_path in zip_files:
            target_dir = temp_root / zip_path.stem
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(target_dir)

        series_dirs = sorted(
            path
            for path in temp_root.rglob("*")
            if path.is_dir()
            and path.name.startswith("dub")
            and (path / "labelmap.txt").exists()
            and (path / "SegmentationObject").exists()
            and (path / "SegmentationClass").exists()
        )
        if series_dirs:
            return temp_root, series_dirs, temp_dir

        temp_dir.cleanup()

    checked = "\n - ".join(str(path) for path in candidates)
    raise ValueError(
        "No usable dub* series found. Checked roots:\n"
        f" - {checked}\n"
        "Expected extracted folders with labelmap.txt/SegmentationObject/SegmentationClass, "
        "or zip files named dub*.zip."
    )


def generate_subvolumes(
    image_volume: np.ndarray,
    label_volume: np.ndarray,
    patch_size: Tuple[int, int, int] = (96, 256, 96),
    stride: Tuple[int, int, int] = (64, 128, 64),
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Split a 3D volume into overlapping fixed-size sub-volumes."""
    if image_volume.shape != label_volume.shape:
        raise ValueError(
            "Image/label shape mismatch before patching: "
            f"{image_volume.shape} vs {label_volume.shape}"
        )

    z_max, y_max, x_max = image_volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    if pz <= 0 or py <= 0 or px <= 0:
        raise ValueError(f"Invalid patch_size: {patch_size}")
    if sz <= 0 or sy <= 0 or sx <= 0:
        raise ValueError(f"Invalid stride: {stride}")

    def axis_starts(axis_size: int, patch_len: int, step: int) -> List[int]:
        if axis_size <= patch_len:
            return [0]

        starts = list(range(0, axis_size - patch_len + 1, step))
        last_start = axis_size - patch_len
        if starts[-1] != last_start:
            # Ensure the tail of each axis is included so edge classes are not dropped.
            starts.append(last_start)
        return starts

    z_starts = axis_starts(z_max, pz, sz)
    y_starts = axis_starts(y_max, py, sy)
    x_starts = axis_starts(x_max, px, sx)

    for z_idx in z_starts:
        for y_idx in y_starts:
            for x_idx in x_starts:
                img_patch = image_volume[z_idx : z_idx + pz, y_idx : y_idx + py, x_idx : x_idx + px]
                lbl_patch = label_volume[z_idx : z_idx + pz, y_idx : y_idx + py, x_idx : x_idx + px]
                if img_patch.shape != patch_size or lbl_patch.shape != patch_size:
                    continue
                yield img_patch, lbl_patch


def prepare_dataset(
    source_root: Path,
    nnunet_root: Path,
    geometry_root: Path,
    dataset_id: int,
    dataset_name: str,
    overwrite: bool,
    patch_size: Tuple[int, int, int] = (96, 256, 96),
    stride: Tuple[int, int, int] = (64, 128, 64),
    skip_empty_patches: bool = True,
) -> Path:
    source_root = source_root.resolve()
    discovered_root, series_dirs, temp_dir = discover_series_dirs_from_roots(source_root)

    dataset_dirname = f"Dataset{dataset_id:03d}_{dataset_name}"
    raw_root = nnunet_root / "nnUNet_raw"
    dataset_root = raw_root / dataset_dirname
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"

    if dataset_root.exists() and overwrite:
        shutil.rmtree(dataset_root)

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for old_file in images_tr.glob("*.nii.gz"):
        old_file.unlink()
    for old_file in labels_tr.glob("*.nii.gz"):
        old_file.unlink()

    source_root = discovered_root

    try:
        ref_labels = parse_labelmap(series_dirs[0] / "labelmap.txt")
        color_to_id: Dict[Tuple[int, int, int], int] = {}
        labels_dict: Dict[str, int] = {"background": 0}
        class_voxel_counts: Dict[int, int] = {}
        training_cases: List[str] = []

        next_id = 1
        for entry in ref_labels:
            if entry.safe_name in BACKGROUND_ALIASES:
                color_to_id[entry.color] = 0
                continue
            key = entry.safe_name
            while key in labels_dict:
                key = f"{entry.safe_name}_{next_id}"
            labels_dict[key] = next_id
            color_to_id[entry.color] = next_id
            class_voxel_counts[next_id] = 0
            next_id += 1

        for series_dir in series_dirs:
            series_name = series_dir.name
            current_labels = parse_labelmap(series_dir / "labelmap.txt")
            if [(entry.original_name, entry.color) for entry in current_labels] != [
                (entry.original_name, entry.color) for entry in ref_labels
            ]:
                raise ValueError(f"Label map mismatch in {series_dir / 'labelmap.txt'}")

            slice_ids = parse_slice_list(series_dir, series_name)
            spacing_xyz = load_spacing(geometry_root, series_name)

            image_slices: List[np.ndarray] = []
            label_slices: List[np.ndarray] = []

            for slice_id in slice_ids:
                image_path = series_dir / "SegmentationObject" / f"{slice_id}.png"
                label_path = series_dir / "SegmentationClass" / f"{slice_id}.png"
                if not image_path.exists() or not label_path.exists():
                    raise FileNotFoundError(f"Missing pair for {series_name}/{slice_id}")

                image_gray = as_gray_array(image_path)
                label_rgb = as_rgb_array(label_path)
                label_ids = convert_mask_rgb_to_ids(label_rgb, color_to_id, label_path)

                image_slices.append(image_gray)
                label_slices.append(label_ids)

            image_volume = np.stack(image_slices, axis=0).astype(np.uint8)
            label_volume = np.stack(label_slices, axis=0).astype(np.uint8)

            class_ids, class_counts = np.unique(label_volume, return_counts=True)
            for class_id, count in zip(class_ids.tolist(), class_counts.tolist()):
                class_int = int(class_id)
                if class_int in class_voxel_counts:
                    class_voxel_counts[class_int] += int(count)

            if image_volume.shape != label_volume.shape:
                raise ValueError(
                    f"Image/label shape mismatch in {series_name}: {image_volume.shape} vs {label_volume.shape}"
                )

            case_counter = 0
            for img_patch, lbl_patch in generate_subvolumes(
                image_volume=image_volume,
                label_volume=label_volume,
                patch_size=patch_size,
                stride=stride,
            ):
                if skip_empty_patches and np.sum(lbl_patch) == 0:
                    continue

                case_name = f"{series_name}patch{case_counter:04d}"
                image_out = images_tr / f"{case_name}_0000.nii.gz"
                label_out = labels_tr / f"{case_name}.nii.gz"

                write_nifti(img_patch, image_out, spacing_xyz)
                write_nifti(lbl_patch, label_out, spacing_xyz)

                training_cases.append(case_name)
                case_counter += 1

            if case_counter == 0:
                raise ValueError(
                    f"No valid patches produced for {series_name}. "
                    "Try reducing --patch-size and/or disabling --skip-empty-patches."
                )

        missing_foreground = [
            label_name
            for label_name, class_id in labels_dict.items()
            if class_id != 0 and class_voxel_counts.get(class_id, 0) == 0
        ]
        if missing_foreground:
            missing_text = ", ".join(sorted(missing_foreground))
            raise ValueError(
                "Prepared dataset has zero voxels for foreground classes: "
                f"{missing_text}. "
                "This typically causes persistent 0.0 pseudo dice. "
                "Check source SegmentationClass masks and labelmap colors for these classes."
            )

        dataset_json = {
            "name": dataset_name,
            "description": "Wood log defect segmentation from Dataset001",
            "channel_names": {"0": "grayscale"},
            "labels": labels_dict,
            "numTraining": len(training_cases),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        with (dataset_root / "dataset.json").open("w", encoding="utf-8") as file_handle:
            json.dump(dataset_json, file_handle, indent=2)

        return dataset_root
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Dataset001 for nnU-Net v2")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./src/nn_UNet/datasets"),
        help="Source root containing dub* folders (supports datasets/ or datasets/Dataset001)",
    )
    parser.add_argument(
        "--nnunet-root",
                type=Path,
                default=Path("./src/nn_UNet/datasets/nnunet_data"),
        help="Root where nnUNet_raw/preprocessed/results folders are stored",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("./src/png"),
        help="Path containing per-series geometry.json folders",
    )
    parser.add_argument("--dataset-id", type=int, default=1, help="nnU-Net dataset id")
    parser.add_argument("--dataset-name", default="BPWoodDefects", help="Dataset suffix in nnU-Net naming")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared dataset")
    parser.add_argument(
        "--patch-size",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        default=(96, 256, 96),
        help="Patch size for sub-volume generation (default: 96 256 96)",
    )
    parser.add_argument(
        "--stride",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        default=(64, 128, 64),
        help="Sliding stride for sub-volume generation (default: 64 128 64)",
    )
    parser.add_argument(
        "--skip-empty-patches",
        action="store_true",
        default=True,
        help="Skip generated patches with only background labels (default: enabled)",
    )
    parser.add_argument(
        "--keep-empty-patches",
        action="store_false",
        dest="skip_empty_patches",
        help="Keep generated patches even when labels are all background",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = prepare_dataset(
        source_root=args.source,
        nnunet_root=args.nnunet_root,
        geometry_root=args.geometry_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        skip_empty_patches=args.skip_empty_patches,
    )
    print(f"Prepared nnU-Net dataset: {dataset_root}")


if __name__ == "__main__":
    main()
