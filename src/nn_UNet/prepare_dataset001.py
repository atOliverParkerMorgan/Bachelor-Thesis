#!/usr/bin/env python3
"""Prepare Dataset001 into nnU-Net v2 raw dataset format."""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image


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
    with labelmap_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            rgb_text = parts[1].strip()
            rgb_values = tuple(int(v.strip()) for v in rgb_text.split(","))
            if len(rgb_values) != 3:
                raise ValueError(f"Invalid RGB mapping in {labelmap_path}: {line}")
            labels.append(LabelEntry(name, make_safe_label_name(name), rgb_values))
    if not labels:
        raise ValueError(f"No labels found in {labelmap_path}")
    return labels


def parse_slice_list(series_dir: Path, series_name: str) -> List[str]:
    list_path = series_dir / "ImageSets" / "Segmentation" / f"{series_name}.txt"
    if not list_path.exists():
        # Fallback to all files if ImageSets list does not exist.
        return sorted(p.stem for p in (series_dir / "SegmentationObject").glob("*.png"))
    items: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
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
    with geom_path.open("r", encoding="utf-8") as f:
        geometry = json.load(f)
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
    h, w, _ = mask_rgb.shape
    mask_ids = np.zeros((h, w), dtype=np.uint8)

    flat_rgb = mask_rgb.reshape(-1, 3)
    unique_colors = np.unique(flat_rgb, axis=0)
    for color_arr in unique_colors:
        color = tuple(int(v) for v in color_arr.tolist())
        if color not in color_to_id:
            raise ValueError(f"Unknown label color {color} in {source_path}")
        class_id = color_to_id[color]
        mask_ids[np.all(mask_rgb == color_arr, axis=2)] = class_id

    return mask_ids


def write_nifti(volume_zyx: np.ndarray, out_path: Path, spacing_xyz: Tuple[float, float, float]) -> None:
    image = sitk.GetImageFromArray(volume_zyx)
    image.SetSpacing(spacing_xyz)
    sitk.WriteImage(image, str(out_path), useCompression=True)


def prepare_dataset(
    source_root: Path,
    nnunet_root: Path,
    geometry_root: Path,
    dataset_id: int,
    dataset_name: str,
    overwrite: bool,
) -> Path:
    dataset_dirname = f"Dataset{dataset_id:03d}_{dataset_name}"
    raw_root = nnunet_root / "nnUNet_raw"
    dataset_root = raw_root / dataset_dirname
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"

    if dataset_root.exists() and overwrite:
        shutil.rmtree(dataset_root)

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    series_dirs = sorted(p for p in source_root.iterdir() if p.is_dir() and p.name.startswith("dub"))
    if not series_dirs:
        raise ValueError(f"No dub* folders found in {source_root}")

    ref_labels = parse_labelmap(series_dirs[0] / "labelmap.txt")
    color_to_id: Dict[Tuple[int, int, int], int] = {}
    labels_dict: Dict[str, int] = {"background": 0}
    training_cases: List[str] = []

    next_id = 1
    for entry in ref_labels:
        if entry.safe_name == "background":
            color_to_id[entry.color] = 0
            continue
        key = entry.safe_name
        while key in labels_dict:
            key = f"{entry.safe_name}_{next_id}"
        labels_dict[key] = next_id
        color_to_id[entry.color] = next_id
        next_id += 1

    for series_dir in series_dirs:
        series_name = series_dir.name
        current_labels = parse_labelmap(series_dir / "labelmap.txt")
        if [(e.original_name, e.color) for e in current_labels] != [
            (e.original_name, e.color) for e in ref_labels
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

        if image_volume.shape != label_volume.shape:
            raise ValueError(f"Image/label shape mismatch in {series_name}: {image_volume.shape} vs {label_volume.shape}")

        image_out = images_tr / f"{series_name}_0000.nii.gz"
        label_out = labels_tr / f"{series_name}.nii.gz"

        write_nifti(image_volume, image_out, spacing_xyz)
        write_nifti(label_volume, label_out, spacing_xyz)
        training_cases.append(series_name)

    dataset_json = {
        "name": dataset_name,
        "description": "Wood log defect segmentation from Dataset001",
        "channel_names": {"0": "grayscale"},
        "labels": labels_dict,
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with (dataset_root / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    return dataset_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Dataset001 for nnU-Net v2")
    parser.add_argument("--source", type=Path, default=Path("src/nn_UNet/Dataset001"), help="Source Dataset001 root")
    parser.add_argument(
        "--nnunet-root",
        type=Path,
        default=Path("src/nn_unet/nnunet_data"),
        help="Root where nnUNet_raw/preprocessed/results folders are stored",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("src/png"),
        help="Path containing per-series geometry.json folders",
    )
    parser.add_argument("--dataset-id", type=int, default=1, help="nnU-Net dataset id")
    parser.add_argument("--dataset-name", default="BPWoodDefects", help="Dataset suffix in nnU-Net naming")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared dataset")
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
    )
    print(f"Prepared nnU-Net dataset: {dataset_root}")


if __name__ == "__main__":
    main()
