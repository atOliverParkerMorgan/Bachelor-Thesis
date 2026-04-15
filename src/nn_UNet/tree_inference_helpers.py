#!/usr/bin/env python3
"""Helpers for whole-tree nnU-Net inference on PNG slice stacks."""

from __future__ import annotations

import json
import shutil
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
from PIL import Image

from src.preprocessing.conversion.ima2png import process_series
from src.preprocessing.utils.upload_to_cvat import upload_specific_file

def _normalize_label_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_only.lower().replace("_", " ").strip()


SEGMENTATION_STYLE_LABELS = {
    "pozadi": "pozadi",
    "zdrave drevo": "zdrave_drevo",
    "suk": "suk",
    "hniloba": "hniloba",
    "kura": "kura",
    "trhlina": "trhlina",
    "trhilina": "trhlina",
    "poskozeni hmyzem": "poskozeni_hmyzem",
}
BACKGROUND_LABEL_ALIASES = {_normalize_label_token(name) for name in ("background", "pozadi", "Pozadí")}
HEALTHY_WOOD_LABEL_ALIASES = {
    _normalize_label_token(name)
    for name in ("zdrave_drevo", "zdrave drevo", "Zdravé dřevo", "healthy_wood", "healthy wood")
}


def sorted_tree_slices(tree_dir: Path) -> list[Path]:
    png_files = sorted(tree_dir.glob("slice_*.png"))
    if not png_files:
        raise FileNotFoundError(f"No slice_*.png files found in {tree_dir}")
    return png_files


def has_png_tree(tree_dir: Path) -> bool:
    return tree_dir.exists() and (tree_dir / "geometry.json").exists() and any(tree_dir.glob("slice_*.png"))


def tree_number(tree_name: str) -> str:
    digits = "".join(character for character in tree_name if character.isdigit())
    if not digits:
        raise ValueError(f"Tree name does not contain a numeric id: {tree_name}")
    return digits.lstrip("0") or "0"


def find_ground_truth_source(tree_name: str, ground_truth_root: Path) -> Path | None:
    exact_dir = ground_truth_root / tree_name
    if exact_dir.exists():
        return exact_dir

    tree_id = tree_number(tree_name)
    normalized_targets = {
        tree_name.lower().replace("_", "").replace("-", ""),
        f"dub{tree_id}",
        f"dub_{tree_id}",
    }

    for candidate in ground_truth_root.iterdir():
        normalized_name = candidate.stem.lower().replace("_", "").replace("-", "")
        if normalized_name in normalized_targets:
            return candidate

    return None


def collect_dicom_like_files(search_root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.IMA", "*.dcm", "*.dicom"):
        files.extend(path for path in search_root.rglob(pattern) if path.is_file())
    if not files:
        files.extend(path for path in search_root.rglob("*") if path.is_file() and not path.suffix)
    return sorted({path.resolve() for path in files})


def prepare_png_tree_from_ground_truth(
    tree_name: str,
    png_root: Path,
    ground_truth_root: Path,
    temp_root: Path,
) -> Path:
    tree_dir = png_root / tree_name
    if has_png_tree(tree_dir):
        return tree_dir

    source_path = find_ground_truth_source(tree_name, ground_truth_root)
    if source_path is None:
        raise FileNotFoundError(
            f"Tree '{tree_name}' was not found in {png_root} or {ground_truth_root}."
        )

    staging_root = temp_root / tree_name / "ground_truth_prepare"
    extracted_root = staging_root / "extracted"
    prepared_input_root = staging_root / "input"
    prepared_series_dir = prepared_input_root / tree_name
    prepared_series_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        extracted_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path) as archive:
            archive.extractall(extracted_root)
        search_root = extracted_root
    else:
        search_root = source_path

    dicom_files = collect_dicom_like_files(search_root)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM/IMA files found for tree '{tree_name}' in {source_path}")

    for index, dicom_file in enumerate(dicom_files, start=1):
        target_name = f"slice_source_{index:06d}{dicom_file.suffix or '.dcm'}"
        shutil.copy2(dicom_file, prepared_series_dir / target_name)

    process_series(
        series_path=prepared_series_dir,
        dicom_files=sorted(prepared_series_dir.iterdir()),
        output_root=png_root,
        input_root=prepared_input_root,
        layout="series",
    )

    if not has_png_tree(tree_dir):
        raise RuntimeError(f"Conversion finished, but PNG tree is still incomplete: {tree_dir}")
    return tree_dir


def load_tree_geometry(tree_dir: Path) -> dict[str, object]:
    geometry_path = tree_dir / "geometry.json"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Missing geometry.json in {tree_dir}")
    with geometry_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _dataset_labels_by_id(dataset_json_path: Path) -> dict[int, str]:
    with dataset_json_path.open("r", encoding="utf-8") as file_handle:
        dataset_json = json.load(file_handle)

    labels = dataset_json.get("labels", {})
    labels_by_id: dict[int, str] = {}

    # Accept both nnU-Net style {"name": id} and {id: "name"} maps.
    for key, value in labels.items():
        if isinstance(value, (int, str)):
            try:
                labels_by_id[int(value)] = str(key)
                continue
            except (TypeError, ValueError):
                pass
        try:
            labels_by_id[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    return labels_by_id


def write_tree_inference_nifti(tree_dir: Path, output_dir: Path, tree_name: str, is_3d: bool) -> list[Path]:
    """Write NIfTI inputs based on model configuration (3D volume vs 2D slices)."""
    png_files = sorted_tree_slices(tree_dir)
    geometry = load_tree_geometry(tree_dir)
    _ensure_clean_dir(output_dir)

    spacing = geometry.get("spacing", [1.0, 1.0, 1.0])
    written: list[Path] = []

    if is_3d:
        # Stack all PNGs into a single 3D volume (Z, Y, X)
        slices = []
        for png_file in png_files:
            with Image.open(png_file) as image:
                pixel_array = np.array(image.convert("L"), dtype=np.uint8)
            slices.append(pixel_array)
        
        volume = np.stack(slices, axis=0)
        itk_image = sitk.GetImageFromArray(volume)
        itk_image.SetSpacing(tuple(float(s) for s in spacing))
        
        out_path = output_dir / f"{tree_name}_0000.nii.gz"
        sitk.WriteImage(itk_image, str(out_path), useCompression=True)
        written.append(out_path)
    else:
        # Write individual 2D slices
        for png_file in png_files:
            with Image.open(png_file) as image:
                pixel_array = np.array(image.convert("L"), dtype=np.uint8)
            volume = pixel_array[np.newaxis, :, :]
            itk_image = sitk.GetImageFromArray(volume)
            itk_image.SetSpacing(tuple(float(s) for s in spacing))

            out_path = output_dir / f"{png_file.stem}_0000.nii.gz"
            sitk.WriteImage(itk_image, str(out_path), useCompression=True)
            written.append(out_path)

    return written


def segmentation_style_label_map(dataset_json_path: Path) -> tuple[dict[int, str], dict[int, str]]:
    dataset_labels = _dataset_labels_by_id(dataset_json_path)

    supported: dict[int, str] = {}
    ignored: dict[int, str] = {}

    for label_id, label_name in dataset_labels.items():
        folder_name = SEGMENTATION_STYLE_LABELS.get(_normalize_label_token(label_name))
        if folder_name is None:
            ignored[label_id] = label_name
            continue
        supported[label_id] = folder_name

    return supported, ignored


def _resolve_background_label_ids(dataset_json_path: Path) -> set[int]:
    dataset_labels = _dataset_labels_by_id(dataset_json_path)
    background_ids = {
        label_id
        for label_id, label_name in dataset_labels.items()
        if _normalize_label_token(label_name) in BACKGROUND_LABEL_ALIASES
    }
    # nnU-Net convention: class 0 is background.
    background_ids.add(0)
    return background_ids


def _resolve_healthy_label_ids(dataset_json_path: Path) -> set[int]:
    dataset_labels = _dataset_labels_by_id(dataset_json_path)
    return {
        label_id
        for label_id, label_name in dataset_labels.items()
        if _normalize_label_token(label_name) in HEALTHY_WOOD_LABEL_ALIASES
    }


def _resolve_defect_label_ids(dataset_json_path: Path) -> set[int]:
    dataset_labels = _dataset_labels_by_id(dataset_json_path)
    return {
        label_id
        for label_id, label_name in dataset_labels.items()
        if _normalize_label_token(label_name) not in BACKGROUND_LABEL_ALIASES
        and _normalize_label_token(label_name) not in HEALTHY_WOOD_LABEL_ALIASES
    }


def _derive_healthy_wood_mask(
    predicted_slice: np.ndarray,
    background_ids: Iterable[int],
    defect_ids: Iterable[int],
) -> np.ndarray:
    background_ids = set(background_ids)
    defect_ids = set(defect_ids)

    occupied = np.zeros(predicted_slice.shape, dtype=bool)
    for class_id in background_ids.union(defect_ids):
        occupied |= predicted_slice == class_id

    return np.where(~occupied, 255, 0).astype(np.uint8)


def _extract_prediction_slice(prediction_image: sitk.Image) -> np.ndarray:
    predicted_slice = sitk.GetArrayFromImage(prediction_image).squeeze()
    if predicted_slice.ndim != 2:
        raise ValueError(f"Expected 2D prediction slice, got shape {predicted_slice.shape}")
    return predicted_slice


def _slice_has_log(predicted_slice: np.ndarray, background_ids: Iterable[int]) -> bool:
    background_ids = set(background_ids)
    if not background_ids:
        # Fallback when dataset has no explicit background class.
        return bool(np.any(predicted_slice > 0))

    log_mask = np.ones(predicted_slice.shape, dtype=bool)
    for background_id in background_ids:
        log_mask &= predicted_slice != background_id
    return bool(np.any(log_mask))


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_prediction_masks(
    prediction_dir: Path,
    tree_dir: Path,
    segmentation_output_dir: Path,
    dataset_json_path: Path,
    tree_name: str,
    is_3d: bool = False
) -> dict[int, str]:
    label_map, ignored_labels = segmentation_style_label_map(dataset_json_path)
    if not label_map:
        raise RuntimeError("No supported segmentation-style labels were found in dataset.json")
    background_ids = _resolve_background_label_ids(dataset_json_path)
    healthy_ids = _resolve_healthy_label_ids(dataset_json_path)
    defect_ids = _resolve_defect_label_ids(dataset_json_path)

    png_files = sorted_tree_slices(tree_dir)

    images_dir = segmentation_output_dir / "images"
    masks_dir = segmentation_output_dir / "masks"
    _ensure_clean_dir(segmentation_output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in sorted(set(label_map.values())):
        (masks_dir / folder_name).mkdir(parents=True, exist_ok=True)

    all_present_ids: set[int] = set()
    exported_slices = 0

    if is_3d:
        # Load the single 3D prediction block and slice it back up
        pred_path = prediction_dir / f"{tree_name}.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing 3D prediction: {pred_path}")
        
        pred_volume = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
        if pred_volume.shape[0] != len(png_files):
            raise ValueError(f"3D prediction depth ({pred_volume.shape[0]}) does not match number of slices ({len(png_files)}).")

        for z, png_file in enumerate(png_files):
            predicted_slice = pred_volume[z]
            if not _slice_has_log(predicted_slice, background_ids):
                continue

            all_present_ids.update(int(v) for v in np.unique(predicted_slice).tolist())
            shutil.copy2(png_file, images_dir / png_file.name)
            for label_id, folder_name in label_map.items():
                if label_id in healthy_ids:
                    mask = _derive_healthy_wood_mask(predicted_slice, background_ids, defect_ids)
                else:
                    mask = np.where(predicted_slice == label_id, 255, 0).astype(np.uint8)
                Image.fromarray(mask).save(masks_dir / folder_name / png_file.name)
            exported_slices += 1
    else:
        # Load individual 2D predictions
        for png_file in png_files:
            pred_path = prediction_dir / f"{png_file.stem}.nii.gz"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction: {pred_path}")

            predicted_slice = _extract_prediction_slice(sitk.ReadImage(str(pred_path)))
            if not _slice_has_log(predicted_slice, background_ids):
                continue

            all_present_ids.update(int(v) for v in np.unique(predicted_slice).tolist())
            shutil.copy2(png_file, images_dir / png_file.name)
            for label_id, folder_name in label_map.items():
                if label_id in healthy_ids:
                    mask = _derive_healthy_wood_mask(predicted_slice, background_ids, defect_ids)
                else:
                    mask = np.where(predicted_slice == label_id, 255, 0).astype(np.uint8)
                Image.fromarray(mask).save(masks_dir / folder_name / png_file.name)
            exported_slices += 1

    if exported_slices == 0:
        raise RuntimeError(
            "No slices with detected log were exported. "
            "Check background label mapping in dataset.json and prediction outputs."
        )

    ignored_present = {lid: name for lid, name in ignored_labels.items() if lid in all_present_ids}
    if ignored_present:
        ignored_text = ", ".join(f"{name}={lid}" for lid, name in sorted(ignored_present.items()))
        print(f"Ignoring predicted labels not used by the segmentation-style export: {ignored_text}")
    print(f"Exported {exported_slices}/{len(png_files)} slices containing log pixels.")

    return label_map


def default_datumaro_output(segmentation_root: Path, tree_name: str) -> Path:
    return segmentation_root.parent / f"datumaro_{tree_name}.zip"


def export_datumaro_for_tree(segmentation_output_dir: Path, datumaro_zip: Path, tree_name: str) -> Path:
    from src.preprocessing.conversion.mask2datumaro import export_datumaro_dataset
    
    return export_datumaro_dataset(
        segmentation_output=segmentation_output_dir,
        output=datumaro_zip,
        task_name=tree_name,
    )


def upload_tree_datumaro(datumaro_zip: Path, organization: str | None = None) -> bool:
    return upload_specific_file(datumaro_zip, organization=organization)