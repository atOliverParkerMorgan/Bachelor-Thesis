from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

from src.nn_UNet.pipeline.env import PROJECT_ROOT, log
from src.preprocessing.conversion.ima2png import process_series as convert_dicom_series_to_png
from src.preprocessing.conversion.mask2datumaro import export_datumaro_dataset
from src.preprocessing.conversion.png2ima import get_3d_direction


def _tree_slug(tree_name: str) -> str:
    return tree_name.lower().replace(" ", "_")


def _series_png_files(series_dir: Path) -> list[Path]:
    return sorted(series_dir.glob("slice_*.png"))


def _load_geometry(series_dir: Path) -> dict:
    geo_path = series_dir / "geometry.json"
    if not geo_path.exists():
        raise FileNotFoundError(f"Missing geometry.json in {series_dir}")
    with open(geo_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_png_volume(series_dir: Path) -> np.ndarray:
    png_files = _series_png_files(series_dir)
    if not png_files:
        raise FileNotFoundError(f"No slice_*.png files found in {series_dir}")
    slices = [np.array(Image.open(png_file)) for png_file in png_files]
    return np.stack(slices, axis=0)


def _write_nifti(volume: np.ndarray, geometry: dict, output_file: Path) -> None:
    itk_img = sitk.GetImageFromArray(volume)
    if "spacing" in geometry:
        itk_img.SetSpacing(geometry["spacing"])
    if "origin" in geometry:
        itk_img.SetOrigin(geometry["origin"])
    if "direction" in geometry:
        itk_img.SetDirection(get_3d_direction(geometry["direction"]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(itk_img, str(output_file))


def _find_tree_source(tree_name: str, ground_truth_root: Path) -> Path:
    candidates = [
        ground_truth_root / tree_name,
        ground_truth_root / tree_name.lower(),
        ground_truth_root / tree_name.upper(),
        ground_truth_root / f"{tree_name}.zip",
        ground_truth_root / f"{tree_name.lower()}.zip",
        ground_truth_root / f"{tree_name.upper()}.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in ground_truth_root.rglob("*"):
        if candidate.name.lower() in {
            tree_name.lower(),
            f"{tree_name.lower()}.zip",
            f"{tree_name.upper()}.zip".lower(),
        }:
            return candidate

    raise FileNotFoundError(f"Could not find source data for tree '{tree_name}' under {ground_truth_root}")


def _extract_if_needed(source_path: Path, temp_root: Path) -> Path:
    if source_path.is_dir():
        return source_path

    if source_path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported tree source: {source_path}")

    extracted_root = temp_root / source_path.stem
    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)

    import zipfile

    with zipfile.ZipFile(source_path, "r") as archive:
        archive.extractall(extracted_root)
    return extracted_root


def _find_dicom_series_dirs(root: Path) -> list[Path]:
    dicom_dirs: list[Path] = []
    seen: set[Path] = set()
    for pattern in ("*.IMA", "*.dcm", "*.dicom"):
        for file_path in root.rglob(pattern):
            parent = file_path.parent
            if parent not in seen:
                seen.add(parent)
                dicom_dirs.append(parent)
    if not dicom_dirs:
        for file_path in root.rglob("*"):
            if file_path.is_file() and not file_path.suffix:
                parent = file_path.parent
                if parent not in seen:
                    seen.add(parent)
                    dicom_dirs.append(parent)
    return sorted(dicom_dirs)


def prepare_png_tree_from_ground_truth(
    tree_name: str,
    png_root: Path,
    ground_truth_root: Path,
    temp_root: Path,
) -> Path:
    source_path = _find_tree_source(tree_name, ground_truth_root)
    work_root = _extract_if_needed(source_path, temp_root)

    tree_slug = _tree_slug(tree_name)
    tree_output_root = png_root / tree_slug
    tree_output_root.mkdir(parents=True, exist_ok=True)

    dicom_dirs = _find_dicom_series_dirs(work_root)
    if not dicom_dirs:
        raise FileNotFoundError(f"No DICOM/IMA files found for tree '{tree_name}' in {source_path}")

    for series_dir in dicom_dirs:
        dicom_files = sorted(
            [p for p in series_dir.iterdir() if p.is_file() and (p.suffix.lower() in {".ima", ".dcm", ".dicom"} or not p.suffix)]
        )
        if dicom_files:
            convert_dicom_series_to_png(series_dir, dicom_files, tree_output_root, work_root, "series")

    geometry_files = sorted(tree_output_root.rglob("geometry.json"))
    if not geometry_files:
        raise FileNotFoundError(f"Failed to generate PNG geometry for tree '{tree_name}'")
    if len(geometry_files) > 1:
        log(f"Warning: found {len(geometry_files)} PNG series for {tree_name}; using {geometry_files[0].parent}")
    return geometry_files[0].parent


def write_tree_inference_nifti(
    tree_dir: Path,
    nifti_in_dir: Path,
    tree_name: str,
    is_3d: bool,
    chunk_size: int | None = None,
) -> list[Path]:
    geometry = _load_geometry(tree_dir)
    volume = _read_png_volume(tree_dir)

    if not is_3d:
        chunk_size = None

    output_files: list[Path] = []
    if chunk_size is None or chunk_size <= 0 or volume.shape[0] <= chunk_size:
        output_file = nifti_in_dir / f"{tree_name}_0000.nii.gz"
        _write_nifti(volume, geometry, output_file)
        return [output_file]

    for chunk_index, start in enumerate(range(0, volume.shape[0], chunk_size)):
        stop = min(start + chunk_size, volume.shape[0])
        chunk_volume = volume[start:stop]
        output_file = nifti_in_dir / f"{tree_name}_part{chunk_index:03d}_0000.nii.gz"
        _write_nifti(chunk_volume, geometry, output_file)
        output_files.append(output_file)

    return output_files


def merge_prediction_chunks(nifti_out_dir: Path, tree_name: str) -> Path:
    chunk_files = sorted(nifti_out_dir.glob(f"{tree_name}_part*_*.nii.gz"))
    if not chunk_files:
        chunk_files = sorted(nifti_out_dir.glob("*.nii.gz"))

    if not chunk_files:
        raise FileNotFoundError(f"No prediction chunks found in {nifti_out_dir}")

    if len(chunk_files) == 1:
        merged_file = nifti_out_dir / f"{tree_name}.nii.gz"
        if chunk_files[0] != merged_file:
            shutil.copy2(chunk_files[0], merged_file)
        return merged_file

    arrays = []
    reference_image = None
    for chunk_file in chunk_files:
        image = sitk.ReadImage(str(chunk_file))
        reference_image = reference_image or image
        arrays.append(sitk.GetArrayFromImage(image))

    merged = np.concatenate(arrays, axis=0)
    merged_image = sitk.GetImageFromArray(merged.astype(np.uint8))
    if reference_image is not None:
        merged_image.SetSpacing(reference_image.GetSpacing())
        merged_image.SetOrigin(reference_image.GetOrigin())
        merged_image.SetDirection(reference_image.GetDirection())

    merged_file = nifti_out_dir / f"{tree_name}.nii.gz"
    sitk.WriteImage(merged_image, str(merged_file))
    return merged_file


def export_prediction_masks(
    prediction_dir: Path,
    tree_dir: Path,
    segmentation_output_dir: Path,
    dataset_json_path: Path,
    tree_name: str,
    is_3d: bool = False,
) -> dict[int, str]:
    del dataset_json_path

    segmentation_output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = segmentation_output_dir / "masks"
    images_dir = segmentation_output_dir / "images"
    masks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for image_file in _series_png_files(tree_dir):
        shutil.copy2(image_file, images_dir / image_file.name)

    prediction_files = sorted(prediction_dir.glob("*.nii.gz"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction NIfTI files found in {prediction_dir}")

    prediction_file = next((path for path in prediction_files if path.name == f"{tree_name}.nii.gz"), prediction_files[0])
    prediction_image = sitk.ReadImage(str(prediction_file))
    prediction = sitk.GetArrayFromImage(prediction_image)

    if not is_3d and prediction.ndim != 3:
        raise ValueError(f"Expected 3D prediction volume, got shape {prediction.shape}")

    for index, slice_array in enumerate(prediction, start=1):
        mask_path = masks_dir / f"slice_{index:04d}.png"
        Image.fromarray(np.asarray(slice_array, dtype=np.uint8), mode="L").save(mask_path)

    return {0: "Pozadí", 1: "Zdravé dřevo", 2: "Suk", 3: "Hniloba", 4: "Kůra", 5: "Trhlina", 6: "Poškození hmyzem"}


def export_datumaro_for_tree(segmentation_output_dir: Path, output_zip: Path, tree_name: str) -> Path:
    export_datumaro_dataset(
        segmentation_output=segmentation_output_dir,
        output=output_zip,
        task_name=_tree_slug(tree_name),
        save_media=True,
        item_id_mode="stem",
    )
    return output_zip