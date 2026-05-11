from __future__ import annotations

import functools
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

import SimpleITK as sitk

from .env import log


def run_cmd(command: List[str], env: Dict[str, str], label: str) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"Executable not found: {command[0]}. Install nnunetv2 and run through poetry.")
    print("$", " ".join(command))
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError:
        log(f"Failed: {label}")
        raise


@functools.lru_cache(maxsize=1)
def detect_gpu_vram_gb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        values = [line.strip() for line in output.splitlines() if line.strip()]
        if not values:
            return None
        return max(int(v) for v in values) / 1024.0
    except Exception:
        return None


def prediction_worker_profile(vram_gb: float | None) -> tuple[int, int]:
    if sys.platform == "win32":
        return 0, 0
    if vram_gb is None:
        return 2, 2
    if vram_gb >= 16:
        return 6, 6
    if vram_gb >= 12:
        return 4, 4
    if vram_gb >= 8:
        return 3, 3
    if vram_gb >= 6:
        return 2, 2
    return 1, 1


def normalize_case_id(name: str) -> str:
    stem = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
    return re.sub(r"_0000(?:_\d+)?$", "", stem)


def convert_dicom_zip_to_nifti(zip_path: Path, output_dir: Path) -> None:
    log(f"Extracting and converting DICOM zip to NIfTI: {zip_path.name}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_path)

        dicom_files: list[Path] = []
        for ext in ("*.IMA", "*.dcm", "*.dicom"):
            dicom_files.extend(tmp_path.rglob(ext))
        if not dicom_files:
            dicom_files = [p for p in tmp_path.rglob("*") if p.is_file() and not p.suffix]
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM/IMA files found in {zip_path}")

        dicom_dir = dicom_files[0].parent
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir)) or [str(p) for p in sorted(dicom_files)]
        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except Exception as exc:
            raise RuntimeError(f"Failed to read DICOM series from {zip_path}: {exc}") from exc

        sitk.WriteImage(image, str(output_dir / f"{zip_path.stem}_0000.nii.gz"), useCompression=True)
        log(f"Converted: {zip_path.stem}_0000.nii.gz")


def _fit_int_list(values: List[int], target_len: int) -> List[int]:
    if target_len <= 0:
        return []
    if not values:
        return [1] * target_len
    if len(values) >= target_len:
        return values[:target_len]
    return values + [values[-1]] * (target_len - len(values))


def decoder_len_for_stages(n_stages: int) -> int:
    return max(1, n_stages - 1)
