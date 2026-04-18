from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


class WoodDefectDataset:
    """Dataset of paired NIfTI image/label volumes.

    Attributes
    ----------
    base_samples : list[dict]
        Unique cases before any oversampling.
    samples : list[dict]
        All cases including rare-class duplicates (kept for compatibility).
    case_classes : dict[str, frozenset[int]]
        Maps each label path → set of class indices present in that volume.
        Computed once at construction time; used by train.py for stratified
        splitting and to apply oversampling only to the training fold.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        rare_label_idx: int = 6,
        oversample_factor: int = 8,
    ) -> None:
        self.case_classes: dict[str, frozenset[int]] = {}
        self.base_samples: list[dict] = []
        self.samples: list[dict] = []

        image_paths = sorted(Path(image_dir).glob("*.nii.gz"))
        if not image_paths:
            raise ValueError(f"No NIfTI volumes found in {image_dir}")

        label_root = Path(label_dir)
        for img_path in image_paths:
            label_name = self._label_name_from_image_name(img_path.name)
            lbl_path = label_root / label_name
            if not lbl_path.exists():
                raise FileNotFoundError(
                    f"Missing label volume for {img_path.name}: {lbl_path}"
                )

            classes = self._get_classes(str(lbl_path))
            self.case_classes[str(lbl_path)] = classes

            entry = {"image": str(img_path), "label": str(lbl_path), "case_id": img_path.stem}
            self.base_samples.append(entry)
            self.samples.append(entry)

        # Case-level oversampling for the rare class
        if oversample_factor > 0 and rare_label_idx is not None:
            rare_samples = [
                s for s in self.base_samples
                if rare_label_idx in self.case_classes.get(s["label"], frozenset())
            ]
            if rare_samples:
                for _ in range(oversample_factor):
                    self.samples.extend(rare_samples)
                print(
                    f"[WoodDefectDataset] Rare-class oversample: "
                    f"{len(rare_samples)} case(s) with label {rare_label_idx} "
                    f"duplicated {oversample_factor}\u00d7 \u2192 "
                    f"{len(self.samples)} total samples"
                )
            else:
                print(
                    f"[WoodDefectDataset] Warning: no cases found with label "
                    f"{rare_label_idx} — rare-class oversampling skipped."
                )

    @staticmethod
    def _label_name_from_image_name(image_name: str) -> str:
        """Map an nnU-Net image filename back to its label filename."""
        if image_name.endswith(".nii.gz"):
            case_name = image_name[:-7]
        else:
            case_name = Path(image_name).stem
        case_name = re.sub(r"_0000(?:_\d+)?$", "", case_name)
        return f"{case_name}.nii.gz"

    @staticmethod
    def _get_classes(label_path: str) -> frozenset[int]:
        """Return the set of all class indices present in the NIfTI label volume."""
        try:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
            return frozenset(int(c) for c in np.unique(arr))
        except Exception:
            return frozenset()

    @staticmethod
    def _has_label(label_path: str, label_idx: int) -> bool:
        """Return True if the NIfTI label volume contains ``label_idx``."""
        return label_idx in WoodDefectDataset._get_classes(label_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
