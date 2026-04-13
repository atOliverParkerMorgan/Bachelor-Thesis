from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


class WoodDefectDataset:
    """Dataset of paired NIfTI image/label volumes.

    Cases that contain ``rare_label_idx`` (default 6 = poskozeni_hmyzem) are
    duplicated ``oversample_factor`` extra times so the DataLoader sees them
    proportionally more often — matching the strategy used by
    nnUNetTrainerRareClassBoostWandb.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        rare_label_idx: int = 6,
        oversample_factor: int = 8,
    ) -> None:
        self.samples: list[dict] = []

        image_paths = sorted(Path(image_dir).glob("*.nii.gz"))
        if not image_paths:
            raise ValueError(f"No NIfTI volumes found in {image_dir}")

        label_root = Path(label_dir)
        for img_path in image_paths:
            case_name = img_path.name
            if case_name.endswith("_0000.nii.gz"):
                label_name = case_name.replace("_0000.nii.gz", ".nii.gz")
            else:
                label_name = case_name

            lbl_path = label_root / label_name
            if not lbl_path.exists():
                raise FileNotFoundError(
                    f"Missing label volume for {img_path.name}: {lbl_path}"
                )

            self.samples.append(
                {"image": str(img_path), "label": str(lbl_path), "case_id": img_path.stem}
            )

        # Case-level oversampling for the rare class (mirrors nnUNet rare-boost)
        if oversample_factor > 0 and rare_label_idx is not None:
            rare_samples = [
                s for s in self.samples if self._has_label(s["label"], rare_label_idx)
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
    def _has_label(label_path: str, label_idx: int) -> bool:
        """Return True if the NIfTI label volume contains ``label_idx``."""
        try:
            img = sitk.ReadImage(str(label_path))
            arr = sitk.GetArrayFromImage(img)
            return int(label_idx) in np.unique(arr)
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
