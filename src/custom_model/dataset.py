from pathlib import Path


class WoodDefectDataset:
    def __init__(self, image_dir, label_dir):
        self.samples = []

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
                raise FileNotFoundError(f"Missing label volume for {img_path.name}: {lbl_path}")

            self.samples.append({"image": str(img_path), "label": str(lbl_path), "case_id": img_path.stem})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
