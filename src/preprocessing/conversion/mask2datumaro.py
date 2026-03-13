import argparse
import logging
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Official Datumaro API imports
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.annotation import Mask, LabelCategories, AnnotationType
from datumaro.components.media import Image


DEFAULT_FOLDER_TO_ID = {"pozadi": 0, "suk": 1, "hniloba": 2, "kura": 3, "trhlina": 4}
DEFAULT_LABEL_NAMES = ["Pozadi", "Suk", "Hniloba", "Kura", "Trhlina"]


class DatasetItemsIterable:
    def __init__(self, mask_files, ref_dir, images_dir, masks_dir, folder_to_id, subset):
        self.mask_files = mask_files
        self.ref_dir = ref_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.folder_to_id = folder_to_id
        self.subset = subset

    def __iter__(self):
        return iter_dataset_items(
            mask_files=self.mask_files,
            ref_dir=self.ref_dir,
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            folder_to_id=self.folder_to_id,
            subset=self.subset,
        )


def iter_dataset_items(mask_files, ref_dir, images_dir, masks_dir, folder_to_id, subset):
    for bg_mask_path in tqdm(mask_files, desc="Building Dataset"):
        rel_path = bg_mask_path.relative_to(ref_dir)
        item_id = rel_path.stem

        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
            candidate = images_dir / rel_path.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        item_annotations = []

        for folder, label_id in folder_to_id.items():
            mask_file = masks_dir / folder / rel_path
            if not mask_file.exists():
                continue

            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask_img is None or not np.any(mask_img > 0):
                continue

            mask_img = (
                (mask_img > 127).astype(np.uint8) if mask_img.max() > 1 else mask_img
            )

            num_labels, labels_im = cv2.connectedComponents(mask_img)

            for component_id in range(1, num_labels):
                instance_mask = (labels_im == component_id).astype(np.uint8)
                item_annotations.append(
                    Mask(
                        image=instance_mask,
                        label=label_id,
                        group=component_id,
                        z_order=len(item_annotations),
                    )
                )

        yield DatasetItem(
            id=item_id,
            subset=subset,
            media=Image.from_file(path=str(image_path)),
            annotations=item_annotations,
        )


def export_datumaro_dataset(
    segmentation_output: Path,
    output: Path,
    task_name: str,
    label_names: list[str] | None = None,
    folder_to_id: dict[str, int] | None = None,
) -> Path:
    label_names = label_names or DEFAULT_LABEL_NAMES
    folder_to_id = folder_to_id or DEFAULT_FOLDER_TO_ID

    label_cat = LabelCategories()
    for label_name in label_names:
        label_cat.add(label_name)

    categories = {AnnotationType.label: label_cat}
    masks_dir = segmentation_output / "masks"
    images_dir = segmentation_output / "images"

    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing masks directory: {masks_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    available_masks = [folder for folder in folder_to_id if (masks_dir / folder).exists()]
    if not available_masks:
        raise FileNotFoundError(f"No mask folders found in {masks_dir}")

    print(f"Found mask types: {', '.join(available_masks)}")
    ref_dir = masks_dir / available_masks[0]
    mask_files = sorted(ref_dir.glob("*.png"))
    if not mask_files:
        mask_files = sorted(ref_dir.rglob("*.png"))

    print(f"Processing {len(mask_files)} images...")

    dataset = Dataset.from_iterable(
        DatasetItemsIterable(
            mask_files=mask_files,
            ref_dir=ref_dir,
            images_dir=images_dir,
            masks_dir=masks_dir,
            folder_to_id=folder_to_id,
            subset=task_name,
        ),
        categories=categories,
    )

    print("Exporting dataset to Datumaro format...")
    temp_dir = output.parent / "temp_datumaro_lib_export"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        dataset.export(save_dir=str(temp_dir), format="datumaro", save_media=True)
        output_base = str(output).replace(".zip", "")
        shutil.make_archive(output_base, "zip", root_dir=temp_dir)
        print(f"Created Datumaro dataset at: {output}")
        return output
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert masks to Datumaro format using the official library."
    )
    parser.add_argument(
        "--segmentation-output",
        "-s",
        type=Path,
        required=True,
        help="Path to the segmentation output (containing 'masks' and 'images' folders)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to save the final .zip file",
    )
    parser.add_argument(
        "--task-name",
        "-n",
        type=str,
        required=True,
        help="Name of the subset (e.g. 'dub1')",
    )
    parser.add_argument(
        "--label-names",
        nargs=5,
        default=DEFAULT_LABEL_NAMES,
        help="Labels for Background, Knot, Decay, Bark, Crack",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    export_datumaro_dataset(
        segmentation_output=args.segmentation_output,
        output=args.output,
        task_name=args.task_name,
        label_names=args.label_names,
    )


if __name__ == "__main__":
    main()
