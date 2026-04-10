import os
import json
import zipfile
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import SimpleITK as sitk
from PIL import Image
from PIL import ImageDraw
from pathlib import Path

# ==========================================
# CVAT RGB to nnU-Net Class Mapping
# ==========================================
# Assuming custom colormap export is used
COLOR_MAP = {
    (163, 56, 212): 1,   # Zdravé dřevo  #a338d4
    (214, 149, 170): 0,  # Pozadí        #d695aa  → background
    (174, 60, 29):   2,  # Suk           #ae3c1d
    (153, 242, 107): 3,  # Hniloba       #99f26b
    (38, 114, 129):  4,  # Kůra          #267281
    (222, 137, 84):  5,  # Trhlina       #de8954
    (63, 19, 205):   6,  # Poškození hmyzem #3f13cd
}

LABEL_TO_CLASS = {
    "Pozadi": 0,
    "pozadi": 0,
    "Pozadí": 0,
    "zdrave drevo": 1,
    "Zdravé dřevo": 1,
    "zdrave_drevo": 1,
    "suk": 2,
    "Suk": 2,
    "hniloba": 3,
    "Hniloba": 3,
    "kura": 4,
    "Kůra": 4,
    "trhlina": 5,
    "Trhlina": 5,
    "poskozeni hmyzem": 6,
    "Poškození hmyzem": 6,
    "poskozeni_hmyzem": 6,
}

def _decode_cvat_rle(rle: str, width: int, height: int) -> np.ndarray:
    counts = [int(x.strip()) for x in rle.split(",") if x.strip()]
    flat = np.zeros(width * height, dtype=bool)
    idx = 0
    value = 0
    for count in counts:
        if count <= 0:
            value = 1 - value
            continue
        end = min(idx + count, flat.size)
        if value == 1:
            flat[idx:end] = True
        idx = end
        value = 1 - value
        if idx >= flat.size:
            break
    return flat.reshape((height, width))


def _polygon_points_to_mask(points_str: str, width: int, height: int) -> np.ndarray:
    if not points_str:
        return np.zeros((height, width), dtype=bool)

    points = []
    for part in points_str.split(";"):
        xy = part.strip().split(",")
        if len(xy) != 2:
            continue
        try:
            x = float(xy[0])
            y = float(xy[1])
        except ValueError:
            continue
        points.append((x, y))

    if len(points) < 3:
        return np.zeros((height, width), dtype=bool)

    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=1)
    return np.array(canvas, dtype=bool)


def _render_slice_from_cvat_image(image_elem: ET.Element) -> np.ndarray:
    width = int(image_elem.get("width"))
    height = int(image_elem.get("height"))
    out = np.zeros((height, width), dtype=np.uint8)

    shapes = sorted(
        [child for child in image_elem if child.tag in {"mask", "polygon"}],
        key=lambda s: int(s.get("z_order", "0")),
    )

    for shape in shapes:
        class_id = LABEL_TO_CLASS.get(shape.get("label"))
        if class_id is None:
            continue

        if shape.tag == "mask":
            rle = shape.get("rle")
            left = int(shape.get("left", "0"))
            top = int(shape.get("top", "0"))
            w = int(shape.get("width", "0"))
            h = int(shape.get("height", "0"))
            if not rle or w <= 0 or h <= 0:
                continue
            local_mask = _decode_cvat_rle(rle, w, h)

            x0 = max(0, left)
            y0 = max(0, top)
            x1 = min(width, left + w)
            y1 = min(height, top + h)
            if x0 >= x1 or y0 >= y1:
                continue

            local_x0 = x0 - left
            local_y0 = y0 - top
            local_x1 = local_x0 + (x1 - x0)
            local_y1 = local_y0 + (y1 - y0)
            region = local_mask[local_y0:local_y1, local_x0:local_x1]
            out[y0:y1, x0:x1][region] = class_id

        elif shape.tag == "polygon":
            points_str = shape.get("points", "")
            polygon_mask = _polygon_points_to_mask(points_str, width, height)
            out[polygon_mask] = class_id

    return out


def _load_masks_from_cvat_xml(xml_path: str) -> tuple[list[str], list[np.ndarray]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_elems = root.findall("image")
    image_elems = sorted(image_elems, key=lambda el: int(el.get("id", "0")))

    image_names: list[str] = []
    masks: list[np.ndarray] = []
    for image_elem in image_elems:
        image_names.append(image_elem.get("name", ""))
        masks.append(_render_slice_from_cvat_image(image_elem))

    return image_names, masks


def generate_dataset_json(output_dir: str | Path):
    output_dir = Path(output_dir)
    labels_tr_dir = output_dir / "labelsTr"
    num_training_cases = len(list(labels_tr_dir.glob("*.nii.gz"))) if labels_tr_dir.exists() else 0

    dataset_info = {
        "channel_names": { "0": "CT" },
        "labels": {
            "background": 0,
            "zdrave_drevo": 1,
            "suk": 2,
            "hniloba": 3,
            "kura": 4,
            "trhlina": 5,
            "poskozeni_hmyzem": 6,
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz"
    }
    
    json_path = output_dir / "dataset.json"
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)

def find_ima_directory(extract_dir):
    for root, dirs, files in os.walk(extract_dir):
        if any(f.lower().endswith('.ima') or f.lower().endswith('.dcm') for f in files):
            return root
    return None

def process_tree(tree_name):
    tree_lower = tree_name.lower()
    tree_upper = tree_name.upper()
    
    zip_path = f"src/ground_truth/{tree_upper}.zip"
    extract_dir = f"src/ground_truth/{tree_upper}"
    cvat_masks_dir = f"src/cvat_exports/cvat/{tree_lower}/images"
    cvat_annotations_xml = f"src/cvat_exports/cvat/{tree_lower}/annotations_fixed.xml"
    nnunet_raw_dir = "src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/" 
    case_identifier = tree_lower 

    # 1. Extraction
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    dicom_dir = find_ima_directory(extract_dir)
    if not dicom_dir:
        raise ValueError(f"Could not find .IMA files in {extract_dir}")

    imagesTr_dir = os.path.join(nnunet_raw_dir, "imagesTr")
    labelsTr_dir = os.path.join(nnunet_raw_dir, "labelsTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    print("Reading DICOM series...")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    dicom_image_3d = reader.Execute()

    if os.path.exists(cvat_annotations_xml):
        print("Loading masks from CVAT XML annotations...")
        mask_files, masks_2d = _load_masks_from_cvat_xml(cvat_annotations_xml)
    else:
        mask_files = sorted([f for f in os.listdir(cvat_masks_dir) if f.endswith('.png')])
        masks_2d = []
    
    # 2. Dynamic Cropping
    start_idx = int(mask_files[0].split('_')[1].split('.')[0]) - 1
    end_idx = int(mask_files[-1].split('_')[1].split('.')[0]) - 1
    
    dicom_cropped = dicom_image_3d[:, :, start_idx : end_idx + 1]
    cropped_size = dicom_cropped.GetSize()

    mask_array_3d = np.zeros((cropped_size[2], cropped_size[1], cropped_size[0]), dtype=np.uint8)

    if masks_2d:
        if len(masks_2d) != cropped_size[2]:
            raise ValueError(
                f"Mismatch between CVAT image count ({len(masks_2d)}) and DICOM cropped depth ({cropped_size[2]})."
            )

        print("Processing masks from CVAT XML (RLE + polygons)...")
        for z, mask_2d_int in enumerate(masks_2d):
            if mask_2d_int.shape != (cropped_size[1], cropped_size[0]):
                raise ValueError(
                    f"Mask shape mismatch at slice {z}: got {mask_2d_int.shape}, expected {(cropped_size[1], cropped_size[0])}."
                )
            mask_array_3d[z, :, :] = mask_2d_int
    else:
        print("Processing masks with Direct Color Matching...")
        for z, mask_file in enumerate(mask_files):
            mask_path = os.path.join(cvat_masks_dir, mask_file)
            # Convert to RGB array
            mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
            # Default to background so any unknown/unmapped color becomes class 0.
            mask_2d_int = np.zeros((cropped_size[1], cropped_size[0]), dtype=np.uint8)

            # Fast direct match for each color in our map
            for rgb_color, class_id in COLOR_MAP.items():
                matches = np.all(mask_rgb == rgb_color, axis=-1)
                mask_2d_int[matches] = class_id

            mask_array_3d[z, :, :] = mask_2d_int

    mask_image_3d = sitk.GetImageFromArray(mask_array_3d)
    mask_image_3d.CopyInformation(dicom_cropped)

    out_image_path = os.path.join(imagesTr_dir, f"{case_identifier}_0000.nii.gz")
    out_mask_path = os.path.join(labelsTr_dir, f"{case_identifier}.nii.gz")
    
    print("Saving NIfTI files...")
    sitk.WriteImage(dicom_cropped, out_image_path)
    sitk.WriteImage(mask_image_3d, out_mask_path)
    
    generate_dataset_json(nnunet_raw_dir)
    print("Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=str, required=True)
    args = parser.parse_args()
    process_tree(args.tree)