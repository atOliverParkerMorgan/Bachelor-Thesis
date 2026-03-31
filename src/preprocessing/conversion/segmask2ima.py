import os
import json
import zipfile
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image

# ==========================================
# CVAT RGB to nnU-Net Class Mapping
# ==========================================
# Assuming custom colormap export is used
COLOR_MAP = {
    (0, 0, 0): 0,       # Pure Black (Unlabeled Background)
    (214, 149, 170): 0, # Pozadi (Wood) -> Merged into 0
    (153, 242, 107): 1, # Hniloba (Rot)
    (38, 114, 129): 2,  # Kura (Bark)
    (63, 19, 205): 3,   # Poškození hmyzem (Insect Damage)
    (174, 60, 29): 4,   # Suk (Knot)
    (222, 137, 84): 5   # Trhlina (Crack)
}

def generate_dataset_json(output_dir):
    dataset_info = {
        "channel_names": { "0": "CT" },
        "labels": {
            "background": 0,
            "Hniloba": 1,
            "Kura": 2,
            "Poskozeni_hmyzem": 3,
            "Suk": 4,
            "Trhlina": 5
        },
        "numTraining": 1, 
        "file_ending": ".nii.gz"
    }
    
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w', encoding='utf-8') as f:
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
    cvat_masks_dir = f"src/cvat_exports/{tree_lower}/SegmentationClass"
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
    
    mask_files = sorted([f for f in os.listdir(cvat_masks_dir) if f.endswith('.png')])
    
    # 2. Dynamic Cropping
    start_idx = int(mask_files[0].split('_')[1].split('.')[0]) - 1
    end_idx = int(mask_files[-1].split('_')[1].split('.')[0]) - 1
    
    dicom_cropped = dicom_image_3d[:, :, start_idx : end_idx + 1]
    cropped_size = dicom_cropped.GetSize()
    
    mask_array_3d = np.zeros((cropped_size[2], cropped_size[1], cropped_size[0]), dtype=np.uint8)
    
    print("Processing masks with Direct Color Matching...")
    for z, mask_file in enumerate(mask_files):
        mask_path = os.path.join(cvat_masks_dir, mask_file)
        # Convert to RGB array
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
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
    parser.add_argument("tree", type=str)
    args = parser.parse_args()
    process_tree(args.tree)