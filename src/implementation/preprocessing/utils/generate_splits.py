#!/usr/bin/env python3
"""
Generate splits_final.json for nnU-Net that keeps all sub-volumes
from the same original volume in the same fold.

This prevents data leakage when original volumes have been split
into sub-volumes for training.

Place the output file at:
  nnUNet_preprocessed/Dataset002_BPWoodDefectsSplit/splits_final.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Generate splits_final.json grouping sub-volumes by source"
    )
    parser.add_argument(
        "--preprocessed_dir",
        required=True,
        help="Path to nnUNet_preprocessed/DatasetXXX folder"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: preprocessed_dir/splits_final.json)"
    )
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)

    # Discover all case IDs from the preprocessed folder
    # nnU-Net preprocessed files are named like: dub_1_part0.npz
    case_ids = set()
    for f in preprocessed_dir.glob("*.npz"):
        case_id = f.stem  # e.g., dub_1_part0
        case_ids.add(case_id)

    if not case_ids:
        # Fallback: try .npy files or just generate from known structure
        print("No .npz files found. Generating splits from known case structure.")
        original_cases = ["dub_1", "dub_5", "dub_11", "dub_37"]
        n_parts = 5
        case_ids = set()
        for orig in original_cases:
            for i in range(n_parts):
                case_ids.add(f"{orig}_part{i}")

    print(f"Found {len(case_ids)} cases: {sorted(case_ids)}")

    # Group by original volume: strip _partN suffix
    groups = defaultdict(list)
    for case_id in case_ids:
        # Find the _partN suffix and extract original name
        if "_part" in case_id:
            orig_name = case_id.rsplit("_part", 1)[0]
        else:
            orig_name = case_id
        groups[orig_name].append(case_id)

    # Sort for determinism
    group_names = sorted(groups.keys())
    for g in group_names:
        groups[g] = sorted(groups[g])
        print(f"  Group '{g}': {groups[g]}")

    n_folds = len(group_names)
    print(f"\nCreating {n_folds}-fold split (one original volume held out per fold)")

    splits = []
    for fold_idx in range(n_folds):
        val_group = group_names[fold_idx]
        val_cases = groups[val_group]

        train_cases = []
        for g in group_names:
            if g != val_group:
                train_cases.extend(groups[g])

        train_cases = sorted(train_cases)
        val_cases = sorted(val_cases)

        splits.append({
            "train": train_cases,
            "val": val_cases
        })

        print(f"\n  Fold {fold_idx}:")
        print(f"    Val  ({len(val_cases)} cases): {val_group} -> {val_cases}")
        print(f"    Train ({len(train_cases)} cases): {[g for g in group_names if g != val_group]}")

    # Save
    output_path = args.output or str(preprocessed_dir / "splits_final.json")
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\nSaved to: {output_path}")
    print(f"\nUsage:")
    print(f"  nnUNetv2_train DATASET_ID 3d_fullres 0 --num_epochs 400")
    print(f"  nnUNetv2_train DATASET_ID 3d_fullres 1 --num_epochs 400")
    print(f"  ... up to fold {n_folds - 1}")


if __name__ == "__main__":
    main()