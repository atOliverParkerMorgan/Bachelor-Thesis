#!/usr/bin/env python3
"""Merge two Datumaro archives and upload the result to a CVAT job."""

import os
import sys
import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# CVAT SDK Imports
from cvat_sdk import Client, Config
from cvat_sdk.core.client import AccessTokenCredentials
from cvat_sdk.api_client.exceptions import ApiException

# Formatting colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _normalize_label_name(value: str) -> str:
    """Normalize label names for robust cross-dataset matching."""
    return (value or "").strip().lower()


def _build_label_name_to_id(payload: dict) -> Dict[str, int]:
    labels = payload.get("categories", {}).get("label", {}).get("labels", [])
    mapping: Dict[str, int] = {}
    for idx, label in enumerate(labels):
        name = _normalize_label_name(label.get("name", ""))
        if name:
            mapping[name] = idx
    return mapping


def _remap_item_annotation_labels(item: dict, id_map: Dict[int, int]) -> tuple[int, int]:
    """Remap annotation labels in one item.

    Returns (remapped_count, dropped_count). Unmapped labels are dropped.
    """
    remapped = 0
    dropped = 0
    new_annotations = []

    for ann in item.get("annotations", []):
        label_key = "label_id" if "label_id" in ann else "label"
        old_label = ann.get(label_key)
        if not isinstance(old_label, int):
            new_annotations.append(ann)
            continue

        if old_label not in id_map:
            dropped += 1
            continue

        new_label = id_map[old_label]
        if new_label != old_label:
            ann[label_key] = new_label
            remapped += 1
        new_annotations.append(ann)

    item["annotations"] = new_annotations
    return remapped, dropped


def merge_two_datumaro_zips(zip1_path: Path, zip2_path: Path) -> Optional[Path]:
    """Merge two Datumaro zip archives.
    
    Combines annotations from both zips, with zip2 annotations added to zip1 annotations.
    Returns path to merged zip, or None if merge fails.
    """
    try:
        print(f"Reading {zip1_path.name}...")
        with zipfile.ZipFile(zip1_path, "r") as z1:
            anno_entries_1 = [
                name for name in z1.namelist()
                if name.startswith("annotations/") and name.endswith(".json")
            ]
            if not anno_entries_1:
                raise ValueError(f"No annotations found in {zip1_path.name}")
            
            # Load annotations from first zip
            anno_name_1 = anno_entries_1[0]
            raw_1 = z1.read(anno_name_1)
            payload_1 = json.loads(raw_1.decode("utf-8"))
            items_1 = {item.get("id", ""): item for item in payload_1.get("items", [])}
            file1_name_to_id = _build_label_name_to_id(payload_1)
        
        print(f"Reading {zip2_path.name}...")
        with zipfile.ZipFile(zip2_path, "r") as z2:
            anno_entries_2 = [
                name for name in z2.namelist()
                if name.startswith("annotations/") and name.endswith(".json")
            ]
            if not anno_entries_2:
                raise ValueError(f"No annotations found in {zip2_path.name}")
            
            # Load annotations from second zip
            anno_name_2 = anno_entries_2[0]
            raw_2 = z2.read(anno_name_2)
            payload_2 = json.loads(raw_2.decode("utf-8"))
            items_2 = payload_2.get("items", [])
            file2_name_to_id = _build_label_name_to_id(payload_2)

        # Build file2->file1 id mapping by label name.
        id_map_2_to_1: Dict[int, int] = {}
        missing_in_file1: Dict[int, str] = {}
        for name, src_id in file2_name_to_id.items():
            if name in file1_name_to_id:
                id_map_2_to_1[src_id] = file1_name_to_id[name]
            else:
                missing_in_file1[src_id] = name

        if missing_in_file1:
            missing_names = ", ".join(sorted(missing_in_file1.values()))
            print(
                f"{YELLOW}Warning:{RESET} file2 contains labels not present in file1 categories: {missing_names}. "
                "Annotations with those labels will be dropped to avoid wrong class assignment."
            )

        print("Label remap (file2 -> file1):")
        for src_name, src_id in sorted(file2_name_to_id.items(), key=lambda kv: kv[1]):
            dst_id = id_map_2_to_1.get(src_id)
            if dst_id is None:
                print(f"  {src_id}:{src_name} -> <no mapping>")
            else:
                print(f"  {src_id}:{src_name} -> {dst_id}:{src_name}")
        
        # Merge: add annotations from zip2 to zip1
        print(f"Merging annotations...")
        merged_count = 0
        remapped_annotations = 0
        dropped_annotations = 0
        for item_2 in items_2:
            remapped, dropped = _remap_item_annotation_labels(item_2, id_map_2_to_1)
            remapped_annotations += remapped
            dropped_annotations += dropped
            item_id = item_2.get("id", "")
            if item_id in items_1:
                # Merge annotations on this frame
                existing_annos = items_1[item_id].get("annotations", [])
                new_annos = item_2.get("annotations", [])
                items_1[item_id]["annotations"] = existing_annos + new_annos
                merged_count += 1
            else:
                # New frame
                items_1[item_id] = item_2
                merged_count += 1
        
        print(f"  Merged {merged_count} items")
        print(f"  Remapped {remapped_annotations} annotation label ids")
        if dropped_annotations:
            print(f"  {YELLOW}Dropped {dropped_annotations} unmapped annotations from file2{RESET}")
        
        # Create merged zip
        payload_1["items"] = list(items_1.values())
        
        with tempfile.NamedTemporaryFile(prefix="datumaro_merged_", suffix=".zip", delete=False) as tmp:
            merged_zip_path = Path(tmp.name)
        
        print(f"Creating merged archive at {merged_zip_path.name}...")
        with zipfile.ZipFile(zip1_path, "r") as src, zipfile.ZipFile(merged_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for entry in src.infolist():
                if entry.filename == anno_name_1:
                    # Write merged annotations
                    dst.writestr(entry, json.dumps(payload_1, ensure_ascii=True).encode("utf-8"))
                else:
                    # Copy other files as-is
                    dst.writestr(entry, src.read(entry.filename))
        
        file_size_mb = merged_zip_path.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}Merged archive created: {file_size_mb:.2f} MB{RESET}")
        return merged_zip_path
        
    except Exception as e:
        print(f"  {RED}Error during merge: {e}{RESET}")
        return None


def upload_to_cvat_job(client: Client, job_id: int, file_path: Path, format_name: str = "Datumaro 1.0"):
    """Upload Datumaro annotations to a CVAT job."""
    try:
        print(f"Retrieving Job ID {job_id}...")
        job = client.jobs.retrieve(job_id)
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"Uploading annotations {file_path.name} ({file_size_mb:.2f} MB)...")
        print(f"{YELLOW}Please wait, do not close the script.{RESET}")
        
        job.import_annotations(
            format_name=format_name,
            filename=str(file_path),
            conv_mask_to_poly=False,
        )
        
        print(f"{GREEN}Success: Annotations uploaded to job {job_id}.{RESET}")
        return True
        
    except ApiException as e:
        print(f"{RED}API Error: {e.status} - {e.reason}{RESET}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return False
    except Exception as e:
        print(f"{RED}Error uploading to CVAT: {e}{RESET}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge two Datumaro archives and upload to a CVAT job"
    )
    parser.add_argument(
        "--file1",
        type=Path,
        required=True,
        help="Path to first Datumaro zip file",
    )
    parser.add_argument(
        "--file2",
        type=Path,
        required=True,
        help="Path to second Datumaro zip file to merge into first",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        required=True,
        help="CVAT job ID to upload merged annotations to",
    )
    parser.add_argument(
        "--organization",
        default=None,
        help="CVAT organization slug (defaults to BP)",
    )
    
    args = parser.parse_args()
    
    # Validate files
    if not args.file1.exists():
        print(f"{RED}Error: {args.file1} not found{RESET}")
        return 1
    if not args.file2.exists():
        print(f"{RED}Error: {args.file2} not found{RESET}")
        return 1
    
    # Load environment
    load_dotenv()
    cvat_token = os.getenv("CVAT_TOKEN")
    cvat_url = os.getenv("CVAT_URL", "https://app.cvat.ai")
    cvat_org = args.organization or os.getenv("CVAT_ORGANIZATION", "BP")
    
    if not cvat_token:
        print(f"{RED}Error: CVAT_TOKEN is required in .env file{RESET}")
        return 1
    
    print(f"Configuration:")
    print(f"  File 1: {args.file1.name}")
    print(f"  File 2: {args.file2.name}")
    print(f"  Job ID: {args.job_id}")
    print(f"  CVAT URL: {cvat_url}")
    if cvat_org:
        print(f"  Organization: {cvat_org}")
    print()
    
    # Step 1: Merge
    print("=" * 60)
    print("Step 1: Merging Datumaro archives...")
    print("=" * 60)
    merged_zip = merge_two_datumaro_zips(args.file1, args.file2)
    if not merged_zip:
        print(f"{RED}Merge failed{RESET}")
        return 1
    
    # Step 2: Upload
    print()
    print("=" * 60)
    print("Step 2: Uploading to CVAT...")
    print("=" * 60)
    config = Config()
    try:
        with Client(url=cvat_url, config=config) as client:
            client.login(AccessTokenCredentials(cvat_token))
            if cvat_org:
                client.organization_slug = cvat_org
            
            success = upload_to_cvat_job(client, args.job_id, merged_zip)
            
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        success = False
    finally:
        # Cleanup
        if merged_zip and merged_zip.exists():
            try:
                merged_zip.unlink()
                print(f"Cleaned up temporary file: {merged_zip.name}")
            except Exception:
                pass
    
    print()
    if success:
        print(f"{GREEN}✓ Complete! Merged annotations successfully uploaded.{RESET}")
        return 0
    else:
        print(f"{RED}✗ Upload failed{RESET}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Aborted by user{RESET}")
        sys.exit(130)
