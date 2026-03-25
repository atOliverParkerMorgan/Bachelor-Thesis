#!/usr/bin/env python3
"""Import Datumaro dataset files to CVAT using the High-Level SDK.

This script imports whole zip files directly into a CVAT Project.
It uses TUS (chunked uploads) automatically, so large files are supported 
without setting massive HTTP timeouts.
"""

import os
import sys
import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# CVAT SDK Imports
from cvat_sdk import Client, Config
from cvat_sdk.core.client import AccessTokenCredentials
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.proxies.jobs import Job
from cvat_sdk.core.proxies.projects import Project

# Formatting colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _item_id_variants(value: str) -> List[str]:
    """Build normalized variants to match CVAT frame names robustly."""
    normalized = str(value).replace("\\", "/")
    base = Path(normalized).name
    full_stem = str(Path(normalized).with_suffix(""))
    base_stem = Path(base).stem
    return [
        normalized.lower(),
        base.lower(),
        full_stem.lower(),
        base_stem.lower(),
    ]


def _frame_target_id(frame_name: str) -> str:
    """Build extensionless target id from CVAT frame name."""
    normalized = str(frame_name).replace("\\", "/")
    return str(Path(normalized).with_suffix(""))


def _build_frame_lookup(frame_names: List[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for frame_name in frame_names:
        target_id = _frame_target_id(frame_name)
        for key in _item_id_variants(frame_name):
            lookup.setdefault(key, target_id)
    return lookup


def _get_job_frame_names(client: Client, job_id: int) -> List[str]:
    """Get frame names from job metadata for id matching/remap."""
    try:
        meta_response = client.api_client.jobs_api.retrieve_data_meta(job_id)
        meta = meta_response[0] if isinstance(meta_response, tuple) else meta_response
        frames = getattr(meta, "frames", None) or []

        names: List[str] = []
        for frame in frames:
            name = getattr(frame, "name", None)
            if name:
                names.append(str(name))
        return names
    except Exception as exc:
        print(f"  {YELLOW}Warning:{RESET} Could not read job frame metadata: {exc}")
        return []


def _remap_and_filter_datumaro_for_job(
    file_path: Path, frame_names: List[str]
) -> Tuple[Optional[Path], int, int, int]:
    """Create a temporary Datumaro zip with ids remapped to known frame names.

    Returns (temp_zip_path, kept_items, dropped_items, remapped_items).
    If no changes are needed or cannot process the archive, temp_zip_path is None.
    """
    if not frame_names:
        return None, 0, 0, 0

    lookup = _build_frame_lookup(frame_names)

    kept_total = 0
    dropped_total = 0
    remapped_total = 0
    changed = False

    with zipfile.ZipFile(file_path, "r") as src_zip:
        file_names = src_zip.namelist()
        anno_entries = [
            name for name in file_names if name.startswith("annotations/") and name.endswith(".json")
        ]

        if not anno_entries:
            return None, 0, 0, 0

        updated_json: Dict[str, bytes] = {}
        for anno_name in anno_entries:
            raw = src_zip.read(anno_name)
            payload = json.loads(raw.decode("utf-8"))
            items = payload.get("items")
            if not isinstance(items, list):
                updated_json[anno_name] = raw
                continue

            new_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue

                item_id = str(item.get("id", ""))
                target_name = None
                for key in _item_id_variants(item_id):
                    if key in lookup:
                        target_name = lookup[key]
                        break

                if target_name is None:
                    dropped_total += 1
                    changed = True
                    continue

                if item_id != target_name:
                    item["id"] = target_name
                    remapped_total += 1
                    changed = True

                kept_total += 1
                new_items.append(item)

            payload["items"] = new_items
            updated_json[anno_name] = json.dumps(payload, ensure_ascii=True).encode("utf-8")

        if not changed:
            return None, kept_total, dropped_total, remapped_total

        with tempfile.NamedTemporaryFile(prefix="datumaro_job_match_", suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with zipfile.ZipFile(file_path, "r") as src, zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for entry in src.infolist():
                if entry.filename in updated_json:
                    dst.writestr(entry, updated_json[entry.filename])
                else:
                    dst.writestr(entry, src.read(entry.filename))

    return tmp_path, kept_total, dropped_total, remapped_total

def find_datumaro_files(base_dir: Path, specific_file: Optional[str] = None) -> List[Path]:
    """Scans standard output directories for Datumaro zip files."""
    search_paths = [base_dir / "src" / "output", base_dir / "output"]
    found_files = []
    
    for path in search_paths:
        if not path.is_dir():
            continue
            
        if specific_file:
            # Look for specific file
            f_name = specific_file if specific_file.endswith(".zip") else f"{specific_file}.zip"
            target = path / f_name
            if target.exists():
                return [target]
        else:
            # Scan for all datumaro*.zip files
            # excluding chunk files if any leftovers exist from previous scripts
            candidates = sorted(path.glob("datumaro*.zip"))
            filtered = [p for p in candidates if "_chunk" not in p.name]
            found_files.extend(filtered)
            
    return list(dict.fromkeys(found_files))


def upload_dataset(client: Client, project_id: int, file_path: Path, format_name: str = "Datumaro 1.0"):
    """
    Retrieves the project and imports the dataset using the high-level API.
    """
    try:
        # 1. Retrieve the project instance
        print(f"  Retrieving Project ID {project_id}...")
        project: Project = client.projects.retrieve(project_id)
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {file_path.name} ({file_size_mb:.2f} MB)...")
        print(f"  {YELLOW}Please wait, do not close the script.{RESET}")

        # 2. Import the dataset
        # The SDK uses TUS protocol, uploading in 10MB chunks automatically.
        project.import_dataset(
            format_name=format_name,
            filename=str(file_path),
            conv_mask_to_poly=False 
        )
        
        print(f"  {GREEN}Success: {file_path.name} imported.{RESET}")
        return True

    except ApiException as e:
        print(f"  {RED}API Error importing {file_path.name}:{RESET}")
        print(f"  Status: {e.status}")
        print(f"  Reason: {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return False
    except Exception as e:
        print(f"  {RED}Unexpected Error importing {file_path.name}:{RESET} {e}")
        return False


def _extract_and_merge_datumaro_annotations(job_id: int, new_anno_path: Path, existing_anno_dict: dict) -> Optional[Path]:
    """Extract new Datumaro annotations and merge with existing job annotations.
    
    Returns path to a merged Datumaro zip that combines both, or None if merge not possible.
    """
    try:
        # Read existing annotations structure
        existing_items: Dict[str, dict] = {}
        for item in existing_anno_dict.get("items", []):
            item_id = item.get("id", "")
            if item_id:
                existing_items[item_id] = item.copy()
        
        merged_zip_path = None
        with zipfile.ZipFile(new_anno_path, "r") as new_zip:
            anno_entries = [
                name for name in new_zip.namelist()
                if name.startswith("annotations/") and name.endswith(".json")
            ]
            
            if not anno_entries:
                return None
            
            # Process each annotation file in the new Datumaro
            merged_json: Dict[str, bytes] = {}
            items_merged_count = 0
            
            for anno_name in anno_entries:
                raw = new_zip.read(anno_name)
                payload = json.loads(raw.decode("utf-8"))
                new_items = payload.get("items", [])
                
                # Merge: keep existing annotations, add new ones
                for new_item in new_items:
                    item_id = new_item.get("id", "")
                    if item_id in existing_items:
                        # Merge annotations on this frame
                        existing_annos = existing_items[item_id].get("annotations", [])
                        new_annos = new_item.get("annotations", [])
                        existing_items[item_id]["annotations"] = existing_annos + new_annos
                    else:
                        # New frame - add as is
                        existing_items[item_id] = new_item
                    items_merged_count += 1
                
                # Update payload with merged items
                payload["items"] = list(existing_items.values())
                merged_json[anno_name] = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            
            # Create merged Datumaro zip
            with tempfile.NamedTemporaryFile(prefix="datumaro_merged_", suffix=".zip", delete=False) as tmp:
                merged_zip_path = Path(tmp.name)
            
            with zipfile.ZipFile(new_anno_path, "r") as new_src, zipfile.ZipFile(merged_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for entry in new_src.infolist():
                    if entry.filename in merged_json:
                        dst.writestr(entry, merged_json[entry.filename])
                    else:
                        dst.writestr(entry, new_src.read(entry.filename))
            
            print(f"    Merged {items_merged_count} items with existing annotations")
            return merged_zip_path
            
    except Exception as e:
        print(f"  {YELLOW}Warning:{RESET} Could not merge annotations: {e}")
        return None


def import_annotations_to_job(client: Client, job_id: int, file_path: Path, format_name: str = "Datumaro 1.0"):
    """Import annotations-only data into an existing CVAT job, merging with existing annotations."""
    temp_file_path: Optional[Path] = None
    merged_file_path: Optional[Path] = None
    try:
        print(f"  Retrieving Job ID {job_id}...")
        job: Job = client.jobs.retrieve(job_id)

        upload_file = file_path
        frame_names = _get_job_frame_names(client, job_id)
        if frame_names:
            (
                temp_file_path,
                kept_items,
                dropped_items,
                remapped_items,
            ) = _remap_and_filter_datumaro_for_job(file_path, frame_names)

            if temp_file_path is not None:
                upload_file = temp_file_path
                print(
                    f"  Prepared job-matched dataset: kept={kept_items}, "
                    f"dropped={dropped_items}, remapped={remapped_items}"
                )
        
        # Retrieve existing annotations to merge with new ones
        print(f"  Fetching existing job annotations...")
        existing_anno = None
        try:
            anno_response = client.api_client.jobs_api.retrieve_annotations(job_id)
            # API returns tuple of (data, http_response)
            existing_anno = anno_response[0] if isinstance(anno_response, tuple) else anno_response
        except Exception as e:
            print(f"    Could not retrieve existing annotations: {e}")
        
        # If we have existing annotations, merge them with new ones
        if existing_anno:
            # Convert annotated object to dict if needed
            if hasattr(existing_anno, 'to_dict'):
                existing_anno = existing_anno.to_dict()
            elif not isinstance(existing_anno, dict):
                existing_anno = vars(existing_anno) if hasattr(existing_anno, '__dict__') else {}
        
        existing_item_count = len(existing_anno.get("items", [])) if existing_anno else 0
        if existing_item_count > 0:
            print(f"  Found {existing_item_count} existing annotations, preparing merge...")
            merged_zip = _extract_and_merge_datumaro_annotations(job_id, upload_file, existing_anno)
            if merged_zip:
                merged_file_path = merged_zip
                upload_file = merged_file_path
                print(f"  Will upload merged dataset (existing + new annotations)")
        else:
            print(f"  No existing annotations found, uploading new annotations only")

        file_size_mb = upload_file.stat().st_size / (1024 * 1024)
        print(f"  Uploading annotations {upload_file.name} ({file_size_mb:.2f} MB)...")
        print(f"  {YELLOW}Please wait, do not close the script.{RESET}")

        # Import the (possibly merged) annotations
        job.import_annotations(
            format_name=format_name,
            filename=str(upload_file),
            conv_mask_to_poly=False,
        )

        print(f"  {GREEN}Success: {file_path.name} annotations imported to job {job_id}.{RESET}")
        return True

    except ApiException as e:
        print(f"  {RED}API Error importing annotations for job {job_id}:{RESET}")
        print(f"  Status: {e.status}")
        print(f"  Reason: {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
            body_text = str(e.body)
            if "Could not match item id" in body_text:
                print(
                    f"  {YELLOW}Hint:{RESET} Datumaro item ids do not match existing job frame names. "
                    "For existing jobs, export annotations-only with extensionless ids "
                    "(for example slice_0083 instead of slice_0083.png)."
                )
        return False
    except Exception as e:
        print(f"  {RED}Unexpected Error importing annotations for job {job_id}:{RESET} {e}")
        return False
    finally:
        if temp_file_path is not None:
            try:
                temp_file_path.unlink(missing_ok=True)
            except Exception:
                pass
        if merged_file_path is not None:
            try:
                merged_file_path.unlink(missing_ok=True)
            except Exception:
                pass


def upload_specific_file(
    file_path: Path,
    organization: Optional[str] = None,
    format_name: str = "Datumaro 1.0",
    job_id: Optional[int] = None,
) -> bool:
    """Upload one Datumaro zip directly using the same env-based CVAT configuration."""
    load_dotenv()

    cvat_token = os.getenv("CVAT_TOKEN")
    cvat_project_id = os.getenv("CVAT_PROJECT_ID")
    cvat_url = os.getenv("CVAT_URL", "https://app.cvat.ai")
    cvat_org = organization or os.getenv("CVAT_ORGANIZATION", "BP")

    if not cvat_token:
        raise RuntimeError("CVAT_TOKEN is required in the environment.")
    if job_id is None and not cvat_project_id:
        raise RuntimeError("CVAT_PROJECT_ID is required unless job_id is provided.")
    if not file_path.exists():
        raise FileNotFoundError(f"Datumaro archive does not exist: {file_path}")

    config = Config()
    with Client(url=cvat_url, config=config) as client:
        client.login(AccessTokenCredentials(cvat_token))
        if cvat_org:
            client.organization_slug = cvat_org
        if job_id is not None:
            return import_annotations_to_job(client, int(job_id), file_path, format_name=format_name)
        return upload_dataset(client, int(cvat_project_id), file_path, format_name=format_name)


def main():
    parser = argparse.ArgumentParser(description="Import Datumaro datasets to CVAT (High-Level SDK)")
    parser.add_argument("-f", "--file", help="Specific file to upload")
    parser.add_argument("--organization", help="CVAT organization slug (overrides env var)")
    parser.add_argument(
        "--job-id",
        type=int,
        help="Import annotations into an existing CVAT job (annotations-only import)",
    )
    args = parser.parse_args()

    # 1. Load Environment Variables
    load_dotenv()
    cvat_token = os.getenv("CVAT_TOKEN")
    cvat_project_id = os.getenv("CVAT_PROJECT_ID")
    cvat_url = os.getenv("CVAT_URL", "https://app.cvat.ai")
    cvat_job_id = args.job_id if args.job_id is not None else os.getenv("CVAT_JOB_ID")
    
    cvat_org = args.organization or os.getenv("CVAT_ORGANIZATION", "BP")

    if not cvat_token:
        print(f"{RED}Error: CVAT_TOKEN is required in .env file.{RESET}")
        return 1
    if cvat_job_id is None and not cvat_project_id:
        print(f"{RED}Error: CVAT_PROJECT_ID is required unless --job-id/CVAT_JOB_ID is provided.{RESET}")
        return 1

    # 2. Setup Client Configuration
    # We use the default configuration. The SDK handles large file chunking automatically.
    config = Config()

    # 3. Locate Files
    base_dir = Path(__file__).resolve().parent.parent.parent
    datumaro_files = find_datumaro_files(base_dir, args.file)

    if not datumaro_files:
        print(f"{YELLOW}No .zip files found in standard output directories.{RESET}")
        return 1

    print(f"Found {len(datumaro_files)} file(s) to import.")
    print(f"Target URL: {cvat_url}")
    if cvat_job_id is not None:
        print(f"Target Job ID: {int(cvat_job_id)} (annotations-only import)")
    else:
        print(f"Target Project ID: {cvat_project_id}")
    if cvat_org:
        print(f"Organization: {cvat_org}")

    # 4. Initialize Client and Execute
    # We instantiate Client directly and login manually to avoid make_client argument issues
    with Client(url=cvat_url, config=config) as client:
        
        # Authenticate
        client.login(AccessTokenCredentials(cvat_token))
        
        # Set Organization Context if provided
        if cvat_org:
            client.organization_slug = cvat_org

        success_count = 0
        for d_file in datumaro_files:
            print(f"\nProcessing: {d_file.name}")
            if cvat_job_id is not None:
                ok = import_annotations_to_job(client, int(cvat_job_id), d_file)
            else:
                ok = upload_dataset(client, int(cvat_project_id), d_file)
            if ok:
                success_count += 1
        
        print(f"\nSummary: {success_count}/{len(datumaro_files)} successful.")
    
    return 0 if success_count == len(datumaro_files) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Aborted by user{RESET}")
        sys.exit(130)