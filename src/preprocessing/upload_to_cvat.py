#!/usr/bin/env python3
"""Import Datumaro dataset files to CVAT using the High-Level SDK.

This script imports whole zip files directly into a CVAT Project.
It uses TUS (chunked uploads) automatically, so large files are supported 
without setting massive HTTP timeouts.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# CVAT SDK Imports
from cvat_sdk import Client, Config
from cvat_sdk.core.client import AccessTokenCredentials
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.proxies.projects import Project

# Formatting colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

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


def main():
    parser = argparse.ArgumentParser(description="Import Datumaro datasets to CVAT (High-Level SDK)")
    parser.add_argument("-f", "--file", help="Specific file to upload")
    parser.add_argument("--organization", help="CVAT organization slug (overrides env var)")
    args = parser.parse_args()

    # 1. Load Environment Variables
    load_dotenv()
    cvat_token = os.getenv("CVAT_TOKEN")
    cvat_project_id = os.getenv("CVAT_PROJECT_ID")
    cvat_url = os.getenv("CVAT_URL", "https://app.cvat.ai")
    
    cvat_org = args.organization or os.getenv("CVAT_ORGANIZATION", "BP")

    if not all([cvat_token, cvat_project_id]):
        print(f"{RED}Error: CVAT_TOKEN and CVAT_PROJECT_ID required in .env file.{RESET}")
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
            if upload_dataset(client, int(cvat_project_id), d_file):
                success_count += 1
        
        print(f"\nSummary: {success_count}/{len(datumaro_files)} successful.")
    
    return 0 if success_count == len(datumaro_files) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Aborted by user{RESET}")
        sys.exit(130)