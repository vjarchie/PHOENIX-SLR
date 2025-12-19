# -*- coding: utf-8 -*-
"""
Download RWTH-PHOENIX-Weather 2014 Dataset

Dataset URL: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
Size: ~53 GB
"""

import os
import subprocess
import sys
from pathlib import Path

DATASET_URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz"
DATASET_DIR = Path(__file__).parent.parent / "data"


def download_phoenix():
    """Download and extract PHOENIX dataset."""
    
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    tar_file = DATASET_DIR / "phoenix-2014.v3.tar.gz"
    
    print("="*60)
    print("RWTH-PHOENIX-Weather 2014 Dataset Download")
    print("="*60)
    print(f"URL: {DATASET_URL}")
    print(f"Destination: {DATASET_DIR}")
    print(f"Size: ~53 GB (will take a while)")
    print("="*60)
    
    # Check if already downloaded
    if (DATASET_DIR / "phoenix2014-release").exists():
        print("Dataset already exists!")
        return
    
    # Download using wget or curl
    print("\nDownloading dataset...")
    
    try:
        # Try wget first
        subprocess.run([
            "wget", "-c", "-O", str(tar_file), DATASET_URL
        ], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try curl
            subprocess.run([
                "curl", "-L", "-C", "-", "-o", str(tar_file), DATASET_URL
            ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Use Python requests
            print("wget/curl not found, using Python requests...")
            import requests
            from tqdm import tqdm
            
            response = requests.get(DATASET_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    print("\nExtracting dataset...")
    subprocess.run([
        "tar", "-xzf", str(tar_file), "-C", str(DATASET_DIR)
    ], check=True)
    
    print("\nDataset downloaded and extracted successfully!")
    print(f"Location: {DATASET_DIR / 'phoenix2014-release'}")


def verify_dataset():
    """Verify dataset structure."""
    expected_dirs = [
        "phoenix2014-release/phoenix-2014-multisigner",
        "phoenix2014-release/phoenix-2014-signerindependent-SI5"
    ]
    
    print("\nVerifying dataset structure...")
    
    for d in expected_dirs:
        path = DATASET_DIR / d
        if path.exists():
            print(f"  [OK] {d}")
        else:
            print(f"  [MISSING] {d}")


if __name__ == "__main__":
    download_phoenix()
    verify_dataset()



