#!/usr/bin/env python
"""
Dataset Download Runner.

Wrapper script to download dermoscopy datasets from ISIC Archive.
This provides a clean entry point from the project root.

Usage:
    python run_download.py --download-all          # Download all datasets
    python run_download.py --download HAM10000     # Download specific dataset
    python run_download.py --verify                # Verify datasets
    python run_download.py --help                  # Show all options
"""

import sys
import runpy
from pathlib import Path

if __name__ == "__main__":
    # Get path to the download module
    project_root = Path(__file__).parent
    download_script = project_root / "src" / "data" / "download.py"
    
    # Run the download module as __main__
    # This avoids importing through src/__init__.py which requires torch
    sys.argv[0] = str(download_script)
    runpy.run_path(str(download_script), run_name="__main__")
