"""
Dataset Download and Setup Scripts for Dermoscopy Datasets.

This module provides utilities to download and organize dermoscopy datasets
from the ISIC Archive API for federated learning experiments:
- HAM10000 (Client 1)
- ISIC 2018 Task 3 (Client 2)
- ISIC 2019 (Client 3)
- ISIC 2020 / SIIM-ISIC (Client 4)

Download Methods:
1. ISIC Archive API (recommended) - No API key required
2. Manual download from ISIC Archive website

API Reference: https://isic-archive.com/api/v2
"""

import csv
import json
import shutil
import time
import logging
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# ISIC Archive API Configuration
# =============================================================================

ISIC_API_BASE_URL = "https://api.isic-archive.com/api/v2"

# Dataset names as they appear in ISIC Archive API
ISIC_DATASET_NAMES = {
    "HAM10000": "HAM10000",
    "ISIC2018": "ISIC_2018_Task_3_Training",
    "ISIC2019": "ISIC_2019_Training", 
    "ISIC2020": "ISIC_2020_Training_JPEG",
}

# Dataset information and expected structure
DATASET_INFO = {
    "HAM10000": {
        "description": "Human Against Machine with 10000 training images",
        "isic_dataset": "HAM10000",
        "source": "https://api.isic-archive.com",
        "archive_url": "https://isic-archive.com/",
        "download_url": "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000",
        "classes": 7,
        "approx_images": 10015,
        "expected_files": [
            "metadata.csv",
            "images/"
        ],
        "client_id": 1
    },
    "ISIC2018": {
        "description": "ISIC 2018 Challenge - Task 3: Lesion Diagnosis",
        "isic_dataset": "ISIC_2018_Task_3_Training",
        "source": "https://api.isic-archive.com",
        "archive_url": "https://challenge.isic-archive.com/data/#2018",
        "download_url": "https://challenge.isic-archive.com/data/#2018",
        "classes": 7,
        "approx_images": 10015,
        "expected_files": [
            "metadata.csv",
            "images/"
        ],
        "client_id": 2
    },
    "ISIC2019": {
        "description": "ISIC 2019 Challenge - Dermoscopic Image Classification",
        "isic_dataset": "ISIC_2019_Training",
        "source": "https://api.isic-archive.com",
        "archive_url": "https://challenge.isic-archive.com/data/#2019",
        "download_url": "https://challenge.isic-archive.com/data/#2019",
        "classes": 8,
        "approx_images": 25331,
        "expected_files": [
            "metadata.csv",
            "images/"
        ],
        "client_id": 3
    },
    "ISIC2020": {
        "description": "ISIC 2020 Challenge - Melanoma Classification (SIIM-ISIC)",
        "isic_dataset": "ISIC_2020_Training_JPEG",
        "source": "https://api.isic-archive.com",
        "archive_url": "https://challenge.isic-archive.com/data/#2020",
        "download_url": "https://challenge.isic-archive.com/data/#2020",
        "classes": 2,
        "approx_images": 33126,
        "expected_files": [
            "metadata.csv",
            "images/"
        ],
        "client_id": 4
    }
}


def get_data_root() -> Path:
    """Get the data root directory."""
    # Try to find project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() or (parent / "src").exists():
            return parent / "data"
    # Fallback
    return Path("./data")


def create_directory_structure(data_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Create the expected directory structure for all datasets.
    
    Returns:
        Dictionary mapping dataset names to their paths
    """
    if data_root is None:
        data_root = get_data_root()
    
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for dataset_name in DATASET_INFO:
        dataset_path = data_root / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        # Create images subdirectory
        (dataset_path / "images").mkdir(exist_ok=True)
        paths[dataset_name] = dataset_path
        
    # Also create raw and processed subdirectories
    (data_root / "raw").mkdir(exist_ok=True)
    (data_root / "processed").mkdir(exist_ok=True)
    
    print(f"Created directory structure at: {data_root}")
    return paths


# =============================================================================
# ISIC Archive API Client
# =============================================================================

class ISICArchiveClient:
    """
    Client for ISIC Archive API v2.
    
    Downloads dermoscopy images and metadata from the official ISIC Archive.
    No API key required.
    
    API Documentation: https://api.isic-archive.com/api/v2/docs
    """
    
    def __init__(
        self,
        base_url: str = ISIC_API_BASE_URL,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        timeout: int = 30,
        max_workers: int = 8
    ):
        """
        Initialize ISIC Archive API client.
        
        Args:
            base_url: ISIC API base URL
            max_retries: Maximum retry attempts for failed requests
            backoff_factor: Exponential backoff factor for retries
            timeout: Request timeout in seconds
            max_workers: Maximum concurrent download workers
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_workers = max_workers
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Set headers
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ISIC-Downloader/1.0 (Federated Learning Research)"
        })
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None
    ) -> requests.Response:
        """Make API request with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            logger.debug("ISIC API request: %s", response.url)
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            raise
    
    def get_image_list(
        self,
        collection: Optional[str] = None,
        limit: int = 10000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get list of images from ISIC Archive.
        
        Args:
            collection: Collection/dataset name filter
            limit: Maximum images to return per request
            offset: Offset for pagination
            
        Returns:
            List of image metadata dictionaries
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset
        }
        if collection:
            # ISIC API may expect either 'collection' or 'collections' depending on version/usage.
            # Include both keys defensively so the server receives the intended filter.
            params["collections"] = collection
            params["collection"] = collection
        
        response = self._make_request("/images/", params=params)
        return response.json().get("results", [])
    
    def get_all_images_for_collection(
        self,
        collection: str,
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get all images from a specific collection with pagination.
        
        Args:
            collection: Collection/dataset name
            batch_size: Number of images per API call
            
        Returns:
            List of all image metadata
        """
        all_images: List[Dict[str, Any]] = []
        seen_ids = set()
        cursor = None

        print(f"  Fetching image list for collection: {collection}")

        with tqdm(desc="  Fetching metadata", unit=" images") as pbar:
            while True:
                params: Dict[str, Any] = {"limit": batch_size}
                if collection:
                    params["collections"] = collection
                    params["collection"] = collection
                if cursor:
                    params["cursor"] = cursor

                resp = self._make_request("/images/", params=params)
                batch = resp.json().get("results", [])

                if not batch:
                    break

                # Filter out duplicates by ISIC id
                new_items = []
                for img in batch:
                    image_id = img.get("isic_id", img.get("_id", ""))
                    if not image_id or image_id in seen_ids:
                        continue
                    seen_ids.add(image_id)
                    new_items.append(img)

                if not new_items:
                    break

                all_images.extend(new_items)
                pbar.update(len(new_items))

                # Determine next cursor from the API's 'next' link
                next_link = resp.json().get("next")
                if not next_link:
                    break
                parsed = urlparse(next_link)
                q = parse_qs(parsed.query)
                cursor = q.get("cursor", [None])[0]

                # Small delay to be nice to the API
                time.sleep(0.1)

        return all_images
    
    def download_image(
        self,
        image_id: str,
        output_path: Path,
        size: str = "full"
    ) -> bool:
        """
        Download a single image from ISIC Archive.
        
        Args:
            image_id: ISIC image ID (e.g., "ISIC_0024306")
            output_path: Path to save the image
            size: Image size - "full", "thumbnail", or pixel size
            
        Returns:
            True if successful
        """
        try:
            # First attempt: use the API files endpoint
            url = f"{self.base_url}/images/{image_id}/files/{size}"
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            # If the API endpoint returns 404, try retrieving the file URL from image metadata
            status = getattr(e.response, 'status_code', None)
            if status == 404:
                try:
                    meta_url = f"{self.base_url}/images/{image_id}"
                    meta_resp = self.session.get(meta_url, timeout=self.timeout)
                    meta_resp.raise_for_status()
                    files = meta_resp.json().get('files', {})
                    file_info = files.get(size) or files.get('full')
                    if file_info and isinstance(file_info, dict):
                        file_url = file_info.get('url')
                        if not file_url:
                            logger.warning(f"No file URL available in metadata for {image_id}")
                            return False
                        response = self.session.get(file_url, timeout=self.timeout, stream=True)
                        response.raise_for_status()
                    else:
                        raise
                except Exception:
                    logger.warning(f"Failed to download {image_id}: {e}")
                    return False
            else:
                logger.warning(f"Failed to download {image_id}: {e}")
                return False
        except Exception as e:
            logger.warning(f"Failed to download {image_id}: {e}")
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.warning(f"Failed to save {image_id}: {e}")
            return False
    
    def _download_worker(
        self,
        task: Tuple[str, Path]
    ) -> Tuple[str, bool]:
        """Worker function for parallel downloads."""
        image_id, output_path = task
        
        if output_path.exists():
            return (image_id, True)  # Skip existing
            
        success = self.download_image(image_id, output_path)
        return (image_id, success)
    
    def download_images_parallel(
        self,
        images: List[Dict[str, Any]],
        output_dir: Path,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download multiple images in parallel.
        
        Args:
            images: List of image metadata dictionaries
            output_dir: Directory to save images
            max_workers: Number of parallel workers
            
        Returns:
            Download statistics
        """
        if max_workers is None:
            max_workers = self.max_workers
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare download tasks
        tasks = []
        for img in images:
            image_id = img.get("isic_id", img.get("_id", ""))
            if not image_id:
                continue
            output_path = output_dir / f"{image_id}.jpg"
            tasks.append((image_id, output_path))
        
        # Check for existing files
        existing = sum(1 for _, path in tasks if path.exists())
        to_download = [(id_, path) for id_, path in tasks if not path.exists()]
        
        print(f"  Found {existing} existing images, downloading {len(to_download)} new images")
        
        if not to_download:
            return {"total": len(tasks), "success": len(tasks), "failed": 0, "skipped": existing}
        
        success_count = existing
        failed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_worker, task): task 
                for task in to_download
            }
            
            with tqdm(total=len(to_download), desc="  Downloading images", unit=" img") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    image_id, success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                    pbar.update(1)
        
        return {
            "total": len(tasks),
            "success": success_count,
            "failed": failed_count,
            "skipped": existing
        }
    
    def save_metadata_csv(
        self,
        images: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """
        Save image metadata to CSV file.
        
        Args:
            images: List of image metadata dictionaries
            output_path: Path to save CSV file
        """
        if not images:
            logger.warning("No images to save metadata for")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten nested metadata
        rows = []
        for img in images:
            row = {
                "isic_id": img.get("isic_id", img.get("_id", "")),
                "attribution": img.get("attribution", ""),
                "copyright_license": img.get("copyright_license", ""),
            }
            
            # Extract diagnosis/label from metadata
            metadata = img.get("metadata", {})
            clinical = metadata.get("clinical", {})
            
            row["diagnosis"] = (
                metadata.get("diagnosis", "") or 
                clinical.get("diagnosis", "") or
                img.get("diagnosis", "")
            )
            row["diagnosis_confirm_type"] = clinical.get("diagnosis_confirm_type", "")
            row["melanocytic"] = clinical.get("melanocytic", "")
            row["benign_malignant"] = (
                metadata.get("benign_malignant", "") or
                clinical.get("benign_malignant", "")
            )
            
            # Patient/acquisition info
            row["age_approx"] = clinical.get("age_approx", "")
            row["sex"] = clinical.get("sex", "")
            row["anatom_site_general"] = clinical.get("anatom_site_general", "")
            
            # Image acquisition
            acquisition = metadata.get("acquisition", {})
            row["image_type"] = acquisition.get("image_type", "")
            row["dermoscopic_type"] = acquisition.get("dermoscopic_type", "")
            
            rows.append(row)
        
        # Write CSV
        fieldnames = list(rows[0].keys()) if rows else []
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"  Saved metadata to {output_path}")


# =============================================================================
# Dataset Download Functions
# =============================================================================

def download_dataset_isic(
    dataset_name: str,
    data_root: Optional[Path] = None,
    max_workers: int = 8,
    force_redownload: bool = False
) -> bool:
    """
    Download a dataset from ISIC Archive.
    
    Args:
        dataset_name: Name of dataset (HAM10000, ISIC2018, ISIC2019, ISIC2020)
        data_root: Root directory for datasets
        max_workers: Number of parallel download workers
        force_redownload: If True, redownload even if files exist
        
    Returns:
        True if successful
    """
    if dataset_name not in DATASET_INFO:
        print(f"Unknown dataset: {dataset_name}")
        return False
    
    if data_root is None:
        data_root = get_data_root()
    
    data_root = Path(data_root)
    dataset_path = data_root / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    info = DATASET_INFO[dataset_name]
    isic_collection = info.get("isic_dataset", dataset_name)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"Collection: {isic_collection}")
    print(f"Expected images: ~{info['approx_images']:,}")
    print(f"{'='*60}")
    
    # Check if already downloaded
    if not force_redownload:
        existing_images = len(list(images_dir.glob("*.jpg")))
        if existing_images >= info["approx_images"] * 0.95:  # 95% threshold
            print(f"Dataset appears complete ({existing_images:,} images found)")
            return True
    
    # Initialize API client
    client = ISICArchiveClient(max_workers=max_workers)
    
    try:
        # Get all images for collection
        print("\nStep 1: Fetching image metadata from ISIC Archive...")
        images = client.get_all_images_for_collection(isic_collection)
        
        if not images:
            print(f"No images found for collection: {isic_collection}")
            print("  This may indicate the collection name has changed.")
            print("  Please check: https://api.isic-archive.com/api/v2/collections/")
            return False
        
        print(f"  Found {len(images):,} images")
        
        # Save metadata
        print("\nStep 2: Saving metadata...")
        metadata_path = dataset_path / "metadata.csv"
        client.save_metadata_csv(images, metadata_path)
        
        # Download images
        print("\nStep 3: Downloading images...")
        stats = client.download_images_parallel(images, images_dir, max_workers)
        
        print(f"\n{'='*60}")
        print(f"Download Summary for {dataset_name}:")
        print(f"  Total images: {stats['total']:,}")
        print(f"  Successfully downloaded: {stats['success']:,}")
        print(f"  Failed: {stats['failed']:,}")
        print(f"  Skipped (existing): {stats['skipped']:,}")
        print(f"{'='*60}")
        
        return stats["failed"] == 0
        
    except Exception as e:
        logger.error(f"Error downloading {dataset_name}: {e}")
        print(f"Download failed: {e}")
        return False


def download_all_datasets(
    data_root: Optional[Path] = None,
    datasets: Optional[List[str]] = None,
    max_workers: int = 8
) -> Dict[str, bool]:
    """
    Download all (or specified) datasets from ISIC Archive.
    
    Args:
        data_root: Root directory for datasets
        datasets: List of datasets to download (None = all)
        max_workers: Number of parallel workers
        
    Returns:
        Dictionary mapping dataset names to success status
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())
    
    results = {}
    
    print("\n" + "=" * 70)
    print("ISIC ARCHIVE DATASET DOWNLOADER")
    print("=" * 70)
    print(f"Datasets to download: {', '.join(datasets)}")
    print(f"Total estimated images: ~{sum(DATASET_INFO[d]['approx_images'] for d in datasets):,}")
    print("=" * 70)
    
    for dataset_name in datasets:
        results[dataset_name] = download_dataset_isic(
            dataset_name,
            data_root=data_root,
            max_workers=max_workers
        )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL DOWNLOAD SUMMARY")
    print("=" * 70)
    for dataset_name, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"  {status} {dataset_name}")
    
    successful = sum(results.values())
    print(f"\nCompleted: {successful}/{len(results)} datasets")
    print("=" * 70)
    
    return results


# =============================================================================
# Verification Functions
# =============================================================================

def verify_dataset(dataset_name: str, data_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Verify that a dataset is properly set up.
    
    Args:
        dataset_name: Name of dataset (HAM10000, ISIC2018, etc.)
        data_root: Root data directory
        
    Returns:
        Dictionary with verification results
    """
    if data_root is None:
        data_root = get_data_root()
    
    data_root = Path(data_root)
    
    if dataset_name not in DATASET_INFO:
        return {"valid": False, "error": f"Unknown dataset: {dataset_name}"}
    
    info = DATASET_INFO[dataset_name]
    dataset_path = data_root / dataset_name
    
    result = {
        "valid": True,
        "dataset": dataset_name,
        "path": str(dataset_path),
        "expected_files": info["expected_files"],
        "found_files": [],
        "missing_files": [],
        "image_count": 0,
        "csv_found": False,
        "completeness": 0.0
    }
    
    # Check expected files
    for expected in info["expected_files"]:
        full_path = dataset_path / expected
        if full_path.exists():
            result["found_files"].append(expected)
            if expected.endswith(".csv"):
                result["csv_found"] = True
        else:
            result["missing_files"].append(expected)
    
    # Count images
    images_dir = dataset_path / "images"
    if images_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            result["image_count"] += len(list(images_dir.glob(ext)))
    
    # Also check root directory for images (backward compatibility)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        result["image_count"] += len(list(dataset_path.glob(ext)))
    
    # Calculate completeness
    expected_images = info["approx_images"]
    result["completeness"] = min(100.0, (result["image_count"] / expected_images) * 100)
    
    # Valid if we have enough images (>90% threshold)
    result["valid"] = result["image_count"] >= expected_images * 0.9
    
    return result


def verify_all_datasets(data_root: Optional[Path] = None) -> Dict[str, Dict]:
    """Verify all datasets and return summary."""
    results = {}
    for dataset_name in DATASET_INFO:
        results[dataset_name] = verify_dataset(dataset_name, data_root)
    return results


def print_verification_report(results: Dict[str, Dict]) -> None:
    """Print a formatted verification report."""
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION REPORT")
    print("=" * 70)
    
    all_valid = True
    total_images = 0
    
    for dataset_name, result in results.items():
        status = "OK" if result["valid"] else "FAIL"
        all_valid = all_valid and result["valid"]
        total_images += result["image_count"]
        
        info = DATASET_INFO[dataset_name]
        expected = info["approx_images"]
        
        print(f"\n{status} {dataset_name} (Client {info['client_id']})")
        print(f"  Path: {result['path']}")
        print(f"  Images: {result['image_count']:,} / ~{expected:,} ({result['completeness']:.1f}%)")
        print(f"  Metadata CSV: {'Found' if result['csv_found'] else 'Missing'}")
        
        if result["missing_files"] and not result["valid"]:
            print(f"  Missing files:")
            for f in result["missing_files"]:
                print(f"    - {f}")
    
    print("\n" + "=" * 70)
    print(f"Total images across all datasets: {total_images:,}")
    if all_valid:
        print("All datasets are properly configured!")
    else:
        print("Some datasets are missing or incomplete.")
        print("  Run: python run_download.py --download-all")
    print("=" * 70)


def print_download_instructions() -> None:
    """Print instructions for downloading datasets."""
    instructions = """
================================================================================
DATASET DOWNLOAD OPTIONS
================================================================================

OPTION 1: Automatic Download via ISIC Archive API (Recommended)
--------------------------------------------------------------------------------
No API key required. Downloads directly from the official ISIC Archive.

    # Download all datasets (~80,000 images, may take several hours)
    python run_download.py --download-all
    
    # Download specific dataset
    python run_download.py --download HAM10000
    python run_download.py --download ISIC2018
    python run_download.py --download ISIC2019
    python run_download.py --download ISIC2020

    # Adjust parallel workers (default: 8)
    python run_download.py --download-all --workers 16

OPTION 2: Manual Download from ISIC Archive Website
--------------------------------------------------------------------------------
Visit the ISIC Archive and download datasets manually:

    HAM10000:  https://isic-archive.com/ → Search "HAM10000"
    ISIC 2018: https://challenge.isic-archive.com/data/#2018
    ISIC 2019: https://challenge.isic-archive.com/data/#2019
    ISIC 2020: https://challenge.isic-archive.com/data/#2020

After downloading, organize files as:
    data/
    ├── HAM10000/
    │   ├── metadata.csv
    │   └── images/
    │       └── *.jpg
    ├── ISIC2018/
    │   ├── metadata.csv
    │   └── images/
    │       └── *.jpg
    ├── ISIC2019/
    │   ├── metadata.csv
    │   └── images/
    │       └── *.jpg
    └── ISIC2020/
        ├── metadata.csv
        └── images/
            └── *.jpg

VERIFICATION
--------------------------------------------------------------------------------
After downloading, verify your datasets:

    python run_download.py --verify

================================================================================
DATASET INFORMATION
================================================================================

| Dataset   | Client | Images  | Classes | Description                    |
|-----------|--------|---------|---------|--------------------------------|
| HAM10000  | 1      | ~10,015 | 7       | Human Against Machine dataset  |
| ISIC 2018 | 2      | ~10,015 | 7       | ISIC 2018 Challenge Task 3     |
| ISIC 2019 | 3      | ~25,331 | 8       | ISIC 2019 Challenge            |
| ISIC 2020 | 4      | ~33,126 | 2       | SIIM-ISIC Melanoma Challenge   |

Total: ~78,487 dermoscopy images

================================================================================
"""
    print(instructions)


# =============================================================================
# Dataset Setup Wizard
# =============================================================================

class DatasetSetupWizard:
    """Interactive wizard for setting up datasets."""
    
    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else get_data_root()
        
    def run(self, auto_download: bool = False) -> None:
        """
        Run the interactive setup wizard.
        
        Args:
            auto_download: If True, automatically download missing datasets
        """
        print("\n" + "=" * 70)
        print("DERMOSCOPY DATASET SETUP WIZARD")
        print("=" * 70)
        
        # Create directory structure
        print("\nStep 1: Creating directory structure...")
        create_directory_structure(self.data_root)
        
        # Verify current state
        print("\nStep 2: Checking existing datasets...")
        results = verify_all_datasets(self.data_root)
        print_verification_report(results)
        
        # Check which datasets are missing
        missing = [name for name, result in results.items() if not result["valid"]]
        
        if not missing:
            print("\nAll datasets are ready! You can proceed with training.")
            return
        
        print(f"\nMissing/incomplete datasets: {', '.join(missing)}")
        
        if auto_download:
            print("\nStep 3: Downloading missing datasets...")
            download_all_datasets(self.data_root, datasets=missing)
        else:
            print_download_instructions()
            print("\nTo download automatically, run:")
            print(f"  python -m src.data.download --download-all")
        
    def quick_verify(self) -> bool:
        """Quick verification of all datasets."""
        results = verify_all_datasets(self.data_root)
        return all(r["valid"] for r in results.values())


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ISIC Archive Dataset Download Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_download.py --verify
  python run_download.py --download-all
  python run_download.py --download HAM10000
  python run_download.py --setup
        """
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for datasets (default: ./data)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing datasets"
    )
    parser.add_argument(
        "--instructions", action="store_true",
        help="Print download instructions"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run interactive setup wizard"
    )
    parser.add_argument(
        "--download", type=str, metavar="DATASET",
        choices=list(DATASET_INFO.keys()),
        help="Download a specific dataset from ISIC Archive"
    )
    parser.add_argument(
        "--download-all", action="store_true",
        help="Download all datasets from ISIC Archive"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel download workers (default: 8)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force redownload even if files exist"
    )
    
    args = parser.parse_args()
    
    # Set data root
    data_root = Path(args.data_root) if args.data_root else None
    
    if args.download:
        # Download specific dataset
        download_dataset_isic(
            args.download,
            data_root=data_root,
            max_workers=args.workers,
            force_redownload=args.force
        )
    elif args.download_all:
        # Download all datasets
        download_all_datasets(
            data_root=data_root,
            max_workers=args.workers
        )
    elif args.verify:
        results = verify_all_datasets(data_root)
        print_verification_report(results)
    elif args.instructions:
        print_download_instructions()
    elif args.setup:
        wizard = DatasetSetupWizard(data_root)
        wizard.run(auto_download=False)
    else:
        # Default: show status and instructions
        results = verify_all_datasets(data_root)
        print_verification_report(results)
        
        missing = [name for name, result in results.items() if not result["valid"]]
        if missing:
            print("\nRun with --download-all to download missing datasets.")
            print("Run with --help for all options.")
