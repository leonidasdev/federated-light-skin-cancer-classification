# TODO: Scripts Review and Download Functionality Analysis

> **Created**: January 30, 2026  
> **Status**: Completed  
> **Objective**: Evaluate entry point scripts, review download functionality, and ensure proper API usage

---

## Overview

This document tracks the analysis and improvements for the project's entry point scripts and download functionality.

---

## 1. Entry Point Scripts Evaluation

### Current Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `run_experiment.py` | Main entry point for centralized, federated, evaluate, compare modes | ✅ Essential |
| `run_download.py` | Dataset download wrapper | ✅ Improved |
| `run_tests.py` | Test suite runner | ✅ Essential |

> **Note**: `run_fl.py` was removed on January 30, 2026 due to redundancy with `run_experiment.py --mode federated`.

### 1.1 `run_experiment.py` Analysis

**Status**: ✅ Essential - Keep

**Functionality**:
- Unified CLI for all experiment modes: `centralized`, `federated`, `evaluate`, `compare`
- Full YAML config support with CLI overrides
- Resume training from checkpoints
- Extensive argument parsing

**Verdict**: This is the primary entry point and should be kept.

---

### 1.2 `run_fl.py` Analysis

**Status**: ✅ REMOVED - Redundant

**Reason for Removal**:
The `run_fl.py` script was redundant because all its functionality was already available in `run_experiment.py --mode federated`. The comparison below shows why it was unnecessary:

| Feature | `run_fl.py` (removed) | `run_experiment.py` |
|---------|----------------------|---------------------|
| FL Training | ✅ | ✅ |
| Config File Support | ❌ | ✅ |
| Resume Training | ✅ (limited) | ✅ (full) |
| Centralized Mode | ❌ | ✅ |
| Evaluate Mode | ❌ | ✅ |
| Model Variant Selection | ❌ | ✅ |
| Non-IID Type Selection | ❌ | ✅ |
| Logging to File | ❌ | ✅ |

**Action Taken**: 
- [x] Removed `run_fl.py` file
- [x] Updated all documentation to remove references
- [x] Updated project structure in README.md
- [x] Updated architecture diagrams in ARCHITECTURE.md and CLAUDE.md
- [x] Removed `mock_fl_parser` fixture and related tests from test_cli.py
- [x] Updated notebooks to use `run_experiment.py`

---

### 1.3 `run_download.py` Analysis

**Status**: ⚠️ INCOMPLETE - Needs Major Revision

**Current State**:
- Only supports ISIC Archive API downloads
- **Missing**: Kaggle API for HAM10000
- **Missing**: Mendeley direct download for PAD-UFES-20
- No PAD-UFES-20 download support at all

**Issues Found**:
1. HAM10000 on Kaggle is different from ISIC API HAM10000 (Kaggle has CSV variants)
2. PAD-UFES-20 not in `DATASET_INFO` in download.py
3. No automated data organization after download

---

## 2. Download Script Deep Analysis

### 2.1 Current `src/data/download.py` Coverage

| Dataset | API Support | Current Status |
|---------|-------------|----------------|
| HAM10000 | ISIC Archive | ⚠️ API works but Kaggle version is standard |
| ISIC2018 | ISIC Archive | ✅ Implemented |
| ISIC2019 | ISIC Archive | ✅ Implemented |
| ISIC2020 | ISIC Archive | ✅ Implemented |
| PAD-UFES-20 | Mendeley | ❌ NOT IMPLEMENTED |

### 2.2 Recommended Download Sources

#### HAM10000 - Kaggle (Preferred)
- **URL**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **API**: Kaggle API (`kaggle datasets download`)
- **Contains**:
  - `HAM10000_metadata.csv` - Main metadata
  - `hmnist_*.csv` - Pre-processed variants (not needed)
  - `HAM10000_images_part_1/` - Images
  - `HAM10000_images_part_2/` - Images
- **Size**: ~2.5GB

#### PAD-UFES-20 - Mendeley Data
- **URL**: https://data.mendeley.com/datasets/zr7vgbcyr2/1
- **Direct Download**: https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/files/download
- **Contains**:
  - `metadata.csv`
  - `imgs_part_1/`, `imgs_part_2/`, `imgs_part_3/` - PNG images
- **Size**: ~1.2GB

#### ISIC Datasets - ISIC Archive API
- Current implementation looks correct
- API endpoint: `https://api.isic-archive.com/api/v2`

---

## 3. Tasks Checklist

### Phase 1: Analysis & Documentation
- [x] Read project documentation
- [x] Analyze `run_experiment.py` functionality
- [x] Analyze `run_fl.py` redundancy
- [x] Review `src/data/download.py` implementation
- [x] Document findings in this TODO file

### Phase 2: Download Script Improvements
- [x] Add Kaggle API support for HAM10000 download
- [x] Add Mendeley direct download for PAD-UFES-20
- [x] Implement automatic data organization after download
- [x] Handle different folder structures between sources
- [x] Add progress tracking for large downloads
- [x] Update `DATASET_INFO` with all 5 datasets
- [x] Update verification functions for new structures
- [x] Update CLI with new options

### Phase 3: Entry Point Scripts
- [x] Remove redundant `run_fl.py`
- [x] Update all documentation to remove references
- [x] Update architecture diagrams
- [x] Remove related tests

### Phase 4: Testing
- [x] Add tests for Kaggle download functionality
- [x] Add tests for Mendeley download functionality
- [x] Add tests for data organization logic
- [x] Verify existing download tests still pass
- [x] Run full test suite

---

## 4. Implementation Plan

### 4.1 Kaggle HAM10000 Download

```python
# Pseudocode for Kaggle download
def download_ham10000_kaggle(data_root: Path) -> bool:
    """
    Download HAM10000 from Kaggle using kaggle API.
    
    Requires: 
    - kaggle package installed
    - ~/.kaggle/kaggle.json with API credentials
    """
    # 1. Check kaggle CLI available
    # 2. Download dataset: kaggle datasets download kmader/skin-cancer-mnist-ham10000
    # 3. Extract to data_root/HAM10000/
    # 4. Reorganize files to expected structure
    pass
```

### 4.2 Mendeley PAD-UFES-20 Download

```python
# Pseudocode for Mendeley download
def download_padufes20_mendeley(data_root: Path) -> bool:
    """
    Download PAD-UFES-20 from Mendeley Data.
    
    Direct download URL (no API key required):
    https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/files/download
    """
    # 1. Download ZIP file from Mendeley
    # 2. Extract to data_root/PAD-UFES-20/
    # 3. Verify structure matches expected layout
    pass
```

### 4.3 Expected Data Structure After Download

```
data/
├── HAM10000/
│   ├── HAM10000_metadata.csv          # From Kaggle
│   ├── HAM10000_images_part_1/        # From Kaggle
│   │   └── *.jpg
│   └── HAM10000_images_part_2/        # From Kaggle
│       └── *.jpg
│
├── ISIC2018/
│   ├── ISIC2018_Task3_Training_GroundTruth.csv  # From ISIC or manual
│   └── ISIC2018_Task3_Training_Input/           # From ISIC or manual
│       └── *.jpg
│
├── ISIC2019/
│   ├── ISIC_2019_Training_GroundTruth.csv
│   └── ISIC_2019_Training_Input/
│       └── *.jpg
│
├── ISIC2020/
│   ├── ISIC_2020_Training_GroundTruth.csv
│   └── ISIC_2020_Training_JPEG/
│       └── *.jpg
│
└── PAD-UFES-20/
    ├── metadata.csv                   # From Mendeley
    ├── imgs_part_1/                   # From Mendeley
    │   └── *.png
    ├── imgs_part_2/
    │   └── *.png
    └── imgs_part_3/
        └── *.png
```

---

## 5. Questions to Resolve

1. **Kaggle vs ISIC HAM10000**: Should we support both sources or prefer one?
   - **Recommendation**: Prefer Kaggle (faster, includes standard CSVs)
   
2. **ISIC API Rate Limits**: Are there rate limits that affect downloads?
   - Current implementation includes retry logic and delays
   
3. **Minimum dataset requirements**: Should downloads fail if <90% complete?
   - Current: 90% threshold for "valid" dataset

---

## 6. Related Files to Modify

| File | Changes Needed |
|------|----------------|
| `src/data/download.py` | Add Kaggle, Mendeley support |
| `run_download.py` | Add new CLI options |
| `tests/test_download.py` | Create new test file |
| `docs/README.md` | Update download instructions |
| `docs/CLAUDE.md` | Update architecture info |

---

## 7. Progress Log

| Date | Action | Status |
|------|--------|--------|
| 2026-01-30 | Initial analysis completed | ✅ |
| 2026-01-30 | TODO document created | ✅ |
| 2026-01-30 | Implement Kaggle download for HAM10000 | ✅ |
| 2026-01-30 | Implement Mendeley download for PAD-UFES-20 | ✅ |
| 2026-01-30 | Add unified download function | ✅ |
| 2026-01-30 | Update DATASET_INFO with all 5 datasets | ✅ |
| 2026-01-30 | Update verification functions | ✅ |
| 2026-01-30 | Update CLI with new options | ✅ |
| 2026-01-30 | Add tests for download functionality (31 tests) | ✅ |
| 2026-01-30 | Run full test suite (133 passed) | ✅ |
| 2026-01-30 | Remove redundant run_fl.py | ✅ |
| 2026-01-30 | Update all documentation | ✅ |

---

## 8. Summary of Changes

### Files Modified

1. **`src/data/download.py`** - Major updates:
   - Added PAD-UFES-20 to `DATASET_INFO`
   - Changed HAM10000 source to "kaggle" (preferred)
   - Updated expected files structure for all datasets
   - Added `check_kaggle_available()` function
   - Added `download_ham10000_kaggle()` function
   - Added `download_padufes20_mendeley()` function
   - Added `download_dataset()` unified function
   - Updated `verify_dataset()` for new folder structures
   - Updated `download_all_datasets()` to use unified function
   - Updated CLI with `--source` option
   - Updated download instructions

2. **`tests/test_download.py`** - New test file:
   - 31 comprehensive tests for download functionality
   - Tests for DATASET_INFO structure
   - Tests for Kaggle and Mendeley download logic
   - Tests for verification functions
   - Tests for ISIC Archive client

3. **Removed `run_fl.py`** - Redundant script:
   - Deleted the file
   - Removed all references from documentation
   - Removed related test fixtures and tests

### Files Updated to Remove run_fl.py References

- `README.md` - Project structure and CLI section
- `docs/ARCHITECTURE.md` - Entry points diagram
- `docs/CLAUDE.md` - Entry points diagram
- `docs/CONFIG_OPTIONS.md` - Federated config usage note
- `docs/CONTRIBUTING.md` - Project structure
- `tests/test_cli.py` - Removed `mock_fl_parser` fixture and 3 tests
- `notebooks/01_dataset_exploration.ipynb` - Next steps section
- `notebooks/03_fl_vs_centralized_comparison.ipynb` - Prerequisites check

### Key Features Added

1. **Kaggle Support for HAM10000**:
   - Automatic download via Kaggle API
   - Falls back to ISIC Archive if Kaggle not configured
   - Proper file organization after download

2. **Mendeley Support for PAD-UFES-20**:
   - Direct download from Mendeley Data API
   - No authentication required
   - Automatic extraction and organization

3. **Unified Download Function**:
   - `download_dataset()` auto-selects best source
   - Consistent interface for all datasets
   - Source override option via `--source` flag

### Conclusion

**`run_fl.py` was REMOVED** because it was redundant:
- All functionality is available in `run_experiment.py --mode federated`
- Missing key features: config file support, logging, checkpoint management

**`run_experiment.py`** is the essential unified entry point.

**`run_download.py`** is now fully functional for all 5 datasets.

**`run_tests.py`** is necessary for test execution.

