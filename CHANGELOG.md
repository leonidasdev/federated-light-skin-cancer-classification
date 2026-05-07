# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **Notebook 02 Export**: Per-sample predictions, labels, and sample IDs now exported to `results_latest.json` in both single-dataset and all-datasets evaluation paths for exact paired statistical testing in Notebook 03
- **Notebook 03 Statistical Testing**: Exact binomial McNemar test with Bonferroni correction for multiple FL variants (replaces summary confidence interval approach)
- **Documentation**: Expanded Notebooks section in README.md with results JSON schema and export structure details

### Fixed

- Pylance `reportAttributeAccessIssue` errors in test `_MockDS` class
- Stale config filenames in README.md, thesis.md, CLAUDE.md, and run_experiment.py docstring
- TODO.md audit: updated dates, config filenames, experiment statuses
- **Notebook 02 Export Cell**: Fixed execution order dependency where `confidence` and `roc_auc` were undefined; both export cells now self-contained

## [2026-03-12]

### Fixed

- **Critical**: `DirichletSubset.__getitem__` referenced `img_paths` instead of `image_paths`, breaking validation transforms and causing raw PIL images to reach the model
- Renamed all federated config files from `dscatnet_federated_{dataset}.yaml` to `dscatnet_federated_{dataset}_non_iid.yaml` for clarity

## [2026-03-11]

### Changed

- CI/CD test coverage threshold raised from 36% to 80%
- Ruff format applied to all 48 Python files
- Updated `num_classes` in `server.py` to match config-driven value

### Fixed

- Checkpoint resume bug: `best_checkpoint.pt` was not saved in federated mode
- MetricsTracker CSV logging: duplicate headers on resume
- Logging configuration: duplicate log handlers on module reload

### Added

- 5 new tests for checkpoint, logging, and resume edge cases
- Test count: 457 passing (80.12% coverage)

## [2026-03-09]

### Changed

- Default model variant switched to `paper` (H=12 attention heads)
- Batch size reduced from 8 to 4 with gradient accumulation increased from 4 to 8 (for 4GB VRAM GPUs)
- Pylance errors fixed across codebase, constants extracted, imports cleaned
- 10 new ruff lint errors resolved

### Added

- IID federated configs (`dscatnet_federated_{dataset}_iid.yaml`) with Dirichlet alpha=1000.0
- All defaults aligned to DSCATNet paper parameters

## [2026-03-08]

### Added

- Pretrained ViT-Small (ImageNet) weight loading into DSCATNet
- Class weights support in FL training for fair benchmark comparison

### Changed

- Modernized type annotations (PEP 604 unions, PEP 585 generics)
- Replaced `DATASET_REGISTRY` global init with direct module-level definition
- Promoted lazy imports to module level
- Removed dead code: `get_default_config()`, unused imports, duplicated simulation logic
- Added `__all__` to evaluation modules
- Fixed FL `drop_last` in data loaders
- Expanded ruff lint rules and fixed all remaining issues

## [2026-03-05 – 2026-03-06]

### Fixed

- **Critical**: Multiple FL bugs (server config, aggregation, client evaluation)
- Config/doc consistency audit: corrected param counts, thesis wording, stale refs
- Broken YAML in `centralized_original` config
- Checkpoint sorting by name instead of mtime for CI stability
- `warmup_epochs` added to `CentralizedConfig`
- Standardized progress bars across centralized and federated training

### Added

- Centralized and federated configs for all 5 datasets
- Reproducibility: `set_seed(42)` added to all training entry points
- FL `save_best_model` now saves full checkpoint for resume support
- Seeded Dirichlet train/val split for reproducible checkpoint resume

### Changed

- Standardized docstrings, config comments, and progress bar output
- Documentation overhaul, code style fixes, lint clean
- Test coverage raised from 28% to 36%

## [2026-01-24 – 2026-01-29]

### Added

- PAD-UFES-20 dataset support
- ISIC Archive API downloader (`run_download.py`)
- Training progress bars (tqdm)
- Evaluation notebook (`02_model_evaluation.ipynb`)
- Dataset registry modularization

### Changed

- Extracted training logic into `centralized/` module
- Formatting consistency pass
- Cleaned unnecessary imports

## [2025-12-26 – 2026-01-24]

### Added

- Initial project structure with DSCATNet model implementation
- Flower-based federated learning client and server
- Federated learning simulation (`FLSimulator`)
- Dataset classes: HAM10000, ISIC2018, ISIC2019, ISIC2020
- Centralized training baseline
- Evaluation metrics and visualization
- Configuration system (YAML-based)
- Jupyter notebooks for exploration and comparison
- CLI entry point (`run_experiment.py`)
- Unit test suite
- CI/CD with GitHub Actions
- Apache 2.0 License
