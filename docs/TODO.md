# Project TODO — Remaining Work Items

> Comprehensive audit of the codebase as of 2026-03-09.
> All items are prioritized: **P0** (blocking), **P1** (important), **P2** (nice-to-have).

---

## 1. Training & Experiments

### P0 — Must-Have Experiments

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Complete centralized baseline on HAM10000 | In Progress | Training resumed from epoch 4 (val_acc=0.6545), currently running. 200 epochs, early stopping patience=15. |
| 1.2 | Run centralized baseline on PAD-UFES-20 | Not Started | Use `dscatnet_centralized_original.yaml` (already configured). |
| 1.3 | Run federated HAM10000 (Dirichlet α=0.5, 4 clients) | Not Started | Use `dscatnet_federated_ham10000.yaml`. |
| 1.4 | Run federated PAD-UFES-20 (Dirichlet α=0.5, 4 clients) | Not Started | Use `dscatnet_federated_padufes20.yaml`. |
| 1.5 | Run final evaluation on best checkpoints | Not Started | Use `--mode evaluate` on each best_checkpoint.pt. |
| 1.6 | Ablation: vary Dirichlet α (0.1, 0.5, 1.0, 10.0) | Not Started | Needed for non-IID analysis in thesis. |
| 1.7 | Ablation: vary number of FL clients (2, 4, 8) | Not Started | Communication cost scales with clients. |
| 1.8 | Run centralized on ISIC2018 (same images as HAM10000) | Optional | Validates label consistency across datasets. |

### P1 — Training Improvements

| # | Task | Notes |
|---|------|-------|
| 1.9 | Investigate declining val_acc (0.6545 → 0.3948 at epoch 6) | ✅ Root cause identified: (1) Class weights created volatile loss landscape, (2) Early stopping at patience=15 killed training prematurely, (3) Gradient clipping (not in paper) may have interfered. Fixed: `use_class_weights: false`, `early_stopping_patience: 200`, `max_grad_norm: null`. |
| 1.10 | Add TensorBoard logging to training loops | ✅ Done. `MetricsTracker` (CSV) and `TensorBoardLogger` wired into both `CentralizedTrainer` and `FLSimulator`. |
| 1.11 | Compute communication cost in FL (total MB transferred) | ✅ Already implemented. `communication_cost_mb` per round + `total_communication_mb` in results. |
| 1.12 | Add test-set evaluation after training completes | ✅ Done. `test_split` config option + `ModelEvaluator` runs on held-out set at end of `run()`. Results saved as `test_metrics` in results.json. |

---

## 2. Model Architecture — Paper Alignment

### Verification Against PONE Paper (Yadav et al., 2024)

| Aspect | Paper | Implementation | Status |
|--------|-------|---------------|--------|
| Image size | 224 × 224 | 224 × 224 | ✅ |
| Fine patch size | 8 × 8 | 8 × 8 | ✅ |
| Coarse patch size | 16 × 16 | 16 × 16 | ✅ |
| Embedding dim (small) | D=192 (fine) / D=768 (coarse) | 384 (unified, adapted for ViT-Small pretrained) | ⚠️ Paper uses asymmetric dims; unified 384 is a practical adaptation for pretrained ViT-Small. |
| Depth (small) | 6 | 6 | ✅ |
| Attention heads | H=12 | 12 (`paper` variant) / 6 (`small` variant) | ✅ Fixed. Use `variant: paper` for paper-faithful 12 heads. |
| MLP ratio | 4.0 | 4.0 | ✅ |
| CLS tokens | Per-scale learnable | Per-scale learnable | ✅ |
| Positional embedding | Learnable | Learnable | ✅ |
| Patch embedding | Conv2d projection | Conv2d projection | ✅ |
| Layer norm | Pre-norm (each sub-layer) | Pre-norm (each sub-layer) | ✅ |
| Cross-attention | Bidirectional fine↔coarse | Bidirectional fine↔coarse | ✅ |
| Self-attention per scale | nn.MultiheadAttention | nn.MultiheadAttention | ✅ |
| Fusion | Concatenate CLS tokens → Linear | Concatenate CLS tokens → Linear | ✅ |
| Classifier | LayerNorm → Dropout → Linear | LayerNorm → Dropout → Linear | ✅ |
| Dropout rate | 0.1 | 0.1 | ✅ |
| Weight init | Truncated normal (std=0.02) | Truncated normal (std=0.02) | ✅ |
| Optimizer | Adam | Adam | ✅ |
| Learning rate | 1e-3 | 1e-3 | ✅ |
| Weight decay | 0.0 | 0.0 | ✅ |
| Batch size | 32 | 8 × 4 grad accum = 32 | ✅ |
| Epochs | 200 | 200 | ✅ |
| Scheduler | None (fixed LR) | None | ✅ |
| Loss | CrossEntropy | CrossEntropy (standard, no class weights) | ✅ Fixed. `use_class_weights: false` in all configs. |
| Augmentation (HAM10000) | None | None | ✅ |
| Augmentation (PAD-UFES-20) | H/V flips (5× oversample) | `level: light` (H/V flips + small rotation) | ✅ Fixed. |
| Gradient clipping | Not mentioned | Configurable via `max_grad_norm` (null = disabled) | ✅ Fixed. |
| Early stopping | Not mentioned (trains 200 epochs) | `patience: 200` (effectively disabled) | ✅ Fixed. |
| Normalization | ImageNet stats | ImageNet stats (default) | ✅ |
| 5-fold cross-validation | Yes | Not implemented (single split) | ⚠️ Paper averages over 5 folds; we use single 85/15 split. Results not directly comparable. |
| Model params (small) | ~22M (paper) | 29.4M | ⚠️ Discrepancy due to separate self-attention per scale + cross-attention Q/K/V projections. Documented in DSCATNet docstring. |

### P1 — Architecture Items

| # | Task | Notes |
|---|------|-------|
| 2.1 | Document parameter count discrepancy vs paper | ✅ Done. Note added to DSCATNet docstring explaining 29.4M vs 22M (separate Q/K/V projections, independent self-attention per scale). |
| 2.2 | Consider adding unweighted CE config option | ✅ Done. `use_class_weights: false` is now the default in all paper-aligned configs. Class weights remain available as a config option for practical/clinical use. |
| 2.3 | Verify ViT weight transfer correctness | ✅ Documented in `load_pretrained_vit_weights()` docstring and log output. 150/286 tensors transferred. Cross-attention, fine-scale embeddings, fusion, classifier randomly initialized. |
| 2.4 | Add `paper` model variant with H=12 heads | ✅ Done. `variant: paper` uses embed_dim=384, depth=6, num_heads=12, mlp_ratio=4.0. Compatible with ViT-Small pretrained weights. |
| 2.5 | Make gradient clipping configurable | ✅ Done. `max_grad_norm` field added to CentralizedConfig/SimulationConfig. `null` (disabled) in paper-aligned configs. |
| 2.6 | Add `paper` to CLI `--model-variant` choices | ✅ Done. `argparse` choices now include `paper`. |

---

## 3. Code Quality

### P2 — Remaining Lint / Style

| # | Task | Notes |
|---|------|-------|
| 3.1 | 7 pre-existing E501 (line too long) violations | 4 in `run_experiment.py`, 3 in `simulation.py`. Ignored by ruff config (`E501` in ignore list). Harmless but could be wrapped. |
| 3.2 | Large file sizes | `simulation.py` ~900 lines, `download.py` ~1200 lines, `datasets.py` ~700 lines. Could be split but risky during active experiments. |

### P1 — Bugs Fixed This Audit

| # | Bug | Fix | Location |
|---|-----|-----|----------|
| 3.11 | `_FEDERATED_SECTIONS` missing 7 training params (batch_size, lr, weight_decay, optimizer, gradient_accumulation_steps, num_classes, pretrained) — silently ignored from YAML | ✅ Added complete mappings | `run_experiment.py` |
| 3.12 | `compute_class_weights()` division by zero when a class has 0 samples | ✅ Added `count > 0` guard | `src/utils/helpers.py` |
| 3.13 | `setup_natural_noniid()` loading images to compute class distribution (O(n) image loads) | ✅ Uses `full_dataset.labels[idx]` directly | `src/federated/simulation.py` |
| 3.14 | `_CENTRALIZED_SECTIONS` missing `pretrained` mapping from model config | ✅ Added mapping | `run_experiment.py` |

### P2 — Dead / Unused Code

| # | Item | Location | Status |
|---|------|----------|--------|
| 3.3 | `CheckpointManager` | `src/utils/checkpoints.py` | Only used in tests, not by actual trainers. Missing scaler state, history, config, best tracking. Low priority—keep for potential future use. |
| 3.4 | `DatasetVerifier` | `src/data/verify.py` | Only used in notebook `01_dataset_exploration.ipynb`, not from CLI. Coverage: 6%. Keep for interactive use. |
| 3.5 | `CLASS_NAMES_8`, `CLASS_NAMES_BINARY` | `src/data/datasets.py` | Used internally for 8-class and binary modes. Not dead—just not used in default 7-class mode. |
| 3.6 | `validate_config()`, `validate_config_dict()` | `src/utils/config_schema.py` | Only reachable via `--validate-config` CLI flag. Not integrated into normal config loading. Could be integrated but risk of breaking existing configs. |
| 3.7 | `ExperimentLogger`, `MetricsTracker` | `src/utils/logging_utils.py` | Coverage: 17%. Not wired into training loops. See item 1.10. |

### P2 — Duplication Assessment

| # | Pattern | Verdict |
|---|---------|---------|
| 3.8 | Per-class accuracy tracking (centralized.py vs client.py) | ~5 lines each, different input types (`.cpu().numpy()` vs `.item()`). Not worth extracting. |
| 3.9 | `from_dict()` classmethod (CentralizedConfig, SimulationConfig) | Identical 1-liner in 2 dataclasses. Too minor to extract. |
| 3.10 | Checkpoint save/load (CentralizedTrainer vs FLSimulator) | Different state shapes (epochs vs rounds, history format). Separate implementations are appropriate. |

---

## 4. Testing

### Current State

- **453 tests passing**, `@slow` tests deselected by default
- **80% line coverage** (`fail_under = 80` in pyproject.toml)
- **Ruff clean** with 12 rule categories
- **Integration tests:** `test_integration.py` covering centralized + FL + env info + test-split eval

### Coverage

Coverage increased from 41% to 80% via comprehensive mocked tests across all core modules.

### P1 — Testing Items

| # | Task | Notes |
|---|------|-------|
| 4.1 | Add integration test for full centralized training (tiny model, synthetic data, 2 epochs) | ✅ Done. `test_integration.py::TestCentralizedIntegration`. |
| 4.2 | Add integration test for FL simulation (tiny model, 2 rounds, 2 clients) | ✅ Done. `test_integration.py::TestFederatedIntegration`. |
| 4.3 | Increase coverage threshold to 80% | ✅ Done. `fail_under = 80` in pyproject.toml. 453 tests. |
| 4.4 | Test checkpoint resume behavior (save → load → continue) | ✅ Covered by integration tests. |

---

## 5. CI/CD

### Current State

- **GitHub Actions** on push/PR to `main`
- **Matrix**: Python 3.10, 3.11, 3.12, 3.13 on `ubuntu-latest`
- **Steps**: Install deps → Ruff lint → Pytest → Coverage threshold (80%)

### P2 — CI Improvements

| # | Task | Notes |
|---|------|-------|
| 5.1 | Add `ruff format --check` to CI | Currently only lints, doesn't check formatting. |
| 5.2 | Add type checking with mypy or pyright | Type hints exist everywhere but aren't verified. |
| 5.3 | Pin dependency versions in CI | `requirements.txt` uses `>=` ranges. CI should pin for reproducibility. Consider `pip-compile` or `uv lock`. |
| 5.4 | Add artifact upload for coverage reports | Currently coverage is only printed, not stored. |

---

## 6. Documentation

### Current State

- **README.md**: Comprehensive (150+ lines), badges, sections, examples
- **CLAUDE.md**: AI assistant context, architecture, conventions
- **config-options-guide.md**: YAML configuration reference
- **architecture.md**: System design overview
- **benchmark-comparison.md**: FL vs centralized fairness audit
- **Docstrings**: Google-style throughout, comprehensive on public APIs
- **Code comments**: Section headers with `# ===` separators

### P1 — Documentation Items

| # | Task | Notes |
|---|------|-------|
| 6.1 | Write thesis document | The main deliverable. See `docs/thesis.md`. |
| 6.2 | Update README results section with final experiment numbers | Currently no results reported. |
| 6.3 | Add CHANGELOG.md | 25+ commits with significant changes. No changelog tracking. |

### P2 — Documentation Nice-to-Haves

| # | Task | Notes |
|---|------|-------|
| 6.4 | Add API reference (pdoc or sphinx) | Low priority for a thesis project. |
| 6.5 | Add architecture diagrams (Mermaid/draw.io) | Text diagrams exist in CLAUDE.md. Visual diagrams would be valuable for thesis. |

---

## 7. Reproducibility & Research Rigor

### P0 — Critical for Thesis

| # | Task | Notes |
|---|------|-------|
| 7.1 | Fix seed for all random sources | `set_seed(42)` covers torch, numpy, random, cuDNN. ✅ |
| 7.2 | Log all hyperparameters to config.json | ✅ Already done. |
| 7.3 | Deterministic train/val splits | ✅ `torch.Generator` with fixed seed. |
| 7.4 | Save model architecture to checkpoint | ✅ Config dict saved in checkpoint. |
| 7.5 | Record hardware/software environment | ✅ Done. `collect_environment_info()` in helpers.py, wired into both CentralizedTrainer and FLSimulator results. |
| 7.6 | Multiple random seeds for statistical significance | Not done. Should run key experiments with seeds {42, 123, 456} and report mean ± std. |
| 7.7 | Compute statistical tests (paired t-test or Wilcoxon) | Needed to claim FL vs centralized differences are significant. |

---

## 8. Summary & Priority Order

### Immediate (before writing thesis)

1. **1.1**: Finish HAM10000 centralized training (in progress)
2. **1.2**: Run PAD-UFES-20 centralized baseline
3. **1.3–1.4**: Run FL experiments on both datasets
4. **1.5**: Final evaluation on all best checkpoints
5. **1.6**: Dirichlet α ablation
6. **6.1**: Write thesis

### Important (strengthen thesis quality)

7. ~~**2.1**: Document param count discrepancy~~ ✅
8. ~~**1.11**: Add communication cost metric~~ ✅
9. ~~**7.5**: Log environment info~~ ✅
10. **7.6**: Multiple seeds for significance
11. ~~**4.1–4.2**: Integration tests~~ ✅

### Also completed (prior sessions + audit)

- ~~**1.12**: Test-set evaluation support~~ ✅
- ~~**4.3**: Coverage raised to 80% (453 tests)~~ ✅
- ~~**4.4**: Checkpoint resume tests~~ ✅
- ~~**2.4**: `paper` variant with H=12 heads~~ ✅
- ~~**2.5**: Configurable gradient clipping (`max_grad_norm`)~~ ✅
- ~~**2.6**: `paper` added to CLI `--model-variant` choices~~ ✅
- **Bug fix**: `img_size` not passed to `create_dscatnet()` in 4 callsites (centralized + FL)
- **Bug fix**: `_FEDERATED_SECTIONS` missing 7 YAML training param mappings (3.11)
- **Bug fix**: `compute_class_weights()` division by zero (3.12)
- **Bug fix**: `setup_natural_noniid()` O(n) image loads for class distribution (3.13)
- **Bug fix**: `_CENTRALIZED_SECTIONS` missing `pretrained` mapping (3.14)

### Completed (2026-03-09 comprehensive audit)

- **Bug fix**: `set_model_parameters()` now places tensors on correct device (GPU-safe) ✅
- **Feature**: Per-class F1 score added to `ModelEvaluator.evaluate()` output ✅
- **Bug fix**: `verify_isic2020()` now checks correct CSV name (`ISIC_2020_Training_GroundTruth.csv`) ✅
- **Feature**: Added `verify_padufes20()` method and PAD-UFES-20 to `verify_all()` ✅
- **Docs**: Added `paper` variant (H=12) to `architecture.md` variants table ✅
- **Config**: `experiment_config.yaml` updated to paper-aligned values (epochs=200, local_epochs=1, use_class_weights=false, gradient_accumulation_steps=4, early_stopping_patience=200) ✅
- **Config**: `fl_config.yaml` updated (local_epochs=1, added PAD-UFES-20 as Client 5, num_clients=5) ✅
- **Config**: `experiment_template.yaml` updated to paper-aligned defaults ✅
- **Deps**: `requirements.txt` Flower version tightened to `>=1.25.0` (matches installed) ✅
- **Build**: Added `[build-system]` section to `pyproject.toml` ✅

### Nice-to-have (polish)

12. **5.1–5.4**: CI improvements
13. **3.1–3.2**: Code style cleanup
14. **6.3**: CHANGELOG
15. **6.5**: Architecture diagrams
