# Project TODO — Remaining Work Items

> Last updated: 2026-03-09.
> All items are prioritized: **P0** (blocking), **P1** (important), **P2** (nice-to-have).

---

## 1. Training & Experiments

### P0 — Must-Have Experiments

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Complete centralized baseline on HAM10000 | In Progress | 200 epochs, `variant: paper`, `dscatnet_centralized_ham10000.yaml`. |
| 1.2 | Run centralized baseline on PAD-UFES-20 | Not Started | Use `dscatnet_centralized_padufes20.yaml`. |
| 1.3 | Run federated HAM10000 non-IID (Dirichlet alpha=0.5) | Not Started | Use `dscatnet_federated_ham10000.yaml`. |
| 1.4 | Run federated HAM10000 IID (Dirichlet alpha=1000.0) | Not Started | Use `dscatnet_federated_ham10000_iid.yaml`. |
| 1.5 | Run federated PAD-UFES-20 non-IID (Dirichlet alpha=0.5) | Not Started | Use `dscatnet_federated_padufes20.yaml`. |
| 1.6 | Run federated PAD-UFES-20 IID (Dirichlet alpha=1000.0) | Not Started | Use `dscatnet_federated_padufes20_iid.yaml`. |
| 1.7 | Run final evaluation on best checkpoints | Not Started | Use `--mode evaluate` on each `best_checkpoint.pt`. |
| 1.8 | Ablation: vary Dirichlet alpha (0.1, 0.5, 1.0, 10.0) | Not Started | Quantifies non-IID sensitivity. |
| 1.9 | Ablation: vary number of FL clients (2, 4, 8) | Not Started | Communication cost scales with clients. |
| 1.10 | Run centralized on ISIC2018 | Optional | Validates label consistency (same images as HAM10000). |

---

## 2. Model Architecture

### Paper Alignment (Yadav et al., PLOS ONE, 2024)

| Aspect | Paper | Implementation | Status |
|--------|-------|---------------|--------|
| Image size | 224 x 224 | 224 x 224 | Done |
| Fine / coarse patch | 8x8 / 16x16 | 8x8 / 16x16 | Done |
| Embedding dim | D=192/768 (asymmetric) | 384 (unified, ViT-Small pretrained) | Adaptation documented |
| Depth | 6 | 6 | Done |
| Attention heads | H=12 | 12 (`paper` variant) | Done |
| MLP ratio | 4.0 | 4.0 | Done |
| Dropout | 0.1 | 0.1 | Done |
| Weight init | Truncated normal (std=0.02) | Truncated normal (std=0.02) | Done |
| Optimizer / LR / WD | Adam / 1e-3 / 0.0 | Adam / 1e-3 / 0.0 | Done |
| Batch size | 32 | 8 x 4 grad accum = 32 | Done |
| Epochs | 200 | 200 | Done |
| Loss | CrossEntropy | CrossEntropy (unweighted) | Done |
| Augmentation (HAM10000) | None | None | Done |
| Augmentation (PAD-UFES-20) | H/V flips | `level: light` | Done |
| 5-fold CV | Yes | Single 85/15 split | Not implemented (documented limitation) |
| Model params | ~22M | 29.4M | Discrepancy documented in DSCATNet docstring |

---

## 3. Code Quality

### P2 — Remaining Items

| # | Task | Notes |
|---|------|-------|
| 3.1 | E501 (line too long) in `run_experiment.py`, `simulation.py` | Suppressed by ruff config. Cosmetic. |
| 3.2 | Large file sizes (`simulation.py` ~900L, `download.py` ~1200L) | Splitting is risky during active experiments. |
| 3.3 | `CheckpointManager` unused by trainers | `src/utils/checkpoints.py` — keep for potential future use. |
| 3.4 | `DatasetVerifier` only used in notebook | `src/data/verify.py` — keep for interactive use. |
| 3.5 | `logging_utils.py` CSV header rewrite logic is fragile | Rebuilds entire file when columns change. Refactor to maintain column schema upfront. |
| 3.6 | `server.py` hardcodes `num_classes=7` | Make it a parameter with default. |
| 3.7 | `simulation.py` uses magic `42` as seed offset | Extract to `RANDOM_SEED_BASE` constant. |
| 3.8 | Expand dataclass docstrings (`CentralizedConfig`, `SimulationConfig`) | Document individual fields and their purposes. |

---

## 4. CI/CD

### P2 — Improvements

| # | Task | Notes |
|---|------|-------|
| 4.1 | Add `ruff format --check` to CI | Currently only lints. |
| 4.2 | Add type checking (mypy or pyright) | Type hints exist but are not verified. |
| 4.3 | Pin dependency versions in CI | `requirements.txt` uses `>=` ranges. |
| 4.4 | Add artifact upload for coverage reports | Currently only printed. |

---

## 5. Documentation

### P1

| # | Task | Notes |
|---|------|-------|
| 5.1 | Write thesis document | Main deliverable. See `docs/thesis.md`. |
| 5.2 | Update README with final experiment results | No results reported yet. |

### P2

| # | Task | Notes |
|---|------|-------|
| 5.3 | Add CHANGELOG.md | No changelog tracking. |
| 5.4 | Add architecture diagrams (Mermaid/draw.io) | Text diagrams exist in CLAUDE.md. |

---

## 6. Reproducibility

### P0

| # | Task | Notes |
|---|------|-------|
| 6.1 | Multiple random seeds (42, 123, 456) | Report mean +/- std for key experiments. |
| 6.2 | Statistical tests (paired t-test or Wilcoxon) | Required to claim FL vs centralized differences are significant. |

---

## 7. Priority Order

### Immediate (before writing thesis)

1. **1.1**: Finish HAM10000 centralized training (in progress)
2. **1.2**: Run PAD-UFES-20 centralized baseline
3. **1.3–1.6**: Run FL experiments (IID + non-IID) on both datasets
4. **1.7**: Final evaluation on all best checkpoints
5. **1.8**: Dirichlet alpha ablation
6. **5.1**: Write thesis

### Important (strengthen thesis quality)

7. **6.1**: Multiple seeds for statistical significance
8. **6.2**: Statistical tests
9. **1.9**: Client count ablation

### Nice-to-have (polish)

10. **4.1–4.4**: CI improvements
11. **3.1–3.2**: Code style cleanup
12. **5.3**: CHANGELOG
13. **5.4**: Architecture diagrams
