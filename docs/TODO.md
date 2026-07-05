# Project TODO — Remaining Work Items

> Last updated: 2026-03-14.
> Priorities: **P0** = pipeline-blocking / immediate fix, **P1** = important for research quality, **P2** = nice-to-have.

---

## P0 — Immediate (Pipeline & Research Blocking)

These items directly affect experiment validity, research results, or reproducibility. Nothing in the next section is worth starting until these are complete.

### Experiments in Progress

| # | Task | Status | Config | Notes |
|---|------|--------|--------|-------|
| 1.1 | Centralized baseline — HAM10000 | **In Progress** | `dscatnet_centralized_ham10000.yaml` | 200 epochs, `paper` variant. Queue after 1.3 finishes GPU. |
| 1.3 | Federated HAM10000 non-IID (Dirichlet α=0.5) | **In Progress — Round 48/100** | `dscatnet_federated_ham10000_non_iid.yaml` | 4 clients, resumed from round 42, current best acc 0.7021. |

### Experiments Not Yet Started (Blocking Research)

| # | Task | Config | Notes |
|---|------|--------|-------|
| 1.4 | Federated HAM10000 IID (Dirichlet α=1000.0) | `dscatnet_federated_ham10000_iid.yaml` | IID baseline for FL vs non-IID comparison. |
| 1.2 | Centralized baseline — PAD-UFES-20 | `dscatnet_centralized_padufes20.yaml` | Required for cross-dataset generalization claim. |
| 1.5 | Federated PAD-UFES-20 non-IID (Dirichlet α=0.5) | `dscatnet_federated_padufes20_non_iid.yaml` | Core FL contribution on clinical dataset. |
| 1.6 | Federated PAD-UFES-20 IID (Dirichlet α=1000.0) | `dscatnet_federated_padufes20_iid.yaml` | IID baseline for PAD-UFES-20. |
| 1.7 | Final evaluation on all best checkpoints | — | Run `--mode evaluate` on each `best_model.pt` after training completes. Produces per-class metrics, confusion matrices, AUC. |

### Result Tables (Blocking Write-up)

| # | Task | Depends On | Notes |
|---|------|------------|-------|
| 1.8 | Fill result tables (accuracy, F1, AUC per dataset) | 1.1–1.7 | Tables in the results section have pending results. |
| 1.9 | Validate normalization consistency across all runs | — | Confirm all configs use `use_dermoscopy_norm: false` (ImageNet normalization). Check any PAD-UFES-20 configs carefully — PAD images are clinical photos, not dermoscopy. |

---

## P1 — Important (Strengthen Research Quality)

| # | Task | Notes |
|---|------|-------|
| 2.1 | Ablation: vary Dirichlet α (0.1, 0.5, 1.0, 10.0) | Quantifies non-IID sensitivity. Required to justify choice of α=0.5 in reporting. Configs can be generated from `federated_template.yaml`. |
| 2.2 | Multiple random seeds (42, 123, 456) for key experiments | Report mean ± std for HAM10000 centralized and FL non-IID. Required for any statistical claim. |
| 2.3 | Statistical significance test (paired t-test or Wilcoxon) | Needed to claim FL vs centralized differences are significant. Apply after 2.2. |
| 2.4 | Run evaluation notebook (`02_model_evaluation.ipynb`) on final checkpoints | Produces per-class breakdown, visualizations, and per-sample predictions exported to `results_latest.json` for reporting and exact statistical testing in Notebook 03. |
| 2.5 | Run FL vs centralized comparison notebook (`03_fl_vs_centralized_comparison.ipynb`) | Final comparison plots and exact McNemar p-values with Bonferroni correction for reporting. Requires `results_latest.json` artifacts from Notebook 02. |
| 2.6 | Update `docs/README.md` and `README.md` with final experiment results | After all experiments complete. |

---

## P2 — Nice-to-Have (Polish)

| # | Task | Notes |
|---|------|-------|
| 3.1 | Ablation: vary number of FL clients (2, 4, 8) | Communication cost vs accuracy trade-off. Adds depth to the analysis but not required. |
| 3.2 | Run centralized on ISIC2018 | Validates label consistency with HAM10000 (shared images). Optional cross-validation. |
| 3.3 | Add type checking (mypy or pyright) to CI | Type hints exist but are not CI-verified. Low urgency. |
| 3.4 | Pin dependency versions in CI | `requirements.txt` uses `>=` ranges. Stability improvement. |
| 3.5 | Add CI artifact upload for coverage HTML reports | Currently only printed. Cosmetic. |
| 3.6 | Add architecture diagram (Mermaid/draw.io) | Text diagrams exist in `CLAUDE.md`. Adds to `docs/architecture.md`. |
| 3.7 | W&B or TensorBoard integration | Training curves are logged to CSV. W&B would unify experiment tracking. |

---

## Completed / Accepted

### Resolved This Session (2026-03-14)

| Item | Resolution |
|------|-----------|
| `splits.py` used global `np.random.seed()` in FL multi-process context | Fixed: all 4 split functions now use isolated `np.random.default_rng(seed)`. No behavior change to existing splits. |
| Notebook glob patterns `centralized_*/` / `federated_*/` missed actual output dirs | Fixed: updated to `dscatnet_centralized_*/` / `dscatnet_federated_*/` in all 3 notebooks. |
| Notebooks used `Path().resolve().parent` which breaks if launched from project root | Fixed: CWD-aware detection `cwd if (cwd / "src").exists() else cwd.parent` in all 3 notebooks. |
| `run_download.py --dataset HAM10000` wrong flag in nb01 | Fixed: `--download HAM10000`. |
| `configs/templates/centralized_template.yaml` defaulted to `variant: small` | Fixed: `variant: paper`, `use_class_weights: false`. |
| `configs/templates/model_template.yaml` lacked `paper` variant, `num_heads: 6` wrong | Fixed: `paper` variant block added, `num_heads: 12`. |
| CI lint missed `run_experiment.py`, `run_download.py`, `run_tests.py` | Fixed: `.github/workflows/ci.yml` now covers root scripts. |
| `docs/README.md` stated 225 tests | Fixed: updated to 457 selected tests. |
| The draft results document had `[PLACEHOLDER]` text in the abstract and a comparison table said "small" | Fixed: de-placeholdered abstract, table updated to "paper variant". |

### Previously Resolved

| Item | Resolution |
|------|-----------|
| `DirichletSubset.__getitem__` wrong attribute name (val transforms never applied) | Fixed: `image_paths` → `img_paths`. |
| `server.py` hardcoded `num_classes=7` | Fixed: parameterized with default=7. |
| CI missing `ruff format --check` | Fixed. |
| `logging_utils.py` CSV header rewrite fragile on resume | Fixed: resume-safe row filtering. |

---

## Paper Alignment Reference (Yadav et al., PLOS ONE 2024)

| Aspect | Paper | Implementation | Status |
|--------|-------|----------------|--------|
| Image size | 224×224 | 224×224 | Done |
| Fine / coarse patch | 8×8 / 16×16 | 8×8 / 16×16 | Done |
| Embedding dim | D=192/768 (asymmetric) | 384 unified (ViT-Small pretrained) | Documented adaptation |
| Depth | 6 | 6 | Done |
| Attention heads | H=12 | 12 (`paper` variant) | Done |
| MLP ratio | 4.0 | 4.0 | Done |
| Dropout | 0.1 | 0.1 | Done |
| Weight init | Truncated normal (σ=0.02) | Truncated normal (σ=0.02) | Done |
| Optimizer / LR / WD | Adam / 1e-3 / 0.0 | Adam / 1e-3 / 0.0 | Done |
| Effective batch size | 32 | 8 × 4 grad accum = 32 | Done |
| Epochs | 200 | 200 | Done |
| Loss | CrossEntropy | CrossEntropy (unweighted) | Done |
| Augmentation (HAM10000) | None | None | Done |
| Augmentation (PAD-UFES-20) | H/V flips | `level: light` | Done |
| 5-fold CV | Yes | Single 85/15 split | Not implemented — documented as limitation in the analysis section |
| Model params | ~22M | 29.4M | Discrepancy documented in `dscatnet.py` docstring |
