# Analysis scripts

Utilities to extract convergence curves and metrics from training logs (`results.json`), generate comparison plots (centralized vs federated), and produce summary tables.

## `extract_logs.py`

Recursively searches for `results.json` files under `outputs/`, extracts training/validation accuracy curves, and generates visualizations and summaries.

### Usage

```bash
# From repository root (recommended)
python scripts/analysis/extract_logs.py --outputs-dir outputs/ --out-dir outputs/analysis

# Custom output directory
python scripts/analysis/extract_logs.py --outputs-dir outputs/ --out-dir my_analysis_output
```

### Outputs (default: `outputs/analysis`)

**Plots by learning type (centralized vs federated):**
- `convergence_all_experiments.png` — all 9 experiments on one plot (overview)
- `convergence_centralized.png` — centralized models only (3 datasets)
- `convergence_federated.png` — federated IID vs Non-IID

**Plots by dataset (recommended for thesis/defense):**
- `convergence_by_dataset_ham10000.png` — HAM10000: C vs F-IID vs F-Non-IID
- `convergence_by_dataset_all_datasets.png` — All Datasets: C vs F-IID vs F-Non-IID
- `convergence_by_dataset_padufes20.png` — PAD-UFES-20: C vs F-IID vs F-Non-IID

**Summary:**
- `experiment_summary.csv` — best/final validation accuracy, test accuracy, training time per experiment

### Requirements

- `pandas`
- `numpy`
- `matplotlib`

### Notes

- The script expects `results.json` files with a `history` dict containing:
  - `epochs` (for centralized) or `rounds` (for federated)
  - `train_accuracy`, `val_accuracy`
- Optionally uses `test_metrics.accuracy` if available for test set accuracy
- Generates 6 PNG plots + 1 CSV summary per run
- GPU not required; runs on CPU
