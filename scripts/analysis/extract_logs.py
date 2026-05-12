#!/usr/bin/env python3
"""Extract convergence curves and metrics from results.json files across all experiments.

Usage:
  python scripts/analysis/extract_logs.py --outputs-dir outputs/ --out-dir outputs/analysis

The script recursively searches for `results.json` files under `outputs/`, extracts training/validation
accuracy curves (by epoch or round), generates comparison plots (centralized vs federated),
and produces a CSV summary with best accuracies and test metrics.
"""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None


def find_results_json(outputs_dir: Path):
    """Find all results.json files under outputs_dir."""
    return sorted(outputs_dir.rglob('results.json'))


def load_experiment(json_path: Path):
    """Load a single results.json and extract key metadata."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to read {json_path}: {e}")
        return None

    exp_name = json_path.parent.name
    is_federated = 'federated' in exp_name.lower()
    
    history = data.get('history', {})
    step_key = 'rounds' if is_federated else 'epochs'
    steps = history.get(step_key, [])
    
    if not steps:
        print(f"Warning: no {step_key} found in {json_path}")
        return None
    
    val_acc = history.get('val_accuracy', [])
    train_acc = history.get('train_accuracy', [])
    
    if not val_acc or not train_acc:
        print(f"Warning: missing accuracy data in {json_path}")
        return None
    
    best_val_idx = np.argmax(val_acc) if val_acc else 0
    best_val_acc = float(val_acc[best_val_idx])
    best_step = int(steps[best_val_idx])
    
    test_acc = data.get('test_metrics', {}).get('accuracy', np.nan)
    final_val_acc = float(val_acc[-1]) if val_acc else np.nan
    
    return {
        'experiment': exp_name,
        'is_federated': is_federated,
        'steps': steps,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'best_val_accuracy': best_val_acc,
        'best_step': best_step,
        'final_val_accuracy': final_val_acc,
        'test_accuracy': test_acc,
        'total_time_seconds': data.get('total_time_seconds', np.nan),
    }


def plot_convergence_curves(experiments: list, out_dir: Path):
    """Plot convergence curves for all experiments, separated by centralized vs federated."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    centralized = [e for e in experiments if not e['is_federated']]
    federated = [e for e in experiments if e['is_federated']]
    
    # Plot 1: Centralized vs Federated comparison
    plt.figure(figsize=(12, 6))
    for exp in centralized:
        plt.plot(exp['steps'], exp['val_accuracy'], label=exp['experiment'], linewidth=2, marker='o')
    for exp in federated:
        plt.plot(exp['steps'], exp['val_accuracy'], label=exp['experiment'], linewidth=2, linestyle='--', marker='s')
    
    plt.xlabel('Epoch / Round')
    plt.ylabel('Validation Accuracy')
    plt.title('Convergence: Centralized vs Federated Learning')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'convergence_all_experiments.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_dir / "convergence_all_experiments.png"}')
    
    # Plot 2: Centralized only
    if centralized:
        plt.figure(figsize=(10, 6))
        for exp in centralized:
            plt.plot(exp['steps'], exp['val_accuracy'], label=exp['experiment'], linewidth=2, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Centralized Learning Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'convergence_centralized.png', dpi=150)
        plt.close()
        print(f'Saved {out_dir / "convergence_centralized.png"}')
    
    # Plot 3: Federated only (IID vs Non-IID)
    if federated:
        iid = [e for e in federated if 'iid' in e['experiment'].lower()]
        non_iid = [e for e in federated if 'non_iid' in e['experiment'].lower()]
        
        plt.figure(figsize=(10, 6))
        for exp in iid:
            plt.plot(exp['steps'], exp['val_accuracy'], label=exp['experiment'], linewidth=2, marker='o', linestyle='-')
        for exp in non_iid:
            plt.plot(exp['steps'], exp['val_accuracy'], label=exp['experiment'], linewidth=2, marker='s', linestyle='--')
        plt.xlabel('Round')
        plt.ylabel('Validation Accuracy')
        plt.title('Federated Learning: IID vs Non-IID')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'convergence_federated.png', dpi=150)
        plt.close()
        print(f'Saved {out_dir / "convergence_federated.png"}')


def plot_convergence_by_dataset(experiments: list, out_dir: Path):
    """Plot convergence grouped by dataset (all_datasets, ham10000, padufes20)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {}
    for exp in experiments:
        # Extract dataset type
        if 'ham10000' in exp['experiment'].lower():
            dataset = 'HAM10000'
        elif 'padufes20' in exp['experiment'].lower():
            dataset = 'PAD-UFES-20'
        elif 'all_datasets' in exp['experiment'].lower():
            dataset = 'All Datasets'
        else:
            continue
        
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(exp)
    
    # Plot each dataset with C + F-IID + F-Non-IID
    for dataset, exps in sorted(datasets.items()):
        plt.figure(figsize=(10, 6))
        
        for exp in exps:
            label_prefix = 'Centralized' if not exp['is_federated'] else ('Federated IID' if 'iid' in exp['experiment'].lower() else 'Federated Non-IID')
            linestyle = '-' if not exp['is_federated'] else ('--' if 'iid' in exp['experiment'].lower() else ':')
            marker = 'o' if not exp['is_federated'] else ('s' if 'iid' in exp['experiment'].lower() else '^')
            linewidth = 2.5 if not exp['is_federated'] else 2
            
            plt.plot(exp['steps'], exp['val_accuracy'], label=label_prefix, linewidth=linewidth, 
                    marker=marker, linestyle=linestyle, markersize=6)
        
        plt.xlabel('Epoch / Round')
        plt.ylabel('Validation Accuracy')
        plt.title(f'{dataset}: Centralized vs Federated Convergence')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'convergence_by_dataset_{dataset.lower().replace(" ", "_").replace("-", "")}.png'
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()
        print(f'Saved {out_dir / filename}')


def create_summary_table(experiments: list, out_dir: Path):
    """Create a CSV summary of all experiments."""
    rows = []
    for exp in experiments:
        rows.append({
            'experiment': exp['experiment'],
            'type': 'Federated' if exp['is_federated'] else 'Centralized',
            'best_val_accuracy': round(exp['best_val_accuracy'], 4),
            'best_step': exp['best_step'],
            'final_val_accuracy': round(exp['final_val_accuracy'], 4),
            'test_accuracy': round(exp['test_accuracy'], 4) if not np.isnan(exp['test_accuracy']) else np.nan,
            'total_epochs_rounds': len(exp['steps']),
            'training_time_hours': round(exp['total_time_seconds'] / 3600, 2),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('test_accuracy', ascending=False, na_position='last')
    df.to_csv(out_dir / 'experiment_summary.csv', index=False)
    print(f'Wrote experiment summary to {out_dir / "experiment_summary.csv"}')
    print('\n' + df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, default='outputs', help='Root outputs directory')
    parser.add_argument('--out-dir', type=str, default='outputs/analysis', help='Destination for analysis artifacts')
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find and load all results.json files
    json_paths = find_results_json(outputs_dir)
    if not json_paths:
        print(f'No results.json files found under {outputs_dir}')
        return
    
    print(f'Found {len(json_paths)} results.json files')
    experiments = []
    for jp in json_paths:
        exp = load_experiment(jp)
        if exp:
            experiments.append(exp)
    
    if not experiments:
        print('Failed to load any experiments')
        return
    
    print(f'Successfully loaded {len(experiments)} experiments\n')
    
    # Generate plots and summary
    print('--- Generating plots by learning type (centralized vs federated) ---')
    plot_convergence_curves(experiments, out_dir)
    
    print('\n--- Generating plots by dataset type ---')
    plot_convergence_by_dataset(experiments, out_dir)
    
    create_summary_table(experiments, out_dir)
    
    print(f'\nAnalysis complete. Outputs saved to {out_dir}')


if __name__ == '__main__':
    main()
