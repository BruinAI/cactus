#!/usr/bin/env python3
"""
Analyze hyperparameter search results and generate visualizations.

This script loads results from a hyperparameter search run and generates:
- Summary tables of best configurations
- Visualizations showing parameter importance
- Detailed comparison plots
- Recommendations for optimal hyperparameters

Usage:
    python analyze_hparam_results.py --results_dir ./hparam_search_results
    python analyze_hparam_results.py --results_file ./hparam_search_results/results.json
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(results_path: Path) -> List[Dict]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert results list to pandas DataFrame."""
    rows = []
    for result in results:
        if not result.get("success"):
            continue  # Skip failed runs

        row = {
            "run_name": result["run_name"],
            "duration_minutes": result["duration_seconds"] / 60,
        }

        # Add hyperparameters
        for key, value in result["hyperparameters"].items():
            row[key] = value

        # Add metrics
        for key, value in result["metrics"].items():
            row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of the results."""
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("="*80)

    print(f"\nTotal successful runs: {len(df)}")
    print(f"\nMetrics available: {[col for col in df.columns if 'loss' in col.lower()]}")

    # Find runs with metrics
    metrics_df = df.dropna(subset=['best_val_loss'])

    if len(metrics_df) > 0:
        print(f"\nRuns with validation metrics: {len(metrics_df)}")

        # Best run
        best_idx = metrics_df['best_val_loss'].idxmin()
        best_run = metrics_df.loc[best_idx]

        print(f"\n{'─'*80}")
        print("BEST CONFIGURATION (by validation loss)")
        print(f"{'─'*80}")
        print(f"Run name: {best_run['run_name']}")
        print(f"Best validation loss: {best_run['best_val_loss']:.4f}")
        if pd.notna(best_run.get('final_train_loss')):
            print(f"Final training loss: {best_run['final_train_loss']:.4f}")
        print(f"Duration: {best_run['duration_minutes']:.1f} minutes")
        print(f"\nHyperparameters:")
        for col in metrics_df.columns:
            if col in ['learning_rate', 'num_epochs', 'lora_rank', 'lora_alpha',
                      'batch_size', 'gradient_accumulation_steps']:
                print(f"  {col}: {best_run[col]}")

        # Top 5 runs
        print(f"\n{'─'*80}")
        print("TOP 5 CONFIGURATIONS")
        print(f"{'─'*80}")
        top5 = metrics_df.nsmallest(5, 'best_val_loss')
        print(top5[['run_name', 'best_val_loss', 'learning_rate', 'num_epochs',
                   'lora_rank', 'lora_alpha']].to_string(index=False))

        # Statistical summary
        print(f"\n{'─'*80}")
        print("VALIDATION LOSS STATISTICS")
        print(f"{'─'*80}")
        print(f"Mean: {metrics_df['best_val_loss'].mean():.4f}")
        print(f"Median: {metrics_df['best_val_loss'].median():.4f}")
        print(f"Std Dev: {metrics_df['best_val_loss'].std():.4f}")
        print(f"Min: {metrics_df['best_val_loss'].min():.4f}")
        print(f"Max: {metrics_df['best_val_loss'].max():.4f}")

    else:
        print("\nNo runs with validation loss metrics found!")


def plot_parameter_importance(df: pd.DataFrame, output_dir: Path):
    """Create box plots showing how each parameter affects validation loss."""
    metrics_df = df.dropna(subset=['best_val_loss'])

    if len(metrics_df) == 0:
        print("\nSkipping parameter importance plots - no validation metrics")
        return

    params = ['learning_rate', 'num_epochs', 'lora_rank', 'lora_alpha',
             'batch_size', 'gradient_accumulation_steps']

    # Filter to params that exist and have multiple values
    params = [p for p in params if p in metrics_df.columns and
             metrics_df[p].nunique() > 1]

    if not params:
        print("\nSkipping parameter importance plots - no varying parameters")
        return

    n_params = len(params)
    n_cols = 2
    n_rows = (n_params + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, param in enumerate(params):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Sort by parameter value for better visualization
        plot_df = metrics_df.sort_values(param)

        # Create box plot
        sns.boxplot(data=plot_df, x=param, y='best_val_loss', ax=ax)
        ax.set_title(f'Validation Loss vs {param}', fontsize=12, fontweight='bold')
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel('Best Validation Loss', fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    output_path = output_dir / "parameter_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved parameter importance plot to {output_path}")
    plt.close()


def plot_loss_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot distribution of validation losses."""
    metrics_df = df.dropna(subset=['best_val_loss'])

    if len(metrics_df) == 0:
        print("\nSkipping loss distribution plot - no validation metrics")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    ax1.hist(metrics_df['best_val_loss'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(metrics_df['best_val_loss'].mean(), color='red',
               linestyle='--', label=f'Mean: {metrics_df["best_val_loss"].mean():.4f}')
    ax1.axvline(metrics_df['best_val_loss'].median(), color='green',
               linestyle='--', label=f'Median: {metrics_df["best_val_loss"].median():.4f}')
    ax1.set_xlabel('Best Validation Loss', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Validation Losses', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(metrics_df['best_val_loss'], vert=True)
    ax2.set_ylabel('Best Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "loss_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss distribution plot to {output_path}")
    plt.close()


def plot_learning_rate_vs_rank(df: pd.DataFrame, output_dir: Path):
    """Create heatmap showing interaction between learning rate and LoRA rank."""
    metrics_df = df.dropna(subset=['best_val_loss'])

    if len(metrics_df) == 0:
        print("\nSkipping LR vs rank plot - no validation metrics")
        return

    if 'learning_rate' not in metrics_df.columns or 'lora_rank' not in metrics_df.columns:
        print("\nSkipping LR vs rank plot - missing parameters")
        return

    # Create pivot table
    pivot = metrics_df.pivot_table(
        values='best_val_loss',
        index='lora_rank',
        columns='learning_rate',
        aggfunc='mean'
    )

    if pivot.empty:
        print("\nSkipping LR vs rank plot - no data for pivot")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', cbar_kws={'label': 'Best Validation Loss'})
    plt.title('Learning Rate vs LoRA Rank (Validation Loss)', fontsize=14, fontweight='bold')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('LoRA Rank', fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "lr_vs_rank_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved LR vs rank heatmap to {output_path}")
    plt.close()


def plot_rank_vs_alpha(df: pd.DataFrame, output_dir: Path):
    """Create heatmap showing interaction between LoRA rank and alpha."""
    metrics_df = df.dropna(subset=['best_val_loss'])

    if len(metrics_df) == 0:
        print("\nSkipping rank vs alpha plot - no validation metrics")
        return

    if 'lora_rank' not in metrics_df.columns or 'lora_alpha' not in metrics_df.columns:
        print("\nSkipping rank vs alpha plot - missing parameters")
        return

    # Create pivot table
    pivot = metrics_df.pivot_table(
        values='best_val_loss',
        index='lora_rank',
        columns='lora_alpha',
        aggfunc='mean'
    )

    if pivot.empty:
        print("\nSkipping rank vs alpha plot - no data for pivot")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', cbar_kws={'label': 'Best Validation Loss'})
    plt.title('LoRA Rank vs Alpha (Validation Loss)', fontsize=14, fontweight='bold')
    plt.xlabel('LoRA Alpha', fontsize=12)
    plt.ylabel('LoRA Rank', fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "rank_vs_alpha_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved rank vs alpha heatmap to {output_path}")
    plt.close()


def plot_training_duration(df: pd.DataFrame, output_dir: Path):
    """Plot training duration vs validation loss."""
    metrics_df = df.dropna(subset=['best_val_loss'])

    if len(metrics_df) == 0:
        print("\nSkipping duration plot - no validation metrics")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['duration_minutes'], metrics_df['best_val_loss'],
               alpha=0.6, s=100)

    # Annotate best run
    best_idx = metrics_df['best_val_loss'].idxmin()
    best_run = metrics_df.loc[best_idx]
    plt.scatter(best_run['duration_minutes'], best_run['best_val_loss'],
               color='red', s=200, marker='*', label='Best Run', zorder=5)

    plt.xlabel('Training Duration (minutes)', fontsize=12)
    plt.ylabel('Best Validation Loss', fontsize=12)
    plt.title('Training Duration vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "duration_vs_loss.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved duration vs loss plot to {output_path}")
    plt.close()


def save_summary_csv(df: pd.DataFrame, output_dir: Path):
    """Save summary results to CSV."""
    metrics_df = df.dropna(subset=['best_val_loss'])

    if len(metrics_df) == 0:
        print("\nNo validation metrics to save to CSV")
        return

    # Sort by validation loss
    sorted_df = metrics_df.sort_values('best_val_loss')

    # Select relevant columns
    cols = ['run_name', 'best_val_loss', 'final_train_loss', 'final_val_loss',
           'learning_rate', 'num_epochs', 'lora_rank', 'lora_alpha',
           'batch_size', 'gradient_accumulation_steps', 'duration_minutes']
    cols = [c for c in cols if c in sorted_df.columns]

    output_path = output_dir / "results_summary.csv"
    sorted_df[cols].to_csv(output_path, index=False)
    print(f"\nSaved results summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter search results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./hparam_search_results",
        help="Directory containing results.json"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Direct path to results.json (overrides results_dir)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save analysis outputs (defaults to same as results_dir)"
    )

    args = parser.parse_args()

    # Determine results file path
    if args.results_file:
        results_path = Path(args.results_file)
        default_output_dir = results_path.parent
    else:
        results_path = Path(args.results_dir) / "results.json"
        default_output_dir = Path(args.results_dir)

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return

    print(f"Loading results from {results_path}")
    results = load_results(results_path)

    if not results:
        print("No results found in file!")
        return

    # Convert to DataFrame
    df = results_to_dataframe(results)

    if len(df) == 0:
        print("No successful runs found in results!")
        return

    # Print summary
    print_summary_statistics(df)

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_distribution(df, output_dir)
    plot_parameter_importance(df, output_dir)
    plot_learning_rate_vs_rank(df, output_dir)
    plot_rank_vs_alpha(df, output_dir)
    plot_training_duration(df, output_dir)

    # Save CSV summary
    save_summary_csv(df, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
