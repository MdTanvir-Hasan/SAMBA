# -*- coding: utf-8 -*-
"""
Visualization utilities for SAMBA experiments
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


def plot_training_curves(
    metrics: dict, save_path: Optional[Path] = None, show: bool = True
):
    """
    Plot training and validation loss curves

    Args:
        metrics: Dictionary containing train_loss_history and val_loss_history
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    train_loss = metrics.get("train_loss_history", [])
    val_loss = metrics.get("val_loss_history", [])
    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, label="Training Loss", linewidth=2, alpha=0.8)
    ax.plot(epochs, val_loss, label="Validation Loss", linewidth=2, alpha=0.8)

    # Mark best epoch
    best_epoch = metrics.get("best_epoch", 0)
    if best_epoch > 0 and best_epoch <= len(val_loss):
        ax.axvline(
            x=best_epoch,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Best Epoch ({best_epoch})",
        )
        ax.scatter([best_epoch], [val_loss[best_epoch - 1]], color="r", s=100, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Training curves saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_boxplot(
    all_results: List[dict], save_path: Optional[Path] = None, show: bool = True
):
    """
    Create box plots for all metrics across runs

    Args:
        all_results: List of result dictionaries from all runs
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ("val_loss", "Validation Loss"),
        ("test_mae", "Test MAE"),
        ("test_rmse", "Test RMSE"),
        ("training_time", "Training Time (seconds)"),
    ]

    for ax, (metric_key, metric_name) in zip(axes, metrics):
        values = [r.get(metric_key, np.nan) for r in all_results]
        values_clean = [v for v in values if not np.isnan(v)]

        # Box plot
        bp = ax.boxplot(
            [values_clean],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Overlay scatter points
        y = values_clean
        x = np.random.normal(1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.4, s=30, color="darkblue")

        # Statistics
        mean_val = np.mean(values_clean)
        std_val = np.std(values_clean)
        min_val = np.min(values_clean)
        max_val = np.max(values_clean)

        ax.axhline(
            y=mean_val,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Mean",
        )

        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}\nMean: {mean_val:.4f} ± {std_val:.4f}")
        ax.set_xticks([])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Box plots saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_distribution(
    all_results: List[dict],
    metric: str = "test_mae",
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot distribution of performance metric

    Args:
        all_results: List of result dictionaries
        metric: Metric to plot
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    values = [r.get(metric, np.nan) for r in all_results]
    values_clean = [v for v in values if not np.isnan(v)]

    # Histogram
    ax.hist(values_clean, bins=20, alpha=0.7, edgecolor="black", color="skyblue")

    # Statistics lines
    mean_val = np.mean(values_clean)
    median_val = np.median(values_clean)
    best_val = np.min(values_clean)

    ax.axvline(
        x=mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.4f}",
    )
    ax.axvline(
        x=median_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.4f}",
    )
    ax.axvline(
        x=best_val,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best: {best_val:.4f}",
    )

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Distribution of {metric.replace('_', ' ').title()} Across {len(values_clean)} Runs"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Distribution plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_val_vs_test_scatter(
    all_results: List[dict], save_path: Optional[Path] = None, show: bool = True
):
    """
    Scatter plot of validation loss vs test performance

    Args:
        all_results: List of result dictionaries
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    val_losses = [r.get("val_loss", np.nan) for r in all_results]
    test_maes = [r.get("test_mae", np.nan) for r in all_results]
    seeds = [r.get("seed", i) for i, r in enumerate(all_results)]

    # Remove NaN values
    valid_indices = [
        i
        for i in range(len(val_losses))
        if not (np.isnan(val_losses[i]) or np.isnan(test_maes[i]))
    ]

    val_losses_clean = [val_losses[i] for i in valid_indices]
    test_maes_clean = [test_maes[i] for i in valid_indices]
    seeds_clean = [seeds[i] for i in valid_indices]

    scatter = ax.scatter(
        val_losses_clean,
        test_maes_clean,
        c=seeds_clean,
        cmap="viridis",
        s=100,
        alpha=0.6,
        edgecolors="black",
    )

    # Add correlation line
    if len(val_losses_clean) > 1:
        z = np.polyfit(val_losses_clean, test_maes_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(val_losses_clean), max(val_losses_clean), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Trend")

        # Calculate correlation
        corr = np.corrcoef(val_losses_clean, test_maes_clean)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel("Validation Loss")
    ax.set_ylabel("Test MAE")
    ax.set_title("Validation Loss vs Test MAE (Colored by Seed)")
    plt.colorbar(scatter, label="Seed", ax=ax)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Scatter plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_analysis(
    all_results: List[dict], save_path: Optional[Path] = None, show: bool = True
):
    """
    Plot convergence analysis (best epoch distribution)

    Args:
        all_results: List of result dictionaries
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Best epoch distribution
    best_epochs = [r.get("best_epoch", np.nan) for r in all_results]
    best_epochs_clean = [e for e in best_epochs if not np.isnan(e)]

    ax1.hist(best_epochs_clean, bins=20, alpha=0.7, edgecolor="black", color="coral")
    ax1.axvline(
        x=np.mean(best_epochs_clean),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(best_epochs_clean):.1f}",
    )
    ax1.set_xlabel("Best Epoch")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Best Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Training time vs performance
    training_times = [
        r.get("training_time", np.nan) / 60 for r in all_results
    ]  # Convert to minutes
    test_maes = [r.get("test_mae", np.nan) for r in all_results]

    valid_indices = [
        i
        for i in range(len(training_times))
        if not (np.isnan(training_times[i]) or np.isnan(test_maes[i]))
    ]

    times_clean = [training_times[i] for i in valid_indices]
    maes_clean = [test_maes[i] for i in valid_indices]

    ax2.scatter(
        times_clean,
        maes_clean,
        s=100,
        alpha=0.6,
        edgecolors="black",
        color="lightgreen",
    )
    ax2.set_xlabel("Training Time (minutes)")
    ax2.set_ylabel("Test MAE")
    ax2.set_title("Training Time vs Test Performance")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Convergence analysis saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_top_k_comparison(
    all_results: List[dict],
    k: int = 5,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Compare top K runs

    Args:
        all_results: List of result dictionaries
        k: Number of top runs to compare
        save_path: Path to save figure
        show: Whether to display plot
    """
    # Sort by validation loss
    sorted_results = sorted(all_results, key=lambda x: x.get("val_loss", float("inf")))[
        :k
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Training curves for top K
    ax1 = axes[0]
    for i, result in enumerate(sorted_results):
        train_loss = result.get("train_loss_history", [])
        val_loss = result.get("val_loss_history", [])
        seed = result.get("seed", i)
        epochs = range(1, len(train_loss) + 1)

        ax1.plot(epochs, val_loss, label=f"Seed {seed}", linewidth=2, alpha=0.7)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title(f"Validation Loss Curves - Top {k} Runs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart comparison
    ax2 = axes[1]
    seeds = [str(r.get("seed", i)) for i, r in enumerate(sorted_results)]
    test_maes = [r.get("test_mae", 0) for r in sorted_results]
    test_rmses = [r.get("test_rmse", 0) for r in sorted_results]

    x = np.arange(len(seeds))
    width = 0.35

    ax2.bar(x - width / 2, test_maes, width, label="Test MAE", alpha=0.8)
    ax2.bar(x + width / 2, test_rmses, width, label="Test RMSE", alpha=0.8)

    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Error")
    ax2.set_title(f"Test Performance Comparison - Top {k} Runs")
    ax2.set_xticks(x)
    ax2.set_xticklabels(seeds)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Top-K comparison saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_all_visualizations(exp_dir: Path, all_results: List[dict], best_run: dict):
    """
    Create all visualization plots for an experiment

    Args:
        exp_dir: Experiment directory
        all_results: List of all run results
        best_run: Best run result dictionary
    """
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60 + "\n")

    # 1. Training curves for best run
    plot_training_curves(
        best_run, save_path=plots_dir / "best_training_curves.png", show=False
    )

    # 2. Metrics box plots
    plot_metrics_boxplot(
        all_results, save_path=plots_dir / "metrics_boxplot.png", show=False
    )

    # 3. Test MAE distribution
    plot_performance_distribution(
        all_results,
        metric="test_mae",
        save_path=plots_dir / "test_mae_distribution.png",
        show=False,
    )

    # 4. Validation vs Test scatter
    plot_val_vs_test_scatter(
        all_results, save_path=plots_dir / "val_vs_test_scatter.png", show=False
    )

    # 5. Convergence analysis
    plot_convergence_analysis(
        all_results, save_path=plots_dir / "convergence_analysis.png", show=False
    )

    # 6. Top-5 comparison
    plot_top_k_comparison(
        all_results, k=5, save_path=plots_dir / "top5_comparison.png", show=False
    )

    print(f"\n✅ All visualizations saved to: {plots_dir}")
    print("=" * 60 + "\n")
