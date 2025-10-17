# -*- coding: utf-8 -*-
"""
Compare Multiple Models/Experiments
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")


def load_experiment_summary(exp_dir: Path):
    """Load experiment summary from directory"""
    summary_path = exp_dir / "experiment_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        return json.load(f)


def compare_experiments(exp_dirs: list, labels: list = None):
    """
    Compare multiple experiments

    Args:
        exp_dirs: List of experiment directories
        labels: List of labels for each experiment
    """
    if labels is None:
        labels = [f"Exp {i + 1}" for i in range(len(exp_dirs))]

    # Load all summaries
    summaries = []
    for exp_dir, label in zip(exp_dirs, labels):
        summary = load_experiment_summary(Path(exp_dir))
        if summary:
            summary["label"] = label
            summaries.append(summary)

    if not summaries:
        print("❌ No valid experiments found")
        return

    # Create comparison plots directory
    output_dir = Path("experiments/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for comparison
    comparison_data = []
    for summary in summaries:
        best_run = summary["best_run"]
        stats = summary["statistics"]

        comparison_data.append(
            {
                "Model": summary["label"],
                "Best Val Loss": best_run.get("val_loss", float("inf")),
                "Best Test MAE": best_run.get("test_mae", float("inf")),
                "Best Test RMSE": best_run.get("test_rmse", float("inf")),
                "Avg Test MAE": stats["test_mae"]["mean"],
                "Std Test MAE": stats["test_mae"]["std"],
                "Avg Test RMSE": stats["test_rmse"]["mean"],
                "Std Test RMSE": stats["test_rmse"]["std"],
                "Avg Training Time": stats["training_time"]["mean"] / 60,
            }
        )

    df = pd.DataFrame(comparison_data)

    # Print comparison table
    print("\n" + "=" * 80)
    print("📊 Model Comparison Table")
    print("=" * 80 + "\n")
    print(df.to_string(index=False))
    print()

    # Create comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Best Performance Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width / 2, df["Best Test MAE"], width, label="MAE", alpha=0.8)
    ax1.bar(x + width / 2, df["Best Test RMSE"], width, label="RMSE", alpha=0.8)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Error")
    ax1.set_title("Best Run Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Average Performance with Error Bars
    ax2 = axes[0, 1]
    ax2.errorbar(
        x,
        df["Avg Test MAE"],
        yerr=df["Std Test MAE"],
        fmt="o-",
        capsize=5,
        capthick=2,
        label="MAE",
        markersize=8,
        linewidth=2,
    )
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Test MAE")
    ax2.set_title("Average Performance Across All Runs")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training Time Comparison
    ax3 = axes[1, 0]
    ax3.bar(x, df["Avg Training Time"], alpha=0.8, color="coral")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Training Time (minutes)")
    ax3.set_title("Average Training Time per Run")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Performance Variance
    ax4 = axes[1, 1]
    ax4.bar(x, df["Std Test MAE"], alpha=0.8, color="lightgreen")
    ax4.set_xlabel("Model")
    ax4.set_ylabel("Standard Deviation (MAE)")
    ax4.set_title("Performance Stability (Lower is Better)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save comparison plot
    comparison_plot_path = output_dir / "models_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Comparison plot saved to: {comparison_plot_path}")
    plt.close()

    # Create detailed comparison plots for each metric
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Detailed MAE comparison
    ax1 = axes[0]
    for summary in summaries:
        runs = summary.get("all_runs_summary", [])
        test_maes = [r.get("test_mae", np.nan) for r in runs]
        test_maes_clean = [m for m in test_maes if not np.isnan(m)]

        ax1.hist(
            test_maes_clean,
            bins=15,
            alpha=0.5,
            label=summary["label"],
            edgecolor="black",
        )

    ax1.set_xlabel("Test MAE")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Test MAE Across Models")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Box plot comparison
    ax2 = axes[1]
    box_data = []
    box_labels = []

    for summary in summaries:
        runs = summary.get("all_runs_summary", [])
        test_maes = [r.get("test_mae", np.nan) for r in runs]
        test_maes_clean = [m for m in test_maes if not np.isnan(m)]
        box_data.append(test_maes_clean)
        box_labels.append(summary["label"])

    bp = ax2.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    ax2.set_ylabel("Test MAE")
    ax2.set_title("Test MAE Distribution by Model")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    detailed_plot_path = output_dir / "detailed_comparison.png"
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Detailed comparison saved to: {detailed_plot_path}")
    plt.close()

    # Save comparison table
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Comparison table saved to: {csv_path}")

    # Generate comparison report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write("## Performance Summary\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Rankings\n\n")

        # Rank by best performance
        df_ranked = df.sort_values("Best Test MAE")
        f.write("### By Best Run Performance (Test MAE)\n\n")
        for idx, row in df_ranked.iterrows():
            f.write(f"{idx + 1}. **{row['Model']}**: {row['Best Test MAE']:.4f}\n")

        f.write("\n### By Average Performance (Test MAE)\n\n")
        df_ranked_avg = df.sort_values("Avg Test MAE")
        for idx, row in df_ranked_avg.iterrows():
            f.write(
                f"{idx + 1}. **{row['Model']}**: {row['Avg Test MAE']:.4f} ± {row['Std Test MAE']:.4f}\n"
            )

        f.write("\n### By Stability (Lower Std is Better)\n\n")
        df_ranked_std = df.sort_values("Std Test MAE")
        for idx, row in df_ranked_std.iterrows():
            f.write(f"{idx + 1}. **{row['Model']}**: {row['Std Test MAE']:.4f}\n")

    print(f"✅ Comparison report saved to: {report_path}")

    print(f"\n{'=' * 80}")
    print("✨ Comparison Complete!")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiments")
    parser.add_argument(
        "--exp_dirs", nargs="+", required=True, help="Experiment directories to compare"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each experiment (optional)",
    )

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.exp_dirs):
        print("❌ Error: Number of labels must match number of experiment directories")
        return

    compare_experiments(args.exp_dirs, args.labels)


if __name__ == "__main__":
    main()
