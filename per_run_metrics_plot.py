# Per-Run Metrics Visualization Script
# Creates graphs showing RMSE, IC, RIC values per run for each dataset

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_per_run_metrics_plot():
    """Create visualization of per-run metrics across datasets"""

    exp_base = Path("experiments")
    datasets = ["DJI", "IXIC", "NYSE"]
    all_run_data = []

    print("🔍 Collecting metrics from all experiment runs...")

    # Collect data from all runs for each dataset
    for dataset in datasets:
        exp_dirs = [d for d in exp_base.glob(f"SAMBA_{dataset}_*") if d.is_dir()]

        if exp_dirs:
            # Get the latest experiment for this dataset
            latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
            run_dirs = [
                d
                for d in latest_exp.iterdir()
                if d.is_dir() and d.name.startswith("run_seed_")
            ]

            print(
                f"📊 Processing {dataset}: Found {len(run_dirs)} runs in {latest_exp.name}"
            )

            for run_dir in run_dirs:
                metrics_file = run_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)

                        # Extract final test metrics
                        all_run_data.append(
                            {
                                "Dataset": dataset,
                                "Run": run_dir.name.replace("run_seed_", ""),
                                "RMSE": metrics.get("test_rmse"),
                                "IC": metrics.get("test_ic"),
                                "RIC": metrics.get("test_ric"),
                            }
                        )
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"⚠️  Error reading {metrics_file}: {e}")
                        continue
                else:
                    print(f"⚠️  Metrics file not found: {metrics_file}")

    if not all_run_data:
        print(
            "❌ No run data found. Make sure experiments have been completed and metrics.json files exist."
        )
        return

    df_runs = pd.DataFrame(all_run_data)
    print(f"✅ Collected data from {len(df_runs)} runs across {len(datasets)} datasets")

    # Convert Run to numeric for proper sorting
    df_runs["Run"] = pd.to_numeric(df_runs["Run"])
    df_runs = df_runs.sort_values(["Dataset", "Run"])

    # Create BOX PLOT visualization
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle(
        "Per-Run Performance Metrics Across Datasets (Box Plot)",
        fontsize=16,
        fontweight="bold",
    )

    metrics = ["RMSE", "IC", "RIC"]
    colors = {"DJI": "#1f77b4", "IXIC": "#ff7f0e", "NYSE": "#2ca02c"}

    for i, metric in enumerate(metrics):
        ax = axes1[i]

        # Create box plot with individual points
        sns.boxplot(
            data=df_runs, x="Dataset", y=metric, ax=ax, palette=colors, showfliers=False
        )
        sns.stripplot(
            data=df_runs, x="Dataset", y=metric, ax=ax, color="black", alpha=0.6, size=4
        )

        ax.set_title(f"{metric} Distribution Across Runs", fontsize=14)
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add value labels on boxes
        for j, dataset in enumerate(datasets):
            dataset_data = df_runs[df_runs["Dataset"] == dataset][metric].dropna()
            if len(dataset_data) > 0:
                mean_val = dataset_data.mean()
                std_val = dataset_data.std()
                ax.text(
                    j,
                    ax.get_ylim()[1] * 0.95,
                    f"μ={mean_val:.4f}\nσ={std_val:.4f}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()

    # Save the box plot
    box_plot_path = "experiments/per_run_metrics_boxplot.png"
    fig1.savefig(box_plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Box plot visualization saved as: {box_plot_path}")

    # Create LINE PLOT visualization
    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 12))
    fig2.suptitle(
        "Per-Run Performance Metrics by Seed Number", fontsize=16, fontweight="bold"
    )

    for i, metric in enumerate(metrics):
        ax = axes2[i]

        for dataset in datasets:
            dataset_data = df_runs[df_runs["Dataset"] == dataset].copy()
            dataset_data = dataset_data.sort_values("Run")

            ax.plot(
                dataset_data["Run"],
                dataset_data[metric],
                marker="o",
                markersize=4,
                linewidth=2,
                label=dataset,
                color=colors[dataset],
                alpha=0.8,
            )

        ax.set_title(f"{metric} vs Run Number", fontsize=14)
        ax.set_xlabel("Run/Seed Number", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add trend lines
        for dataset in datasets:
            dataset_data = df_runs[df_runs["Dataset"] == dataset].copy()
            if len(dataset_data) > 1:
                x = dataset_data["Run"].values
                y = dataset_data[metric].values
                # Simple linear trend line
                try:
                    from numpy import polyfit

                    z = polyfit(x, y, 1)
                    ax.plot(
                        x,
                        z[0] * x + z[1],
                        "--",
                        alpha=0.5,
                        color=colors[dataset],
                        label=f"{dataset} trend",
                    )
                except Exception:
                    pass  # Skip trend line if polyfit fails

    plt.tight_layout()

    # Save the line plot
    line_plot_path = "experiments/per_run_metrics_lineplot.png"
    fig2.savefig(line_plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Line plot visualization saved as: {line_plot_path}")

    # Show both plots
    plt.figure(fig1.number)  # Show box plot
    plt.show()

    plt.figure(fig2.number)  # Show line plot
    plt.show()

    # Print summary statistics
    print("\n📊 Summary Statistics:")
    summary = df_runs.groupby("Dataset")[["RMSE", "IC", "RIC"]].describe()
    print(summary)

    # Save summary to CSV
    summary.to_csv("experiments/per_run_metrics_summary.csv")
    print("✅ Summary statistics saved as: experiments/per_run_metrics_summary.csv")


if __name__ == "__main__":
    create_per_run_metrics_plot()
