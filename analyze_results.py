# -*- coding: utf-8 -*-
"""
Analyze Results from Completed Experiments
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from experiment_manager import ExperimentManager
from utils.visualization_utils import create_all_visualizations
from utils.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--exp_dir", type=str, required=True, help="Experiment directory to analyze"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="val_loss",
        help="Metric for selecting best run (default: val_loss)",
    )
    parser.add_argument(
        "--regenerate_plots", action="store_true", help="Regenerate all plots"
    )
    parser.add_argument(
        "--regenerate_reports",
        action="store_true",
        help="Regenerate HTML and LaTeX reports",
    )

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    if not exp_dir.exists():
        print(f"❌ Error: Experiment directory not found: {exp_dir}")
        return

    # Load experiment summary
    summary_path = exp_dir / "experiment_summary.json"
    if not summary_path.exists():
        print(f"❌ Error: No experiment summary found in {exp_dir}")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Extract base configuration
    base_config = summary["experiment_info"]

    # Create experiment manager
    exp_manager = ExperimentManager(
        base_config={},
        num_runs=summary["experiment_info"]["num_runs"],
        base_dir=exp_dir.parent,
    )
    exp_manager.base_dir = exp_dir  # Override to use existing directory

    # Load all results
    exp_manager.results = exp_manager.load_all_results()

    if not exp_manager.results:
        print(f"❌ Error: No results found in {exp_dir}")
        return

    # Analysis Header
    print("\n" + "=" * 70)
    print(f"📊 Analysis of: {exp_dir.name}")
    print("=" * 70 + "\n")

    # Get best run
    best_run, best_seed = exp_manager.select_best_run(
        criterion=args.criterion, mode="min"
    )

    print(f"🏆 Best Run (by {args.criterion}): Seed {best_seed}")
    print(f"{'─' * 70}")
    print(f"   Validation Loss: {best_run.get('val_loss', float('inf')):.4f}")
    print(f"   Test MAE: {best_run.get('test_mae', float('inf')):.4f}")
    print(f"   Test RMSE: {best_run.get('test_rmse', float('inf')):.4f}")
    print(f"   Best Epoch: {best_run.get('best_epoch', 'N/A')}")
    print(f"   Training Time: {best_run.get('training_time', 0) / 60:.2f} minutes")
    if best_run.get("early_stopped", False):
        print(f"   Status: Early stopped")

    # Statistics
    stats = exp_manager.compute_statistics()
    print(f"\n📈 Statistics Across {stats['num_runs']} Runs:")
    print(f"{'─' * 70}")
    print(f"   Successful Runs: {stats['num_successful']}/{stats['num_runs']}")
    print(f"\n   Validation Loss:")
    print(
        f"      Mean: {stats['val_loss']['mean']:.4f} ± {stats['val_loss']['std']:.4f}"
    )
    print(f"      Min:  {stats['val_loss']['min']:.4f}")
    print(f"      Max:  {stats['val_loss']['max']:.4f}")
    print(f"\n   Test MAE:")
    print(
        f"      Mean: {stats['test_mae']['mean']:.4f} ± {stats['test_mae']['std']:.4f}"
    )
    print(f"      Min:  {stats['test_mae']['min']:.4f}")
    print(f"      Max:  {stats['test_mae']['max']:.4f}")
    print(f"\n   Test RMSE:")
    print(
        f"      Mean: {stats['test_rmse']['mean']:.4f} ± {stats['test_rmse']['std']:.4f}"
    )
    print(f"      Min:  {stats['test_rmse']['min']:.4f}")
    print(f"      Max:  {stats['test_rmse']['max']:.4f}")
    print(f"\n   Convergence:")
    print(
        f"      Avg Best Epoch: {stats['best_epoch']['mean']:.1f} ± {stats['best_epoch']['std']:.1f}"
    )
    print(f"\n   Training Time:")
    print(f"      Total: {stats['training_time']['total'] / 3600:.2f} hours")
    print(f"      Mean:  {stats['training_time']['mean'] / 60:.2f} minutes/run")

    # Top K runs
    print(f"\n🌟 Top 10 Runs:")
    print(f"{'─' * 70}")
    print(
        f"{'Rank':<6}{'Seed':<8}{'Val Loss':<12}{'Test MAE':<12}{'Test RMSE':<12}{'Epoch':<8}"
    )
    print(f"{'─' * 70}")

    top_10 = exp_manager.get_top_k_runs(k=10, criterion=args.criterion, mode="min")
    for rank, run in enumerate(top_10, 1):
        print(
            f"{rank:<6}{run.get('seed', 'N/A'):<8}{run.get('val_loss', 0):<12.4f}"
            f"{run.get('test_mae', 0):<12.4f}{run.get('test_rmse', 0):<12.4f}"
            f"{run.get('best_epoch', 'N/A'):<8}"
        )

    # Regenerate visualizations if requested
    if args.regenerate_plots:
        print(f"\n📸 Regenerating visualizations...")
        create_all_visualizations(exp_dir, exp_manager.results, best_run)

    # Regenerate reports if requested
    if args.regenerate_reports:
        print(f"\n📄 Regenerating reports...")
        report_gen = ReportGenerator(exp_dir)
        report_gen.generate_html_report(best_run, stats, exp_manager.results)
        report_gen.generate_latex_table(exp_manager.results, top_k=10)
        report_gen.generate_summary_text(best_run, stats)

    # Export table
    print(f"\n📝 Exporting results table...")
    df = exp_manager.export_results_table(format="csv")

    print(f"\n{'=' * 70}")
    print("✅ Analysis Complete!")
    print(f"{'=' * 70}")
    print(f"\n📁 Results Location: {exp_dir}")
    print(f"📊 View HTML Report: {exp_dir}/reports/experiment_report.html")
    print(f"📈 Plots Directory: {exp_dir}/plots/")
    print(f"📋 Results Table: {exp_dir}/all_results.csv\n")


if __name__ == "__main__":
    main()
