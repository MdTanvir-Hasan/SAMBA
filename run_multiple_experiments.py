# -*- coding: utf-8 -*-
"""
Run Multiple SAMBA Experiments with Different Random Seeds
"""

import argparse
import sys
import torch
from tqdm import tqdm
from pathlib import Path

from paper_config import get_paper_config
from models import SAMBA
from utils import prepare_data, init_seed, print_model_parameters, All_Metrics
from trainer import Trainer
from experiment_manager import ExperimentManager
from utils.visualization_utils import create_all_visualizations
from utils.report_generator import ReportGenerator


def masked_mae_loss(scaler, mask_value):
    """Masked MAE loss function"""

    def loss(preds, labels):
        # Ensure preds and labels are on the same device
        if torch.is_tensor(preds) and torch.is_tensor(labels):
            labels = labels.to(preds.device)

        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        from utils.metrics import MAE_torch

        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


def run_single_experiment(config: dict, seed: int, data_file: str):
    """
    Run a single training experiment with given seed

    Args:
        config: Configuration dictionary
        seed: Random seed
        data_file: Path to data CSV file

    Returns:
        Dictionary of metrics
    """
    # Set all random seeds
    init_seed(seed)

    # Prepare data
    train_loader, val_loader, test_loader, scaler, num_nodes = prepare_data(
        csv_file=data_file,
        window=config.get("lag"),
        predict=config.get("horizon"),
        test_ratio=config.get("test_ratio"),
        val_ratio=config.get("val_ratio"),
    )

    # Get model arguments
    from paper_config import get_paper_config

    model_args, _ = get_paper_config()

    # Create SAMBA model
    model = SAMBA(
        model_args,
        hidden=config.get("rnn_units"),
        inp=config.get("lag"),
        out=config.get("horizon"),
        embed=config.get("embed_dim"),
        cheb_k=config.get("cheb_k"),
    )

    # Move to device
    device = torch.device(
        config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # Initialize loss, optimizer, scheduler
    loss = masked_mae_loss(scaler, mask_value=None)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.get("lr_init"))

    lr_scheduler = None
    if config.get("lr_decay"):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=config.get("lr_decay_step"),
            gamma=config.get("lr_decay_rate"),
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=config,
        lr_scheduler=lr_scheduler,
    )

    # Train and get metrics
    try:
        metrics, y_pred, y_true = trainer.train()
        return metrics
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return {
            "seed": seed,
            "val_loss": float("inf"),
            "test_mae": float("inf"),
            "test_rmse": float("inf"),
            "converged": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run multiple SAMBA experiments")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=20,
        help="Number of independent runs (default: 20)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DJI",
        choices=["DJI", "IXIC", "NYSE"],
        help="Dataset to use (DJI, IXIC, or NYSE)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: use config value)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: use config value)",
    )

    args = parser.parse_args()

    # Get base configuration
    model_args, config = get_paper_config()
    base_config = config.to_dict()

    # Override with command line arguments if provided
    if args.epochs:
        base_config["epochs"] = args.epochs
    if args.batch_size:
        base_config["batch_size"] = args.batch_size

    # Set dataset
    base_config["dataset"] = args.dataset
    data_file = f"Dataset/combined_dataframe_{args.dataset}.csv"

    # Verify data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        sys.exit(1)

    # Create experiment manager
    exp_manager = ExperimentManager(
        base_config=base_config, num_runs=args.num_runs, base_dir=args.base_dir
    )

    print("\n" + "=" * 70)
    print("🚀 SAMBA Multi-Run Experiment")
    print("=" * 70)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔢 Number of runs: {args.num_runs}")
    print(f"📁 Results directory: {exp_manager.base_dir}")
    print(f"⚙️  Epochs per run: {base_config.get('epochs')}")
    print(f"📦 Batch size: {base_config.get('batch_size')}")
    print("=" * 70 + "\n")

    # Run all experiments
    successful_runs = 0
    failed_runs = 0

    for i, seed in enumerate(tqdm(exp_manager.seeds, desc="Running experiments"), 1):
        print(f"\n{'─' * 70}")
        print(f"🔄 Run {i}/{args.num_runs} | Seed: {seed}")
        print(f"{'─' * 70}")

        # Get configuration for this run
        run_config = exp_manager.get_run_config(seed, i)

        # Set seeds
        exp_manager.set_all_seeds(seed)

        # Run experiment
        try:
            metrics = run_single_experiment(run_config, seed, data_file)

            # Save results
            exp_manager.save_run_results(seed, metrics)

            if metrics.get("converged", True):
                successful_runs += 1
                print(f"\n✅ Run {i} completed successfully!")
                print(f"   Val Loss: {metrics.get('val_loss', float('inf')):.4f}")
                print(f"   Test MAE: {metrics.get('test_mae', float('inf')):.4f}")
                print(f"   Test RMSE: {metrics.get('test_rmse', float('inf')):.4f}")
                print(f"   Best Epoch: {metrics.get('best_epoch', 'N/A')}")
            else:
                failed_runs += 1
                print(f"\n⚠️  Run {i} did not converge properly")

        except Exception as e:
            failed_runs += 1
            print(f"\n❌ Error in run {i} with seed {seed}: {str(e)}")
            # Save error metrics
            error_metrics = {
                "seed": seed,
                "val_loss": float("inf"),
                "test_mae": float("inf"),
                "test_rmse": float("inf"),
                "converged": False,
                "error": str(e),
            }
            exp_manager.save_run_results(seed, error_metrics)
            continue

    print(f"\n{'=' * 70}")
    print("📊 All Experiments Completed!")
    print(f"{'=' * 70}")
    print(f"✅ Successful runs: {successful_runs}/{args.num_runs}")
    if failed_runs > 0:
        print(f"❌ Failed runs: {failed_runs}/{args.num_runs}")
    print(f"{'=' * 70}\n")

    # Analyze results
    print("\n" + "=" * 70)
    print("📈 Analyzing Results...")
    print("=" * 70 + "\n")

    # Select best run
    best_run, best_seed = exp_manager.select_best_run(criterion="val_loss", mode="min")
    print(f"🏆 Best run: Seed {best_seed}")
    print(f"   Validation Loss: {best_run.get('val_loss', float('inf')):.4f}")
    print(f"   Test MAE: {best_run.get('test_mae', float('inf')):.4f}")
    print(f"   Test RMSE: {best_run.get('test_rmse', float('inf')):.4f}")
    print(f"   Best Epoch: {best_run.get('best_epoch', 'N/A')}")

    # Compute statistics
    stats = exp_manager.compute_statistics()
    print(f"\n📊 Statistics across {args.num_runs} runs:")
    print(
        f"   Test MAE: {stats['test_mae']['mean']:.4f} ± {stats['test_mae']['std']:.4f}"
    )
    print(
        f"   Test RMSE: {stats['test_rmse']['mean']:.4f} ± {stats['test_rmse']['std']:.4f}"
    )
    print(
        f"   Avg Best Epoch: {stats['best_epoch']['mean']:.1f} ± {stats['best_epoch']['std']:.1f}"
    )
    print(f"   Total Training Time: {stats['training_time']['total'] / 3600:.2f} hours")

    # Save summary
    exp_manager.save_summary(best_run, stats, criterion="val_loss")

    # Export results table
    print(f"\n📝 Exporting results...")
    exp_manager.export_results_table(format="all")

    # Create visualizations
    print(f"\n📸 Creating visualizations...")
    create_all_visualizations(exp_manager.base_dir, exp_manager.results, best_run)

    # Generate reports
    print(f"\n📄 Generating reports...")
    report_gen = ReportGenerator(exp_manager.base_dir)
    report_gen.generate_html_report(best_run, stats, exp_manager.results)
    report_gen.generate_latex_table(exp_manager.results, top_k=10)
    report_gen.generate_summary_text(best_run, stats)

    # Print top 5 runs
    print(f"\n🌟 Top 5 Runs:")
    print(f"{'─' * 70}")
    top_5 = exp_manager.get_top_k_runs(k=5, criterion="val_loss", mode="min")
    for rank, run in enumerate(top_5, 1):
        print(
            f"{rank}. Seed {run.get('seed'):3d} | Val Loss: {run.get('val_loss'):.4f} | "
            f"Test MAE: {run.get('test_mae'):.4f} | Test RMSE: {run.get('test_rmse'):.4f}"
        )

    print(f"\n{'=' * 70}")
    print("✨ Experiment Complete!")
    print(f"📁 All results saved to: {exp_manager.base_dir}")
    print(f"📊 View HTML report: {exp_manager.base_dir}/reports/experiment_report.html")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
