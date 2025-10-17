# -*- coding: utf-8 -*-
"""
Experiment Manager for running multiple training runs with different random seeds
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class ExperimentManager:
    """Manages multiple training runs with different random seeds"""

    def __init__(
        self, base_config: dict, num_runs: int = 20, base_dir: str = "experiments"
    ):
        """
        Args:
            base_config: Base configuration dictionary
            num_runs: Number of independent runs
            base_dir: Base directory for all experiments
        """
        self.base_config = base_config.copy()
        self.num_runs = num_runs
        self.base_dir = Path(base_dir)
        self.seeds = list(range(1, num_runs + 1))  # Seeds: 1, 2, 3, ..., num_runs
        self.results = []

        # Create base experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.base_config.get('model', 'model')}_{self.base_config.get('dataset', 'data')}_{timestamp}"
        self.base_dir = self.base_dir / self.experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Save base configuration
        self._save_base_config()

    def _save_base_config(self):
        """Save base configuration"""
        config_path = self.base_dir / "base_config.json"
        with open(config_path, "w") as f:
            # Convert non-serializable items
            config_copy = {}
            for k, v in self.base_config.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    config_copy[k] = v
                else:
                    config_copy[k] = str(v)
            json.dump(config_copy, f, indent=4)

    def get_run_config(self, seed: int, run_id: int) -> dict:
        """Create configuration for a specific run"""
        config = self.base_config.copy()

        # Set unique seed
        config["seed"] = seed
        config["run_id"] = run_id

        # Create unique log directory for this run
        run_dir = self.base_dir / f"run_seed_{seed:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config["log_dir"] = str(run_dir)

        return config

    def set_all_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def save_run_results(self, seed: int, metrics: dict):
        """Save results for a single run"""
        run_dir = self.base_dir / f"run_seed_{seed:03d}"
        results_path = run_dir / "metrics.json"

        # Add seed and timestamp to metrics
        metrics["seed"] = seed
        metrics["timestamp"] = datetime.now().isoformat()

        # Convert numpy types to Python native types
        metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics_clean[k] = v.tolist()
            elif isinstance(v, (np.int64, np.int32)):
                metrics_clean[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                metrics_clean[k] = float(v)
            elif (
                isinstance(v, list)
                and len(v) > 0
                and isinstance(v[0], (np.int64, np.float64))
            ):
                metrics_clean[k] = [float(x) for x in v]
            else:
                metrics_clean[k] = v

        with open(results_path, "w") as f:
            json.dump(metrics_clean, f, indent=4)

        self.results.append(metrics_clean)

    def load_all_results(self) -> List[dict]:
        """Load results from all runs"""
        all_results = []

        for seed in self.seeds:
            run_dir = self.base_dir / f"run_seed_{seed:03d}"
            results_path = run_dir / "metrics.json"

            if results_path.exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
                    all_results.append(results)

        return all_results

    def select_best_run(
        self, criterion: str = "val_loss", mode: str = "min"
    ) -> Tuple[dict, int]:
        """
        Select the best run based on validation metric

        Args:
            criterion: Metric to use for selection ('val_loss', 'val_mae', etc.)
            mode: 'min' for loss-like metrics, 'max' for accuracy-like metrics

        Returns:
            Tuple of (best_run_metrics, best_seed)
        """
        if not self.results:
            self.results = self.load_all_results()

        if not self.results:
            raise ValueError("No results found. Run experiments first.")

        # Sort based on criterion
        if mode == "min":
            best_run = min(self.results, key=lambda x: x.get(criterion, float("inf")))
        else:
            best_run = max(self.results, key=lambda x: x.get(criterion, float("-inf")))

        best_seed = best_run["seed"]

        return best_run, best_seed

    def get_top_k_runs(
        self, k: int = 5, criterion: str = "val_loss", mode: str = "min"
    ) -> List[dict]:
        """Get top k performing runs"""
        if not self.results:
            self.results = self.load_all_results()

        sorted_results = sorted(
            self.results,
            key=lambda x: x.get(
                criterion, float("inf") if mode == "min" else float("-inf")
            ),
            reverse=(mode == "max"),
        )

        return sorted_results[:k]

    def compute_statistics(self) -> dict:
        """Compute statistics across all runs"""
        if not self.results:
            self.results = self.load_all_results()

        # Extract metrics
        val_losses = [r.get("val_loss", np.nan) for r in self.results]
        test_maes = [r.get("test_mae", np.nan) for r in self.results]
        test_rmses = [r.get("test_rmse", np.nan) for r in self.results]
        test_ics = [r.get("test_ic", np.nan) for r in self.results]
        test_rics = [r.get("test_ric", np.nan) for r in self.results]
        training_times = [r.get("training_time", np.nan) for r in self.results]
        best_epochs = [r.get("best_epoch", np.nan) for r in self.results]

        stats = {
            "num_runs": len(self.results),
            "num_successful": sum(1 for r in self.results if r.get("converged", True)),
            "val_loss": {
                "mean": float(np.nanmean(val_losses)),
                "std": float(np.nanstd(val_losses)),
                "min": float(np.nanmin(val_losses)),
                "max": float(np.nanmax(val_losses)),
                "median": float(np.nanmedian(val_losses)),
            },
            "test_mae": {
                "mean": float(np.nanmean(test_maes)),
                "std": float(np.nanstd(test_maes)),
                "min": float(np.nanmin(test_maes)),
                "max": float(np.nanmax(test_maes)),
                "median": float(np.nanmedian(test_maes)),
            },
            "test_rmse": {
                "mean": float(np.nanmean(test_rmses)),
                "std": float(np.nanstd(test_rmses)),
                "min": float(np.nanmin(test_rmses)),
                "max": float(np.nanmax(test_rmses)),
                "median": float(np.nanmedian(test_rmses)),
            },
            "test_ic": {
                "mean": float(np.nanmean(test_ics)),
                "std": float(np.nanstd(test_ics)),
                "min": float(np.nanmin(test_ics)),
                "max": float(np.nanmax(test_ics)),
                "median": float(np.nanmedian(test_ics)),
            },
            "test_ric": {
                "mean": float(np.nanmean(test_rics)),
                "std": float(np.nanstd(test_rics)),
                "min": float(np.nanmin(test_rics)),
                "max": float(np.nanmax(test_rics)),
                "median": float(np.nanmedian(test_rics)),
            },
            "training_time": {
                "mean": float(np.nanmean(training_times)),
                "std": float(np.nanstd(training_times)),
                "total": float(np.nansum(training_times)),
                "min": float(np.nanmin(training_times)),
                "max": float(np.nanmax(training_times)),
            },
            "best_epoch": {
                "mean": float(np.nanmean(best_epochs)),
                "std": float(np.nanstd(best_epochs)),
                "min": float(np.nanmin(best_epochs)),
                "max": float(np.nanmax(best_epochs)),
            },
        }

        return stats

    def save_summary(self, best_run: dict, stats: dict, criterion: str = "val_loss"):
        """Save experiment summary"""
        summary = {
            "experiment_info": {
                "experiment_name": self.experiment_name,
                "num_runs": self.num_runs,
                "seeds": self.seeds,
                "selection_criterion": criterion,
                "timestamp": datetime.now().isoformat(),
            },
            "best_run": best_run,
            "statistics": stats,
            "all_runs_summary": [
                {
                    "seed": r["seed"],
                    "val_loss": r.get("val_loss", None),
                    "test_mae": r.get("test_mae", None),
                    "test_rmse": r.get("test_rmse", None),
                    "test_ic": r.get("test_ic", None),
                    "test_ric": r.get("test_ric", None),
                    "best_epoch": r.get("best_epoch", None),
                    "training_time": r.get("training_time", None),
                }
                for r in self.results
            ],
        }

        summary_path = self.base_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"✅ Summary saved to: {summary_path}")

    def export_results_table(self, format: str = "csv") -> pd.DataFrame:
        """Export results as table (CSV, Markdown, LaTeX)"""
        if not self.results:
            self.results = self.load_all_results()

        # Create DataFrame with key metrics
        data = []
        for r in self.results:
            data.append(
                {
                    "Seed": r["seed"],
                    "Val Loss": r.get("val_loss", np.nan),
                    "Test MAE": r.get("test_mae", np.nan),
                    "Test RMSE": r.get("test_rmse", np.nan),
                    "Test IC": r.get("test_ic", np.nan),
                    "Test RIC": r.get("test_ric", np.nan),
                    "Best Epoch": r.get("best_epoch", np.nan),
                    "Training Time (min)": r.get("training_time", np.nan) / 60
                    if r.get("training_time")
                    else np.nan,
                    "Converged": r.get("converged", True),
                }
            )

        df = pd.DataFrame(data)

        # Sort by validation loss
        df = df.sort_values("Val Loss")

        # Save in different formats
        if format == "csv" or format == "all":
            csv_path = self.base_dir / "all_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV table saved to: {csv_path}")

        if format == "markdown" or format == "all":
            md_path = self.base_dir / "all_results.md"
            with open(md_path, "w") as f:
                f.write("# Experiment Results\n\n")
                f.write(df.to_markdown(index=False, floatfmt=".4f"))
            print(f"✅ Markdown table saved to: {md_path}")

        if format == "latex" or format == "all":
            latex_path = self.base_dir / "all_results.tex"
            with open(latex_path, "w") as f:
                f.write(df.to_latex(index=False, float_format="%.4f"))
            print(f"✅ LaTeX table saved to: {latex_path}")

        return df

    def get_experiment_info(self) -> dict:
        """Get experiment information"""
        return {
            "experiment_name": self.experiment_name,
            "base_dir": str(self.base_dir),
            "num_runs": self.num_runs,
            "seeds": self.seeds,
            "completed_runs": len(self.results)
            if self.results
            else len(self.load_all_results()),
        }
