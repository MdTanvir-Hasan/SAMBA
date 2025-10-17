# -*- coding: utf-8 -*-
"""
Report Generator for SAMBA Experiments
Creates HTML reports, PDF summaries, and LaTeX tables
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """Generate professional reports for experiment results"""

    def __init__(self, exp_dir: Path):
        """
        Args:
            exp_dir: Experiment directory
        """
        self.exp_dir = Path(exp_dir)
        self.reports_dir = self.exp_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def generate_html_report(
        self, best_run: dict, stats: dict, all_results: List[dict]
    ):
        """Generate comprehensive HTML report"""

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAMBA Experiment Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .metric-card .subvalue {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        table tr:hover {{
            background-color: #f0f0f0;
        }}
        .highlight {{
            background-color: #ffeaa7 !important;
            font-weight: bold;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-container {{
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .image-container p {{
            margin-top: 10px;
            color: #666;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        .success {{
            color: #27ae60;
        }}
        .info {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SAMBA Experiment Report</h1>
        <p>Graph-Mamba Approach for Stock Price Prediction</p>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="section">
        <h2>📊 Experiment Overview</h2>
        <div class="info">
            <strong>Total Runs:</strong> {stats["num_runs"]}<br>
            <strong>Successful Runs:</strong> {stats["num_successful"]}<br>
            <strong>Selection Criterion:</strong> Validation Loss (Lower is Better)<br>
            <strong>Total Training Time:</strong> {stats["training_time"]["total"] / 3600:.2f} hours
        </div>
    </div>

    <div class="section">
        <h2>🏆 Best Run Results</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Seed</div>
                <div class="value">{best_run.get("seed", "N/A")}</div>
            </div>
            <div class="metric-card">
                <div class="label">Best Epoch</div>
                <div class="value">{best_run.get("best_epoch", "N/A")}</div>
            </div>
            <div class="metric-card">
                <div class="label">Validation Loss</div>
                <div class="value">{best_run.get("val_loss", 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Test MAE</div>
                <div class="value">{best_run.get("test_mae", 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Test RMSE</div>
                <div class="value">{best_run.get("test_rmse", 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Training Time</div>
                <div class="value">{best_run.get("training_time", 0) / 60:.1f}</div>
                <div class="subvalue">minutes</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Statistical Summary (Across All Runs)</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Test MAE</div>
                <div class="value">{stats["test_mae"]["mean"]:.4f}</div>
                <div class="subvalue">± {stats["test_mae"]["std"]:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Test RMSE</div>
                <div class="value">{stats["test_rmse"]["mean"]:.4f}</div>
                <div class="subvalue">± {stats["test_rmse"]["std"]:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Val Loss Range</div>
                <div class="value">[{stats["val_loss"]["min"]:.4f}, {stats["val_loss"]["max"]:.4f}]</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Convergence</div>
                <div class="value">{stats["best_epoch"]["mean"]:.0f}</div>
                <div class="subvalue">epochs</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📋 Top 10 Runs</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Seed</th>
                    <th>Val Loss</th>
                    <th>Test MAE</th>
                    <th>Test RMSE</th>
                    <th>Best Epoch</th>
                    <th>Time (min)</th>
                </tr>
            </thead>
            <tbody>
"""

        # Sort and get top 10
        sorted_results = sorted(
            all_results, key=lambda x: x.get("val_loss", float("inf"))
        )[:10]
        for rank, result in enumerate(sorted_results, 1):
            row_class = "highlight" if rank == 1 else ""
            html_content += f"""
                <tr class="{row_class}">
                    <td>{rank}</td>
                    <td>{result.get("seed", "N/A")}</td>
                    <td>{result.get("val_loss", 0):.4f}</td>
                    <td>{result.get("test_mae", 0):.4f}</td>
                    <td>{result.get("test_rmse", 0):.4f}</td>
                    <td>{result.get("best_epoch", "N/A")}</td>
                    <td>{result.get("training_time", 0) / 60:.1f}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>📸 Visualizations</h2>
        <div class="image-grid">
            <div class="image-container">
                <img src="../plots/best_training_curves.png" alt="Training Curves">
                <p>Best Run Training Curves</p>
            </div>
            <div class="image-container">
                <img src="../plots/metrics_boxplot.png" alt="Metrics Distribution">
                <p>Metrics Distribution Across All Runs</p>
            </div>
            <div class="image-container">
                <img src="../plots/test_mae_distribution.png" alt="MAE Distribution">
                <p>Test MAE Distribution</p>
            </div>
            <div class="image-container">
                <img src="../plots/val_vs_test_scatter.png" alt="Validation vs Test">
                <p>Validation vs Test Performance</p>
            </div>
            <div class="image-container">
                <img src="../plots/convergence_analysis.png" alt="Convergence Analysis">
                <p>Convergence Analysis</p>
            </div>
            <div class="image-container">
                <img src="../plots/top5_comparison.png" alt="Top 5 Comparison">
                <p>Top 5 Runs Comparison</p>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>SAMBA: Graph-Mamba Approach for Stock Price Prediction</p>
        <p>Report generated automatically by ExperimentManager</p>
    </div>
</body>
</html>
"""

        # Save HTML report
        report_path = self.reports_dir / "experiment_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML report saved to: {report_path}")
        return report_path

    def generate_latex_table(self, all_results: List[dict], top_k: int = 10):
        """Generate LaTeX table for paper"""

        sorted_results = sorted(
            all_results, key=lambda x: x.get("val_loss", float("inf"))
        )[:top_k]

        latex_content = r"""\begin{table}[ht]
\centering
\caption{Top Performing Runs - SAMBA Model}
\label{tab:samba_results}
\begin{tabular}{cccccc}
\hline
\textbf{Rank} & \textbf{Seed} & \textbf{Val Loss} & \textbf{Test MAE} & \textbf{Test RMSE} & \textbf{Epoch} \\
\hline
"""

        for rank, result in enumerate(sorted_results, 1):
            latex_content += f"""{rank} & {result.get("seed", "-")} & {result.get("val_loss", 0):.4f} & {result.get("test_mae", 0):.4f} & {result.get("test_rmse", 0):.4f} & {result.get("best_epoch", "-")} \\\\
"""

        latex_content += r"""\hline
\end{tabular}
\end{table}
"""

        # Save LaTeX table
        latex_path = self.reports_dir / "results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex_content)

        print(f"✅ LaTeX table saved to: {latex_path}")
        return latex_path

    def generate_summary_text(self, best_run: dict, stats: dict):
        """Generate text summary for README or presentation"""

        summary = f"""
# SAMBA Experiment Results Summary

## Best Performance
- **Seed:** {best_run.get("seed", "N/A")}
- **Validation Loss:** {best_run.get("val_loss", 0):.4f}
- **Test MAE:** {best_run.get("test_mae", 0):.4f}
- **Test RMSE:** {best_run.get("test_rmse", 0):.4f}
- **Best Epoch:** {best_run.get("best_epoch", "N/A")}
- **Training Time:** {best_run.get("training_time", 0) / 60:.2f} minutes

## Statistical Summary (All Runs)
- **Number of Runs:** {stats["num_runs"]}
- **Successful Runs:** {stats["num_successful"]}
- **Average Test MAE:** {stats["test_mae"]["mean"]:.4f} ± {stats["test_mae"]["std"]:.4f}
- **Average Test RMSE:** {stats["test_rmse"]["mean"]:.4f} ± {stats["test_rmse"]["std"]:.4f}
- **Best Test MAE:** {stats["test_mae"]["min"]:.4f}
- **Total Training Time:** {stats["training_time"]["total"] / 3600:.2f} hours

## Key Findings
- Model converged on average at epoch {stats["best_epoch"]["mean"]:.0f} (±{stats["best_epoch"]["std"]:.0f})
- Performance variance (Test MAE std): {stats["test_mae"]["std"]:.4f}
- Best run outperformed average by {(stats["test_mae"]["mean"] - best_run.get("test_mae", 0)):.4f} MAE points

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        summary_path = self.reports_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)

        print(f"✅ Summary saved to: {summary_path}")
        return summary_path
