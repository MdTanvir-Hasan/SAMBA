# -*- coding: utf-8 -*-
"""
Interactive Demo Script for SAMBA Presentation
Shows results dynamically with professional formatting
"""

import sys
import json
from pathlib import Path
from time import sleep
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style

# Initialize colorama for colored terminal output
init(autoreset=True)


class SAMBADemo:
    """Interactive demo presenter for SAMBA experiments"""

    def __init__(self, exp_dir: str):
        """
        Args:
            exp_dir: Path to experiment directory
        """
        self.exp_dir = Path(exp_dir)
        self.summary = None
        self.load_summary()

    def load_summary(self):
        """Load experiment summary"""
        summary_path = self.exp_dir / "experiment_summary.json"
        if not summary_path.exists():
            print(f"{Fore.RED}❌ Experiment summary not found in {self.exp_dir}")
            sys.exit(1)

        with open(summary_path, "r") as f:
            self.summary = json.load(f)

    def print_header(self, text: str, color=Fore.CYAN):
        """Print a formatted header"""
        print(f"\n{color}{'=' * 80}")
        print(f"{color}{text:^80}")
        print(f"{color}{'=' * 80}\n")

    def print_section(self, text: str):
        """Print a section title"""
        print(f"\n{Fore.YELLOW}{'─' * 80}")
        print(f"{Fore.YELLOW}{text}")
        print(f"{Fore.YELLOW}{'─' * 80}")

    def animate_text(self, text: str, delay: float = 0.03):
        """Animate text character by character"""
        for char in text:
            print(char, end="", flush=True)
            sleep(delay)
        print()

    def show_intro(self):
        """Show introduction"""
        self.print_header("🚀 SAMBA: Graph-Mamba for Stock Price Prediction", Fore.CYAN)

        print(
            f"{Fore.WHITE}Paper: {Fore.GREEN}A Graph-Mamba Approach for Stock Price Prediction"
        )
        print(f"{Fore.WHITE}Task:  {Fore.GREEN}Multi-step Time Series Forecasting")
        print(
            f"{Fore.WHITE}Model: {Fore.GREEN}SAMBA (Combining GNN + Mamba Architecture)"
        )

        sleep(1)

    def show_experiment_overview(self):
        """Show experiment overview"""
        self.print_header("📊 Experiment Overview", Fore.MAGENTA)

        info = self.summary["experiment_info"]
        stats = self.summary["statistics"]

        print(
            f"{Fore.WHITE}Experiment Name: {Fore.CYAN}{info.get('experiment_name', 'N/A')}"
        )
        print(f"{Fore.WHITE}Number of Runs:  {Fore.CYAN}{info['num_runs']}")
        print(
            f"{Fore.WHITE}Successful Runs: {Fore.GREEN}{stats['num_successful']}/{info['num_runs']}"
        )
        print(
            f"{Fore.WHITE}Total Time:      {Fore.CYAN}{stats['training_time']['total'] / 3600:.2f} hours"
        )

        sleep(1)

    def show_best_results(self):
        """Show best run results"""
        self.print_header("🏆 Best Model Performance", Fore.GREEN)

        best_run = self.summary["best_run"]

        print(f"{Fore.YELLOW}Configuration:")
        print(f"  {Fore.WHITE}Seed:        {Fore.CYAN}{best_run.get('seed', 'N/A')}")
        print(
            f"  {Fore.WHITE}Best Epoch:  {Fore.CYAN}{best_run.get('best_epoch', 'N/A')}"
        )
        print(
            f"  {Fore.WHITE}Training Time: {Fore.CYAN}{best_run.get('training_time', 0) / 60:.2f} minutes"
        )

        print(f"\n{Fore.YELLOW}Performance Metrics:")
        print(
            f"  {Fore.WHITE}Validation Loss: {Fore.GREEN}{best_run.get('val_loss', 0):.4f}"
        )
        print(
            f"  {Fore.WHITE}Test MAE:        {Fore.GREEN}{best_run.get('test_mae', 0):.4f}"
        )
        print(
            f"  {Fore.WHITE}Test RMSE:       {Fore.GREEN}{best_run.get('test_rmse', 0):.4f}"
        )

        if best_run.get("early_stopped", False):
            print(f"\n  {Fore.CYAN}✓ Model used early stopping")
        if best_run.get("converged", True):
            print(f"  {Fore.GREEN}✓ Training converged successfully")

        sleep(1.5)

    def show_statistics(self):
        """Show statistical summary"""
        self.print_header("📈 Statistical Analysis Across All Runs", Fore.BLUE)

        stats = self.summary["statistics"]

        print(f"{Fore.YELLOW}Test MAE Statistics:")
        print(
            f"  {Fore.WHITE}Mean:   {Fore.CYAN}{stats['test_mae']['mean']:.4f} {Fore.WHITE}± {Fore.CYAN}{stats['test_mae']['std']:.4f}"
        )
        print(f"  {Fore.WHITE}Best:   {Fore.GREEN}{stats['test_mae']['min']:.4f}")
        print(f"  {Fore.WHITE}Worst:  {Fore.RED}{stats['test_mae']['max']:.4f}")
        print(f"  {Fore.WHITE}Median: {Fore.CYAN}{stats['test_mae']['median']:.4f}")

        print(f"\n{Fore.YELLOW}Test RMSE Statistics:")
        print(
            f"  {Fore.WHITE}Mean:   {Fore.CYAN}{stats['test_rmse']['mean']:.4f} {Fore.WHITE}± {Fore.CYAN}{stats['test_rmse']['std']:.4f}"
        )
        print(f"  {Fore.WHITE}Best:   {Fore.GREEN}{stats['test_rmse']['min']:.4f}")
        print(f"  {Fore.WHITE}Worst:  {Fore.RED}{stats['test_rmse']['max']:.4f}")

        print(f"\n{Fore.YELLOW}Convergence Statistics:")
        print(
            f"  {Fore.WHITE}Avg Best Epoch: {Fore.CYAN}{stats['best_epoch']['mean']:.1f} {Fore.WHITE}± {Fore.CYAN}{stats['best_epoch']['std']:.1f}"
        )
        print(
            f"  {Fore.WHITE}Fastest:        {Fore.GREEN}{stats['best_epoch']['min']:.0f} {Fore.WHITE}epochs"
        )
        print(
            f"  {Fore.WHITE}Slowest:        {Fore.RED}{stats['best_epoch']['max']:.0f} {Fore.WHITE}epochs"
        )

        # Show improvement over average
        best_run = self.summary["best_run"]
        improvement = (
            (stats["test_mae"]["mean"] - best_run.get("test_mae", 0))
            / stats["test_mae"]["mean"]
        ) * 100
        print(f"\n{Fore.YELLOW}Performance Gain:")
        print(
            f"  {Fore.WHITE}Best run outperforms average by: {Fore.GREEN}{improvement:.2f}%"
        )

        sleep(2)

    def show_top_runs(self, k: int = 5):
        """Show top K runs"""
        self.print_header(f"🌟 Top {k} Performing Runs", Fore.YELLOW)

        all_runs = self.summary.get("all_runs_summary", [])
        sorted_runs = sorted(all_runs, key=lambda x: x.get("val_loss", float("inf")))[
            :k
        ]

        print(
            f"{Fore.WHITE}{'Rank':<6}{'Seed':<8}{'Val Loss':<12}{'Test MAE':<12}{'Test RMSE':<12}{'Epoch':<8}"
        )
        print(f"{Fore.WHITE}{'-' * 70}")

        for rank, run in enumerate(sorted_runs, 1):
            color = Fore.GREEN if rank == 1 else Fore.CYAN
            print(
                f"{color}{rank:<6}{run.get('seed', 'N/A'):<8}"
                f"{run.get('val_loss', 0):<12.4f}"
                f"{run.get('test_mae', 0):<12.4f}"
                f"{run.get('test_rmse', 0):<12.4f}"
                f"{run.get('best_epoch', 'N/A'):<8}"
            )

        sleep(1)

    def show_visualizations(self):
        """Show available visualizations"""
        self.print_header("📸 Available Visualizations", Fore.MAGENTA)

        plots_dir = self.exp_dir / "plots"

        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))

            print(
                f"{Fore.WHITE}Generated {Fore.CYAN}{len(plot_files)}{Fore.WHITE} visualization plots:"
            )
            print()

            for i, plot in enumerate(plot_files, 1):
                print(
                    f"  {Fore.CYAN}{i}. {Fore.WHITE}{plot.stem.replace('_', ' ').title()}"
                )

            print(f"\n{Fore.YELLOW}📁 Location: {Fore.WHITE}{plots_dir}")
        else:
            print(f"{Fore.RED}❌ No plots directory found")

        sleep(1)

    def show_reports(self):
        """Show available reports"""
        self.print_header("📄 Generated Reports", Fore.CYAN)

        reports_dir = self.exp_dir / "reports"

        if reports_dir.exists():
            files = {
                "experiment_report.html": "📊 Interactive HTML Report",
                "results_table.tex": "📝 LaTeX Table for Paper",
                "summary.txt": "📋 Text Summary",
            }

            print(f"{Fore.WHITE}Available reports:")
            print()

            for filename, description in files.items():
                filepath = reports_dir / filename
                if filepath.exists():
                    print(f"  {Fore.GREEN}✓ {Fore.WHITE}{description}")
                    print(f"    {Fore.CYAN}{filepath}")
                else:
                    print(f"  {Fore.RED}✗ {Fore.WHITE}{description}")

            print(f"\n{Fore.YELLOW}📁 Location: {Fore.WHITE}{reports_dir}")
        else:
            print(f"{Fore.RED}❌ No reports directory found")

        sleep(1)

    def show_conclusion(self):
        """Show conclusion"""
        self.print_header("✨ Summary", Fore.GREEN)

        best_run = self.summary["best_run"]
        stats = self.summary["statistics"]

        print(f"{Fore.WHITE}Key Takeaways:")
        print(
            f"  {Fore.GREEN}✓ {Fore.WHITE}Completed {Fore.CYAN}{stats['num_runs']} {Fore.WHITE}independent runs"
        )
        print(
            f"  {Fore.GREEN}✓ {Fore.WHITE}Best Test MAE: {Fore.CYAN}{best_run.get('test_mae', 0):.4f}"
        )
        print(
            f"  {Fore.GREEN}✓ {Fore.WHITE}Average performance: {Fore.CYAN}{stats['test_mae']['mean']:.4f} ± {stats['test_mae']['std']:.4f}"
        )
        print(
            f"  {Fore.GREEN}✓ {Fore.WHITE}Model shows {Fore.CYAN}stable performance {Fore.WHITE}across different initializations"
        )

        print(f"\n{Fore.YELLOW}Next Steps:")
        print(f"  {Fore.WHITE}• View detailed HTML report for comprehensive analysis")
        print(f"  {Fore.WHITE}• Examine visualizations for insights")
        print(f"  {Fore.WHITE}• Compare with baseline models")
        print(f"  {Fore.WHITE}• Use LaTeX table for paper publication")

        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.CYAN}{'Thank you for your attention!':^80}")
        print(f"{Fore.CYAN}{'=' * 80}\n")

    def run_demo(self, pause_between_sections: bool = True):
        """Run the complete demo"""
        sections = [
            ("Introduction", self.show_intro),
            ("Experiment Overview", self.show_experiment_overview),
            ("Best Results", self.show_best_results),
            ("Statistical Analysis", self.show_statistics),
            ("Top Runs", self.show_top_runs),
            ("Visualizations", self.show_visualizations),
            ("Reports", self.show_reports),
            ("Conclusion", self.show_conclusion),
        ]

        for i, (name, func) in enumerate(sections, 1):
            func()

            if pause_between_sections and i < len(sections):
                input(f"\n{Fore.WHITE}Press Enter to continue to next section...")

        print(f"\n{Fore.GREEN}✅ Demo completed!\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive SAMBA Demo")
    parser.add_argument(
        "--exp_dir", type=str, required=True, help="Experiment directory to present"
    )
    parser.add_argument(
        "--auto", action="store_true", help="Run demo automatically without pauses"
    )

    args = parser.parse_args()

    # Create and run demo
    demo = SAMBADemo(args.exp_dir)
    demo.run_demo(pause_between_sections=not args.auto)


if __name__ == "__main__":
    main()
