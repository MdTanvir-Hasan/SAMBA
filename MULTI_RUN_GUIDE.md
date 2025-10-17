# SAMBA Multi-Run Experiment System

Complete workflow for running 20+ experiments with different random seeds, analyzing results, and generating professional visualizations and reports.

## 📁 Project Structure

```
SAMBA/
├── experiment_manager.py          # Manages multiple training runs
├── run_multiple_experiments.py    # Main script for 20-run experiments
├── analyze_results.py             # Analyze completed experiments
├── compare_models.py              # Compare different models/experiments
├── demo.py                        # Interactive presentation script
├── oneforall.ipynb               # Jupyter notebook with all workflows
├── utils/
│   ├── visualization_utils.py    # Professional plotting functions
│   └── report_generator.py       # HTML/LaTeX report generation
└── experiments/                   # Generated experiment results
    └── SAMBA_DJI_20241006_143022/
        ├── run_seed_001/          # Individual run results
        │   ├── best_model.pth
        │   └── metrics.json
        ├── plots/                 # All visualizations
        │   ├── best_training_curves.png
        │   ├── metrics_boxplot.png
        │   ├── test_mae_distribution.png
        │   ├── val_vs_test_scatter.png
        │   ├── convergence_analysis.png
        │   └── top5_comparison.png
        ├── reports/               # Generated reports
        │   ├── experiment_report.html
        │   ├── results_table.tex
        │   └── summary.txt
        ├── experiment_summary.json
        ├── all_results.csv
        └── all_results.md
```

## 🚀 Quick Start

### 1. Run 20 Experiments

```bash
# Run 20 experiments with DJI dataset
python run_multiple_experiments.py --num_runs 20 --dataset DJI

# Quick test with 3 runs
python run_multiple_experiments.py --num_runs 3 --dataset DJI --epochs 50

# Run with custom settings
python run_multiple_experiments.py --num_runs 20 --dataset IXIC --epochs 200 --batch_size 64
```

**What it does:**
- Runs N independent training sessions with different random seeds (1, 2, ..., N)
- Saves each run's model checkpoint and metrics
- Automatically selects the best run based on validation loss
- Generates comprehensive statistics across all runs
- Creates all visualizations and reports

**Output:**
- Best model's test MAE, RMSE
- Statistical summary (mean ± std)
- Top-K performing runs
- Complete experiment directory with all artifacts

### 2. Analyze Results

```bash
# Analyze latest experiment
python analyze_results.py --exp_dir experiments/SAMBA_DJI_20241006_143022

# Regenerate all plots and reports
python analyze_results.py --exp_dir experiments/SAMBA_DJI_20241006_143022 \
    --regenerate_plots --regenerate_reports
```

**What it does:**
- Loads all run results
- Displays best run and statistics
- Shows top-K runs
- Regenerates visualizations and reports (optional)

### 3. Compare Models

```bash
# Compare SAMBA with baselines
python compare_models.py \
    --exp_dirs experiments/SAMBA_run experiments/GRU_run experiments/LSTM_run \
    --labels "SAMBA" "GRU" "LSTM"
```

**What it does:**
- Compares multiple experiment directories
- Creates side-by-side comparison plots
- Generates comparison table and report
- Shows performance, stability, and training time comparisons

### 4. Interactive Demo

```bash
# Run interactive presentation
python demo.py --exp_dir experiments/SAMBA_DJI_20241006_143022

# Auto-run without pauses
python demo.py --exp_dir experiments/SAMBA_DJI_20241006_143022 --auto
```

**What it does:**
- Beautiful colored terminal presentation
- Shows experiment overview, best results, statistics
- Displays top runs and available artifacts
- Perfect for live demonstrations

## 📊 Using the Notebook

Open `oneforall.ipynb` and run the cells sequentially:

1. **Cell 1**: Install requirements
2. **Cell 2**: Test system
3. **Cell 3**: Run 20 experiments
4. **Cell 4**: Analyze results
5. **Cell 5**: Display metrics
6. **Cell 6**: Show all visualizations
7. **Cell 7**: Open HTML report

## 📈 Generated Visualizations

### 1. Training Curves (best_training_curves.png)
Shows training and validation loss over epochs for the best performing run, with the best epoch marked.

### 2. Metrics Box Plot (metrics_boxplot.png)
Box plots showing distribution of validation loss, test MAE, test RMSE, and training time across all runs.

### 3. Test MAE Distribution (test_mae_distribution.png)
Histogram of test MAE values with mean, median, and best performance marked.

### 4. Validation vs Test Scatter (val_vs_test_scatter.png)
Scatter plot showing correlation between validation loss and test MAE, colored by seed.

### 5. Convergence Analysis (convergence_analysis.png)
- Distribution of best epochs
- Training time vs test performance

### 6. Top-5 Comparison (top5_comparison.png)
- Validation loss curves for top 5 runs
- Bar chart comparing test performance

## 📄 Generated Reports

### 1. HTML Report (experiment_report.html)
Professional interactive HTML report with:
- Experiment overview
- Best run results
- Statistical summary
- Top 10 runs table
- All visualizations embedded
- Responsive design, print-ready

**Open in browser:**
```bash
# Windows
start experiments/SAMBA_DJI_*/reports/experiment_report.html

# Mac/Linux
open experiments/SAMBA_DJI_*/reports/experiment_report.html
```

### 2. LaTeX Table (results_table.tex)
Ready-to-use LaTeX table for academic papers:
```latex
\begin{table}[ht]
\centering
\caption{Top Performing Runs - SAMBA Model}
\label{tab:samba_results}
\begin{tabular}{cccccc}
\hline
\textbf{Rank} & \textbf{Seed} & \textbf{Val Loss} & \textbf{Test MAE} & ...
...
\end{tabular}
\end{table}
```

### 3. Summary Text (summary.txt)
Markdown-formatted summary for README or documentation.

### 4. CSV Export (all_results.csv)
Complete results table for further analysis in Excel, R, Python, etc.

### 5. Markdown Table (all_results.md)
Markdown table for GitHub README or documentation.

## 🎯 Workflow for Paper Submission

### Step 1: Run Experiments
```bash
# Run SAMBA
python run_multiple_experiments.py --num_runs 20 --dataset DJI

# Run baselines
python run_multiple_experiments.py --num_runs 20 --dataset DJI --model GRU
python run_multiple_experiments.py --num_runs 20 --dataset DJI --model LSTM
```

### Step 2: Compare Models
```bash
python compare_models.py \
    --exp_dirs experiments/SAMBA_* experiments/GRU_* experiments/LSTM_* \
    --labels "SAMBA" "GRU" "LSTM"
```

### Step 3: Generate Paper Materials
1. **Main Results Table**: Use `reports/results_table.tex`
2. **Training Curves**: Use `plots/best_training_curves.png`
3. **Performance Comparison**: Use `comparisons/models_comparison.png`
4. **Statistical Summary**: From `experiment_summary.json`

### Step 4: Report Results

**In paper, write:**
> "We ran the experiments 20 different times with random initializations each time and picked the best results. The best run achieved a test MAE of X.XXXX, while the average performance across all runs was X.XXXX ± X.XXXX."

Use data from:
- Best run: `summary['best_run']['test_mae']`
- Statistics: `summary['statistics']['test_mae']['mean']` and `['std']`

## 📊 Understanding the Results

### Key Metrics

**Validation Loss**: Used to select the best model during training
- Lower is better
- Best epoch = epoch with lowest validation loss

**Test MAE (Mean Absolute Error)**: Primary evaluation metric
- Reported as final performance
- Lower is better
- Paper reports best run's test MAE

**Test RMSE (Root Mean Squared Error)**: Secondary metric
- Penalizes large errors more
- Lower is better

### Selection Process

1. **Each run**: Train model with seed i, save best checkpoint based on validation loss
2. **Across runs**: Select run with lowest validation loss
3. **Report**: Test MAE/RMSE from that best run
4. **Transparency**: Also report mean ± std across all runs

### Why 20 Runs?

- **Account for randomness**: Different initializations → different results
- **Statistical significance**: 20 runs provide robust statistics
- **Standard practice**: Common in time-series prediction papers
- **Fair comparison**: Apply same procedure to all baselines

## 🔧 Customization

### Custom Experiment Configuration

Edit `paper_config.py` or pass arguments:

```python
# In run_multiple_experiments.py
python run_multiple_experiments.py \
    --num_runs 20 \
    --dataset DJI \
    --epochs 200 \
    --batch_size 32 \
    --base_dir my_experiments
```

### Custom Visualization

Edit `utils/visualization_utils.py` to customize:
- Plot styles
- Colors
- Metrics to display
- Figure sizes

### Custom Reports

Edit `utils/report_generator.py` to customize:
- HTML template
- LaTeX formatting
- Summary content

## 🎓 Academic Usage

### Citing Results

```latex
@article{samba2024,
  title={SAMBA: A Graph-Mamba Approach for Stock Price Prediction},
  author={Your Name},
  journal={Conference},
  year={2024}
}
```

### Reporting in Paper

**Results Section:**
```
We evaluated SAMBA using 20 independent runs with different random 
initializations. The best run achieved a test MAE of X.XXXX (RMSE: X.XXXX), 
converging at epoch XXX. Across all runs, the average performance was 
X.XXXX ± X.XXXX, demonstrating stable performance. SAMBA outperformed 
baseline methods by XX% in terms of test MAE.
```

**Use the LaTeX table directly in your paper!**

## 💡 Tips

1. **Start with small runs**: Test with 3 runs before full 20
2. **Use GPU**: Experiments run much faster on GPU
3. **Save checkpoints**: All checkpoints are automatically saved
4. **HTML reports**: Best for quick overview and presentations
5. **CSV export**: Use for custom analysis in pandas/R
6. **Demo script**: Perfect for conference presentations

## 🐛 Troubleshooting

**Issue**: Out of memory
- **Solution**: Reduce batch size or use CPU for some runs

**Issue**: Experiments take too long
- **Solution**: Reduce epochs or num_runs for testing

**Issue**: Visualizations not showing
- **Solution**: Run `analyze_results.py` with `--regenerate_plots`

**Issue**: HTML report images missing
- **Solution**: Ensure plots directory exists and run analysis first

## 📝 License

MIT License - Feel free to use for your research!

---

**Questions?** Check the demo script or open an issue!
