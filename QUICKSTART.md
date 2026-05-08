# Quickstart

This is the minimal command sequence for running the experiments locally.

## Prerequisites

- Python 3.10 or 3.11 or 3.12
- ~500 MB free disk space
- ~4 GB RAM
- ~20 minutes for the full experiment (CPU only)

## One-time setup

```bash
# 1. Get the code
cd path/to/wherever/you/keep/projects
# (clone or unpack here)
cd demoaug

# 2. Create a virtual environment (recommended, not required)
python -m venv venv
source venv/bin/activate              # Linux/macOS
# OR on Windows:
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the install
pytest tests/ -v
# Expected: 17 passed in ~5 seconds.
```

## The three commands you'll actually run

```bash
# Quick demo (~1 minute) — sanity check the pipeline
python scripts/run_demo.py

# Full experiment (~20 minutes on CPU) — produces results/experiment_results.csv
python scripts/run_experiment.py

# Generate figures and analysis report
python scripts/run_analysis.py
```

## Where output lands

After running the three commands above:

```
demoaug/
├── results/
│   ├── experiment_results.csv      # 100 rows: 5 methods × 2 regimes × 10 seeds
│   ├── experiment_summary.json     # statistical tests, correlations
│   └── analysis_report.md          # human-readable summary
└── figures/
    ├── fig_correlation.png         # the headline figure
    ├── fig_pareto.png              # Pareto frontier of methods
    └── fig_six_metrics.png         # all 6 metrics × 2 regimes
```

## What if something goes wrong?

### "ModuleNotFoundError: No module named 'src'"
Run from the repo root, not from inside a subdirectory.

### "ModuleNotFoundError: No module named 'imblearn'"
Run `pip install -r requirements.txt` again.

### Tests fail
Check your Python version (must be 3.10+) and that `pip install -r requirements.txt` completed without errors. Some tests depend on numpy ≥ 2.0.

### Experiment hangs / runs out of memory
Edit `src/config.py`:
```python
N_SEEDS = 3              # was 10
DATA_SCALE = 0.2         # was 0.4
METHOD3_K = 10           # was 25
```
Re-run.

### "ConvergenceWarning" warnings
These are harmless. The MLP probes don't always converge in 200 iterations on small data; the results are still valid.

### Want to run only one regime
```bash
python scripts/run_experiment.py --regime DistributionalBias
# or
python scripts/run_experiment.py --regime LabelNoiseBias
```

### Run interrupted mid-experiment
Just run the same command again. Resume support skips already-completed (regime, seed) pairs.

## What to read after running

1. `results/analysis_report.md` — auto-generated report with all tables.
2. `figures/fig_correlation.png` — the headline figure.
3. `docs/05_results_interpretation.md` — how to read all the numbers and figures.
