# 4. Reproducing the Experiments

This document describes exactly how to reproduce the empirical results in the paper.

## What's in the experiment

The main experiment (`scripts/run_experiment.py`) sweeps:

- **5 methods**: Original, SMOTE, LLM-Unconstrained, LLM+SMOTE, DEMOAUG
- **2 regimes**: DistributionalBias (clean labels, harder topic), LabelNoiseBias (10% G=1 label flips, heavier contamination)
- **10 seeds**: 42, 1337, 2025, 7919, 31337, 5555, 8888, 1234, 9876, 17171
- **6 metrics**: AUC, ABROCA, Hard-bias_IH, DI gap, EO gap, Calib gap

Total: 5 × 2 × 10 = 100 runs, each producing 6 metric values + filter logs.

## Hardware requirements

The toy version (default) runs on a laptop CPU:

- **RAM**: 4 GB peak
- **Time**: ~50 seconds per (regime, seed) combination
- **Total runtime**: ~17 minutes for 10 seeds × 2 regimes
- **Disk**: ~30 MB for outputs

For the real-data run with bert-base-uncased + Mistral-7B-Instruct, see [`06_real_data_migration.md`](06_real_data_migration.md). Estimated ~1500 A100-GPU-hours for the full ablation matrix.

## Setup

```bash
# 1. Clone or unpack the repo
cd demoaug

# 2. Create a fresh Python environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify by running unit tests
pytest tests/ -v
# Expected: 17 passed in ~5 seconds
```

## Quick demo (~1 minute)

```bash
python scripts/run_demo.py
```

This runs a single seed of all 5 methods on a small (40%-scale) dataset and prints a comparison table. Use this to verify the pipeline is working before launching the full experiment.

## Full experiment (~17 minutes)

```bash
python scripts/run_experiment.py
```

This produces:

- `results/experiment_results.csv` — one row per (method, seed, regime), 100 rows total.
- `results/experiment_summary.json` — aggregated stats + significance tests.
- Live progress to stdout.

The script supports **resume-on-interruption**. If the run is killed, re-running picks up where it stopped (skips already-saved (regime, seed) combos in the CSV).

## Generate figures

```bash
python scripts/run_analysis.py
```

This reads `results/experiment_results.csv` and writes:

- `figures/fig_correlation.png` — headline figure: pooled Pearson r ≈ 0 between ΔHard-bias and ΔABROCA.
- `figures/fig_pareto.png` — Pareto frontier of methods on (Hard-bias, ABROCA) plane with 95% CI bars.
- `figures/fig_six_metrics.png` — all 6 metrics × 2 regimes bar chart.
- `figures/fig_density_gap.png` — log-density of real vs SMOTE vs LLM points.
- `results/analysis_report.md` — full Markdown report with tables + interpretation.

## Customizing the experiment

Edit `src/config.py`:

```python
# Reduce seeds for fast iteration
N_SEEDS = 3

# Change dataset size
DATA_SCALE = 0.2  # 20% of full Sha-Forum size, ~750 posts

# Adjust filter thresholds
TAU_LABEL = 0.7
DIVERSITY_DELTA = 0.5

# Reduce Method-3 iterations
METHOD3_K = 10
```

Then re-run `scripts/run_experiment.py`.

## What if the experiment crashes?

1. **Check `results/experiment_results.csv`** — partial results are saved after each (regime, seed). Re-running picks up from there.
2. **Reduce `N_SEEDS`** if memory is the issue.
3. **Reduce `METHOD3_K`** if individual runs are too slow.
4. **Run regimes separately** — `scripts/run_experiment.py` accepts `--regime DistributionalBias` or `--regime LabelNoiseBias` to run one at a time.

## Reproducibility notes

The experiment is deterministic given:

- Fixed seeds (controlled by `SEED` and seed list in `scripts/run_experiment.py`).
- Pinned package versions in `requirements.txt`.
- Single-threaded for sklearn calls (we use `random_state=SEED` everywhere).

You should reproduce the headline findings (pooled $r$ near 0) within ±0.02 across hardware. Per-seed point estimates may differ slightly across NumPy/SciPy versions due to floating-point accumulation order, but aggregate findings are stable.

## Expected output

After a successful run, you should see (approximately):

| Regime | DEMOAUG ABROCA | DEMOAUG Hard-bias_IH | Pooled $r$(ΔHard-bias, ΔABROCA) |
|---|---|---|---|
| DistributionalBias | 0.032 ± 0.016 | 0.099 ± 0.027 | +0.06 |
| LabelNoiseBias | 0.116 ± 0.039 | 0.041 ± 0.013 | +0.00 |

These are the numbers that back the paper's headline claim.

## Verifying the install

Before submitting your reproduction, run:

```bash
pytest tests/ -v
```

All 17 tests must pass. The `test_density_smote_below_real` test is particularly relevant — it verifies the geometric claim about SMOTE points lying below real data in PCA-whitened density.
