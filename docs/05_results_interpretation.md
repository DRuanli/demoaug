# 5. Reading the Results

This document explains how to interpret each figure and table the experiment produces.

## The four figures

### `figures/fig_correlation.png` — the headline figure

Two scatter plots (one per regime). Each point is a (method, seed) combination. The x-axis is `Δ Hard-bias_IH` (vs Original), the y-axis is `Δ ABROCA` (vs Original). The yellow box shows pooled Pearson correlation across all 40 (4 methods × 10 seeds) points in that regime.

**What you should see**:

- A horizontal-band shape: methods spread along the x-axis (some reduce Hard-bias more than others) but the y-axis is roughly flat (no consistent vertical pattern).
- Pooled $r$ near zero in both regimes, with 95% CI containing zero.
- The "ideal" quadrant (lower-right: both metrics improved) is not preferentially populated.

**Why it matters**: This visual is the strongest evidence for the Pareto-independence thesis. If Hard-bias improvement *caused* ABROCA improvement, you would see a clear positive linear trend (lower-left to upper-right diagonal). You don't.

### `figures/fig_pareto.png` — the Pareto frontier

Per-regime scatter plot of all 50 (5 methods × 10 seeds) points, with mean ± 95% CI markers for each method. X-axis is Hard-bias_IH, y-axis is ABROCA. Both lower = fairer, so the lower-left corner is best.

**What you should see**:

- **Original** sits in the upper-right (worst on both).
- **DEMOAUG** sits furthest left (lowest Hard-bias), middle on ABROCA.
- **SMOTE** sits lowest (lowest ABROCA), middle on Hard-bias.
- **LLM-Unconstrained** sits between Original and the augmentation methods — partial improvement on Hard-bias.
- **LLM+SMOTE** tracks SMOTE — the SMOTE step dominates.

**Why it matters**: No method dominates. DEMOAUG and SMOTE are both Pareto-optimal — neither is uniformly better.

### `figures/fig_six_metrics.png` — multi-metric panel

A 2×6 grid of bar charts: 2 regimes (rows) × 6 metrics (columns). Each bar is a method's mean with 95% CI.

**What you should see**:

- **AUC** column: nearly identical bars across methods (all ~0.98 in DistBias, all ~0.89 in LabelNoise). Augmentation doesn't hurt accuracy.
- **ABROCA** column: similar bars across all augmented methods, all lower than Original.
- **Hard-bias_IH** column: dramatic separation — Original is huge, DEMOAUG is smallest, others in between.
- **DI/EO/Calib** columns: overlapping CIs — augmentation has small/no effect on these metrics.

**Why it matters**: confirms that the Hard-bias finding doesn't generalize to other fairness metrics. DEMOAUG specifically optimizes Hard-bias and that's where it wins.

### `figures/fig_density_gap.png` — geometric evidence

Bar chart of mean $\log \hat p(z)$ for three sources: real text, SMOTE-interpolated points, LLM-generated points.

**What you should see**:

- Real text: highest log-density (by construction, since the KDE is fit on real text).
- SMOTE: ~3 nats below real.
- LLM (toy): ~10-14 nats below real (because bigram outputs don't match the encoder's training distribution well).

**Why it matters**: this operationalizes the "off-manifold" critique of SMOTE. With real Mistral, the LLM gap should shrink dramatically (Mistral outputs are much closer to BERT's training distribution than bigram outputs).

## Reading the summary tables

The CLI output and `results/analysis_report.md` show the same tables. Per regime:

```
Method        | AUC                  | ABROCA               | Hard-bias_IH         | DI_gap | EO_gap | Calib_gap
Original      | 0.9857 ± 0.0043      | 0.0291 ± 0.0150      | 0.4550 ± 0.0487      | ...
SMOTE         | 0.9848 ± 0.0046      | 0.0260 ± 0.0132      | 0.0921 ± 0.0181      | ...
LLM-UC        | 0.9840 ± 0.0047      | 0.0276 ± 0.0144      | 0.2495 ± 0.0329      | ...
LLM+SMOTE     | 0.9845 ± 0.0045      | 0.0259 ± 0.0141      | 0.1333 ± 0.0287      | ...
DEMOAUG       | 0.9815 ± 0.0056      | 0.0322 ± 0.0157      | 0.0986 ± 0.0270      | ...
```

**How to read**:

1. **Cross-method comparison** within a column: are means reasonably separated relative to std? Use the statistical rigor table (next section) for the formal answer.
2. **Cross-regime comparison** for one metric: is the metric harder in LabelNoise than DistributionalBias? Usually yes for AUC and ABROCA.
3. **DEMOAUG row**: notice DEMOAUG has the lowest Hard-bias_IH but not the lowest ABROCA. **This is the central finding.**

## Reading the statistical rigor tables

For each (regime, baseline, metric) the analysis prints:

```
DEMOAUG vs SMOTE
  AUC          : Δ=-0.0032, d=-2.01, p=0.000***, TOST=0.000(EQUIV), BF₀₁=0.00[strong-against-null]
  ABROCA       : Δ=+0.0062, d=+1.02, p=0.011**, TOST=0.043(EQUIV), BF₀₁=0.12[moderate-against-null]
  Hard-bias_IH : Δ=+0.0065, d=+0.18, p=0.574, TOST=0.002(EQUIV), BF₀₁=4.45[moderate-for-null]
  ...
```

**Decoding**:

- `Δ` = mean(DEMOAUG) - mean(baseline). Positive Δ on a fairness metric means DEMOAUG is *worse*; on AUC means DEMOAUG is *better*.
- `d` = Cohen's d. Rule of thumb: |d| < 0.2 trivial, 0.5 medium, 0.8 large, >2 massive.
- `p` = paired t-test p-value. Stars: ` *` < 0.10, `**` < 0.05, `***` < 0.01.
- `TOST` = equivalence test p-value, with verdict `EQUIV` or `diff`. **EQUIV is positive evidence of practical equivalence**, not just failure to reject difference.
- `BF₀₁` = Bayes factor for null vs alternative. >3 = moderate evidence for null. <1/3 = moderate against null.

**Reading patterns**:

- `Δ small, p>0.05, TOST=EQUIV, BF>3`: **strong evidence the two methods produce equivalent results** on this metric. Important for the paper's claim that DEMOAUG ≈ SMOTE on most fairness metrics other than Hard-bias.
- `Δ large, p<0.001, d>2, BF<<1`: **strong evidence of difference**. We see this for Hard-bias_IH (DEMOAUG vs Original) and AUC.
- `Δ small, p>0.1, TOST=diff, BF ambiguous`: underpowered. With more seeds we'd see one direction or the other.

## The correlation analysis

```
DistributionalBias
  SMOTE        : pearson r = +0.254 [-0.395, +0.925], p=0.478
  LLM-UC       : pearson r = +0.371 [-0.818, +0.844], p=0.291
  LLM+SMOTE    : pearson r = +0.172 [-0.348, +0.718], p=0.634
  DEMOAUG      : pearson r = +0.308 [-0.332, +0.932], p=0.387
  POOLED       : pearson r = +0.062 [-0.238, +0.360], p=0.706
```

**How to read**:

- Per-method correlations are noisy with n=10 seeds — wide CIs.
- The **pooled** row aggregates all methods (n=40 deltas) and is the publishable result.
- Pooled CI containing zero is the headline empirical claim.
- We report Pearson and Spearman; both should agree. If they disagree dramatically, suspect outliers.

## When numbers don't match exactly

Across hardware/library versions, expect ±0.02 on individual cell means and ±0.05 on individual correlation point estimates. The qualitative findings — DEMOAUG dominant on Hard-bias, similar on other metrics, pooled correlation near zero — should reproduce robustly.

If you see qualitatively different patterns (e.g., DEMOAUG dominates ABROCA), check:

1. Did `data_synth.py` get changed? The plant of demographic bias is sensitive to the contamination rate.
2. Are filter thresholds in `config.py` modified?
3. Did the seeds list change in `scripts/run_experiment.py`?

## Going beyond the headline

The repo includes raw CSV outputs (`results/experiment_results.csv`) so you can:

- Compute additional fairness metrics not in `metrics.py`.
- Plot per-seed trajectories.
- Run regression analysis on which dataset properties predict ABROCA gain.

The Markdown report (`results/analysis_report.md`) is a starting point — it's regenerated from CSV every time you run `scripts/run_analysis.py`.
