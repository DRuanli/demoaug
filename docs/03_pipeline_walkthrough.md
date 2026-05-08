# 3. Pipeline Walkthrough — From Code

This document walks through what each module does and how they connect. Read alongside the source files in `src/`.

## High-level flow

```
data_synth.py  ──┐
                 │
                 ▼
       ┌──────────────────────┐
       │  encoder.py          │   text → ℝᵈ embeddings
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  pipeline.py          │   orchestrates everything below
       └─┬─┬─┬─┬─┬─┬───────────┘
         │ │ │ │ │ │
         ▼ ▼ ▼ ▼ ▼ ▼
   density.py
   hardness.py
   vocab_debias.py
   llm_gen.py
   filter_label.py
   filter_probe.py
   filter_diversity.py
   filter_mmd.py
   method3.py
                  │
                  ▼
       ┌──────────────────────┐
       │  metrics.py          │   ABROCA, DI, EO, Hard-bias, etc.
       └──────────────────────┘
```

## Module index

### `src/config.py`
Single source of truth for all hyperparameters. Edit here, not in modules. Important fields:

- `USE_REAL_MODELS` — flip to `True` to use BERT + Mistral instead of toy versions.
- `DATA_SCALE` — fraction of full Sha-Forum size; 1.0 = ~3700 posts.
- `N_SEEDS` — how many seeds the experiment runs.
- `METHOD3_K` — number of resampling iterations in Method-3.
- `TOP_K_DISC_VOCAB` — size of $V_{\text{disc}}$ (banned vocabulary).
- All filter thresholds: `TAU_LABEL`, `PROBE_VARIANCE_TAU`, `DIVERSITY_DELTA`, `MMD_P_VALUE_MIN`.

### `src/data_synth.py`
Generates a synthetic Forum-Post-Classification-like dataset matching Sha et al. 2023 Table 1 cell counts (1413/812/925/553) and length distributions (G=0 ≈ 130 words, G=1 ≈ 115 words). Plants demographic linguistic signal:

- Group-typical vocabulary (formality markers vs direct markers).
- L2-typo pass for G=1 (article omission, agreement slips).
- Group-correlated label noise (8% of G=1 content posts get flipped — mimics annotator bias).

The `make_realistic_forum(scale, seed)` function is the main entry point.

### `src/encoder.py`
Two encoder classes with the same interface:

```python
class ToyEncoder:
    """TF-IDF + TruncatedSVD to 128 dims. L2-normalized."""
    def fit(self, texts: List[str]) -> None: ...
    def transform(self, texts: List[str]) -> np.ndarray: ...

class BertEncoder:
    """bert-base-uncased [CLS] embeddings. Used when USE_REAL_MODELS=True."""
    # Same interface.
```

The `get_encoder()` factory returns the right one based on config.

### `src/llm_gen.py`
Three LLM classes:

- `ToyClassConditionalLLM` — bigram model fit per (G, Y) cell. **Inherits group bias by design.** Used by the LLM-Unconstrained baseline to demonstrate what unfiltered LLM augmentation does.
- `ToyNeutralizingLLM` — bigram model fit per Y class only (group-agnostic). Stand-in for what a real LLM does when prompted with few-shot examples from both groups.
- `MistralLLM` — real Mistral-7B-Instruct wrapper with HuggingFace `LogitsProcessor` for hard token banning. **NOT integration-tested in v1** — flagged for the real-data run.

All three expose the same `generate(cell, banned_tokens, max_tokens, n)` signature.

### `src/density.py` (Step 0)
`ManifoldDensity` class wrapping PCA whitening + KDE. Used to quantify the off-manifold-ness of synthetic embeddings. Not part of the augmentation pipeline — it's a measurement tool for Section 4 of the paper.

### `src/hardness.py` (Step A)
`IHClass.fit_and_score(Z, Y)` runs 5-fold CV with the {LR, SVM-RBF, RandomForest} ensemble and returns IH for every original point, then refits on full data for `score_new(Z_new, Y_new)` calls.

`hard_bias_ih(ih_values, A, Y, y_target)` computes the symmetric KL between IH distributions across groups within a Y class.

### `src/vocab_debias.py` (Step B1)
`extract_disc_vocab(texts, G)` returns the top-K most discriminative tokens, balanced (K/2 from each group's tail of the L2-LR coefficient distribution). Also provides `expand_to_subwords(words, tokenizer)` for real Mistral integration.

### `src/filter_label.py` (Step C1)
`LabelFidelityFilter` is a thin wrapper around a logistic regression classifier trained on real (Z, Y) data. Exposes `confidence(Z, Y_target)` and `keep_mask(Z, Y_target)`.

### `src/filter_probe.py` (Step C2)
The most architecturally important filter. `ProbeEnsemble.fit(Z_real, G_real, Z_synth_known, G_synth_known)` trains $M=5$ MLPs with inverse-propensity reweighting over the mixed real+synthetic training set.

After fitting, three methods are exposed:

- `keep_mask_calibrated(Z, intended_g, k_sigma)` — hard cut at $k\sigma$ from the real-data mean for the intended group.
- `quality_score(Z, intended_g)` — soft Gaussian similarity score for Method-3 weighting.
- `keep_mask(Z)` — old neutrality criterion (around 0.5), kept for ablation studies.

### `src/filter_diversity.py` (Step C3)
`safe_logdet_cov(Z)` — Ledoit-Wolf-shrunk log-det. `diversity_subset(Z, threshold)` greedily removes points until threshold is met, returns indices to keep.

### `src/filter_mmd.py` (Step C4)
`mmd_permutation_test(X, Y, n_perm)` — RBF kernel with median bandwidth. Short-circuits to $p=1$ when observed MMD² ≤ 0 (necessary to control Type I error rate; this was caught by a unit test).

### `src/method3.py` (Step D)
`method3_select(Z_orig, A_orig, Y_orig, Z_synth_pool, ih_orig, ih_synth, quality_scores, K, target_n)` runs $K$ resamplings and returns the one minimizing $\Gamma_A^{\text{IH}}$.

### `src/metrics.py` (Step E)
- `abroca(scores, y_true, sensitive)` — primary fairness metric.
- `disparate_impact(scores, y_true, sensitive)`, `equalized_odds_gap(...)`, `calibration_gap(...)` — secondary fairness metrics.
- `evaluate_pipeline(Z_train, A_train, Y_train, Z_test, A_test, Y_test, method_name)` — full evaluation in one call.

### `src/stats_rigor.py`
- `tost_equivalence(x, y, equivalence_margin, alpha)` — TOST test for equivalence.
- `cohens_d_paired(x, y)` — effect size.
- `bayes_factor_bic(x, y)` — Bayes factor via BIC approximation.
- `correlation_analysis(deltas_a, deltas_b)` — Pearson + Spearman with bootstrap CI.
- `comprehensive_comparison(df, method_a, method_b, metric, regime, equivalence_margin)` — runs all four in one call.

### `src/pipeline.py`
The orchestrator. Five top-level functions, one per method:

```python
def run_original(Z_train, A_train, Y_train) -> (Z, A, Y): ...
def run_smote(Z_train, A_train, Y_train, target_n) -> (Z, A, Y): ...
def run_llm_unconstrained(df_train, encoder, target_n) -> (Z, A, Y): ...
def run_llm_then_smote(df_train, encoder, target_n) -> (Z, A, Y): ...
def run_demoaug(df_train, encoder, target_n, verbose) -> (Z, A, Y, log): ...
```

The DEMOAUG function is the most complex. Reading from top to bottom you'll see the Step A → B1 → B2 → C1 → C2 → C3 → C4 → D sequence in roughly 200 lines. The `log` return value contains attrition counters (how many synthetic samples survived each filter), final $\Gamma_A^{\text{IH}}$, and probe calibration statistics.

## What gets imported where

```python
# Common imports for any new analysis script:
from src.data_synth import make_realistic_forum
from src.encoder import get_encoder
from src.pipeline import run_demoaug, run_smote, run_original, run_llm_unconstrained, run_llm_then_smote
from src.metrics import evaluate_pipeline
from src.stats_rigor import comprehensive_comparison, correlation_analysis
```

## Adding a new method

1. Implement a `run_my_method(df_train, encoder, target_n)` that returns `(Z_balanced, A_balanced, Y_balanced)`.
2. Add to the methods list in `scripts/run_experiment.py`.
3. Re-run the experiment.

## Adding a new metric

1. Implement `my_metric(scores, y_true, sensitive) -> float` in `src/metrics.py`.
2. Add to the dict returned by `evaluate_pipeline()`.
3. Add to the `METRICS` constant in `scripts/run_experiment.py`.
4. Re-run.

## Common pitfalls

- **Don't fit the encoder on test data**. The pipeline does this correctly (encoder fits on `df_train` only) but if you write your own scripts, check carefully.
- **Don't mix up A and Y**. By convention throughout the code: `A` = sensitive attribute (group), `Y` = task label.
- **Don't forget to set seeds**. Every script seeds `numpy`, `random`, and any sklearn calls. If you add stochastic logic, follow this pattern.
- **The probe filter relies on calibration to real data**. If you swap encoders, re-tune `PROBE_MEAN_LOW`/`PROBE_MEAN_HIGH` based on what real-data probe outputs look like under the new encoder.
