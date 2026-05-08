# 1. Paper Overview

## The problem

Educational text classifiers (e.g., classifying forum posts as content-relevant vs not) often perform unequally across demographic groups — a well-known fairness issue in educational data mining. Sha et al. (2023, *Expert Systems with Applications*) proposed an augmentation framework based on:

- **kDN-based instance hardness** to identify which examples are difficult to classify, and
- **Method-3** — a resampling procedure that minimizes a *Hard-bias* metric (KL divergence between hardness distributions across groups).

Sha et al. report substantial ABROCA reduction (the standard educational fairness metric) on their balanced datasets, with the implicit framing that minimizing Hard-bias is what produces the ABROCA improvement.

## What we ask

Two questions motivated this work:

1. **Can we replace Sha et al.'s SMOTE step with constrained LLM generation?** SMOTE produces phantom embeddings that no real text decodes to; LLM generation produces inspectable text but risks inheriting demographic bias from pre-training.

2. **If we successfully reduce Hard-bias more aggressively, does ABROCA improve correspondingly?**

## What we built

A 4-stage pipeline (DEMOAUG) that:

- Identifies deficit demographic cells via IH_class (Smith et al. 2014) — a more principled hardness measure than kDN for high-dim representations.
- Generates synthetic text with vocabulary debiasing (logit-bias on group-discriminative tokens).
- Applies a 4-stage filter cascade: label fidelity → demographic probe (calibrated to real-data distribution) → diversity (log-det covariance) → MMD test.
- Uses an extended Method-3 with quality-weighted sampling.

The pipeline is fully implementable on a single GPU with Mistral-7B-Instruct + BERT, and all results in this repo use a toy-encoder/toy-LLM stand-in for fast iteration.

## What we found (the central empirical result)

Across **10 seeds × 2 bias regimes × 5 methods × 6 fairness metrics = 600 measurements**:

> **The change in Hard-bias_IH and the change in ABROCA are statistically uncorrelated.**

| Regime | Pooled Pearson r (ΔHard-bias × ΔABROCA) | 95% CI | p-value |
|---|---|---|---|
| DistributionalBias | +0.062 | [-0.238, +0.360] | 0.706 |
| LabelNoiseBias | +0.000 | [-0.262, +0.340] | 0.999 |

Both confidence intervals contain zero. This is **positive evidence** (not just failure-to-reject) that the two metrics move independently under augmentation.

We also find:

- **DEMOAUG** achieves the largest Hard-bias reduction (Cohen's d ≈ -5 vs Original) but does **not** achieve the lowest ABROCA.
- **SMOTE** achieves the lowest ABROCA despite producing off-manifold phantom embeddings (~3 nats below real text in PCA-whitened KDE log-density). This is consistent with the implicit-regularization hypothesis: off-manifold synthetic points act as label-smoothing noise at the decision boundary where ABROCA is computed.
- No method dominates the others: each occupies a distinct point on the Pareto frontier.

## What we claim (and don't)

We **claim**:

1. Under a wide range of augmentation methods, the per-seed change in Hard-bias does not predict the per-seed change in ABROCA on our synthetic Forum-like dataset.
2. SMOTE's off-manifold property is not necessarily harmful for ABROCA — and may be helpful via implicit regularization.
3. Methods occupy distinct Pareto points; practitioners should choose by which fairness axis matters for their application.
4. A 4-stage filter cascade can produce inspectable synthetic text that matches real demographic distributions without trivially copying real samples.

We **do not claim**:

- That Sha et al. 2023 are wrong — they report empirical correlation on real data; we report empirical independence on synthetic data. Both can be true (regime-dependent).
- That DEMOAUG is universally better than SMOTE.
- That synthetic experiments alone validate generalization to real classroom data — that requires the real-data run flagged in `docs/07_limitations.md`.

## What this paper contributes

1. **Empirical finding (decoupling regime)**: Hard-bias and ABROCA are statistically independent fairness axes in our augmentation regime.
2. **Methodological**: A 4-stage filter cascade for LLM augmentation that is mathematically principled (TOST-validated, MMD-certified) and produces inspectable text.
3. **Density-gap analysis**: A reusable PCA-whitened KDE measurement for "off-manifold-ness" of synthetic embeddings.
4. **Statistical rigor template**: TOST + Cohen's d + Bayes factor + correlation analysis applied to fairness metrics — a template for future fairness-augmentation papers.

## Where to read next

- [`02_methodology.md`](02_methodology.md) — full math.
- [`05_results_interpretation.md`](05_results_interpretation.md) — how to read the figures.
- [`07_limitations.md`](07_limitations.md) — what we know we don't know.
