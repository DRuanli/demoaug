# 7. Limitations and Future Work

This document is deliberately frank about what the v1 paper does **not** do. Future revisions will address these one by one.

## What this paper does not show

### 1. Real-data validation is missing

All experiments in v1 use a **synthetic Forum-Post-like dataset** that mirrors Sha et al. 2023 Table 1 in cell counts and average post lengths but plants demographic bias through controlled vocabulary and length disparities.

The synthetic dataset is engineered to isolate the *Hard-bias × ABROCA* relationship. We claim only that the relationship is decoupled in this controlled regime. We do not claim it generalizes to real classroom forum posts without further validation.

**Plan**: a real-data run on Sha et al.'s Forum dataset (if obtainable) or a public alternative such as OULAD discussion-forum extracts is the next deliverable. See `docs/06_real_data_migration.md` for the migration steps.

### 2. The toy LLM is a bigram model

The `ToyClassConditionalLLM` and `ToyNeutralizingLLM` classes use n=2 Markov chains fit to the training data. This is intentionally weak — it lets the full pipeline run in <20 minutes on a CPU — but it is not a fair stand-in for Mistral-7B for two reasons:

- **Semantic coverage**: a bigram model cannot generate sentences with consistent topical structure beyond 2-token windows. Real Mistral can.
- **Density gap**: in our experiments, toy-LLM-generated text falls ~10-14 nats below real text in PCA-whitened KDE log-density. Mistral output should fall much closer to real (~1-2 nats), narrowing the gap with SMOTE's ~3-nat offset.

The toy LLM is most likely the reason DEMOAUG underperforms LLM+SMOTE on ABROCA in v1 results. With Mistral, this could go either way.

**Plan**: real Mistral integration is implemented in `MistralLLM` but **not integration-tested** in v1.

### 3. Two regimes is not a comprehensive sweep

We test two bias regimes (DistributionalBias and LabelNoiseBias) chosen to span "easy" and "hard" extremes. Real educational fairness problems can have many other bias structures:

- **Intersectional bias**: ≥2 sensitive attributes simultaneously.
- **Continuous-valued sensitive attributes**: e.g., socioeconomic status as income.
- **Hierarchical bias**: nested cells (e.g., institution → cohort → individual).
- **Adversarial bias**: bias planted by an actor (e.g., for evaluation).

We do not test any of these. The Pareto-independence claim is regime-specific.

### 4. ABROCA is the headline fairness metric

While we report DI gap, EO gap, and Calibration gap as secondary metrics, the central correlation analysis pairs Hard-bias_IH with ABROCA only. ABROCA is the metric of choice in Sha et al. 2023 (the work we extend), but Reviewer 2 correctly notes that ABROCA is a narrow lens for educational fairness.

**Plan**: revision will rerun the correlation analysis pairing Hard-bias_IH with each of DI, EO, Calibration gap to test whether the decoupling generalizes across fairness metrics.

### 5. No human evaluation of generated text

The paper claims "interpretable text" as a virtue of LLM augmentation over SMOTE, but no educator has read DEMOAUG-generated samples and rated them for plausibility or fairness. Reviewer 2 flagged this as a deal-breaker for educational venues.

**Plan**: a 5-10 educator evaluation of 30-50 generated posts per (cell × method) for the real-data run, with Krippendorff's α inter-rater agreement on plausibility, fluency, and demographic neutrality.

### 6. No ethics analysis

For an *AI-generated student text* paper aimed at educational systems, this is a real gap. Specific issues:

- **IRB / consent**: deploying DEMOAUG-trained classifiers on real students raises questions about whether students should know their evaluator was trained on partly-synthetic data.
- **Stereotype amplification**: Mistral has known biases against L2-English speakers; even with our V_disc filter, stereotypes may persist in subtler features.
- **Disparate downstream impact**: a classifier that flags posts as "non-content-relevant" might systematically flag certain groups, leading to instructors not engaging with their posts.

**Plan**: a dedicated Ethics & Limitations section in the revision, drawing on Holstein et al. 2019 ("Improving fairness in machine learning systems: What do industry practitioners need?"), Suresh & Guttag 2021 ("A Framework for Understanding Sources of Harm Throughout the Machine Learning Life Cycle"), and recent EDM/LAK ethics guidance.

## Things we got partially right but could be stronger

### Statistical analysis

We report Cohen's d, TOST equivalence, and Bayes factors. But:

- TOST equivalence margins are chosen heuristically (0.01 for ABROCA). A pre-registered analysis with theoretically-motivated margins would be stronger.
- Bayes factors use BIC approximation, which is less reliable for small n than full conjugate-prior analysis.
- Bonferroni correction for multiple comparisons is not applied across the 100+ pairwise tests.

### Reproducibility infrastructure

`requirements.txt` is pinned, `pytest` runs, `scripts/` has clear entry points. But:

- No Docker image. Different OS / CUDA versions can produce small numerical differences.
- No DVC or similar for results-versioning. CSV outputs are git-ignored.
- The `MistralLLM` class is implemented but not integration-tested. Section 6 of the docs is honest about this.

### Code testing

17 unit tests covering `hardness`, `filter_mmd`, `metrics`, `stats_rigor`, `vocab_debias`, `density`. But:

- No tests for `pipeline.py` (only end-to-end smoke tests via `scripts/run_demo.py`).
- No property-based testing (e.g., Hypothesis).
- Coverage is not measured.

## Things we explicitly chose not to do

### We did not implement neural augmentation methods (back-translation, paraphrasing, MixUp on text)

These are reasonable baselines for educational NLP. We argue they are out of scope for a paper specifically extending Sha et al.'s SMOTE-on-embeddings framework. A separate paper could compare the broader space.

### We did not optimize hyperparameters by grid search

Filter thresholds and LLM generation parameters are set by hand based on small pilot runs. A grid search would likely close some of DEMOAUG's ABROCA gap to SMOTE, but at the cost of the paper's parsimony claim. We deliberately test "out of the box" performance.

### We did not extend to multi-class or regression tasks

The pipeline is binary-classification only. Sha et al. also focused on binary. Extension to multi-class is straightforward (cell partition becomes $|G| \times |Y|$); regression requires reformulating instance-hardness, which is non-trivial.

### We did not investigate causal mechanisms for the decoupling

We *empirically* show Hard-bias_IH and ABROCA are uncorrelated. We do not formally prove they must be uncorrelated under any hypothesis class. A theoretical analysis (e.g., showing that Hard-bias bounds ABROCA only under specific regularity assumptions) would strengthen the contribution but is its own paper.

## How to read this v1 paper

The right framing is: **"A first empirical investigation of LLM-based fair augmentation, with a surprising negative result that motivates further study."**

It is not: "DEMOAUG is the new SOTA for educational fairness."

It is also not: "Sha et al. 2023 are wrong."

It is: "When you measure carefully, the chain Hard-bias → ABROCA does not always hold; here is a controlled regime where it doesn't, here is the methodology that surfaces this, and here are the open questions."

This is a contribution. It is a **first version**. The revision plan is concrete: real-data validation, human evaluation, ethics analysis, broader fairness-metric correlations.

## Roadmap to a revised version

| Item | Estimated effort | Priority |
|---|---|---|
| Real-data run on Sha Forum or OULAD | 4-6 weeks | High |
| Human evaluation by 5-10 educators | 2-3 weeks | High |
| Ethics & Limitations section | 1 week | High |
| Multi-metric correlation analysis | 1 week | Medium |
| Multi-temperature LLM ablation | 2 weeks | Medium |
| Causal analysis of Hard-bias / ABROCA gap | 4+ weeks | Medium |
| Cost-analysis section with real numbers | 1 week | Low |
| Docker image + DVC | 1 week | Low |

Total revision effort: ~3-4 months for a thorough revision.
