# 2. Methodology

This document specifies the math and algorithms. For code-level walkthrough, see [`03_pipeline_walkthrough.md`](03_pipeline_walkthrough.md).

## Notation

- $D = \{(x_i, y_i, a_i)\}_{i=1}^N$ — training set with text $x_i$, task label $y_i \in \{0, 1\}$, sensitive attribute $a_i \in \{0, 1\}$.
- $z_i = \phi(x_i) \in \mathbb{R}^d$ — encoder embedding (BERT [CLS] in production; TF-IDF + SVD in toy).
- A *cell* is a $(g, y)$ pair; the four cells partition $D$.
- $D_{g,y} = \{i : a_i = g, y_i = y\}$ — indices in cell $(g, y)$.

## Step 0: Manifold density (measurement, not augmentation)

We operationalize "off-manifold-ness" of synthetic embeddings via a PCA-whitened kernel density estimator:

$$
\hat p(z) = \text{KDE}_{\sigma}\big(\,W^{-1/2} U^\top z\,\big)
$$

where $U \Sigma V^\top$ is the SVD of the centered training-set embedding matrix and $W = \text{diag}(\Sigma^2)$ is the eigenvalue diagonal. We fit $\hat p$ on real embeddings only, then evaluate $\log \hat p(z)$ for SMOTE-interpolated and LLM-generated points.

The empirical claim of Section 4 is:
$$
\mathbb{E}_{z \sim \text{real}}[\log \hat p(z)] \;>\; \mathbb{E}_{z \sim \text{SMOTE}}[\log \hat p(z)]
$$
with a typical gap of ~3 nats. This makes "off-manifold" a measurable quantity rather than handwaving.

## Step A: IH_class — instance hardness

Following Smith et al. (2014), we define instance hardness as the ensemble miss-rate:

$$
\text{IH}_{\text{class}}(z, y) = 1 - \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} P_l(y \mid z)
$$

where $\mathcal{L} = \{\text{LR, RBF-SVM, RandomForest}\}$ is a small ensemble of fast, diverse learners.

For original data points we compute IH via 5-fold CV. For synthetic candidates we use inference-only predictions from learners trained once on the full original training set. This avoids the $O(K \cdot |\mathcal{L}|)$ retrain cost that a naive per-iteration approach would incur in Method-3.

## Hard-bias under IH_class

We extend Sha et al.'s Hard-bias from kDN to IH_class. The formal definition:

$$
\Gamma_A^{\text{IH}}(y) = \tfrac{1}{2}\Big[\text{KL}\big(f_y^0 \,\|\, f_y^1\big) + \text{KL}\big(f_y^1 \,\|\, f_y^0\big)\Big]
$$

where $f_y^g$ is the empirical density (Gaussian KDE on $[0,1]$, 100-point grid) of $\{\text{IH}(x_i, y) : a_i = g, y_i = y\}$. Larger $\Gamma_A^{\text{IH}}$ means greater across-group disparity in difficulty.

The dataset-level Hard-bias is the average over $y$:
$$
\Gamma_A^{\text{IH}}(D) = \tfrac{1}{2}\big(\Gamma_A^{\text{IH}}(0) + \Gamma_A^{\text{IH}}(1)\big)
$$

## Step B1: Discriminative vocabulary

Let $X \in \mathbb{R}^{N \times V}$ be the TF-IDF matrix (unigrams + bigrams, max 3000 features). Fit L2-regularized logistic regression to predict $a$ from $X$:

$$
\hat w = \arg\min_{w} \frac{1}{N}\sum_{i=1}^N \log\big(1 + e^{-a_i (w^\top X_i)}\big) + \lambda \|w\|_2^2
$$

Take the top $K/2$ tokens with most negative $\hat w_v$ (G=0 markers) and the top $K/2$ with most positive $\hat w_v$ (G=1 markers). This balanced selection is essential: an L1 version we tried produced one-sided $V_{\text{disc}}$ (only G=1 markers survived L1 selection), which degraded the pipeline. See `vocab_debias.py` for the implementation.

## Step B2: Constrained LLM generation

For each deficit cell $(g, y)$, we generate $n$ samples conditioned on:

1. **Few-shot examples** from the cell's real samples (5 examples).
2. **Hard token banning** via logit-bias: for each banned word $v \in V_{\text{disc}}$, we expand to all subword token ids in the Mistral tokenizer (with leading-space and capitalization variants) and set logit = $-\infty$ at decode time.

Hard banning is strictly stronger than instruction-level negative prompting (which has 30-40% violation rate on 7B-class models per IFEval).

## Step C1: Label fidelity filter

Train a logistic-regression classifier on the original $(z, y)$ data. For a generated sample $\tilde z$ for cell $(g, y)$, keep it iff:
$$
P(y_{\text{intended}} \mid \tilde z) \geq \tau_{\text{label}}
$$
We use $\tau_{\text{label}} = 0.65$.

## Step C2: Demographic probe (calibrated)

Train an ensemble of $M = 5$ MLPs to predict $a$ from $z$. The training data mixes real samples with a held-out chunk of LLM-generated samples (with their *intended* group label as ground truth), with inverse-propensity reweighting to combat real/synthetic distribution shift.

Each probe outputs $P_m(a=1 \mid z)$. After fitting, we compute calibration statistics on real samples per group:
$$
\mu_g = \mathbb{E}_{z : a=g} \Big[\tfrac{1}{M}\sum_m P_m(a=1 \mid z)\Big],
\quad
\sigma_g = \text{std}_{z : a=g} \Big[\tfrac{1}{M}\sum_m P_m(a=1 \mid z)\Big]
$$

For a synthetic sample $\tilde z$ generated for cell $(g, y)$:
$$
\text{keep iff } \bigg|\tfrac{1}{M}\sum_m P_m(a=1 \mid \tilde z) - \mu_g\bigg| \leq k\sigma_g \quad \text{AND} \quad \text{Var}_m P_m(a=1 \mid \tilde z) < \epsilon
$$

with $k = 3.0$ (lenient hard cut) and $\epsilon = 0.05$. The first condition says "the sample looks like a typical $a=g$ point"; the second says "the probes agree".

We also compute a **soft quality score** $q(\tilde z) = \exp\big(-\tfrac{1}{2}((\bar p - \mu_g)/\sigma_g)^2\big)$ used by Method-3 for weighted sampling.

### Why calibration, not neutrality

An earlier version of this filter required $\bar p \in [0.4, 0.6]$ (demographic untraceability). This was theoretically wrong: it forced synthetic samples to be group-erased, which created distribution shift between training and test sets. The calibration variant says "look like an authentic member of your intended group" instead, which preserves the data manifold.

## Step C3: Diversity filter (log-det)

For each cell $(g, y)$, compute $\log\det \hat\Sigma_{g,y}^{\text{real}}$ on the real cell embeddings using Ledoit-Wolf shrinkage (handles small-sample case). For a batch of generated cell-samples to be accepted:
$$
\log\det \hat\Sigma_{g,y}^{\text{gen}} \geq \log\det \hat\Sigma_{g,y}^{\text{real}} - \delta
$$

with $\delta = 1.0$. If a batch fails, we greedily remove the points whose removal most increases the batch's log-det, until either the threshold is met or no improvement is possible.

This filter prevents intra-cell stereotype collapse — a known failure mode of class-conditional generation.

## Step C4: MMD distribution-level test

For each cell, run a permutation MMD test with median-heuristic RBF kernel between generated and real cell-embeddings:
$$
\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x, x')] + \mathbb{E}_{y,y' \sim Q}[k(y, y')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)]
$$

with bandwidth set to the median pairwise distance on the *combined* sample (Gretton et al. 2012). 200 permutations gives an empirical p-value. We accept the augmented cell if $p > 0.05$.

**Important detail caught by unit tests**: the unbiased MMD² estimator can be slightly negative under H₀ (sampling artifact). When observed MMD² ≤ 0 we return $p = 1$ directly, since negative values cannot be evidence against H₀. Without this short-circuit, Type I error inflates to ~20%.

## Step D: Method-3 with quality-weighted resampling

The original Sha et al. Method-3:
$$
D^* = \arg\min_{D \in \mathcal{D}_K} \Gamma_A^{\text{IH}}(D) \quad \text{s.t. } \Delta_A(D) = 0
$$

where $\mathcal{D}_K$ is a set of $K = 25$ candidate balanced datasets and $\Delta_A$ is Dist-bias.

**Our extension**: each cell's sampling pool is $\{\text{originals}\} \cup \{\text{filtered synthetics}\}$, and synthetic samples are drawn with probability proportional to their soft quality score $q(\tilde z)$. Originals get weight 1.0. This concentrates Method-3's selection on synthetic samples that pass calibration most strongly.

## Step E: Evaluation

We report 6 fairness/utility metrics:

| Metric | Definition | Direction |
|---|---|---|
| AUC | Standard ROC-AUC | higher = better |
| ABROCA | $\int_0^1 \|\text{ROC}_{a=0}(t) - \text{ROC}_{a=1}(t)\|\,dt$ | lower = fairer |
| Hard-bias_IH | $\Gamma_A^{\text{IH}}(D)$ as defined above | lower = fairer |
| DI gap | $\|1 - \frac{P(\hat y = 1 \mid a=0)}{P(\hat y = 1 \mid a=1)}\|$ | lower = fairer |
| EO gap | $\max(\|TPR_0 - TPR_1\|, \|FPR_0 - FPR_1\|)$ at threshold 0.5 | lower = fairer |
| Calib gap | mean over score-bins of $\|\mathbb{E}[Y \mid \text{bin}, a=0] - \mathbb{E}[Y \mid \text{bin}, a=1]\|$ | lower = fairer |

## Statistical analysis (per metric, per regime, per baseline)

For DEMOAUG vs each baseline on each metric, we report:

1. **Paired t-test** $p$-value
2. **Cohen's d** for paired samples: $d = \bar{(x - y)} / \text{std}(x - y)$
3. **TOST** (Two One-Sided Tests) for equivalence: tests $H_1: |\mu_x - \mu_y| < \delta$ at metric-specific equivalence margins (AUC: 0.01, ABROCA: 0.01, Hard-bias: 0.05, DI/EO/Calib: 0.02-0.05).
4. **Bayes factor** $BF_{01}$ via BIC approximation. $BF_{01} > 3$ is moderate evidence for the null.

## Correlation analysis (Section 5 main result)

For each method and each seed, compute:
$$
\Delta\text{Hard-bias}_s^m = \text{Hard-bias}^m_s - \text{Hard-bias}^{\text{Original}}_s
$$
$$
\Delta\text{ABROCA}_s^m = \text{ABROCA}^m_s - \text{ABROCA}^{\text{Original}}_s
$$

Pool across $m \in \{$SMOTE, LLM-UC, LLM+SMOTE, DEMOAUG$\}$ and $s \in \{$10 seeds$\}$ giving 40 deltas per regime. Compute Pearson and Spearman correlations with bootstrap 95% CI (1000 resamples).

The headline finding (`fig_correlation.png`):

| Regime | Pooled $r$ | 95% CI | p |
|---|---|---|---|
| DistributionalBias | +0.062 | [-0.238, +0.360] | 0.706 |
| LabelNoiseBias | +0.000 | [-0.262, +0.340] | 0.999 |

Both CIs contain zero in both regimes. This is the empirical ground for the Pareto independence thesis.
