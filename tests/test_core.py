"""Unit tests for math-critical modules.

Run with: pytest tests/ -v

Focus is on modules where mathematical correctness matters most:
    - hardness.py    (IH_class definition, KL divergence)
    - filter_mmd.py  (MMD permutation test, including Type I error rate)
    - method3.py     (selection algorithm)
    - stats_rigor.py (TOST, Bayes factor, correlation)
    - metrics.py     (ABROCA, DI, EO, calibration)
    - vocab_debias.py (balanced V_disc extraction)
    - density.py     (manifold density gap claim)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest


# =============================================================================
# Hardness (IH_class)
# =============================================================================

def test_ih_perfect_separation():
    """IH should be ~0 for perfectly separable data."""
    from hardness import IHClass
    rng = np.random.RandomState(42)
    Z = np.vstack([rng.randn(50, 4) - 5, rng.randn(50, 4) + 5])
    Y = np.array([0] * 50 + [1] * 50)
    ih_clf = IHClass()
    ih = ih_clf.fit_and_score(Z, Y)
    assert ih.mean() < 0.1, f"Expected low IH for separable data, got {ih.mean():.3f}"


def test_ih_random_data():
    """IH should be ~0.5 for completely random labels."""
    from hardness import IHClass
    rng = np.random.RandomState(42)
    Z = rng.randn(100, 4)
    Y = rng.randint(0, 2, 100)
    ih_clf = IHClass()
    ih = ih_clf.fit_and_score(Z, Y)
    assert 0.3 < ih.mean() < 0.7, f"Expected IH near 0.5 for random data, got {ih.mean():.3f}"


def test_hard_bias_zero_when_distributions_match():
    """KL divergence-based Hard-bias should be ~0 when groups have identical IH."""
    from hardness import hard_bias_ih
    rng = np.random.RandomState(42)
    n = 200
    ih = rng.beta(2, 5, n)
    Y = np.ones(n, dtype=int)
    A = np.array([0] * (n // 2) + [1] * (n // 2))
    hb = hard_bias_ih(ih, A, Y, y_target=1)
    assert hb < 0.5, f"Expected small Hard-bias for identical groups, got {hb:.3f}"


def test_hard_bias_positive_when_distributions_differ():
    """Hard-bias should be > 0 when groups have different IH distributions."""
    from hardness import hard_bias_ih
    rng = np.random.RandomState(42)
    n = 200
    ih_a0 = rng.beta(2, 8, n // 2)   # easy
    ih_a1 = rng.beta(8, 2, n // 2)   # hard
    ih = np.concatenate([ih_a0, ih_a1])
    A = np.array([0] * (n // 2) + [1] * (n // 2))
    Y = np.ones(n, dtype=int)
    hb = hard_bias_ih(ih, A, Y, y_target=1)
    assert hb > 1.0, f"Expected Hard-bias > 1 for different distributions, got {hb:.3f}"


# =============================================================================
# MMD permutation test
# =============================================================================

def test_mmd_type_i_error_rate():
    """MMD test should have Type I error rate near alpha=0.05 across many
    same-distribution trials. We allow up to 15% as the one-sided ceiling."""
    from filter_mmd import mmd_permutation_test
    n_trials = 20
    rejections = 0
    for s in range(n_trials):
        rng = np.random.RandomState(s)
        X = rng.randn(80, 8)
        Y = rng.randn(80, 8)
        _, p = mmd_permutation_test(X, Y, n_perm=200, seed=s)
        if p < 0.05:
            rejections += 1
    rate = rejections / n_trials
    assert rate <= 0.15, f"Type I error rate too high: {rate:.2f} (expected ~0.05)"


def test_mmd_rejects_different_distributions():
    """MMD test should reject H0 (p < 0.05) for clearly different distributions."""
    from filter_mmd import mmd_permutation_test
    rng = np.random.RandomState(42)
    X = rng.randn(80, 8)
    Y = rng.randn(80, 8) + 3.0
    mmd2, p = mmd_permutation_test(X, Y, n_perm=200)
    assert p < 0.05, f"Expected p < 0.05 for shifted dist, got p={p:.3f}"
    assert mmd2 > 0.1, f"Expected mmd2 > 0.1 for shifted dist, got {mmd2:.4f}"


# =============================================================================
# Statistical rigor: TOST, Bayes factor, correlation
# =============================================================================

def test_tost_equivalent_when_means_close():
    """TOST should conclude equivalence when means are very close."""
    from stats_rigor import tost_equivalence
    rng = np.random.RandomState(42)
    x = rng.randn(50) * 0.1
    y = rng.randn(50) * 0.1 + 0.005
    res = tost_equivalence(x, y, equivalence_margin=0.1)
    assert res["equivalent"], f"Expected equivalence, got p_tost={res['p_tost']:.3f}"


def test_tost_not_equivalent_when_means_differ():
    """TOST should NOT conclude equivalence when means differ substantially."""
    from stats_rigor import tost_equivalence
    rng = np.random.RandomState(42)
    x = rng.randn(50) * 0.1
    y = rng.randn(50) * 0.1 + 0.5
    res = tost_equivalence(x, y, equivalence_margin=0.1)
    assert not res["equivalent"], f"Expected non-equivalence, got p_tost={res['p_tost']:.3f}"


def test_bf_favors_null_when_no_effect():
    """Bayes factor should favor null (BF_01 > 1) when data has no effect."""
    from stats_rigor import bayes_factor_bic
    rng = np.random.RandomState(42)
    x = rng.randn(60)
    y = rng.randn(60)
    res = bayes_factor_bic(x, y)
    assert res["BF_01"] > 1.0, f"Expected BF_01 > 1 for null data, got {res['BF_01']:.2f}"


def test_correlation_strong_positive():
    """Pearson r should be ~1 for perfectly linear data."""
    from stats_rigor import correlation_analysis
    x = np.linspace(0, 1, 30)
    y = 2 * x + 0.001 * np.random.RandomState(42).randn(30)
    res = correlation_analysis(x, y)
    assert res["pearson_r"] > 0.99, f"Expected strong positive corr, got r={res['pearson_r']:.3f}"


def test_correlation_zero_independent():
    """Pearson r should be ~0 for independent data."""
    from stats_rigor import correlation_analysis
    rng = np.random.RandomState(42)
    x = rng.randn(50)
    y = rng.randn(50)
    res = correlation_analysis(x, y)
    assert abs(res["pearson_r"]) < 0.4, f"Expected r near 0, got r={res['pearson_r']:.3f}"


# =============================================================================
# Method-3 selection
# =============================================================================

def test_method3_returns_balanced_dataset():
    """Output should have target_n samples per cell."""
    from method3 import method3_select
    rng = np.random.RandomState(42)
    Z = rng.randn(60, 8)
    A = np.array([0, 1] * 30)
    Y = np.array([0, 0, 1, 1] * 15)
    ih = rng.uniform(0, 1, 60)
    synth_pool = {(0, 0): rng.randn(20, 8), (1, 0): rng.randn(20, 8),
                  (0, 1): rng.randn(20, 8), (1, 1): rng.randn(20, 8)}
    ih_synth = {k: rng.uniform(0, 1, 20) for k in synth_pool}
    result, gamma = method3_select(Z, A, Y, synth_pool, ih, ih_synth,
                                    K=5, target_n=20)
    assert result is not None
    Z_b, A_b, Y_b, _ = result
    for g in [0, 1]:
        for y in [0, 1]:
            n = ((A_b == g) & (Y_b == y)).sum()
            assert n == 20, f"Cell ({g},{y}) has {n}, expected 20"


# =============================================================================
# Metrics
# =============================================================================

def test_abroca_zero_for_perfect_classifier():
    """ABROCA should be ~0 when both groups have identical perfect ROC."""
    from metrics import abroca
    n = 200
    scores = np.concatenate([np.linspace(0, 0.4, n // 2),
                              np.linspace(0.6, 1.0, n // 2)])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    sensitive = np.tile([0, 1], n // 2)
    ab = abroca(scores, y, sensitive)
    assert ab < 0.05, f"Expected ABROCA near 0 for matched perfect classifier, got {ab:.3f}"


def test_abroca_positive_when_groups_differ():
    """ABROCA should be > 0 when groups have different ROC curves."""
    from metrics import abroca
    rng = np.random.RandomState(42)
    n = 400
    sensitive = np.array([0] * (n // 2) + [1] * (n // 2))
    y = rng.randint(0, 2, n)
    scores = np.zeros(n)
    # Group 0: clean signal
    scores[sensitive == 0] = y[sensitive == 0] + 0.1 * rng.randn((sensitive == 0).sum())
    # Group 1: noisy signal
    scores[sensitive == 1] = y[sensitive == 1] + 1.5 * rng.randn((sensitive == 1).sum())
    ab = abroca(scores, y, sensitive)
    assert ab > 0.05, f"Expected ABROCA > 0.05 for divergent groups, got {ab:.3f}"


def test_disparate_impact_zero_at_parity():
    """DI gap should be ~0 when both groups have same prediction rate."""
    from metrics import disparate_impact
    n = 100
    rng = np.random.RandomState(42)
    scores = rng.uniform(0, 1, n)
    sensitive = np.tile([0, 1], n // 2)
    y = (scores > 0.5).astype(int)
    di = disparate_impact(scores, y, sensitive)
    # Both groups draw from same scores -> roughly same prediction rates -> DI gap small
    assert di < 0.3, f"Expected small DI gap, got {di:.3f}"


# =============================================================================
# Vocab debiasing
# =============================================================================

def test_vocab_debias_returns_balanced():
    """V_disc should contain markers from both groups (after Round 4 fix)."""
    from vocab_debias import extract_disc_vocab
    texts = []
    G = []
    rng = np.random.RandomState(42)
    g0_words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    g1_words = ["one", "two", "three", "four", "five"]
    shared = ["the", "and", "is", "of", "in"]
    for _ in range(80):
        body = " ".join(rng.choice(g0_words + shared, 30))
        texts.append(body); G.append(0)
    for _ in range(80):
        body = " ".join(rng.choice(g1_words + shared, 30))
        texts.append(body); G.append(1)
    disc, coefs = extract_disc_vocab(texts, np.array(G))
    n_negative = (coefs < 0).sum()
    n_positive = (coefs > 0).sum()
    assert n_negative > 0 and n_positive > 0, \
        f"V_disc should have markers from both groups, got n_neg={n_negative}, n_pos={n_positive}"


# =============================================================================
# Density gap
# =============================================================================

def test_density_smote_below_real():
    """Density estimator should rank real > SMOTE-interpolated points (the
    central empirical claim of Section 4)."""
    from density import ManifoldDensity
    rng = np.random.RandomState(42)
    # Real data: clusters
    Z_real = np.vstack([rng.randn(60, 8) - 3, rng.randn(60, 8) + 3])
    md = ManifoldDensity(n_components=4, bandwidth=0.5).fit(Z_real)

    # SMOTE: interpolations between distant clusters land in low-density region
    Z_smote = []
    for _ in range(40):
        i = rng.choice(60)        # from cluster 0
        j = rng.choice(60) + 60   # from cluster 1
        alpha = rng.uniform(0.3, 0.7)
        Z_smote.append(alpha * Z_real[i] + (1 - alpha) * Z_real[j])
    Z_smote = np.array(Z_smote)

    real_lp = md.log_prob(Z_real).mean()
    smote_lp = md.log_prob(Z_smote).mean()
    assert real_lp > smote_lp, \
        f"Expected real density > SMOTE density, got real={real_lp:.2f}, smote={smote_lp:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
