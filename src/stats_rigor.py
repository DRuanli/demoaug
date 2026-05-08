"""Statistical rigor module.

Provides:
  - TOST (Two One-Sided Tests) for equivalence claims
  - Pearson correlation between per-seed metric deltas
  - Cohen's d effect sizes
  - Bayes factor approximation (BIC-based) for 'no effect' claims

These are the four standard tools for going beyond p-values: equivalence
testing for null claims, effect sizes for practical significance, and
Bayes factors for evidence calibration.
"""
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d_paired(x, y):
    """Effect size for paired samples. Standard threshold: |d|<0.2 small,
    0.5 medium, 0.8 large."""
    diff = np.asarray(x) - np.asarray(y)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def tost_equivalence(x, y, equivalence_margin: float, alpha: float = 0.05):
    """Two One-Sided Tests for equivalence of paired samples.

    H0: |mean(x) - mean(y)| >= equivalence_margin  (NOT equivalent)
    H1: |mean(x) - mean(y)| <  equivalence_margin  (equivalent)

    Returns p_tost (max of two one-sided p-values). If p_tost < alpha,
    reject H0 -> conclude equivalence.

    Important: failure to reject difference is
    NOT evidence of equivalence. TOST gives positive equivalence evidence.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    n = len(diff)
    if n < 2:
        return {"p_tost": 1.0, "equivalent": False, "diff_mean": float(diff.mean()),
                "diff_se": float("nan"), "margin": equivalence_margin, "n": n}
    se = diff.std(ddof=1) / np.sqrt(n)
    if se < 1e-12:
        # Zero variance — equivalent if diff is small
        return {"p_tost": 0.0 if abs(diff.mean()) < equivalence_margin else 1.0,
                "equivalent": abs(diff.mean()) < equivalence_margin,
                "diff_mean": float(diff.mean()), "diff_se": 0.0,
                "margin": equivalence_margin, "n": n}

    # Test 1: H0_1: diff >= +margin  vs  H1_1: diff < +margin
    t1 = (diff.mean() - equivalence_margin) / se
    p1 = stats.t.cdf(t1, df=n - 1)
    # Test 2: H0_2: diff <= -margin  vs  H1_2: diff > -margin
    t2 = (diff.mean() - (-equivalence_margin)) / se
    p2 = 1 - stats.t.cdf(t2, df=n - 1)
    p_tost = max(p1, p2)
    return {
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < alpha),
        "diff_mean": float(diff.mean()),
        "diff_se": float(se),
        "margin": equivalence_margin,
        "n": n,
        "alpha": alpha,
    }


def bayes_factor_bic(x, y):
    """Bayes factor approximation via BIC for paired t-test.

    BF_01 = exp((BIC_alt - BIC_null) / 2)
    BF_01 > 3:  moderate evidence FOR null (no effect)
    BF_01 > 10: strong evidence FOR null
    BF_01 < 1/3: moderate evidence AGAINST null
    """
    diff = np.asarray(x) - np.asarray(y)
    n = len(diff)
    if n < 3:
        return {"BF_01": float("nan"), "n": n}
    # Null: diff ~ N(0, sigma^2) — 1 parameter
    sigma2_null = np.mean(diff ** 2)
    ll_null = -0.5 * n * (np.log(2 * np.pi * sigma2_null) + 1)
    bic_null = -2 * ll_null + 1 * np.log(n)
    # Alt: diff ~ N(mu, sigma^2) — 2 parameters
    sigma2_alt = np.var(diff, ddof=1)
    ll_alt = -0.5 * n * (np.log(2 * np.pi * sigma2_alt) + 1)
    bic_alt = -2 * ll_alt + 2 * np.log(n)
    bf_01 = float(np.exp((bic_alt - bic_null) / 2))
    return {"BF_01": bf_01, "n": n,
            "interpretation": (
                "strong-for-null" if bf_01 > 10 else
                "moderate-for-null" if bf_01 > 3 else
                "ambiguous" if bf_01 > 1/3 else
                "moderate-against-null" if bf_01 > 1/10 else
                "strong-against-null"
            )}


def correlation_analysis(deltas_a: np.ndarray, deltas_b: np.ndarray):
    """Pearson + Spearman correlation between two delta vectors,
    with bootstrap 95% CI."""
    deltas_a = np.asarray(deltas_a)
    deltas_b = np.asarray(deltas_b)
    n = len(deltas_a)
    if n < 3:
        return {"pearson_r": float("nan"), "n": n}

    pearson_r, pearson_p = stats.pearsonr(deltas_a, deltas_b)
    spearman_r, spearman_p = stats.spearmanr(deltas_a, deltas_b)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_r = []
    for _ in range(1000):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(idx)) < 2:
            continue
        try:
            r, _ = stats.pearsonr(deltas_a[idx], deltas_b[idx])
            if not np.isnan(r):
                boot_r.append(r)
        except Exception:
            continue
    if boot_r:
        ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "ci_95_low": float(ci_lo),
        "ci_95_high": float(ci_hi),
        "n": n,
    }


def comprehensive_comparison(df_results, method_a: str, method_b: str,
                              metric: str, regime: str = None,
                              equivalence_margin: float = None) -> dict:
    """Full comparison: t-test, Cohen's d, TOST, BF."""
    if regime is not None:
        sub = df_results[df_results["regime"] == regime]
    else:
        sub = df_results
    a = sub[sub["method"] == method_a].sort_values("seed")[metric].values
    b = sub[sub["method"] == method_b].sort_values("seed")[metric].values
    if len(a) != len(b) or len(a) < 2:
        return {"error": "mismatched or insufficient samples"}

    # Standard t-test
    t_stat, p_value = stats.ttest_rel(a, b)
    d = cohens_d_paired(a, b)

    # Equivalence
    if equivalence_margin is None:
        equivalence_margin = 0.5 * np.std(np.concatenate([a, b]))
    tost = tost_equivalence(a, b, equivalence_margin)

    # Bayes factor
    bf = bayes_factor_bic(a, b)

    return {
        "method_a": method_a, "method_b": method_b, "metric": metric, "regime": regime,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff_mean": float(a.mean() - b.mean()),
        "t_stat": float(t_stat), "p_value": float(p_value),
        "cohens_d": d,
        "tost": tost,
        "bayes_factor_01": bf,
        "n_pairs": len(a),
    }
