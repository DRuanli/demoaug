"""Step C4: distribution-level certification via MMD permutation test.

Given two sets of BERT embeddings (X = generated samples for G=g, Y=y;
Y = real samples for the same Y but G != g), we test whether their
distributions match using Maximum Mean Discrepancy with an RBF kernel.

Bandwidth: median heuristic on the *combined* sample (Gretton et al. 2012).

Permutation test gives a p-value. We accept the augmented dataset if
p > MMD_P_VALUE_MIN, meaning we cannot reject H0: P = Q.

Note on direction of test: the goal is for synthetic samples to have
the same distribution as real samples in the *target* group, since the
overall purpose is to make the augmented dataset's per-group distributions
indistinguishable across G.
"""
import numpy as np
from config import MMD_P_VALUE_MIN, MMD_PERMUTATIONS, SEED


def _median_heuristic(Z: np.ndarray) -> float:
    """Median pairwise distance — standard RBF bandwidth."""
    n = len(Z)
    if n < 2:
        return 1.0
    # Subsample for speed if needed
    if n > 200:
        idx = np.random.RandomState(SEED).choice(n, 200, replace=False)
        Z = Z[idx]
    # Pairwise squared distances
    sq = np.sum(Z**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2 * Z @ Z.T
    D2 = np.maximum(D2, 0)
    iu = np.triu_indices_from(D2, k=1)
    med = np.median(np.sqrt(D2[iu]))
    return float(med) if med > 1e-8 else 1.0


def _rbf_kernel_matrix(Z: np.ndarray, sigma: float) -> np.ndarray:
    sq = np.sum(Z**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2 * Z @ Z.T
    D2 = np.maximum(D2, 0)
    return np.exp(-D2 / (2 * sigma**2))


def mmd2_unbiased(Kxx, Kyy, Kxy):
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / max(m * (m - 1), 1)
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / max(n * (n - 1), 1)
    sum_xy = Kxy.mean()
    return float(sum_xx + sum_yy - 2 * sum_xy)


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray, n_perm: int = MMD_PERMUTATIONS, seed: int = SEED):
    """Returns (mmd2, p_value). p_value = fraction of perm MMDs >= observed.

    Note: unbiased MMD2 estimator can be slightly negative when X, Y are from
    the same distribution (sampling artifact). When observed mmd2 <= 0, we
    return p=1.0 directly (cannot reject H0)."""
    if len(X) < 5 or len(Y) < 5:
        return 0.0, 1.0
    Z = np.vstack([X, Y])
    sigma = _median_heuristic(Z)
    K = _rbf_kernel_matrix(Z, sigma)
    m, n = len(X), len(Y)
    Kxx = K[:m, :m]
    Kyy = K[m:, m:]
    Kxy = K[:m, m:]
    obs = mmd2_unbiased(Kxx, Kyy, Kxy)

    # Short-circuit: negative or zero MMD2 means no evidence against H0
    if obs <= 0:
        return float(obs), 1.0

    rng = np.random.RandomState(seed)
    count = 0
    total = m + n
    for _ in range(n_perm):
        perm = rng.permutation(total)
        Kp = K[np.ix_(perm, perm)]
        Kxx_p = Kp[:m, :m]
        Kyy_p = Kp[m:, m:]
        Kxy_p = Kp[:m, m:]
        if mmd2_unbiased(Kxx_p, Kyy_p, Kxy_p) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(obs), float(p)


def passes_mmd(X_gen: np.ndarray, X_real: np.ndarray) -> tuple:
    """Return (pass: bool, mmd2: float, p_value: float)."""
    mmd2, p = mmd_permutation_test(X_gen, X_real)
    return (p > MMD_P_VALUE_MIN), mmd2, p
