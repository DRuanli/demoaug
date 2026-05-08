"""Step C3: diversity filter (intra-cell stereotype prevention).

For each (G=g, Y=y) cell, compute log-det of the empirical covariance
matrix of real BERT embeddings: log_det_real(g, y).

A batch of generated samples for that cell is accepted iff:
    log_det(cov(Z_gen_batch)) >= log_det_real(g, y) - delta

This prevents the LLM from generating stylistically homogeneous text
inside a cell (a known failure mode of class-conditional generation).

Implementation note: log-det of a covariance matrix in d dimensions
requires at least d+1 samples for a non-degenerate estimate. We use a
shrinkage estimator (Ledoit-Wolf) to handle the small-sample case.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from config import DIVERSITY_DELTA


def safe_logdet_cov(Z: np.ndarray) -> float:
    """Log-det of Ledoit-Wolf-shrunk covariance. Numerically safe."""
    if len(Z) < 3:
        return -np.inf
    lw = LedoitWolf().fit(Z)
    sign, logdet = np.linalg.slogdet(lw.covariance_)
    if sign <= 0:
        return -np.inf
    return float(logdet)


def diversity_threshold(Z_real_cell: np.ndarray, delta: float = DIVERSITY_DELTA) -> float:
    return safe_logdet_cov(Z_real_cell) - delta


def passes_diversity(Z_gen_cell: np.ndarray, threshold: float) -> bool:
    return safe_logdet_cov(Z_gen_cell) >= threshold


def diversity_subset(Z_gen_cell: np.ndarray, threshold: float, max_iter: int = 5):
    """If the full batch fails, try removing the most outlying points
    (those that most reduce diversity) and check again. Returns indices
    to keep.

    For the prototype we just return all indices if it passes, else a
    random subsample sized to maximize log-det estimate.
    """
    if passes_diversity(Z_gen_cell, threshold):
        return np.arange(len(Z_gen_cell))
    # Try removing one point at a time (greedy)
    n = len(Z_gen_cell)
    keep = np.ones(n, dtype=bool)
    for _ in range(min(max_iter, n - 3)):
        # Find the point whose removal most increases logdet
        current = safe_logdet_cov(Z_gen_cell[keep])
        best_gain = -np.inf
        best_i = -1
        for i in np.where(keep)[0]:
            keep[i] = False
            new = safe_logdet_cov(Z_gen_cell[keep])
            gain = new - current
            if gain > best_gain:
                best_gain = gain
                best_i = i
            keep[i] = True
        if best_i < 0 or best_gain <= 0:
            break
        keep[best_i] = False
        if passes_diversity(Z_gen_cell[keep], threshold):
            return np.where(keep)[0]
    return np.where(keep)[0]
