"""Step D: Method-3 selection — extension to LLM pool with quality weighting.

Algorithm (extends Sha et al. 2023, Method-3):
    Input: filtered pool of synthetic samples + original training set
           + per-sample quality scores from probe filter
    Goal:  produce a balanced dataset minimizing Hard-bias (IH-based)

    For k = 1 ... K:
        For each (G=g, Y=y) cell:
            Sample N_target indices from {original_cell ∪ synth_cell}
                with probability proportional to quality_score
                (originals get score=1.0; synth get their probe-calibration score)
        Compute Gamma^IH on the resulting balanced dataset.
        Track the dataset achieving min Gamma^IH.
    Return D* = argmin_k Gamma^IH(D_k)
"""
import numpy as np
from typing import Dict
from config import METHOD3_K, SEED, TARGET_BALANCE
from hardness import hard_bias_ih


def method3_select(
    Z_orig: np.ndarray, A_orig: np.ndarray, Y_orig: np.ndarray,
    Z_synth_pool: Dict[tuple, np.ndarray],
    ih_values_orig: np.ndarray,
    ih_values_synth: Dict[tuple, np.ndarray],
    quality_scores: Dict[tuple, np.ndarray] = None,
    K: int = METHOD3_K,
    target_n: int = TARGET_BALANCE,
    seed: int = SEED,
):
    """Returns indices into a virtual pool, plus the achieved Gamma^IH."""
    rng = np.random.RandomState(seed)
    cells = [(g, y) for g in [0, 1] for y in [0, 1]]
    orig_idx_by_cell = {
        (g, y): np.where((A_orig == g) & (Y_orig == y))[0]
        for (g, y) in cells
    }

    best = None
    best_score = np.inf

    for k in range(K):
        Z_parts, A_parts, Y_parts, IH_parts = [], [], [], []
        for (g, y) in cells:
            orig_pool = orig_idx_by_cell[(g, y)]
            synth_pool_z = Z_synth_pool.get((g, y), np.empty((0, Z_orig.shape[1])))
            synth_pool_ih = ih_values_synth.get((g, y), np.array([]))
            n_orig = len(orig_pool)
            n_synth = len(synth_pool_z)
            n_avail = n_orig + n_synth
            if n_avail == 0:
                continue

            # Quality-weighted sampling probabilities
            if quality_scores is not None and (g, y) in quality_scores and len(quality_scores[(g, y)]) == n_synth:
                synth_w = np.maximum(quality_scores[(g, y)], 0.05)
                probs = np.concatenate([np.ones(n_orig), synth_w])
            else:
                probs = np.ones(n_avail)
            probs = probs / probs.sum()

            if target_n <= n_avail:
                # Try without replacement (random subset weighted by quality is hard;
                # approximate with weighted-with-replacement when target_n is large
                # relative to pool)
                if target_n > 0.7 * n_avail:
                    indices = rng.choice(n_avail, target_n, replace=True, p=probs)
                else:
                    # Subsample weighted: convert to ranking
                    indices = rng.choice(n_avail, target_n, replace=False, p=probs)
            else:
                indices = rng.choice(n_avail, target_n, replace=True, p=probs)

            for idx in indices:
                if idx < n_orig:
                    real_i = orig_pool[idx]
                    Z_parts.append(Z_orig[real_i])
                    IH_parts.append(ih_values_orig[real_i])
                else:
                    syn_i = idx - n_orig
                    Z_parts.append(synth_pool_z[syn_i])
                    IH_parts.append(synth_pool_ih[syn_i] if syn_i < len(synth_pool_ih) else 0.5)
                A_parts.append(g)
                Y_parts.append(y)

        if not Z_parts:
            continue

        Z_b = np.array(Z_parts)
        A_b = np.array(A_parts)
        Y_b = np.array(Y_parts)
        IH_b = np.array(IH_parts)

        gamma_y0 = hard_bias_ih(IH_b, A_b, Y_b, y_target=0)
        gamma_y1 = hard_bias_ih(IH_b, A_b, Y_b, y_target=1)
        gamma = 0.5 * (gamma_y0 + gamma_y1)

        if gamma < best_score:
            best_score = gamma
            best = (Z_b, A_b, Y_b, IH_b)

    return best, float(best_score)
