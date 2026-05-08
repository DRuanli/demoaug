"""Pipeline orchestration.

Implements:
  - Baseline 1: Original (no augmentation)
  - Baseline 2: SMOTE in embedding space (Sha et al.)
  - Baseline 3: LLM-Generated, no constraints (bias inheritance)
  - Baseline 4: LLM-Generated + SMOTE (intermediate)
  - DEMOAUG: full pipeline with all four filters + Method-3
"""
import numpy as np
from typing import Dict
from imblearn.over_sampling import SMOTE
from config import (TARGET_BALANCE, GEN_OVERSAMPLE_FACTOR, SEED)
from llm_gen import (ToyClassConditionalLLM, ToyNeutralizingLLM,
                     get_unconstrained_llm, get_neutral_llm)
from vocab_debias import extract_disc_vocab
from filter_label import LabelFidelityFilter
from filter_probe import ProbeEnsemble
from filter_diversity import diversity_threshold, diversity_subset
from filter_mmd import passes_mmd
from method3 import method3_select
from hardness import IHClass


# ============================================================
# Baseline 1: Original
# ============================================================
def run_original(Z_train, A_train, Y_train):
    return Z_train.copy(), A_train.copy(), Y_train.copy()


# ============================================================
# Baseline 2: SMOTE in embedding space (Sha et al.)
# ============================================================
def run_smote(Z_train, A_train, Y_train, target_n=TARGET_BALANCE):
    """SMOTE per Method-2 of Sha et al.: balance over (A, Y) cells."""
    # Construct a synthetic 4-class label: 0=00, 1=01, 2=10, 3=11
    cell_label = A_train * 2 + Y_train
    # Target: target_n per cell
    counts = {c: int((cell_label == c).sum()) for c in range(4)}
    sampling_strategy = {c: max(counts[c], target_n) for c in range(4)}
    try:
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=SEED, k_neighbors=3)
        Zb, cell_b = sm.fit_resample(Z_train, cell_label)
        Ab = cell_b // 2
        Yb = cell_b % 2
        return Zb, Ab, Yb
    except ValueError as e:
        print(f"  SMOTE failed: {e}")
        return Z_train, A_train, Y_train


# ============================================================
# Baseline 3: LLM-Generated, no constraints
# ============================================================
def run_llm_unconstrained(df_train, encoder, target_n=TARGET_BALANCE):
    llm = ToyClassConditionalLLM()
    llm.fit(df_train)
    Z_orig = encoder.transform(df_train["text"].tolist())
    A_orig = df_train["G"].to_numpy()
    Y_orig = df_train["Y"].to_numpy()

    new_texts, new_A, new_Y = [], [], []
    for (g, y) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mask = (A_orig == g) & (Y_orig == y)
        cur = mask.sum()
        if cur < target_n:
            n_gen = target_n - cur
            gens = llm.generate((g, y), banned_tokens=set(),
                                max_tokens=60, n=n_gen, seed=SEED + g * 17 + y * 31)
            new_texts.extend(gens)
            new_A.extend([g] * n_gen)
            new_Y.extend([y] * n_gen)

    if not new_texts:
        return Z_orig, A_orig, Y_orig

    Z_new = encoder.transform(new_texts)
    Z_full = np.vstack([Z_orig, Z_new])
    A_full = np.concatenate([A_orig, np.array(new_A)])
    Y_full = np.concatenate([Y_orig, np.array(new_Y)])
    return Z_full, A_full, Y_full


# ============================================================
# Baseline 4: LLM-Generated + SMOTE
# ============================================================
def run_llm_then_smote(df_train, encoder, target_n=TARGET_BALANCE):
    Z_aug, A_aug, Y_aug = run_llm_unconstrained(df_train, encoder, target_n=target_n // 2)
    return run_smote(Z_aug, A_aug, Y_aug, target_n=target_n)


# ============================================================
# DEMOAUG: full pipeline
# ============================================================
def run_demoaug(df_train, encoder, target_n=TARGET_BALANCE, verbose=True):
    log = {}

    # Encode original
    Z_orig = encoder.transform(df_train["text"].tolist())
    A_orig = df_train["G"].to_numpy()
    Y_orig = df_train["Y"].to_numpy()

    # ----- Step A: IH_class on original -----
    ih_clf = IHClass()
    ih_orig = ih_clf.fit_and_score(Z_orig, Y_orig)
    log["ih_orig_mean"] = float(ih_orig.mean())

    # Identify deficit cells
    deficit_cells = []
    for (g, y) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        n_cell = int(((A_orig == g) & (Y_orig == y)).sum())
        if n_cell < target_n:
            deficit_cells.append(((g, y), target_n - n_cell))
    log["deficit_cells"] = [(c, n) for c, n in deficit_cells]
    if verbose:
        print(f"  [DEMOAUG] Deficit cells: {log['deficit_cells']}")

    # ----- Step B1: discriminative vocab -----
    disc_vocab, disc_coefs = extract_disc_vocab(
        df_train["text"].tolist(), A_orig
    )
    log["n_disc_vocab"] = len(disc_vocab)
    log["disc_vocab_examples"] = disc_vocab[:10]
    if verbose:
        print(f"  [DEMOAUG] |V_disc| = {len(disc_vocab)}, top-5: {disc_vocab[:5]}")

    # ----- Step B2: LLM generation with banned tokens -----
    # CLASS-CONDITIONAL generation: we want samples to look like authentic
    # members of their intended cell. V_disc banning removes only the most
    # discriminative shortcut tokens — remaining demographic signal is OK
    # because it's group-AUTHENTIC (matches real-data distribution).
    llm = ToyClassConditionalLLM()
    llm.fit(df_train)
    banned = set(disc_vocab)

    cell_pool_z: Dict[tuple, np.ndarray] = {}
    cell_pool_text: Dict[tuple, list] = {}
    for (cell, n_needed) in deficit_cells:
        n_gen = int(n_needed * GEN_OVERSAMPLE_FACTOR)
        gens = llm.generate(cell, banned_tokens=banned,
                            max_tokens=60, n=n_gen, seed=SEED + cell[0] * 17 + cell[1] * 31)
        cell_pool_text[cell] = gens
        cell_pool_z[cell] = encoder.transform(gens) if gens else np.empty((0, Z_orig.shape[1]))

    log["n_generated_total"] = int(sum(len(v) for v in cell_pool_text.values()))

    # ----- Step C1: Filter 1 (label fidelity) -----
    label_filter = LabelFidelityFilter()
    label_filter.fit(Z_orig, Y_orig)
    for (cell, _) in deficit_cells:
        if len(cell_pool_z[cell]) == 0:
            continue
        Y_target = np.full(len(cell_pool_z[cell]), cell[1])
        keep = label_filter.keep_mask(cell_pool_z[cell], Y_target)
        cell_pool_z[cell] = cell_pool_z[cell][keep]
        cell_pool_text[cell] = [t for t, k in zip(cell_pool_text[cell], keep) if k]
    log["after_filter_label"] = int(sum(len(v) for v in cell_pool_z.values()))
    if verbose:
        print(f"  [DEMOAUG] After Filter 1 (label): {log['after_filter_label']} synth")

    # ----- Step C2: Filter 2 (probe ensemble) -----
    # Train probes on real + a held-out chunk of synthetic
    holdout_z, holdout_g = [], []
    keep_z, keep_g = {}, {}
    for cell in cell_pool_z:
        z = cell_pool_z[cell]
        g_arr = np.full(len(z), cell[0])
        if len(z) > 0:
            n_hold = max(1, len(z) // 4)
            holdout_z.append(z[:n_hold])
            holdout_g.append(g_arr[:n_hold])
            keep_z[cell] = z[n_hold:]
            keep_g[cell] = g_arr[n_hold:]
        else:
            keep_z[cell] = z
            keep_g[cell] = g_arr
    if holdout_z:
        Zh = np.vstack(holdout_z)
        Gh = np.concatenate(holdout_g)
    else:
        Zh, Gh = None, None

    probe = ProbeEnsemble()
    probe.fit(Z_orig, A_orig, Z_synth_known=Zh, G_synth_known=Gh)

    # Compute soft quality scores per sample (used by Method-3)
    quality_scores: dict = {}
    for cell in list(cell_pool_z.keys()):
        z = keep_z[cell]
        if len(z) == 0:
            cell_pool_z[cell] = z
            quality_scores[cell] = np.array([])
            continue
        # Lenient hard cut: k_sigma=3.0 (only reject extreme outliers)
        keep, info = probe.keep_mask_calibrated(z, intended_g=cell[0], k_sigma=3.0)
        cell_pool_z[cell] = z[keep]
        # Soft quality score for the survivors
        if len(cell_pool_z[cell]) > 0:
            quality_scores[cell] = probe.quality_score(cell_pool_z[cell], intended_g=cell[0])
        else:
            quality_scores[cell] = np.array([])
    log["after_filter_probe"] = int(sum(len(v) for v in cell_pool_z.values()))
    log["probe_calibration"] = {
        f"g={g}": {"mu": probe.real_mean_per_g.get(g, None),
                   "sigma": probe.real_std_per_g.get(g, None)}
        for g in [0, 1]
    }
    log["mean_quality_score"] = float(np.mean([s.mean() for s in quality_scores.values() if len(s) > 0]))
    if verbose:
        print(f"  [DEMOAUG] After Filter 2 (probe-calibrated, lenient): {log['after_filter_probe']} synth")
        print(f"            Mean quality score: {log['mean_quality_score']:.3f}")

    # ----- Step C3: Filter 3 (diversity, log-det) -----
    for cell in cell_pool_z:
        z_real_cell = Z_orig[(A_orig == cell[0]) & (Y_orig == cell[1])]
        if len(z_real_cell) < 4:
            continue
        thr = diversity_threshold(z_real_cell)
        if len(cell_pool_z[cell]) > 3:
            keep_idx = diversity_subset(cell_pool_z[cell], thr)
            cell_pool_z[cell] = cell_pool_z[cell][keep_idx]
            # Keep quality scores aligned
            if cell in quality_scores and len(quality_scores[cell]) > 0:
                quality_scores[cell] = quality_scores[cell][keep_idx]
    log["after_filter_diversity"] = int(sum(len(v) for v in cell_pool_z.values()))
    if verbose:
        print(f"  [DEMOAUG] After Filter 3 (diversity): {log['after_filter_diversity']} synth")

    # ----- Step C4: MMD certification (informational only) -----
    mmd_log = {}
    for cell in cell_pool_z:
        z_gen = cell_pool_z[cell]
        z_real_cell = Z_orig[(A_orig == cell[0]) & (Y_orig == cell[1])]
        if len(z_gen) < 5 or len(z_real_cell) < 5:
            continue
        ok, m2, p = passes_mmd(z_gen, z_real_cell)
        mmd_log[str(cell)] = {"mmd2": m2, "p": p, "passed": ok}
    log["mmd_per_cell"] = mmd_log

    # ----- Step A applied to synth pool: compute IH for candidates -----
    ih_synth: Dict[tuple, np.ndarray] = {}
    for cell in cell_pool_z:
        z = cell_pool_z[cell]
        if len(z) == 0:
            ih_synth[cell] = np.array([])
        else:
            y_arr = np.full(len(z), cell[1])
            ih_synth[cell] = ih_clf.score_new(z, y_arr)

    # ----- Step D: Method-3 selection -----
    result, gamma_star = method3_select(
        Z_orig, A_orig, Y_orig,
        cell_pool_z, ih_orig, ih_synth,
        quality_scores=quality_scores,
        target_n=target_n,
    )
    if result is None:
        if verbose:
            print("  [DEMOAUG] Method-3 found no valid set; falling back to original")
        return Z_orig.copy(), A_orig.copy(), Y_orig.copy(), log
    Z_b, A_b, Y_b, _ = result
    log["gamma_ih_star"] = float(gamma_star)
    if verbose:
        print(f"  [DEMOAUG] Method-3 best Gamma^IH = {gamma_star:.4f}")
        print(f"  [DEMOAUG] Final balanced size = {len(Z_b)}")

    return Z_b, A_b, Y_b, log
