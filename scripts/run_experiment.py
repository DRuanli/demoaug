"""Full experiment: 10 seeds × 2 regimes × 5 methods × 6 metrics.

Saves results incrementally to results/experiment_results.csv so the run
can be resumed if interrupted.

Usage:
    python scripts/run_experiment.py                    # full experiment
    python scripts/run_experiment.py --regime DistributionalBias  # one regime
    python scripts/run_experiment.py --n-seeds 3        # quick check
"""
import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Make the src/ directory importable when running from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from data_synth import (make_realistic_forum, _l2_typo_pass,
                         G0_FORMAL_MARKERS, G1_DIRECT_MARKERS,
                         G0_HEDGES, G1_GENERIC_NOUNS,
                         CONTENT_REL_CORE, CONTENT_REL_COLLOQUIAL,
                         NON_CONTENT_CORE, SHARED_FORUM_VOCAB)
from encoder import get_encoder
from pipeline import (run_original, run_smote, run_llm_unconstrained,
                       run_llm_then_smote, run_demoaug)
from metrics import evaluate_pipeline


# ============================================================
# Two-regime data generators
# ============================================================
def _generate_regime_a(g, y, rng, contam_rate):
    """DistributionalBias: shared markers, weaker keyword density for G=1."""
    target_len = max(20, int(rng.gauss(130 if g == 0 else 115, 35 if g == 0 else 30)))
    all_markers = G0_FORMAL_MARKERS + G1_DIRECT_MARKERS

    if y == 1:
        if g == 0:
            primary = CONTENT_REL_CORE * 3 + CONTENT_REL_COLLOQUIAL
            primary_weight = 0.70
        else:
            primary = CONTENT_REL_CORE + CONTENT_REL_COLLOQUIAL * 2
            primary_weight = 0.45
        contamination = NON_CONTENT_CORE
    else:
        primary = NON_CONTENT_CORE * 2
        primary_weight = 0.65 if g == 0 else 0.45
        contamination = CONTENT_REL_CORE + CONTENT_REL_COLLOQUIAL

    words = []
    for _ in range(target_len):
        r = rng.random()
        if r < 0.20:
            words.append(rng.choice(all_markers))
        elif r < 0.65:
            if rng.random() < primary_weight:
                if rng.random() < contam_rate:
                    words.append(rng.choice(contamination))
                else:
                    words.append(rng.choice(primary))
            else:
                words.append(rng.choice(SHARED_FORUM_VOCAB))
        else:
            words.append(rng.choice(SHARED_FORUM_VOCAB))

    if g == 1:
        words = _l2_typo_pass(words, rng)
    return " ".join(words), len(words)


def _generate_regime_b(g, y, rng, contam_rate):
    """LabelNoiseBias: distinct group markers, heavy contamination."""
    target_len = max(20, int(rng.gauss(130 if g == 0 else 115, 35 if g == 0 else 30)))
    group_marker_pool = G0_FORMAL_MARKERS if g == 0 else G1_DIRECT_MARKERS
    group_secondary_pool = G0_HEDGES if g == 0 else G1_GENERIC_NOUNS

    if y == 1:
        primary = (CONTENT_REL_CORE * 3 + CONTENT_REL_COLLOQUIAL) if g == 0 \
                  else (CONTENT_REL_CORE + CONTENT_REL_COLLOQUIAL * 2)
        contamination = NON_CONTENT_CORE
    else:
        primary = NON_CONTENT_CORE * 2
        contamination = CONTENT_REL_CORE + CONTENT_REL_COLLOQUIAL

    words = []
    for _ in range(target_len):
        r = rng.random()
        if r < 0.30:
            words.append(rng.choice(group_marker_pool))
        elif r < 0.38:
            words.append(rng.choice(group_secondary_pool))
        elif r < 0.65:
            if rng.random() < contam_rate:
                words.append(rng.choice(contamination))
            else:
                words.append(rng.choice(primary))
        else:
            words.append(rng.choice(SHARED_FORUM_VOCAB))

    if g == 1:
        words = _l2_typo_pass(words, rng)
    return " ".join(words), len(words)


def _make_with_params(seed, scale, flip_rate, contam_rate, generator_fn):
    rng = random.Random(seed)
    base_counts = {
        (0, 1): int(1413 * scale),
        (0, 0): int(812 * scale),
        (1, 1): int(925 * scale),
        (1, 0): int(553 * scale),
    }
    rows = []
    for (g, y), n in base_counts.items():
        for _ in range(n):
            text, length = generator_fn(g, y, rng, contam_rate)
            actual_y = y
            if g == 1 and y == 1 and rng.random() < flip_rate:
                actual_y = 0
            rows.append({"text": text, "G": g, "Y": actual_y, "length": length})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def make_regime_a(seed, scale=0.4):
    return _make_with_params(seed, scale, flip_rate=0.0, contam_rate=0.35,
                              generator_fn=_generate_regime_a)


def make_regime_b(seed, scale=0.4):
    return _make_with_params(seed, scale, flip_rate=0.10, contam_rate=0.40,
                              generator_fn=_generate_regime_b)


# ============================================================
# Per-seed runner
# ============================================================
def run_one(make_fn, seed, regime_name):
    df = make_fn(seed=seed)
    df_tr, df_te = train_test_split(
        df, test_size=0.2, random_state=seed,
        stratify=df["G"].astype(str) + "_" + df["Y"].astype(str),
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    enc = get_encoder().fit(df_tr["text"].tolist())
    Z_tr = enc.transform(df_tr["text"].tolist())
    A_tr = df_tr["G"].to_numpy()
    Y_tr = df_tr["Y"].to_numpy()
    Z_te = enc.transform(df_te["text"].tolist())
    A_te = df_te["G"].to_numpy()
    Y_te = df_te["Y"].to_numpy()
    target_n = max(df_tr.groupby(["G", "Y"]).size())

    out = []
    for (name, fn) in [
        ("Original",  lambda: run_original(Z_tr, A_tr, Y_tr)),
        ("SMOTE",     lambda: run_smote(Z_tr, A_tr, Y_tr, target_n=target_n)),
        ("LLM-UC",    lambda: run_llm_unconstrained(df_tr, enc, target_n=target_n)),
        ("LLM+SMOTE", lambda: run_llm_then_smote(df_tr, enc, target_n=target_n)),
        ("DEMOAUG",   lambda: run_demoaug(df_tr, enc, target_n=target_n, verbose=False)[:3]),
    ]:
        try:
            Z_b, A_b, Y_b = fn()
            r = evaluate_pipeline(Z_b, A_b, Y_b, Z_te, A_te, Y_te, name)
            r["seed"] = seed
            r["regime"] = regime_name
            out.append(r)
        except Exception as e:
            print(f"     {name} FAILED: {e}", flush=True)
    return out


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["DistributionalBias", "LabelNoiseBias", "both"],
                        default="both")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--csv-path", default=None,
                        help="Output CSV path (default: results/experiment_results.csv)")
    args = parser.parse_args()

    csv_path = args.csv_path or os.path.join(ROOT, "results", "experiment_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    regimes = {
        "DistributionalBias": make_regime_a,
        "LabelNoiseBias": make_regime_b,
    }
    if args.regime != "both":
        regimes = {args.regime: regimes[args.regime]}

    seeds = [42, 1337, 2025, 7919, 31337, 5555, 8888, 1234, 9876, 17171][:args.n_seeds]

    print("=" * 72)
    print(f"DEMOAUG full experiment")
    print(f"  Regimes: {list(regimes.keys())}")
    print(f"  Seeds:   {len(seeds)} → {seeds}")
    print(f"  Output:  {csv_path}")
    print("=" * 72)

    # Resume support
    all_rows = []
    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        done = set(zip(prev["regime"], prev["seed"]))
        all_rows = prev.to_dict("records")
        print(f"\nResuming: {len(done)} (regime, seed) pairs already done")

    for regime_name, make_fn in regimes.items():
        print(f"\n>>> Regime: {regime_name}")
        for s in seeds:
            if (regime_name, s) in done:
                print(f"   seed {s}: SKIP (already done)")
                continue
            t0 = time.time()
            rows = run_one(make_fn, s, regime_name)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(csv_path, index=False)
            print(f"   seed {s}: {time.time() - t0:.1f}s [saved {len(all_rows)} rows]")

    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nDone. {len(all_rows)} rows saved to {csv_path}")
    print("Run `python scripts/run_analysis.py` to generate figures and reports.")


if __name__ == "__main__":
    main()
