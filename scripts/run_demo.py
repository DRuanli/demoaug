"""Quick demo script: 1 seed × 5 methods on a small synthetic dataset.

Use this to verify the pipeline runs end-to-end before launching the full
experiment. Runtime: ~50 seconds on a laptop CPU.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Make the src/ directory importable when running from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from data_synth import make_realistic_forum
from encoder import get_encoder
from pipeline import (run_original, run_smote, run_llm_unconstrained,
                       run_llm_then_smote, run_demoaug)
from metrics import evaluate_pipeline
from density import ManifoldDensity
from llm_gen import get_unconstrained_llm
from config import SEED


def main():
    print("=" * 72)
    print("DEMOAUG demo — 1 seed × 5 methods on small synthetic Forum")
    print("=" * 72)

    # Use a small dataset for fast demo
    df = make_realistic_forum(scale=0.4, seed=SEED)
    print(f"\nDataset: {len(df)} posts")
    print(f"Cell distribution: {df.groupby(['G','Y']).size().to_dict()}")

    df_tr, df_te = train_test_split(
        df, test_size=0.2, random_state=SEED,
        stratify=df["G"].astype(str) + "_" + df["Y"].astype(str),
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    encoder = get_encoder().fit(df_tr["text"].tolist())
    Z_tr = encoder.transform(df_tr["text"].tolist())
    A_tr = df_tr["G"].to_numpy()
    Y_tr = df_tr["Y"].to_numpy()
    Z_te = encoder.transform(df_te["text"].tolist())
    A_te = df_te["G"].to_numpy()
    Y_te = df_te["Y"].to_numpy()
    target_n = max(df_tr.groupby(["G", "Y"]).size())

    # Density gap (Step 0)
    print("\nManifold density gap analysis...")
    md = ManifoldDensity().fit(Z_tr)
    rng = np.random.RandomState(SEED)
    minority = np.where((A_tr == 1) & (Y_tr == 0))[0]
    Z_smote_demo = []
    for _ in range(50):
        if len(minority) >= 2:
            i, j = rng.choice(minority, 2, replace=False)
            alpha = rng.random()
            Z_smote_demo.append(alpha * Z_tr[i] + (1 - alpha) * Z_tr[j])
    Z_smote_demo = np.array(Z_smote_demo) if Z_smote_demo else Z_tr[:5]
    llm = get_unconstrained_llm()
    llm.fit(df_tr)
    gens = llm.generate((1, 0), banned_tokens=set(), max_tokens=100, n=50, seed=SEED)
    Z_llm_demo = encoder.transform(gens)
    rep = md.report_gap(Z_tr, Z_smote_demo, Z_llm_demo)
    print(f"  Real text log p̂(z):  {rep['real_log_p_mean']:+.2f}")
    print(f"  SMOTE points log p̂(z): {rep['smote_log_p_mean']:+.2f}  (gap: {rep['smote_offset_vs_real']:+.2f})")
    print(f"  LLM points log p̂(z):   {rep['llm_log_p_mean']:+.2f}  (gap: {rep['llm_offset_vs_real']:+.2f})")

    # All 5 methods
    print("\nRunning 5 methods...")
    results = []
    for name, run_fn in [
        ("Original",  lambda: run_original(Z_tr, A_tr, Y_tr)),
        ("SMOTE",     lambda: run_smote(Z_tr, A_tr, Y_tr, target_n=target_n)),
        ("LLM-UC",    lambda: run_llm_unconstrained(df_tr, encoder, target_n=target_n)),
        ("LLM+SMOTE", lambda: run_llm_then_smote(df_tr, encoder, target_n=target_n)),
        ("DEMOAUG",   lambda: run_demoaug(df_tr, encoder, target_n=target_n, verbose=False)[:3]),
    ]:
        Z_b, A_b, Y_b = run_fn()
        r = evaluate_pipeline(Z_b, A_b, Y_b, Z_te, A_te, Y_te, name)
        results.append(r)
        print(f"  {name:<12}: AUC={r['AUC']:.4f}  ABROCA={r['ABROCA']:.4f}  HB_IH={r['Hard-bias_IH']:.4f}")

    print("\n" + "=" * 72)
    print("Demo done. For the full statistical experiment run:")
    print("  python scripts/run_experiment.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
