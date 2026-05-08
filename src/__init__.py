"""DEMOAUG — demographic-aware LLM augmentation for fair educational text classification.

Modules:
    config              — all hyperparameters
    data_synth          — synthetic dataset generator
    encoder             — toy + BERT encoders
    llm_gen             — toy + Mistral LLM generators
    density             — manifold density estimator (Step 0)
    hardness            — IH_class + Hard-bias (Step A)
    vocab_debias        — discriminative vocabulary extraction (Step B1)
    filter_label        — label fidelity filter (Step C1)
    filter_probe        — demographic probe ensemble (Step C2)
    filter_diversity    — log-det diversity filter (Step C3)
    filter_mmd          — MMD permutation test (Step C4)
    method3             — Method-3 selection (Step D)
    metrics             — fairness/utility metrics (Step E)
    stats_rigor         — TOST + Cohen's d + Bayes factor + correlation
    pipeline            — full orchestration
"""

__version__ = "1.0.0"
