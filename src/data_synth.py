"""Realistic synthetic Forum-Post-Classification-like dataset.

Engineered to reproduce the *qualitative* statistical structure of
Sha et al. (2023) Table 1 (Forum dataset, sex attribute):

    Real Forum (Sha et al. Table 1):
        female content-relevant:     1413
        female non-content-relevant:  812
        male   content-relevant:      925
        male   non-content-relevant:  553

We additionally plant:
    1. Demographic linguistic signal — group X uses partially distinct
       vocabulary (formality markers, hedging, discourse markers).
    2. Label-relevant content vocabulary that is partly group-dependent
       (educational forums research: minority-group L2-English students
       phrase content-relevant points using fewer canonical academic markers).
    3. Length disparity matching Sha et al. Table 1.
    4. Group-correlated label noise: 8% of G=1, Y=1 flipped (annotator bias).
"""
import numpy as np
import random
import pandas as pd
from config import SEED


# === Vocabulary banks (modelled after academic-forum corpora) ===

G0_FORMAL_MARKERS = [
    "moreover", "furthermore", "consequently", "nevertheless", "however",
    "therefore", "specifically", "particularly", "subsequently", "additionally",
    "essentially", "fundamentally", "predominantly", "ultimately", "respectively",
    "thereby", "henceforth", "notwithstanding", "hereby", "thus",
]
G0_HEDGES = [
    "presumably", "arguably", "ostensibly", "tentatively", "plausibly",
    "conceivably", "demonstrably", "purportedly",
]
G1_DIRECT_MARKERS = [
    "very", "really", "actually", "basically", "totally",
    "definitely", "obviously", "always", "never", "just",
    "well", "also", "still", "even", "maybe",
]
G1_GENERIC_NOUNS = [
    "thing", "people", "way", "part", "stuff",
    "kind", "type", "side", "case", "point",
]
SHARED_FORUM_VOCAB = [
    "course", "class", "lecture", "discussion", "module",
    "professor", "tutor", "peer", "group", "team",
    "post", "thread", "reply", "comment", "topic",
    "question", "answer", "response", "feedback", "view",
    "assignment", "exercise", "task", "exam", "quiz",
    "reading", "video", "slide", "notes", "textbook",
    "i", "we", "you", "this", "that", "the", "a", "is", "are",
    "and", "or", "but", "for", "with", "about", "from", "to",
]
CONTENT_REL_CORE = [
    "concept", "theory", "framework", "model", "principle",
    "hypothesis", "argument", "evidence", "analysis", "interpretation",
    "definition", "implication", "assumption", "method", "approach",
    "literature", "research", "study", "finding", "result",
    "data", "variable", "correlation", "causation", "validity",
]
CONTENT_REL_COLLOQUIAL = [
    "idea", "viewpoint", "explanation", "reason",
    "example", "story", "experiment", "answer", "thought", "observation",
]
NON_CONTENT_CORE = [
    "deadline", "submission", "extension", "schedule", "office",
    "grade", "mark", "rubric", "absent", "missed",
    "late", "available", "appointment", "consultation", "session",
    "thanks", "regards", "hello", "hi", "introduction",
    "welcome", "congratulations", "weekend", "holiday", "break",
]


def _l2_typo_pass(words, rng):
    out = []
    for w in words:
        if w in {"a", "an", "the"} and rng.random() < 0.30:
            continue
        if w == "is" and rng.random() < 0.05:
            out.append("are"); continue
        if w == "are" and rng.random() < 0.05:
            out.append("is"); continue
        if w == "in" and rng.random() < 0.05:
            out.append("on"); continue
        out.append(w)
    return out


def _generate_post(g: int, y: int, rng: random.Random) -> tuple:
    """Realistic generation with HIGH cross-class contamination.

    Real forum posts from L1 students are NOT trivially separable —
    a content-relevant post often discusses logistics ("the deadline for
    submitting the analysis is..."), and a non-content post might use
    academic vocabulary ("hi everyone, I think the framework for
    submitting questions could be improved").

    To match this regime, we use HEAVY contamination (40%) and reduce
    vocabulary specificity (50% of the bucket goes to topic, vs 65% before).
    """
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

    contam_rate = 0.40   # HEAVY contamination — drives AUC down to ~0.92

    words = []
    for _ in range(target_len):
        r = rng.random()
        if r < 0.30:
            words.append(rng.choice(group_marker_pool))
        elif r < 0.38:
            words.append(rng.choice(group_secondary_pool))
        elif r < 0.65:
            # Reduced topic budget (was 0.75)
            if rng.random() < contam_rate:
                words.append(rng.choice(contamination))
            else:
                words.append(rng.choice(primary))
        else:
            # Higher shared-vocab fraction
            words.append(rng.choice(SHARED_FORUM_VOCAB))

    if g == 1:
        words = _l2_typo_pass(words, rng)
    return " ".join(words), len(words)


def make_realistic_forum(scale: float = 1.0, seed: int = SEED):
    """Generate Forum-Post-like dataset.
    scale=1.0 -> ~3700 posts (matches Sha et al.).
    """
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
            text, length = _generate_post(g, y, rng)
            actual_y = y
            if g == 1 and y == 1 and rng.random() < 0.08:
                actual_y = 0
            rows.append({"text": text, "G": g, "Y": actual_y, "length": length})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = make_realistic_forum(scale=1.0)
    print(f"Realistic Forum dataset: {len(df)} posts")
    print("\n=== Cell distribution ===")
    print(df.groupby(["G", "Y"]).size())
    print("\n=== Avg words/post by group ===")
    print(df.groupby("G")["length"].agg(["mean", "std"]))
    print("\n=== Label rate (Y=1) by group ===")
    print(df.groupby("G")["Y"].mean())
    print("\n=== Sample posts ===")
    for (g, y), grp in df.groupby(["G", "Y"]):
        s = grp.iloc[0]
        print(f"  [G={g} Y={y}] {s.text[:140]}...")
