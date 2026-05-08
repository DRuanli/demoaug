"""Step B1: extract V_disc — vocabulary that discriminates G=0 vs G=1.

Procedure (revised for balance):
  1. TF-IDF vectorize texts (unigrams + bigrams).
  2. Fit L2-LR (NOT L1, since we want a DENSE coefficient vector).
  3. Take top K/2 tokens with most negative coefficient (G=0 markers)
     AND top K/2 tokens with most positive coefficient (G=1 markers).
  4. This guarantees both groups' demographic signals get banned.

Rationale:
  An earlier L1 version produced a one-sided V_disc (only G=1 markers
  survived L1 selection). Banning only G=1 markers makes generated text
  *more* G=0-like, which is exactly the opposite of demographic neutrality.
  L2 with explicit balance fixes this.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import (TFIDF_MAX_FEATURES, TFIDF_NGRAM, LR_C_L1,
                    BONFERRONI_ALPHA, TOP_K_DISC_VOCAB, SEED)


def extract_disc_vocab(texts, G):
    """Return:
        disc_tokens: List[str] — balanced bans (K/2 from each direction)
        coefs:       np.ndarray of corresponding coefficients
    """
    G = np.asarray(G)
    if len(np.unique(G)) < 2:
        return [], np.array([])

    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM,
        min_df=2,
    )
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out()
    V = len(feature_names)
    if V == 0:
        return [], np.array([])

    # L2 logistic regression — gives a dense coef vector
    clf = LogisticRegression(
        penalty="l2", solver="liblinear", C=LR_C_L1,
        max_iter=500, random_state=SEED,
    )
    clf.fit(X, G)
    coefs = clf.coef_.ravel()  # shape (V,) — sign indicates direction

    half = TOP_K_DISC_VOCAB // 2
    # Top half from each tail
    g1_idx = np.argsort(-coefs)[:half]   # most positive — G=1 markers
    g0_idx = np.argsort(coefs)[:half]    # most negative — G=0 markers
    selected = np.concatenate([g0_idx, g1_idx])
    return list(feature_names[selected]), coefs[selected]


def expand_to_subwords(words, tokenizer=None):
    """For real Mistral: expand each banned word to subword tokens."""
    if tokenizer is None:
        return set(words)
    banned_token_ids = set()
    for w in words:
        for variant in [w, " " + w, w.capitalize(), " " + w.capitalize()]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            banned_token_ids.update(ids)
    return banned_token_ids
