"""Step C1: label fidelity filter.

Train a classifier on the original (real) (Z, Y) data, then for each
generated sample (Z_gen, intended Y), check if the classifier's
confidence on the intended label exceeds tau_label. If not, drop.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from config import TAU_LABEL, SEED


class LabelFidelityFilter:
    def __init__(self, tau: float = TAU_LABEL):
        self.tau = tau
        self.clf = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
        self._fitted = False

    def fit(self, Z, Y):
        self.clf.fit(Z, Y)
        self._fitted = True
        return self

    def confidence(self, Z, Y_target):
        """Return P_hat(Y=Y_target | Z) for each row."""
        assert self._fitted
        proba = self.clf.predict_proba(Z)
        classes = self.clf.classes_
        col_of = {c: i for i, c in enumerate(classes)}
        n = len(Z)
        out = np.zeros(n)
        for i in range(n):
            yt = Y_target[i]
            if yt in col_of:
                out[i] = proba[i, col_of[yt]]
        return out

    def keep_mask(self, Z, Y_target):
        return self.confidence(Z, Y_target) >= self.tau
