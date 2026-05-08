"""Step A: Instance Hardness via classifier scores (IH_class).

Smith et al. (2014, eq for IH_class):
    IH_class(x, y) = 1 - (1/|L|) sum_l P_l(y | x)

where L is a small ensemble of fast, diverse learners. We use 5-fold CV
to compute IH_class for ORIGINAL training points, and inference-only
predictions for synthetic candidates.

We also redefine Hard-bias under IH_class:
    Gamma_A^IH(y) = KL( f({IH | A=1, Y=y}) || f({IH | A=0, Y=y}) )
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gaussian_kde
from config import N_CV_FOLDS, HARDNESS_LEARNERS, SEED


def _make_learner(name):
    if name == "lr":
        return LogisticRegression(max_iter=500, random_state=SEED)
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    raise ValueError(name)


class IHClass:
    """Computes IH_class for points in a fixed training set, plus an
    inference interface for new candidates."""

    def __init__(self):
        self.learners = [_make_learner(n) for n in HARDNESS_LEARNERS]
        self._fitted_full = False

    def fit_and_score(self, Z: np.ndarray, Y: np.ndarray):
        """5-fold CV: train each learner on 4 folds, score the 5th.
        Returns IH_class for every point in Z.

        Then re-trains all learners on the full (Z, Y) so we can score new candidates.
        """
        n = len(Z)
        # Sum of P_l(y | x) across learners across folds
        score_sum = np.zeros(n)
        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
        for fold_idx, (tr, te) in enumerate(skf.split(Z, Y)):
            for learner_name in HARDNESS_LEARNERS:
                clf = _make_learner(learner_name)
                clf.fit(Z[tr], Y[tr])
                # P_l(y_true | x) for points in test fold
                proba = clf.predict_proba(Z[te])
                # Map class -> column
                classes = clf.classes_
                class_to_col = {c: i for i, c in enumerate(classes)}
                for local_i, global_i in enumerate(te):
                    y_true = Y[global_i]
                    if y_true in class_to_col:
                        score_sum[global_i] += proba[local_i, class_to_col[y_true]]

        n_learners = len(HARDNESS_LEARNERS)
        ih = 1.0 - score_sum / n_learners
        # Refit on full data for inference on new candidates
        self.learners = [_make_learner(n) for n in HARDNESS_LEARNERS]
        for clf in self.learners:
            clf.fit(Z, Y)
        self._fitted_full = True
        return ih

    def score_new(self, Z_new: np.ndarray, Y_new: np.ndarray) -> np.ndarray:
        """IH_class for new candidates using learners trained on full original set."""
        assert self._fitted_full
        n = len(Z_new)
        score_sum = np.zeros(n)
        for clf in self.learners:
            proba = clf.predict_proba(Z_new)
            classes = clf.classes_
            class_to_col = {c: i for i, c in enumerate(classes)}
            for i in range(n):
                yt = Y_new[i]
                if yt in class_to_col:
                    score_sum[i] += proba[i, class_to_col[yt]]
        return 1.0 - score_sum / len(self.learners)


def hard_bias_ih(ih_values: np.ndarray, A: np.ndarray, Y: np.ndarray, y_target: int = 1) -> float:
    """Symmetric KL between IH distributions across A within Y=y_target.
    Returns 0 if either group has too few samples."""
    mask = (Y == y_target)
    ih_y = ih_values[mask]
    a_y = A[mask]
    g0 = ih_y[a_y == 0]
    g1 = ih_y[a_y == 1]
    if len(g0) < 5 or len(g1) < 5:
        return 0.0

    grid = np.linspace(0, 1, 100)
    eps = 1e-6
    try:
        d0 = gaussian_kde(g0)(grid) + eps
        d1 = gaussian_kde(g1)(grid) + eps
    except Exception:
        return 0.0
    d0 /= d0.sum()
    d1 /= d1.sum()
    kl_01 = float(np.sum(d0 * np.log(d0 / d1)))
    kl_10 = float(np.sum(d1 * np.log(d1 / d0)))
    return 0.5 * (kl_01 + kl_10)
