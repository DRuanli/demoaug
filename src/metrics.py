"""Step E: evaluation metrics.

Reports six metrics in one call:
    AUC                — overall predictive accuracy
    ABROCA             — absolute between-ROC-area between groups (Sha et al. 2023)
    Dist-bias          — |P(Y=1|A=0) - P(Y=1|A=1)|
    Hard-bias_IH       — proposed IH-based hardness disparity
    Disparate Impact   — |1 - P(Y=1|A=0)/P(Y=1|A=1)| at threshold 0.5
    Equalized Odds gap — max(|TPR_diff|, |FPR_diff|) at threshold 0.5
    Calibration gap    — |E[Y|score=s, A=0] - E[Y|score=s, A=1]| binned

Multiple metrics provide a broader fairness lens than ABROCA alone.
"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from hardness import IHClass, hard_bias_ih
from config import ABROCA_GRID, SEED


def abroca(scores: np.ndarray, y_true: np.ndarray, sensitive: np.ndarray, n_grid: int = ABROCA_GRID) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0
    grid = np.linspace(0, 1, n_grid)
    rocs = {}
    for g in groups:
        mask = sensitive == g
        if mask.sum() < 5 or len(np.unique(y_true[mask])) < 2:
            return 0.0
        fpr, tpr, _ = roc_curve(y_true[mask], scores[mask])
        rocs[g] = np.interp(grid, fpr, tpr)
    diff = np.abs(rocs[groups[0]] - rocs[groups[1]])
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(diff, grid))


def dist_bias(A: np.ndarray, Y: np.ndarray) -> float:
    if len(np.unique(A)) < 2:
        return 0.0
    p0 = (Y[A == 0] == 1).mean() if (A == 0).sum() > 0 else 0
    p1 = (Y[A == 1] == 1).mean() if (A == 1).sum() > 0 else 0
    return float(abs(p0 - p1))


def disparate_impact(scores, y_true, sensitive, threshold=0.5):
    """DI = P(Yhat=1|A=0) / P(Yhat=1|A=1). Closer to 1 = fairer.
    Returns |1 - DI| so 0 = perfectly fair (parity)."""
    yhat = (scores >= threshold).astype(int)
    p0 = yhat[sensitive == 0].mean() if (sensitive == 0).sum() > 0 else 0.0
    p1 = yhat[sensitive == 1].mean() if (sensitive == 1).sum() > 0 else 0.0
    if p1 < 1e-9:
        return 1.0
    di = p0 / p1
    return float(abs(1.0 - di))


def equalized_odds_gap(scores, y_true, sensitive, threshold=0.5):
    """EO gap = max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|) at threshold."""
    yhat = (scores >= threshold).astype(int)
    def _tpr_fpr(g):
        mask = sensitive == g
        y, p = y_true[mask], yhat[mask]
        pos = (y == 1)
        neg = (y == 0)
        tpr = (p[pos] == 1).mean() if pos.sum() > 0 else 0.0
        fpr = (p[neg] == 1).mean() if neg.sum() > 0 else 0.0
        return tpr, fpr
    t0, f0 = _tpr_fpr(0)
    t1, f1 = _tpr_fpr(1)
    return float(max(abs(t0 - t1), abs(f0 - f1)))


def calibration_gap(scores, y_true, sensitive, n_bins: int = 10):
    """Mean across bins of |E[Y|score in bin, A=0] - E[Y|score in bin, A=1]|."""
    edges = np.linspace(0, 1, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() < 4:
            continue
        m0 = mask & (sensitive == 0)
        m1 = mask & (sensitive == 1)
        if m0.sum() < 2 or m1.sum() < 2:
            continue
        gaps.append(abs(y_true[m0].mean() - y_true[m1].mean()))
    return float(np.mean(gaps)) if gaps else 0.0


def evaluate_pipeline(
    Z_train: np.ndarray, A_train: np.ndarray, Y_train: np.ndarray,
    Z_test: np.ndarray, A_test: np.ndarray, Y_test: np.ndarray,
    method_name: str = "method",
):
    clf = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    clf.fit(Z_train, Y_train)
    scores = clf.predict_proba(Z_test)[:, list(clf.classes_).index(1)] \
        if 1 in clf.classes_ else np.zeros(len(Z_test))

    auc = roc_auc_score(Y_test, scores) if len(np.unique(Y_test)) > 1 else float("nan")

    ih_clf = IHClass()
    ih_train = ih_clf.fit_and_score(Z_train, Y_train)
    hb_y0 = hard_bias_ih(ih_train, A_train, Y_train, y_target=0)
    hb_y1 = hard_bias_ih(ih_train, A_train, Y_train, y_target=1)
    hb = 0.5 * (hb_y0 + hb_y1)

    return {
        "method": method_name,
        "AUC": float(auc),
        "ABROCA": abroca(scores, Y_test, A_test),
        "Dist-bias": dist_bias(A_train, Y_train),
        "Hard-bias_IH": float(hb),
        "DI_gap": disparate_impact(scores, Y_test, A_test),
        "EO_gap": equalized_odds_gap(scores, Y_test, A_test),
        "Calib_gap": calibration_gap(scores, Y_test, A_test),
        "n_train": len(Z_train),
    }
