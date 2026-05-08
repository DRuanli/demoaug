"""Step C2: demographic CALIBRATION probe filter.

The probe ensemble certifies that synthetic samples for cell (g, y) match
the conditional distribution P(text | G=g, Y=y) observed in real data.

Design rationale:
    The naive filter design ('demographic untraceability', probe outputs
    near 0.5) is theoretically wrong. It would force synthetic samples to
    be group-erased, creating distribution shift between training and test
    sets. The calibration variant says 'look like an authentic member of
    your intended group' instead, preserving the data manifold.

Mathematics:
    Acceptance criterion for sample x generated for cell (g_intended, y):
        keep iff probe(x) ∈ [μ_real(g_intended) - k·σ, μ_real(g_intended) + k·σ]
        AND  Var_m(probe_m(x)) < ε

    where μ_real(g) and σ_real(g) are the mean/std of probe outputs on
    REAL samples of group g.

Combined with V_disc banning (which removes the strongest demographic
shortcuts) and Method-3 (which balances cells), this achieves fairness
without distribution shift.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from config import (N_PROBES, PROBE_HIDDEN, PROBE_EPOCHS, PROBE_LR, SEED)


def _binary_entropy(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class ProbeEnsemble:
    def __init__(self, n_probes: int = N_PROBES):
        self.n_probes = n_probes
        self.probes = []
        self._fitted = False
        self.real_mean_per_g = {}
        self.real_std_per_g = {}

    def fit(self, Z_real, G_real, Z_synth_known=None, G_synth_known=None):
        if Z_synth_known is not None and len(Z_synth_known) > 0:
            Z = np.vstack([Z_real, Z_synth_known])
            G = np.concatenate([G_real, G_synth_known])
            from sklearn.linear_model import LogisticRegression
            d = np.concatenate([np.zeros(len(Z_real)), np.ones(len(Z_synth_known))])
            domain_clf = LogisticRegression(max_iter=300, random_state=SEED).fit(Z, d)
            propensity = domain_clf.predict_proba(Z)[:, 1]
            w = np.where(d == 1,
                         1.0 / np.clip(propensity, 0.05, 0.95),
                         1.0 / np.clip(1 - propensity, 0.05, 0.95))
            w = w / w.mean()
        else:
            Z, G = Z_real, G_real
            w = np.ones(len(Z))

        self.probes = []
        for i in range(self.n_probes):
            mlp = MLPClassifier(
                hidden_layer_sizes=(PROBE_HIDDEN,),
                max_iter=PROBE_EPOCHS,
                learning_rate_init=PROBE_LR,
                random_state=SEED + i * 97,
            )
            n = len(Z)
            probs = w / w.sum()
            idx = np.random.RandomState(SEED + i).choice(n, size=n, replace=True, p=probs)
            mlp.fit(Z[idx], G[idx])
            self.probes.append(mlp)

        # Calibration: compute real-data probe distribution per group
        P_real = self._predict_g_probs(Z_real)
        mean_real = P_real.mean(axis=0)
        for g in np.unique(G_real):
            mask = G_real == g
            if mask.sum() > 0:
                self.real_mean_per_g[int(g)] = float(mean_real[mask].mean())
                self.real_std_per_g[int(g)] = float(mean_real[mask].std())
        self._fitted = True
        return self

    def _predict_g_probs(self, Z):
        out = []
        for mlp in self.probes:
            classes = mlp.classes_
            col_of_1 = np.where(classes == 1)[0]
            if len(col_of_1) == 0:
                out.append(np.zeros(len(Z)))
            else:
                out.append(mlp.predict_proba(Z)[:, col_of_1[0]])
        return np.vstack(out)

    def predict_g_probs(self, Z):
        return self._predict_g_probs(Z)

    def quality_score(self, Z, intended_g):
        """Soft quality score in [0, 1] indicating how well a sample matches
        the intended group's real-data probe distribution. Higher = better.

        Score = exp( -((probe_mean - mu_real)/sigma_real)^2 )
        """
        assert self._fitted
        P = self._predict_g_probs(Z)
        mean_pred = P.mean(axis=0)
        mu = self.real_mean_per_g.get(int(intended_g), 0.5)
        sigma = self.real_std_per_g.get(int(intended_g), 0.2)
        sigma = max(sigma, 0.05)
        z_score = (mean_pred - mu) / sigma
        return np.exp(-0.5 * z_score**2)

    def keep_mask_calibrated(self, Z, intended_g, k_sigma: float = 2.0):
        """Keep iff probe mean is within k_sigma of real-data probe
        distribution for the INTENDED group."""
        assert self._fitted
        P = self._predict_g_probs(Z)
        mean_pred = P.mean(axis=0)
        var_pred = P.var(axis=0)

        mu = self.real_mean_per_g.get(int(intended_g), 0.5)
        sigma = self.real_std_per_g.get(int(intended_g), 0.2)
        sigma = max(sigma, 0.05)

        keep = (
            (mean_pred >= mu - k_sigma * sigma)
            & (mean_pred <= mu + k_sigma * sigma)
            & (var_pred < 0.05)
        )
        return keep, {
            "probe_mean": mean_pred, "probe_var": var_pred,
            "target_mu": mu, "target_sigma": sigma,
        }

    # Backward-compat: old neutrality method for ablation experiments
    def keep_mask(self, Z):
        from config import (PROBE_VARIANCE_TAU, PROBE_MEAN_LOW,
                            PROBE_MEAN_HIGH, PROBE_ENTROPY_MIN)
        P = self._predict_g_probs(Z)
        mean_p = P.mean(axis=0)
        var_p = P.var(axis=0)
        ent = _binary_entropy(mean_p)
        keep = (
            (var_p < PROBE_VARIANCE_TAU)
            & (mean_p >= PROBE_MEAN_LOW)
            & (mean_p <= PROBE_MEAN_HIGH)
            & (ent >= PROBE_ENTROPY_MIN)
        )
        return keep, {"probe_mean": mean_p, "probe_var": var_p, "probe_entropy": ent}
