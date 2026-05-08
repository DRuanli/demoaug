"""Step 0: operationalize 'off-manifold' as low log-density.

We fit a PCA-whitened kernel density estimator on real BERT embeddings.
Then for any candidate point z, log p_hat(z) is a measurable proxy for
'how natural' it is. SMOTE-interpolated points should score significantly
lower than real points or LLM-generated points.

This is a *measurement tool*, not part of the augmentation pipeline.
It's used in the paper's Section 4 to make the geometric critique
empirically grounded.
"""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA


class ManifoldDensity:
    def __init__(self, n_components: int = 16, bandwidth: float = 0.5):
        self.pca = PCA(n_components=n_components, whiten=True)
        self.kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self._fitted = False

    def fit(self, Z: np.ndarray):
        n_components = min(self.pca.n_components, Z.shape[1] - 1, Z.shape[0] - 1)
        self.pca = PCA(n_components=n_components, whiten=True)
        Zp = self.pca.fit_transform(Z)
        self.kde.fit(Zp)
        self._fitted = True
        return self

    def log_prob(self, Z: np.ndarray) -> np.ndarray:
        assert self._fitted, "ManifoldDensity not fitted"
        Zp = self.pca.transform(Z)
        return self.kde.score_samples(Zp)

    def report_gap(self, Z_real, Z_smote, Z_llm):
        """Return a dict summarizing log-density of each source."""
        return {
            "real_log_p_mean": float(np.mean(self.log_prob(Z_real))),
            "smote_log_p_mean": float(np.mean(self.log_prob(Z_smote))),
            "llm_log_p_mean": float(np.mean(self.log_prob(Z_llm))),
            "smote_offset_vs_real": float(
                np.mean(self.log_prob(Z_smote)) - np.mean(self.log_prob(Z_real))
            ),
            "llm_offset_vs_real": float(
                np.mean(self.log_prob(Z_llm)) - np.mean(self.log_prob(Z_real))
            ),
        }
