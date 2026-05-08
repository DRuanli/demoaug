"""Encoder interface. Toy implementation = TF-IDF + TruncatedSVD.
For the real run, swap to bert-base-uncased / Legal-BERT [CLS] features.

Both implementations expose the same interface:
    encoder.fit(texts: List[str])
    encoder.transform(texts: List[str]) -> np.ndarray of shape (N, D)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from config import EMBED_DIM, TFIDF_MAX_FEATURES, TFIDF_NGRAM, USE_REAL_MODELS, SEED


class ToyEncoder:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM,
            min_df=2,
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=SEED)
        self._fitted = False

    def fit(self, texts):
        X = self.tfidf.fit_transform(texts)
        # If vocabulary smaller than requested dim, shrink dim to fit
        max_components = min(self.dim, X.shape[1] - 1)
        if max_components < self.dim:
            self.svd = TruncatedSVD(n_components=max_components, random_state=SEED)
            self.dim = max_components
        self.svd.fit(X)
        self._fitted = True
        return self

    def transform(self, texts):
        assert self._fitted, "Encoder not fitted"
        X = self.tfidf.transform(texts)
        Z = self.svd.transform(X)
        # L2-normalize so cosine ≈ Euclidean (matches how BERT [CLS] behaves
        # after layer-norm)
        Z = normalize(Z, norm="l2")
        return Z


class BertEncoder:
    """Real BERT encoder. Only constructed when USE_REAL_MODELS=True."""
    def __init__(self, model_name: str = "bert-base-uncased"):
        from transformers import AutoModel, AutoTokenizer  # lazy
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.dim = self.model.config.hidden_size

    def fit(self, texts):
        return self  # nothing to fit

    def transform(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=256, return_tensors="pt")
            if self.torch.cuda.is_available():
                tok = {k: v.cuda() for k, v in tok.items()}
            with self.torch.no_grad():
                outputs = self.model(**tok)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            out.append(cls)
        Z = np.vstack(out)
        from sklearn.preprocessing import normalize
        return normalize(Z, norm="l2")


def get_encoder():
    if USE_REAL_MODELS:
        return BertEncoder()
    return ToyEncoder()
