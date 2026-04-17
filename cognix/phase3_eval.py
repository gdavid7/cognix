"""Reading-time prediction evaluation for Phase 3.

Holds the testable pieces of the analysis: feature loading, leave-one-out
ridge regression, permutation null, surprisal residualization, and Provo
cloze parsing. The Phase 3 analysis notebook imports these so the smoke
test exercises identical code paths.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict


DEFAULT_ALPHAS = np.logspace(-2, 6, 25)


def run_loocv(X: np.ndarray, y: np.ndarray, alphas: np.ndarray = DEFAULT_ALPHAS) -> dict:
    """Leave-one-out CV with RidgeCV.

    Returns dict with: preds (n_samples,), r2, rho (Spearman), p_rho.
    """
    preds = cross_val_predict(RidgeCV(alphas=alphas), X, y, cv=LeaveOneOut())
    r2 = r2_score(y, preds)
    rho, p_rho = stats.spearmanr(y, preds)
    return {"preds": preds, "r2": float(r2), "rho": float(rho), "p_rho": float(p_rho)}


def perm_null(
    X: np.ndarray,
    y: np.ndarray,
    n_perms: int = 200,
    seed: int = 0,
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> np.ndarray:
    """Permutation null distribution of LOO R²: shuffle y, re-run, record R²."""
    rng = np.random.default_rng(seed)
    null_r2s = np.empty(n_perms)
    for i in range(n_perms):
        y_shuf = rng.permutation(y)
        preds = cross_val_predict(RidgeCV(alphas=alphas), X, y_shuf, cv=LeaveOneOut())
        null_r2s[i] = r2_score(y_shuf, preds)
    return null_r2s


def load_features(directory: Path, hashes: list[str], expected_dim: int) -> np.ndarray:
    """Stack per-passage feature vectors in `hashes` order.

    Each vector lives at `directory/{hash}.npy`. Validates shape and absence
    of NaN/Inf. Raises FileNotFoundError if any hash is missing.
    """
    rows = []
    missing = []
    for h in hashes:
        path = Path(directory) / f"{h}.npy"
        if not path.exists():
            missing.append(h)
            continue
        v = np.load(path)
        if v.shape != (expected_dim,):
            raise ValueError(f"{path.name}: expected ({expected_dim},), got {v.shape}")
        if np.isnan(v).any() or np.isinf(v).any():
            raise ValueError(f"{path.name}: contains NaN or Inf")
        rows.append(v)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} vectors in {directory}: {missing[:3]}..."
        )
    return np.stack(rows).astype(np.float32)


def passage_cloze(text_id: int, norms_path: Path) -> float:
    """Mean cloze probability per passage from Provo predictability norms.

    For each (Text_ID, Word_Number), the cloze probability is the
    Response_Proportion of the row where Response equals Word
    (case-insensitive). Words with no matching response get cloze 0.
    Returns mean across all words in the passage.
    """
    word_cloze: dict[int, float] = {}
    word_actual: dict[int, str] = {}
    with open(norms_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(row["Text_ID"])
            except (ValueError, KeyError):
                continue
            if tid != text_id:
                continue
            wn = int(row["Word_Number"])
            actual = row["Word"].strip().lower()
            response = row["Response"].strip().lower()
            word_actual[wn] = actual
            if response == actual:
                # Use max in case the actual word appears as a response from
                # multiple respondents and gets aggregated — Provo de-dupes
                # already, so this is a defensive guard.
                word_cloze[wn] = max(word_cloze.get(wn, 0.0), float(row["Response_Proportion"]))
    if not word_actual:
        raise ValueError(f"No data for text_id={text_id} in {norms_path}")
    clozes = [word_cloze.get(wn, 0.0) for wn in word_actual]
    return float(np.mean(clozes))


def residualize(y: np.ndarray, predictor: np.ndarray) -> np.ndarray:
    """Linear-regress `predictor` out of `y`. Returns y minus its OLS projection on predictor.

    The residual has zero correlation with `predictor`. Useful for asking
    "what does X add *beyond* `predictor`?"
    """
    s_centered = predictor - predictor.mean()
    y_centered = y - y.mean()
    denom = (s_centered * s_centered).sum() + 1e-12
    w = (s_centered * y_centered).sum() / denom
    pred = w * s_centered + y.mean()
    return y - pred
