"""Tests for Phase 3 evaluation logic + end-to-end smoke test on synthetic data.

The unit tests cover residualize / run_loocv / perm_null with controlled inputs
where the answer is known. The smoke test runs the same analysis pipeline the
notebook runs, with synthetic vectors sized like Provo (n=55), to verify nothing
crashes before we burn Colab time on real inference.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from cognix.phase3_eval import (
    DEFAULT_ALPHAS,
    load_features,
    passage_cloze,
    perm_null,
    residualize,
    run_loocv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
NORMS_PATH = REPO_ROOT / "data" / "provo" / "Provo_Corpus-Predictability_Norms.csv"
CHUNKS_PATH = REPO_ROOT / "data" / "phase3_provo_chunks.jsonl"


# ---------------------------------------------------------------------------
# residualize
# ---------------------------------------------------------------------------

def test_residualize_returns_zero_correlation_with_predictor():
    rng = np.random.default_rng(0)
    predictor = rng.normal(size=200)
    y = 3.0 * predictor + rng.normal(size=200) * 0.5  # y has predictor as part of signal
    resid = residualize(y, predictor)
    r, _ = np.corrcoef(predictor, resid)[0], None  # avoid scipy import in test
    # Pearson r between predictor and residual must be ~0
    actual_r = np.corrcoef(predictor, resid)[0, 1]
    assert abs(actual_r) < 1e-10, f'residual still correlated with predictor: r={actual_r}'


def test_residualize_constant_predictor_falls_back_to_intercept_only():
    """If predictor has no variance, OLS reduces to y_pred = mean(y), so residual = y - mean(y).

    This is the correct OLS behavior — a constant predictor carries no information,
    so the best linear fit is just the mean. We test it doesn't crash on the zero
    denominator and that variance is preserved (only the mean shift is removed).
    """
    y = np.array([1.0, 2.0, 3.0, 4.0])
    constant_predictor = np.array([5.0, 5.0, 5.0, 5.0])
    resid = residualize(y, constant_predictor)
    np.testing.assert_allclose(resid, y - y.mean(), atol=1e-9)
    # Variance preserved because a constant predictor has no signal to subtract
    assert abs(resid.var() - y.var()) < 1e-9


def test_residualize_perfect_correlation_zeros_out_signal():
    """If y is exactly proportional to predictor, residual should be zero."""
    predictor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.5 * predictor + 7.0  # affine
    resid = residualize(y, predictor)
    np.testing.assert_allclose(resid, np.zeros_like(y), atol=1e-9)


# ---------------------------------------------------------------------------
# run_loocv
# ---------------------------------------------------------------------------

def test_run_loocv_returns_expected_keys_and_types():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 10))
    y = X[:, 0] + 0.1 * rng.normal(size=30)  # genuine signal
    out = run_loocv(X, y)
    assert set(out.keys()) == {'preds', 'r2', 'rho', 'p_rho'}
    assert out['preds'].shape == (30,)
    assert isinstance(out['r2'], float)
    assert isinstance(out['rho'], float)
    assert isinstance(out['p_rho'], float)


def test_run_loocv_recovers_signal_when_present():
    """With genuine signal in feature 0, LOO R² should be clearly positive."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5))
    y = 2.0 * X[:, 0] + rng.normal(size=50) * 0.3
    out = run_loocv(X, y)
    assert out['r2'] > 0.8, f'Expected strong recovery, got R²={out["r2"]}'
    assert out['rho'] > 0.8


def test_run_loocv_pure_noise_gives_low_r2():
    """No signal → predictions ≈ mean → R² ≈ 0 or negative."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = rng.normal(size=30)  # independent of X
    out = run_loocv(X, y)
    assert out['r2'] < 0.3, f'Pure-noise R² unexpectedly high: {out["r2"]}'


def test_run_loocv_handles_n_features_gg_n_samples():
    """The Provo case: 20484 features, 55 samples. Must not crash."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(55, 1000))  # smaller p for test speed but same regime
    y = rng.normal(size=55)
    out = run_loocv(X, y)
    assert -2.0 < out['r2'] < 1.0  # valid R² range, can be negative


# ---------------------------------------------------------------------------
# perm_null
# ---------------------------------------------------------------------------

def test_perm_null_distribution_centers_near_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    y = X[:, 0] + 0.5 * rng.normal(size=40)
    null = perm_null(X, y, n_perms=20)  # small for speed
    assert null.shape == (20,)
    # Permuted R² should mostly be ≤ 0 (no learnable signal after shuffle)
    assert null.mean() < 0.3


# ---------------------------------------------------------------------------
# load_features
# ---------------------------------------------------------------------------

def test_load_features_stacks_in_hash_order_and_validates():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Make 3 fake feature vectors at known hashes
        hashes = ["aaaa", "bbbb", "cccc"]
        for i, h in enumerate(hashes):
            np.save(d / f"{h}.npy", np.full(7, fill_value=float(i), dtype=np.float32))
        X = load_features(d, hashes, expected_dim=7)
        assert X.shape == (3, 7)
        # Order must follow `hashes` order, not filesystem order
        np.testing.assert_array_equal(X[0], np.zeros(7))
        np.testing.assert_array_equal(X[1], np.ones(7))
        np.testing.assert_array_equal(X[2], np.full(7, 2.0))


def test_load_features_raises_on_missing_hash():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        np.save(d / "aaaa.npy", np.zeros(5, dtype=np.float32))
        try:
            load_features(d, ["aaaa", "missing"], expected_dim=5)
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "missing" in str(e)


def test_load_features_raises_on_shape_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        np.save(d / "aaaa.npy", np.zeros(7, dtype=np.float32))  # wrong shape
        try:
            load_features(d, ["aaaa"], expected_dim=10)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "expected (10,)" in str(e)


def test_load_features_raises_on_nan():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bad = np.zeros(5, dtype=np.float32)
        bad[2] = np.nan
        np.save(d / "aaaa.npy", bad)
        try:
            load_features(d, ["aaaa"], expected_dim=5)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "NaN" in str(e)


# ---------------------------------------------------------------------------
# passage_cloze (uses real Provo data)
# ---------------------------------------------------------------------------

def test_passage_cloze_returns_value_in_unit_interval_for_known_passage():
    """Cloze on real Provo data: must be in [0, 1] and not crash."""
    if not NORMS_PATH.exists():
        return  # data not downloaded; skip silently in this env
    cloze = passage_cloze(text_id=1, norms_path=NORMS_PATH)
    assert 0.0 <= cloze <= 1.0, f"Cloze out of [0,1]: {cloze}"


def test_passage_cloze_unknown_text_id_raises():
    if not NORMS_PATH.exists():
        return
    try:
        passage_cloze(text_id=99999, norms_path=NORMS_PATH)
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "99999" in str(e)


def test_passage_cloze_all_55_passages_succeed():
    """Every Provo passage should yield a finite cloze value."""
    if not (NORMS_PATH.exists() and CHUNKS_PATH.exists()):
        return
    chunks = [json.loads(l) for l in open(CHUNKS_PATH)]
    clozes = [passage_cloze(c["text_id"], NORMS_PATH) for c in chunks]
    assert len(clozes) == 55
    assert all(0.0 <= c <= 1.0 for c in clozes)
    # Sanity: cloze should vary across passages (some content words are predictable, some aren't)
    assert np.std(clozes) > 0.005, f"Cloze suspiciously uniform: std={np.std(clozes)}"


# ---------------------------------------------------------------------------
# End-to-end smoke test: same pipeline the notebook runs
# ---------------------------------------------------------------------------

def test_phase3_pipeline_smoke_on_synthetic_provo_sized_data():
    """Mimic the notebook's full grid + residualization on synthetic data.

    Goal: catch any integration bug (shape mismatches, NaN propagation,
    unexpected return formats) before running on real Colab outputs.

    Synthetic setup:
      - 55 passages (matches Provo)
      - 3 feature sources at TRIBE/LLaMA/ST dims
      - 5 targets, all derived from a shared latent + noise
      - Surprisal predictor with known relationship to targets
    """
    n = 55
    rng = np.random.default_rng(0)

    # Latent "true" cognitive load per passage
    latent = rng.normal(size=n)

    # Feature matrices — all 3 sources weakly carry the latent + their own noise
    X_tribe = (latent.reshape(-1, 1) * rng.normal(size=(1, 20484))
               + rng.normal(size=(n, 20484)) * 2.0).astype(np.float32)
    X_llama = (latent.reshape(-1, 1) * rng.normal(size=(1, 3072))
               + rng.normal(size=(n, 3072)) * 2.0).astype(np.float32)
    X_st = (latent.reshape(-1, 1) * rng.normal(size=(1, 384))
            + rng.normal(size=(n, 384)) * 2.0).astype(np.float32)

    # Reading-time-like targets, all positive ms-scale, anchored on latent
    targets = ['mean_ffd_skip_excl', 'mean_fprt_skip_excl',
               'mean_trt_skip_incl', 'mean_trt_skip_excl', 'mean_passage_total_trt']
    y_dict = {t: 200.0 + 30.0 * latent + rng.normal(size=n) * 5.0 for t in targets}

    # Surprisal predictor partially correlated with latent
    surprisal = 0.5 * latent + rng.normal(size=n) * 0.5

    # ---- Same loop as notebook cell 8 ----
    FEATURES = {'TRIBE': X_tribe, 'LLaMA': X_llama, 'ST': X_st}
    results_raw = {}
    for fname, X in FEATURES.items():
        results_raw[fname] = {}
        for tname in targets:
            r = run_loocv(X, y_dict[tname])
            assert 'r2' in r and 'preds' in r
            assert r['preds'].shape == (n,)
            assert not np.isnan(r['preds']).any()
            results_raw[fname][tname] = r

    # ---- Permutation null (notebook cell 10) ----
    primary = 'mean_trt_skip_incl'
    for fname, X in FEATURES.items():
        null = perm_null(X, y_dict[primary], n_perms=10)
        assert null.shape == (10,)
        assert not np.isnan(null).any()

    # ---- Residualization (notebook cells 13-14) ----
    resid = residualize(y_dict[primary], surprisal)
    assert resid.shape == (n,)
    actual_r = np.corrcoef(surprisal, resid)[0, 1]
    assert abs(actual_r) < 1e-10

    results_resid = {}
    for fname, X in FEATURES.items():
        results_resid[fname] = {}
        for tname in targets:
            y_resid = residualize(y_dict[tname], surprisal)
            r = run_loocv(X, y_resid)
            results_resid[fname][tname] = r

    # All runs produced numeric R² values within a sane range
    for fname in FEATURES:
        for tname in targets:
            for results in (results_raw, results_resid):
                r2 = results[fname][tname]['r2']
                assert -2.0 < r2 < 1.0, f'{fname}/{tname}: R² out of range: {r2}'
