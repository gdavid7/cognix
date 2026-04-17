"""Unit tests for Provo loader aggregation logic.

The full live-data run (`python -m cognix.provo_loader`) is the integration
test. These tests cover the edge cases that a casual eyeball of the output
won't catch: NA handling, all-skipped words, single-subject sanity.
"""

from cognix.provo_loader import (
    _parse_float_or_none,
    aggregate_passage_rt,
    text_hash,
)


def test_parse_float_or_none_handles_na_and_empty():
    assert _parse_float_or_none("NA") is None
    assert _parse_float_or_none("") is None
    assert _parse_float_or_none("147") == 147.0
    assert _parse_float_or_none("0") == 0.0
    assert _parse_float_or_none("garbage") is None


def test_text_hash_deterministic_and_short():
    h1 = text_hash("hello world")
    h2 = text_hash("hello world")
    h3 = text_hash("hello world!")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def _row(participant, word_number, ffd, fprt, trt):
    return {"participant": participant, "word_number": word_number,
            "ffd": ffd, "fprt": fprt, "trt": trt}


def test_aggregate_single_subject_no_skips():
    rows = [
        _row("S1", 1, ffd=200.0, fprt=200.0, trt=250.0),
        _row("S1", 2, ffd=180.0, fprt=180.0, trt=200.0),
    ]
    agg = aggregate_passage_rt(rows)
    assert agg["n_subjects"] == 1
    assert agg["mean_ffd_skip_excl"] == 190.0
    assert agg["mean_fprt_skip_excl"] == 190.0
    assert agg["mean_trt_skip_excl"] == 225.0
    assert agg["mean_trt_skip_incl"] == 225.0  # no skips, same as excl
    assert agg["mean_passage_total_trt"] == 450.0  # 250 + 200
    assert agg["skip_rate"] == 0.0


def test_aggregate_handles_skips():
    # Word 1 fixated, word 2 skipped (TRT=0, FFD/FPRT NA)
    rows = [
        _row("S1", 1, ffd=200.0, fprt=200.0, trt=200.0),
        _row("S1", 2, ffd=None, fprt=None, trt=0.0),
    ]
    agg = aggregate_passage_rt(rows)
    # Skip-excluded means use only fixated words
    assert agg["mean_ffd_skip_excl"] == 200.0
    assert agg["mean_fprt_skip_excl"] == 200.0
    assert agg["mean_trt_skip_excl"] == 200.0
    # Skip-included mean averages 200 and 0
    assert agg["mean_trt_skip_incl"] == 100.0
    assert agg["skip_rate"] == 0.5
    assert agg["mean_passage_total_trt"] == 200.0


def test_aggregate_multi_subject_averages_passage_total_correctly():
    # S1 reads passage in 500ms total, S2 in 700ms. Mean should be 600ms.
    rows = [
        _row("S1", 1, ffd=200.0, fprt=200.0, trt=200.0),
        _row("S1", 2, ffd=300.0, fprt=300.0, trt=300.0),
        _row("S2", 1, ffd=300.0, fprt=300.0, trt=300.0),
        _row("S2", 2, ffd=400.0, fprt=400.0, trt=400.0),
    ]
    agg = aggregate_passage_rt(rows)
    assert agg["n_subjects"] == 2
    assert agg["mean_passage_total_trt"] == 600.0
    # Per-word per-subject mean TRT: (200+300+300+400)/4 = 300
    assert agg["mean_trt_skip_incl"] == 300.0


def test_aggregate_all_skipped_returns_zeros_not_crash():
    """Edge case: every observation is a skip. Means over empty lists must return 0.0."""
    rows = [
        _row("S1", 1, ffd=None, fprt=None, trt=0.0),
        _row("S1", 2, ffd=None, fprt=None, trt=0.0),
    ]
    agg = aggregate_passage_rt(rows)
    assert agg["mean_ffd_skip_excl"] == 0.0
    assert agg["mean_fprt_skip_excl"] == 0.0
    assert agg["mean_trt_skip_excl"] == 0.0
    assert agg["mean_trt_skip_incl"] == 0.0
    assert agg["skip_rate"] == 1.0


def test_aggregate_handles_empty_rows():
    """No data at all → safe defaults, no division by zero."""
    agg = aggregate_passage_rt([])
    assert agg["n_subjects"] == 0
    assert agg["mean_passage_total_trt"] == 0.0
    assert agg["skip_rate"] == 0.0


def test_aggregate_drops_rows_with_none_trt():
    """A row with TRT=None (malformed) should be silently skipped, not counted."""
    rows = [
        _row("S1", 1, ffd=200.0, fprt=200.0, trt=200.0),
        _row("S1", 2, ffd=None, fprt=None, trt=None),  # malformed
    ]
    agg = aggregate_passage_rt(rows)
    assert agg["mean_trt_skip_incl"] == 200.0  # only the valid row counted
    assert agg["skip_rate"] == 0.0  # malformed row not counted as skip
