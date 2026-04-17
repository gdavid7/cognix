"""Load Provo Corpus eye-tracking data and aggregate to passage level.

Data source: Luke & Christianson (2018), https://osf.io/sjefs/

Inputs (place in data/provo/):
  - Provo_Corpus-Predictability_Norms.csv  (passage texts + cloze norms)
  - Provo_Corpus-Eyetracking_Data.csv      (per-(subject, word) fixation data)

Output:
  - data/phase3_provo_chunks.jsonl  (one row per passage)

Each output row:
  {
    "text_id":      int,        # Provo Text_ID, 1-55
    "hash":         str,        # SHA-256 of text, first 16 chars (matches Round 2 cache key)
    "text":         str,        # full passage text (whitespace-stripped)
    "n_words":      int,        # word count of text
    "n_subjects":   int,        # number of subjects with data on this passage

    # Reading-time targets, all in milliseconds, all per word per subject means.
    # "skip_excl": only fixated words (NA dropped). "skip_incl": skipped = 0ms.
    # FFD = first fixation duration. FPRT = first-pass reading time (gaze duration).
    # TRT = total reading time including refixations.
    "mean_ffd_skip_excl":  float,
    "mean_fprt_skip_excl": float,
    "mean_trt_skip_excl":  float,
    "mean_trt_skip_incl":  float,
    "mean_passage_total_trt": float,  # total TRT for the passage, avg across subjects
    "skip_rate":   float,       # fraction of (subject, word) pairs with TRT == 0
  }

Usage:
  python -m cognix.provo_loader

Or as a library:
  from cognix.provo_loader import load_provo_chunks
  chunks = load_provo_chunks()
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROVO_DIR = DATA_DIR / "provo"

NORMS_PATH = PROVO_DIR / "Provo_Corpus-Predictability_Norms.csv"
EYETRACK_PATH = PROVO_DIR / "Provo_Corpus-Eyetracking_Data.csv"
OUTPUT_PATH = DATA_DIR / "phase3_provo_chunks.jsonl"

# Filter threshold — TRIBE requires paragraph-length input. CLAUDE.md notes that
# texts under ~10 words often fail or produce trivial time steps. ≥20 is a
# conservative cutoff. Provo passages are 39-62 words so this is a no-op for
# Provo, but kept for explicitness and reuse.
MIN_WORDS = 20

# Files use latin-1 (Windows-1252) encoding, not UTF-8.
ENCODING = "latin-1"


def text_hash(text: str) -> str:
    """SHA-256 first 16 chars. Matches Round 2 cache key convention."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_float_or_none(value: str) -> float | None:
    """Parse a CSV value that may be 'NA' or empty."""
    if value in ("NA", "", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_passage_texts(norms_path: Path = NORMS_PATH) -> dict[int, str]:
    """Return {text_id: passage_text} from the predictability norms CSV.

    The Text column is repeated for every word in the passage; we take the
    first occurrence per Text_ID.
    """
    if not norms_path.exists():
        raise FileNotFoundError(
            f"Missing {norms_path}. Download from https://osf.io/sjefs/ "
            f"and place in {PROVO_DIR}/."
        )

    texts: dict[int, str] = {}
    with open(norms_path, encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["Text_ID"])
            if tid not in texts:
                texts[tid] = row["Text"].strip()
    return texts


def load_fixations(
    eyetrack_path: Path = EYETRACK_PATH,
) -> dict[int, list[dict]]:
    """Return {text_id: [{participant, word_number, ffd, fprt, trt}, ...]}.

    NA values become None. TRT zeros are preserved (they encode skips).
    """
    if not eyetrack_path.exists():
        raise FileNotFoundError(
            f"Missing {eyetrack_path}. Download from https://osf.io/sjefs/ "
            f"and place in {PROVO_DIR}/."
        )

    fixations: dict[int, list[dict]] = defaultdict(list)
    with open(eyetrack_path, encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(row["Text_ID"])
            except (ValueError, KeyError):
                continue  # malformed row
            fixations[tid].append({
                "participant": row["Participant_ID"],
                "word_number": int(row["Word_Number"]) if row["Word_Number"] not in ("", "NA") else None,
                "ffd": _parse_float_or_none(row["IA_FIRST_FIXATION_DURATION"]),
                "fprt": _parse_float_or_none(row["IA_FIRST_RUN_DWELL_TIME"]),
                "trt": _parse_float_or_none(row["IA_DWELL_TIME"]),
            })
    return dict(fixations)


def aggregate_passage_rt(rows: list[dict]) -> dict:
    """Aggregate per-(subject, word) fixation rows to passage-level summary."""
    # Per-subject totals for "passage total reading time"
    per_subject_total_trt: dict[str, float] = defaultdict(float)
    per_subject_n_words: dict[str, int] = defaultdict(int)

    ffd_fixated: list[float] = []   # FFD when fixated
    fprt_fixated: list[float] = []  # FPRT when fixated
    trt_fixated: list[float] = []   # TRT when fixated (TRT > 0)
    trt_all: list[float] = []       # TRT including 0s for skips

    skipped = 0
    total = 0

    for r in rows:
        if r["trt"] is None:
            continue  # malformed row; skip
        total += 1
        trt = r["trt"]
        per_subject_total_trt[r["participant"]] += trt
        per_subject_n_words[r["participant"]] += 1
        trt_all.append(trt)
        if trt == 0:
            skipped += 1
        else:
            trt_fixated.append(trt)
            if r["ffd"] is not None:
                ffd_fixated.append(r["ffd"])
            if r["fprt"] is not None:
                fprt_fixated.append(r["fprt"])

    n_subjects = len(per_subject_total_trt)
    mean_passage_total_trt = (
        sum(per_subject_total_trt.values()) / n_subjects if n_subjects else 0.0
    )

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_subjects": n_subjects,
        "mean_ffd_skip_excl": _mean(ffd_fixated),
        "mean_fprt_skip_excl": _mean(fprt_fixated),
        "mean_trt_skip_excl": _mean(trt_fixated),
        "mean_trt_skip_incl": _mean(trt_all),
        "mean_passage_total_trt": mean_passage_total_trt,
        "skip_rate": skipped / total if total else 0.0,
    }


def load_provo_chunks(
    norms_path: Path = NORMS_PATH,
    eyetrack_path: Path = EYETRACK_PATH,
    min_words: int = MIN_WORDS,
) -> list[dict]:
    """Load and aggregate Provo into per-passage records, filtered by min_words."""
    texts = load_passage_texts(norms_path)
    fixations = load_fixations(eyetrack_path)

    chunks = []
    for text_id, text in sorted(texts.items()):
        n_words = len(text.split())
        if n_words < min_words:
            continue
        rows = fixations.get(text_id, [])
        if not rows:
            continue
        agg = aggregate_passage_rt(rows)
        chunks.append({
            "text_id": text_id,
            "hash": text_hash(text),
            "text": text,
            "n_words": n_words,
            **agg,
        })
    return chunks


def write_chunks(chunks: Iterable[dict], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def main():
    chunks = load_provo_chunks()
    write_chunks(chunks)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")
    print(f"\nSummary:")
    print(f"  Total passages: {len(chunks)}")
    print(f"  Word count range: {min(c['n_words'] for c in chunks)}-{max(c['n_words'] for c in chunks)}")
    print(f"  Subjects per passage: {min(c['n_subjects'] for c in chunks)}-{max(c['n_subjects'] for c in chunks)}")
    print(f"  Mean per-word TRT (skip incl): "
          f"{sum(c['mean_trt_skip_incl'] for c in chunks) / len(chunks):.1f} ms "
          f"(range {min(c['mean_trt_skip_incl'] for c in chunks):.0f}-{max(c['mean_trt_skip_incl'] for c in chunks):.0f})")
    print(f"  Mean skip rate: "
          f"{100 * sum(c['skip_rate'] for c in chunks) / len(chunks):.1f}% "
          f"(range {100*min(c['skip_rate'] for c in chunks):.0f}-{100*max(c['skip_rate'] for c in chunks):.0f}%)")


if __name__ == "__main__":
    main()
