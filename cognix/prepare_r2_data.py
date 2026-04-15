"""Prepare Round 2 dataset: merge handcrafted pairs, STS-B, and Wikipedia random baseline.

Outputs:
  data/validation_pairs_r2.jsonl   — all pairs with source/category labels
  data/r2_unique_texts.jsonl       — deduplicated text list for TRIBE inference

Usage:
  python -m cognix.prepare_r2_data
"""

import json
import hashlib
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
HANDCRAFTED_PATH = DATA_DIR / "validation_pairs_r2_handcrafted.jsonl"
OUTPUT_PAIRS_PATH = DATA_DIR / "validation_pairs_r2.jsonl"
OUTPUT_TEXTS_PATH = DATA_DIR / "r2_unique_texts.jsonl"

STS_SAMPLE_SIZE = 400
WIKI_SAMPLE_SIZE = 300  # number of pairs (needs 2x paragraphs)
WIKI_MIN_WORDS = 20
WIKI_MAX_WORDS = 150

RANDOM_SEED = 42


def load_handcrafted() -> list[dict]:
    """Load handcrafted divergence pairs."""
    pairs = []
    with open(HANDCRAFTED_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} handcrafted pairs")
    return pairs


def load_stsb(n: int = STS_SAMPLE_SIZE) -> list[dict]:
    """Download STS-B and sample pairs spanning the similarity range."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        raise

    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    print(f"STS-B test set: {len(ds)} pairs")

    # Bin by similarity score (0-5) and sample evenly across bins
    bins = {i: [] for i in range(6)}
    for row in ds:
        score = row["score"]
        bin_idx = min(int(score), 5)
        bins[bin_idx].append(row)

    rng = random.Random(RANDOM_SEED)
    per_bin = n // 6
    sampled = []
    for bin_idx in sorted(bins.keys()):
        available = bins[bin_idx]
        take = min(per_bin, len(available))
        sampled.extend(rng.sample(available, take))

    # Fill remaining from any bin
    remaining = n - len(sampled)
    if remaining > 0:
        all_rows = [r for r in ds if r not in sampled]
        sampled.extend(rng.sample(all_rows, min(remaining, len(all_rows))))

    pairs = []
    for i, row in enumerate(sampled):
        pairs.append({
            "id": 10000 + i,
            "source": "stsb",
            "category": "stsb",
            "text_a": row["sentence1"],
            "text_b": row["sentence2"],
            "expected": f"sts_score_{row['score']:.1f}",
            "sts_score": round(row["score"], 2),
        })

    print(f"Sampled {len(pairs)} STS-B pairs across score range")
    return pairs


def load_wikipedia_random(n_pairs: int = WIKI_SAMPLE_SIZE) -> list[dict]:
    """Download random Wikipedia paragraphs and pair them randomly."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        raise

    # Use wikimedia/wikipedia which is the maintained version
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    rng = random.Random(RANDOM_SEED)
    paragraphs = []
    target = n_pairs * 2 + 200  # collect extra to filter

    print(f"Sampling ~{target} Wikipedia paragraphs (streaming)...")
    for i, article in enumerate(ds):
        if len(paragraphs) >= target:
            break
        # Split article into paragraphs, take ones in word-count range
        for para in article["text"].split("\n\n"):
            para = para.strip()
            word_count = len(para.split())
            if WIKI_MIN_WORDS <= word_count <= WIKI_MAX_WORDS:
                paragraphs.append(para)
                if len(paragraphs) >= target:
                    break
        if i % 5000 == 0 and i > 0:
            print(f"  Scanned {i} articles, collected {len(paragraphs)} paragraphs...")

    print(f"Collected {len(paragraphs)} Wikipedia paragraphs")

    # Shuffle and pair randomly
    rng.shuffle(paragraphs)
    pairs = []
    for i in range(min(n_pairs, len(paragraphs) // 2)):
        pairs.append({
            "id": 20000 + i,
            "source": "wikipedia_random",
            "category": "random_baseline",
            "text_a": paragraphs[2 * i],
            "text_b": paragraphs[2 * i + 1],
            "expected": "random",
        })

    print(f"Created {len(pairs)} random Wikipedia pairs")
    return pairs


def deduplicate_texts(pairs: list[dict]) -> list[dict]:
    """Extract all unique texts across all pairs."""
    seen = {}
    for p in pairs:
        for text in [p["text_a"], p["text_b"]]:
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            if h not in seen:
                seen[h] = text

    texts = [{"hash": h, "text": t} for h, t in seen.items()]
    print(f"Deduplicated to {len(texts)} unique texts from {len(pairs)} pairs")
    return texts


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Load all sources
    handcrafted = load_handcrafted()
    stsb = load_stsb()
    wiki = load_wikipedia_random()

    # Merge
    all_pairs = handcrafted + stsb + wiki
    print(f"\nTotal pairs: {len(all_pairs)}")

    # Summary
    sources = {}
    for p in all_pairs:
        src = p["source"]
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    categories = {}
    for p in all_pairs:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\nCategories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Deduplicate texts
    unique_texts = deduplicate_texts(all_pairs)

    # Write outputs
    with open(OUTPUT_PAIRS_PATH, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(all_pairs)} pairs to {OUTPUT_PAIRS_PATH}")

    with open(OUTPUT_TEXTS_PATH, "w") as f:
        for t in unique_texts:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {len(unique_texts)} unique texts to {OUTPUT_TEXTS_PATH}")


if __name__ == "__main__":
    main()
