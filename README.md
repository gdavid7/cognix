# Cognix

Brain-grounded embedding system where similarity means "processed similarly by the human brain" — not just "means the same thing."

Standard embedding models (sentence-transformers, CLIP) capture **semantic similarity**. Cognix captures **cognitive similarity** by routing text through [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2), a model that predicts fMRI brain responses from text, audio, and video.

## How it works

```
text → TRIBE v2 → predicted brain response (T × 20,484 cortical vertices)
     → temporal mean pooling → brain fingerprint (20,484-d)
     → learned projection → Cognix embedding (512-d)
     → cosine similarity
```

TRIBE v2 internally converts text to speech, extracts word-level timestamps, runs LLaMA 3.2-3B for language features, and maps those through an 8-layer transformer (trained on 1,000+ hours of real fMRI data from 720+ subjects) onto the cortical surface.

## Why this matters

Two texts can be semantically unrelated but cognitively similar:

| Text A | Text B | Semantic sim | Expected brain sim |
|--------|--------|-------------|-------------------|
| Dense legal clause about indemnification | Dense math proof about eigenvalues | Low | High (both demand heavy working memory) |
| Soldier watching a friend die | Reading a terminal diagnosis letter | Low | High (same emotional arousal pattern) |
| Kicking a ball across a field | Stomping on a brake pedal | Low | High (same motor cortex activation) |
| Vast empty desert | Open ocean, no land in sight | Moderate | High (same spatial scene processing) |

## Current status

**Phase 1: Validation experiment.** Testing whether brain-predicted similarity actually diverges from standard semantic similarity on 100 curated text pairs. If correlation is < 0.8, the hypothesis holds and we proceed to build the full system.

## Project structure

```
cognix/
  cognix/              # Python package
    tribe_wrapper.py   # TRIBE v2 loading, inference, caching
  notebooks/
    00_validation_experiment.ipynb  # Colab-ready validation notebook
  data/
    validation_pairs.jsonl          # 100 curated text pairs (10 categories)
```

## Running the validation experiment

Requires Google Colab with A100 GPU (free for students via [Colab Pro](https://colab.research.google.com/signup)).

1. Add your [HuggingFace token](https://huggingface.co/settings/tokens) as a Colab secret named `HF_TOKEN` (requires access to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B))
2. Open `notebooks/00_validation_experiment.ipynb` from GitHub in Colab
3. Set runtime to **A100 GPU + High RAM**
4. Run all cells (~25 min)

## Dependencies

- [TRIBE v2](https://github.com/facebookresearch/tribev2) (Meta, CC BY-NC 4.0)
- PyTorch, sentence-transformers, scipy, scikit-learn

## License

This project depends on TRIBE v2, which is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Cognix inherits that constraint: **non-commercial use only**.
