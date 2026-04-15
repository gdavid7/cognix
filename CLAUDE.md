# Cognix

Brain-grounded embedding system that maps text into vectors where similarity reflects predicted similarity of human brain processing, not standard semantic similarity. Built on Meta's TRIBE v2.

## Core pipeline

```
text -> TRIBE v2 (LLaMA 3.2 -> 8-layer Transformer -> brain projection)
     -> predicted brain tensor (T, 20484)
     -> temporal mean pooling -> (20484,)
     -> MLP projection head -> (512,)
     -> cosine similarity
```

## Key facts

- TRIBE v2 outputs 20,484 cortical vertices (fsaverage5 mesh), not 70k
- TRIBE v2 needs 40GB+ GPU VRAM (A100 minimum) — cannot run on Mac Air
- TRIBE v2 license: CC BY-NC (non-commercial only)
- TRIBE v2 internally uses LLaMA 3.2-3B (requires HuggingFace gated access)
- Product name is **Cognix**, not CSE
- Design doc: `cognitive_similarity_embedding_system_design_v2.md`
- Full game plan: `.claude/plans/zazzy-forging-rivest.md`

## Current status

Working on: **Phase 0 (environment setup) and Phase 1 (validation round 1, 100 pairs)**

Everything else (scaffold, training, eval, HuggingFace release) comes later — only after validation proves brain similarity diverges from semantic similarity.

## Dev setup

- Local machine: Mac Air (no GPU) — used for code, data curation, sentence-transformer baselines
- GPU compute: TBD (likely Colab Pro+ with A100)
- Workflow: build locally, push to GitHub, run TRIBE inference on Colab, download cached vectors

## Architecture decisions

- Projection head: `20484 -> 1024 -> ReLU -> Dropout(0.1) -> 512`
- Training: contrastive loss (InfoNCE) on cached brain vectors
- Baselines: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

## Primary risk

The embedding may collapse to behave like a standard sentence-transformer because TRIBE's brain predictions are a learned transformation of LLaMA features. The validation experiment (Phase 1) tests this before building the full system.

## TRIBE v2 setup

```bash
pip uninstall -y numpy
pip install 'numpy>=1.26.4,<2.1.0'
git clone https://github.com/facebookresearch/tribev2.git
cd tribev2
pip install -e ".[plotting]"
huggingface-cli login  # needs access to meta-llama/Llama-3.2-3B
```

```python
from tribev2 import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
events = model.get_events_dataframe(text_path="input.txt")
preds, segments = model.predict(events=events)
# preds.shape = (T, 20484)
```

## Validation experiment (Phase 1)

100 text pairs across 4 categories:
- ~25 paraphrases (expect both scores high)
- ~25 unrelated (expect both scores low)
- ~25 "different meaning, same brain" (the interesting divergence pairs)
- ~25 "same topic, different processing" (controls)

Compare `sim_brain` (cosine of pooled TRIBE vectors) vs `sim_semantic` (sentence-transformer cosine).

Decision gate:
- r > 0.95 -> project likely not viable
- r = 0.8-0.95 -> weak signal, investigate
- r < 0.8 -> meaningful divergence, proceed

## Repo structure (target)

```
cognix/
  pyproject.toml
  requirements.txt
  README.md
  cognix/
    __init__.py
    config.py
    tribe_wrapper.py
    pooling.py
    projection.py
    similarity.py
    datasets.py
    train.py
    evaluate.py
    cache_features.py
    inference.py
  notebooks/
    00_validation_experiment.ipynb
  configs/
    mvp_text.yaml
  data/
    validation_pairs.jsonl
  artifacts/
    cached_vectors/
    checkpoints/
```
