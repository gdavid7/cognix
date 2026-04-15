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

## Real-world implications

**Search and retrieval** — Find content by how it feels to process, not just what it's about. "Find me articles that are as cognitively demanding as this legal brief" or "find video clips with a similar attentional profile to this one."

**Content accessibility** — Flag content that will be processed similarly to known-difficult material. If a patient information leaflet triggers the same brain pattern as a dense legal contract, it's probably too complex for its audience.

**Education** — Match learning materials to a student's cognitive level. Two explanations of the same concept can have identical semantic content but wildly different processing demands. Cognix could quantify that difference.

**Mental health and media** — Identify content with high emotional arousal signatures without relying on keyword matching. A passage about war and a passage about a medical diagnosis might trigger the same distress pattern — semantic filters miss this, brain-grounded embeddings wouldn't.

**Recommendation systems** — Recommend content based on cognitive engagement patterns rather than topic. "You were deeply engaged by this thriller — here's a medical drama with a similar suspense-processing profile" instead of "here's another thriller."

**AI alignment research** — Compare how language models represent text vs. how the human brain processes it. Cognix embeddings could serve as a human-grounded reference point for evaluating whether model representations are becoming more or less brain-like.

## How Cognix differs from related work

| System | Direction | What it captures |
|--------|-----------|-----------------|
| **Sentence-transformers** | text → embedding | Semantic meaning (topic, paraphrase detection) |
| **CLIP** | image/text → shared embedding | Cross-modal semantic alignment |
| **BrainCLIP** | real fMRI → CLIP space | Brain decoding ("what was this person looking at?") |
| **MindEye2** | real fMRI → image | Image reconstruction from brain scans |
| **Cognix** | text → predicted brain response → new embedding | Cognitive processing similarity ("how similarly does the brain handle these?") |

**The simplest way to understand the difference:**

- **BrainCLIP:** Someone is in an fMRI scanner looking at a dog. BrainCLIP reads their brain scan and says "they're looking at a dog." It needs a real brain scan as input.
- **Cognix:** You give it two pieces of text — a legal contract and a math proof. It predicts what the brain would do with each one, and says "these would be processed similarly" even though they're about completely different topics. No brain scanner needed.

## Validation results

Phase 1 validation (100 pairs, 10 categories) confirmed that brain similarity and semantic similarity are **barely correlated** (Pearson r = 0.24, p = 0.017).

![Scatter plot](notebooks/validation_scatter.png)

### Per-category breakdown

| Category | Semantic sim | Brain sim | Divergence |
|----------|-------------|-----------|------------|
| Syntactic surprise | 0.090 | 0.872 | **+0.782** |
| Cognitive load | 0.087 | 0.845 | **+0.758** |
| Spatial scene | 0.329 | 0.944 | **+0.615** |
| Narrative suspense | 0.237 | 0.829 | **+0.592** |
| Theory of mind | 0.260 | 0.824 | **+0.564** |
| Sensorimotor | 0.396 | 0.942 | **+0.547** |
| Emotional arousal | 0.285 | 0.735 | **+0.450** |
| Control (diff processing) | 0.322 | 0.587 | +0.265 |
| Paraphrase | 0.812 | 0.907 | +0.095 |

Every divergence category we predicted works. The control category (same topic, different complexity) has the **lowest** brain similarity — confirming the brain distinguishes processing demands even within the same topic.

## Current status

**Phase 1: PASSED.** Proceeding to Round 2 (1,000 pairs) and full system build.

## Project structure

```
cognix/
  cognix/              # Python package
    tribe_wrapper.py   # TRIBE v2 loading, inference, caching
  notebooks/
    00_validation_experiment.ipynb  # Colab notebook with full results
    validation_scatter.png          # Scatter plot from Phase 1
  data/
    validation_pairs.jsonl          # 100 curated text pairs (10 categories)
    validation_results.jsonl        # Per-pair similarity scores
  artifacts/
    cached_vectors/                 # Cached TRIBE brain vectors (reusable)
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

## Built with

| Provider | What we use | Role in Cognix |
|----------|------------|----------------|
| **Meta** | [TRIBE v2](https://github.com/facebookresearch/tribev2), [LLaMA 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B) | Brain response prediction, text feature extraction |
| **Google** | [Colab](https://colab.research.google.com), [gTTS](https://pypi.org/project/gTTS/) | GPU compute, text-to-speech for temporal alignment |
| **OpenAI** | [Whisper](https://github.com/openai/whisper) (via [WhisperX](https://github.com/m-bain/whisperX)) | Word-level timestamp extraction |
| **Hugging Face** | [Transformers](https://huggingface.co/docs/transformers), [sentence-transformers](https://www.sbert.net/) | Model hosting, semantic similarity baseline |
| **PyTorch** | [torch](https://pytorch.org/) | ML framework |

## License

This project depends on TRIBE v2, which is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Cognix inherits that constraint: **non-commercial use only**.
