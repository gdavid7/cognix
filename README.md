# Cognix

**Can predicted brain responses tell us something about text that standard embeddings miss?**

Standard text embeddings capture what content is *about* — its semantic meaning. They don't capture how the brain *processes* it: how cognitively demanding it is, what emotions it triggers, whether it activates motor circuits. Two texts about completely different topics can engage the brain in remarkably similar ways.

Cognix uses [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2) — a model that predicts fMRI brain responses from text — to explore whether predicted brain activity can serve as a useful representation space for measuring **cognitive similarity**.

## The question

TRIBE v2 predicts ~20,484 cortical vertices of brain activity from text. If we mean-pool those predictions into a vector and compute cosine similarity between two texts, do we get something meaningfully different from semantic similarity?

Early results suggest yes:

| Text A | Text B | Semantic sim | Brain sim |
|--------|--------|-------------|-----------|
| Dense legal clause | Dense math proof | 0.09 | 0.85 |
| Soldier watching a friend die | Receiving a terminal diagnosis | 0.29 | 0.74 |
| Kicking a ball across wet grass | Stomping on a brake pedal | 0.40 | 0.94 |
| Vast empty desert, no movement | Open ocean, no land in sight | 0.33 | 0.94 |
| Center-embedded sentence | Garden-path sentence | 0.09 | 0.87 |

Round 1 (100 pairs): Pearson r = 0.24 between brain and semantic similarity. They're measuring different things.

## What we don't know yet

This is research. Several things could undermine the finding:

1. **Does the brain mapping actually add value?** TRIBE uses LLaMA 3.2-3B internally. If LLaMA's own embeddings show the same divergence, then the brain mapping is just re-encoding LLaMA features in a noisier space. This is the critical experiment — Round 2 includes a direct LLaMA 3.2 embedding comparison.

2. **Is the high baseline a problem?** Even unrelated texts average 0.82 brain similarity. The useful signal lives in a narrow range above that floor. Baseline removal (mean-centering) may resolve this, but it hasn't been tested yet.

3. **Does this hold at scale?** Round 1 had only 5 pairs per divergence category. Round 2 scales to 923 pairs with 20–25 per category.

## If it holds up

If the brain mapping genuinely reshapes the similarity geometry beyond what LLaMA already captures, the next steps are:

1. **Region-specific pooling** — instead of averaging all 20,484 brain vertices, pool within specific brain regions (prefrontal for cognitive load, limbic for emotion, motor cortex for sensorimotor). This should recover signals that whole-brain averaging drowns out.
2. **Distilled embedding model** — train a small, fast model that approximates TRIBE's cognitive similarity in milliseconds without needing a 40GB GPU. Contrastive learning on the cached brain vectors, producing a 512-d cognitive embedding.
3. **Applications** — cognitive readability scoring, cognitively-targeted advertising, cross-topic similarity based on processing demands, AI alignment evaluation, multimodal extension via TRIBE's video/audio pathways.

Each step is gated on the previous one succeeding. These are research directions, not product promises.

## Pipeline

```
text → TRIBE v2 → brain tensor (T, 20484) → mean pooling → (20484,) → cosine similarity
```

TRIBE v2 internally: text → gTTS → WhisperX → LLaMA 3.2-3B → 8-layer Transformer → brain projection

## How Cognix differs from related work

| System | Direction | What it captures |
|--------|-----------|-----------------|
| Sentence-transformers | text → embedding | Semantic meaning |
| BrainCLIP / MindEye2 | real fMRI → content | Brain decoding ("what were they looking at?") |
| **Cognix** | text → predicted brain response | Cognitive processing characteristics |

BrainCLIP and MindEye2 go **brain → content** (decoding what someone perceived). Cognix goes **content → brain** (predicting how the brain would process it). No scanner needed.

## Current status

**Round 2 in progress.** 923 pairs, three experiments:
1. Scale: does r ≈ 0.24 hold at 10× more pairs?
2. LLaMA baseline: does brain sim ≠ LLaMA sim? (the make-or-break test)
3. Baseline removal: does mean-centering fix the 0.82 floor?

See the [design doc](cognitive_similarity_embedding_system_design_v2.md) for full details.

## Running the experiments

Requires Google Colab with A100 GPU.

1. Add your [HuggingFace token](https://huggingface.co/settings/tokens) as a Colab secret named `HF_TOKEN` (requires access to [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B))
2. Open `notebooks/01_r2_tribe_inference.ipynb` in Colab
3. Set runtime to **A100 GPU + High RAM**
4. Run all cells (~19 hours for full dataset, resumable)

## Built with

| Provider | What | Role |
|----------|------|------|
| **Meta** | [TRIBE v2](https://github.com/facebookresearch/tribev2), [LLaMA 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B) | Brain response prediction, text features |
| **Google** | [Colab](https://colab.research.google.com), [gTTS](https://pypi.org/project/gTTS/) | GPU compute, text-to-speech |
| **OpenAI** | [Whisper](https://github.com/openai/whisper) (via [WhisperX](https://github.com/m-bain/whisperX)) | Timestamp extraction |
| **Hugging Face** | [Transformers](https://huggingface.co/docs/transformers), [sentence-transformers](https://www.sbert.net/) | Model hosting, semantic baseline |

## License

Depends on TRIBE v2 ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)). **Non-commercial use only.**
