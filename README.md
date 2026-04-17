# Cognix

**Can predicted brain responses tell us something about text that standard embeddings miss?**

Standard text embeddings capture what content is *about* — its semantic meaning. They don't capture how the brain *processes* it: how cognitively demanding it is, what emotions it triggers, whether it activates motor circuits. Two texts about completely different topics can engage the brain in remarkably similar ways.

Cognix uses [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2) — a model that predicts fMRI brain responses from text — to explore whether predicted brain activity can serve as a useful representation space for measuring **cognitive similarity**.

## Results

**The brain mapping reshapes the similarity geometry beyond what LLaMA encodes.**

Round 2 (923 pairs, 1,835 texts) compared three similarity metrics per pair:

| Comparison | Pearson r |
|---|---|
| Brain vs Semantic | 0.43 |
| Brain vs LLaMA 3.2-3B | 0.44 |
| LLaMA vs Semantic | 0.83 |

LLaMA and sentence-transformers largely agree with each other (r=0.83), but brain vectors diverge from both (r≈0.43). The brain mapping isn't re-encoding LLaMA — it captures something different.

Examples of what brain similarity captures that semantics misses:

| Text A | Text B | Semantic sim | Brain sim |
|--------|--------|-------------|-----------|
| Dense legal clause | Dense math proof | 0.05 | 0.83 |
| Soldier watching a friend die | Receiving a terminal diagnosis | 0.30 | 0.75 |
| Kicking a ball across wet grass | Stomping on a brake pedal | 0.43 | 0.91 |
| Vast empty desert, no movement | Open ocean, no land in sight | 0.38 | 0.95 |
| Simple explanation of boiling | Technical nucleation physics | 0.39 | 0.78 |

All per-category divergences are statistically significant (p < 0.001). Permutation test p=0.0000.

## Next steps

1. **Region-specific pooling** — instead of averaging all 20,484 brain vertices, pool within specific brain regions (prefrontal for cognitive load, limbic for emotion, motor cortex for sensorimotor). Should recover signals that whole-brain averaging drowns out (especially emotional arousal, which underperforms with mean pooling). Determines whether the distilled model outputs a structured embedding (per-region scores) or a single vector.
2. **Pooling method ablation** — compare mean, variance, max, and z-scored pooling (both whole-brain and per-region). Selects the best input representation for the distilled model.
3. **Distilled embedding model** — train a small, fast model that approximates TRIBE's cognitive similarity in milliseconds without needing a 40GB GPU.
4. **Downstream evaluation** — validate on human behavioral data (reading times, eye-tracking) to prove the signal is useful, not just different.

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

**Round 2 complete. Phase 3 (region-specific pooling) in progress.** See the [design doc](cognitive_similarity_embedding_system_design_v2.md) for full details.

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
