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

Examples of what brain similarity captures that semantics misses (brain rates pairs as *more* similar than their topics suggest):

| Text A | Text B | Semantic sim | Brain sim |
|--------|--------|-------------|-----------|
| Dense legal clause | Dense math proof | 0.05 | 0.83 |
| Soldier watching a friend die | Receiving a terminal diagnosis | 0.30 | 0.75 |
| Kicking a ball across wet grass | Stomping on a brake pedal | 0.43 | 0.91 |
| Vast empty desert, no movement | Open ocean, no land in sight | 0.38 | 0.95 |
| Simple explanation of boiling | Technical nucleation physics | 0.39 | 0.78 |

And examples where brain similarity goes the other way — the geometry isn't a one-way inflation:

| Text A | Text B | Semantic sim | LLaMA sim | Brain sim |
|--------|--------|-------------|-----------|-----------|
| Plain-language description of dogs as loyal pets | Technical paper on *Canis lupus* domestication via mitochondrial DNA divergence | 0.15 | 0.63 | 0.65 |
| Mother screams as stroller rolls toward subway tracks | Widower visits late wife's grave every Sunday for eleven years | 0.26 | 0.80 | 0.39 |
| Gillespie sent a letter to CBS President Leslie Moonves asking for a historical review or a disclaimer | Republican National Committee Chairman Ed Gillespie issued a letter Friday to CBS Television President Leslie Moonves | 0.80 | 0.82 | 0.65 |

The first row is the "control" pattern — same topic, very different processing demands. Brain rates them moderately similar (same domain) but well below the ~0.85 typical brain score, showing the metric distinguishes processing complexity within a topic. The second is the emotional-arousal anomaly: two grief-themed texts that LLaMA rates as very similar, but brain rates *less* similar than chance — evidence that mean pooling drowns the limbic signal in 20,484 vertices, and the main reason region-specific pooling is on the roadmap. The third shows brain disagreeing downward on near-paraphrases.

Note that mean-pooled brain similarity has a high baseline (~0.81 for random pairs) — most of the "low" examples are still well above 0; it's the *direction relative to LLaMA and semantic* that matters. Mean-centering compresses the random-pair floor to ~0.26, which is the geometry the distilled model will be trained against.

All per-category divergences are statistically significant (p < 0.001). Permutation test p=0.0000.

## Next steps

1. **Downstream validation** *(next)* — Round 2 proved Cognix diverges from LLaMA, but not that the divergence is useful. Test whether mean-pooled brain vectors predict per-word reading times on an eye-tracking corpus (Provo / Dundee / GECO) better than LLaMA and sentence-transformer baselines. Binary go/no-go for everything below.
2. **Region-specific pooling** *(if validation passes)* — pool within specific brain regions (prefrontal for cognitive load, limbic for emotion, motor cortex for sensorimotor) to recover signals that whole-brain averaging drowns out, and decide whether the distilled model outputs a structured per-region embedding or a single vector.
3. **Pooling method ablation** — compare mean, variance, max, and z-scored pooling (whole-brain and per-region) to pick the best input representation for the distilled model.
4. **Distilled embedding model** — train a small, fast model that approximates TRIBE's cognitive similarity in milliseconds without a 40GB GPU.

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

**Round 2 complete. Phase 3 (downstream validation on eye-tracking data) is next.** Region pooling is deferred until validation confirms Cognix has signal worth structuring. See the [design doc](cognitive_similarity_embedding_system_design_v2.md) for full details.

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
