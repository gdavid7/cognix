# Cognix

Brain-grounded embedding system using Meta's TRIBE v2. Similarity = "how similarly the brain processes two inputs" rather than standard semantic similarity.

## Pipeline

```
text -> TRIBE v2 (gTTS -> WhisperX -> LLaMA 3.2 -> 8-layer Transformer -> brain projection)
     -> predicted brain tensor (T, 20484) at 1 Hz
     -> temporal mean pooling -> (20484,)
     -> MLP projection head -> (512,)   [not yet built]
     -> cosine similarity
```

## Key facts

- TRIBE v2 outputs 20,484 cortical vertices (fsaverage5 mesh), left hemi 0-10241, right hemi 10242-20483
- TRIBE v2 needs 40GB+ GPU VRAM (A100). ~38 sec/text, bottleneck is TTS + WhisperX + LLaMA
- L4 (24GB) may work for text-only and costs 3x fewer Colab compute units
- License: CC BY-NC (non-commercial only)
- TRIBE internally uses LLaMA 3.2-3B (requires HuggingFace gated access)
- Short texts (<10 words) often fail or produce trivial time steps — use paragraph-length minimum
- Product name: **Cognix**
- Design doc: `cognitive_similarity_embedding_system_design_v2.md`
- Game plan: `.claude/plans/zazzy-forging-rivest.md`

## Validation results (Round 1, April 2026)

**Pearson r = 0.24, Spearman r = 0.32, p < 0.02. PASSED.**

Brain similarity and semantic similarity are barely correlated. All 7 divergence categories (cognitive load, syntactic surprise, spatial scene, narrative suspense, theory of mind, sensorimotor, emotional arousal) diverge as predicted. Control category (same topic, different complexity) has the lowest brain similarity (0.587), confirming the brain distinguishes processing demands.

### Red flags to address

1. **High baseline brain similarity.** 81% of pairs have brain sim > 0.7. Even unrelated texts average 0.824. Mean-pooled whole-brain vectors are dominated by a shared language-processing baseline. Useful signal is in relative differences above this floor.

2. **r=0.24 interpretation.** Low correlation partly because brain space returns ~0.8 for everything while semantic space uses its full range. The divergence is real but the brain space has compressed dynamic range.

3. **Unrelated pairs aren't low.** Expected both scores low. Semantic is near zero (correct), brain is 0.824 (wrong). Brain space can't cleanly separate "cognitively similar" from "any two texts."

4. **5 pairs per category.** Not statistically robust per category. Round 2 needs 25+ per category.

5. **LLaMA baseline not yet tested.** Need to check if raw LLaMA embeddings show the same pattern — if so, TRIBE's brain mapping adds nothing.

## Current status

**Round 2 next.** Same method, scale to 1,000 pairs. Don't change variables — confirm the signal first. Fingerprinting method experiments come AFTER Round 2 as a separate phase.

## Round 2 plan
- 1,000 pairs, 25+ per divergence category, paragraph-length minimum
- Same pipeline (mean-pooled TRIBE vectors, cosine sim vs. sentence-transformer)
- Add LLaMA embedding baseline and random-pair baseline
- Use L4 GPU to save compute units

## Dev setup

- Local: Mac Air (no GPU) — code, data curation, lightweight models
- GPU: Google Colab Pro (free via UCI student program), A100 or L4
- Workflow: build locally, push to GitHub, run TRIBE inference on Colab, download cached vectors

## Colab setup

```bash
!apt-get update -qq && apt-get install -y -qq ffmpeg
!pip uninstall -y numpy && pip install 'numpy>=1.26.4,<2.1.0'
!git clone https://github.com/facebookresearch/tribev2.git /content/tribev2-repo
!pip install -e '/content/tribev2-repo/.[plotting]'
!pip install torchcodec==0.2.1 sentence-transformers scipy scikit-learn matplotlib tqdm
```

HuggingFace token via Colab secrets (key icon > `HF_TOKEN`). Clone to `/content/tribev2-repo` (not `/content/tribev2`) to avoid import conflicts.

## Future fingerprinting methods to test (after Round 2)

- Mean-centering (subtract corpus-average brain vector)
- MVP z-scoring (z-score across vertices per time point)
- Region-specific pooling (emotional regions, motor cortex, prefrontal, language network)
- Variance/max pooling
- Correlation distance, CKA
- Per-modality baseline vectors for multimodal
