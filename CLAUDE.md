# Cognix

Brain-grounded embedding system using Meta's TRIBE v2. Core question: **does the brain mapping reshape the representation space, or is it just re-encoding LLaMA features?**

Similarity = "how similarly the brain processes two inputs" rather than standard semantic similarity.

## Pipeline

```
text → TRIBE v2 (gTTS → WhisperX → LLaMA 3.2 → 8-layer Transformer → brain projection)
     → predicted brain tensor (T, 20484) at 1 Hz
     → temporal mean pooling → (20484,)
     → cosine similarity (with baseline removal)
```

## Key facts

- TRIBE v2 outputs 20,484 cortical vertices (fsaverage5 mesh), left hemi 0–10241, right hemi 10242–20483
- TRIBE v2 needs 40GB+ GPU VRAM (A100). ~38 sec/text, bottleneck is TTS + WhisperX + LLaMA
- L4 (24GB) may work for text-only and costs 3x fewer Colab compute units
- License: CC BY-NC (non-commercial only)
- TRIBE internally uses LLaMA 3.2-3B (requires HuggingFace gated access)
- Short texts (<10 words) often fail or produce trivial time steps — use paragraph-length minimum
- Product name: **Cognix**
- Design doc: `cognitive_similarity_embedding_system_design_v2.md`

## Validation results (Round 1, April 2026)

**Pearson r = 0.24, Spearman r = 0.32, p < 0.02.**

Brain similarity and semantic similarity are barely correlated. All 7 divergence categories diverged as predicted. Note: Round 1 used short garden-path sentences (syntactic_surprise) and length-mismatched control pairs. Both were fixed for Round 2 — syntactic_surprise replaced with paragraph-length syntactic_complexity, control simple texts extended to match complex text lengths. But:

1. **High baseline.** 81% of pairs have brain sim > 0.7. Mean-pooled whole-brain vectors are dominated by a shared language-processing baseline. Baseline removal (mean-centering) is required before results are interpretable.
2. **LLaMA baseline untested.** We don't yet know if this divergence comes from the brain mapping or from LLaMA 3.2 features that sentence-transformers don't capture. This is the critical experiment.
3. **5 pairs per category.** Not statistically robust. Round 2 scales to 20–25 per category.

## Current status

**Round 2 in progress.** 923 pairs, 1,835 unique texts. Three key experiments:

1. **Scale test:** Does the r ≈ 0.24 divergence hold at 10× more pairs?
2. **LLaMA baseline:** Extract LLaMA 3.2-3B embeddings (last hidden state, mean-pooled) and compare. If brain sim ≈ LLaMA sim, the brain mapping adds nothing.
3. **Baseline removal:** Mean-center brain vectors (subtract corpus mean) before computing similarity. Required to address the 0.82 floor.

## Critical: LLaMA baseline

The analysis notebook (`02_r2_analysis.ipynb`) currently uses `all-mpnet-base-v2` as a proxy for the LLaMA test. **This is wrong** — mpnet is a different model entirely. The real test requires extracting actual LLaMA 3.2-3B embeddings. The inference notebook (`01_r2_tribe_inference.ipynb`) includes a LLaMA embedding extraction step for this purpose.

## Dev setup

- Local: Mac Air (no GPU) — code, data curation, lightweight models
- GPU: Google Colab Pro (free via UCI student program), A100 or L4
- Workflow: build locally, push to GitHub, run TRIBE + LLaMA inference on Colab, download cached vectors

## Colab setup

```bash
!apt-get update -qq && apt-get install -y -qq ffmpeg
!pip uninstall -y numpy && pip install 'numpy>=1.26.4,<2.1.0'
!git clone https://github.com/facebookresearch/tribev2.git /content/tribev2-repo
!pip install -e '/content/tribev2-repo/.[plotting]'
!pip install torchcodec==0.2.1 sentence-transformers scipy scikit-learn matplotlib tqdm
```

HuggingFace token via Colab secrets (key icon > `HF_TOKEN`). Clone to `/content/tribev2-repo` (not `/content/tribev2`) to avoid import conflicts.

## After Round 2 (gated on LLaMA baseline results)

### Phase 3: Region-specific pooling
Mean pooling drowns localized signals (limbic emotion in 500 of 20,484 vertices). Map vertices to brain regions via Desikan-Killiany atlas, pool per region. Test whether region-specific vectors recover categories that mean pooling misses (especially emotional arousal). Runs on cached raw tensors, no GPU needed, seconds on Mac.

### Phase 4: Distilled cognitive embedding model
Train a projection head (MLP) on frozen LLaMA features using TRIBE brain similarities as supervision (contrastive learning). Goal: fast 512-d cognitive embedding that runs in milliseconds without TRIBE. Train on ~1.7M pairs from cached vectors (CPU, minutes). Evaluate on held-out benchmark pairs (AUC: does Cognix separate cognitively similar from different better than raw LLaMA?).

### Phase 5: Applications
Cognitive readability scoring, cognitively-targeted advertising, cross-topic similarity, AI alignment evaluation, multimodal extension, knowledge distillation to standalone small transformer.
