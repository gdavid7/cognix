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

## Round 2 results (April 2026)

923 pairs, 1,835 unique texts. All three experiments completed.

### LLaMA baseline: PASSED

| Comparison | Pearson r |
|---|---|
| Brain vs Semantic | 0.43 |
| Brain vs LLaMA | 0.44 |
| LLaMA vs Semantic | 0.83 |

LLaMA and sentence-transformers largely agree (r=0.83). Brain vectors diverge from both (r≈0.43). **The brain mapping reshapes the similarity geometry beyond what LLaMA encodes.**

### Per-category results (handcrafted pairs, brain vs LLaMA divergence)

| Category | N | Sem | LLaMA | Brain | Brain−LLaMA |
|---|---|---|---|---|---|
| cognitive_load | 25 | 0.050 | 0.509 | 0.833 | **+0.324** |
| spatial_scene | 20 | 0.383 | 0.870 | 0.949 | +0.079 |
| sensorimotor | 20 | 0.428 | 0.861 | 0.905 | +0.044 |
| syntactic_complexity | 20 | 0.201 | 0.870 | 0.909 | +0.039 |
| narrative_suspense | 20 | 0.250 | 0.794 | 0.824 | +0.030 |
| theory_of_mind | 20 | 0.253 | 0.831 | 0.838 | +0.007 |
| emotional_arousal | 20 | 0.297 | 0.819 | 0.749 | **−0.070** |

Emotional arousal going negative supports the hypothesis that mean pooling drowns limbic signal — Phase 3 region pooling should recover this.

### Baseline removal

Mean-centering drops random-pair brain sim from 0.812 to 0.264. Centered brain vs semantic: r=0.55.

### Robustness checks

- Text length correlation: r=−0.31 (not a confound)
- Permutation test: p=0.0000 (divergence is significant)
- All per-category brain vs semantic t-tests: p < 0.001

## Current status

**Round 2 complete. Phase 3 (region-specific pooling) in progress.**

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
