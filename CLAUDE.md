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

**Round 2 complete. Phase 3 (downstream validation on human behavioral data) is next.**

Region-specific pooling is deferred until downstream validation proves Cognix has signal worth structuring. Round 2 answered "does brain ≠ LLaMA?" (yes, r=0.44) but not "is the divergence useful?" Region pooling shapes architecture; it doesn't prove value.

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

## Phase 3: Downstream validation (next)

Round 2 proved divergence (r=0.44 brain vs LLaMA). It did not prove the divergence is *useful*. Without an external signal to ground against, every downstream phase is built on a circular foundation: handcrafted pairs designed to diverge confirm divergence. The smallest possible test of usefulness comes first.

### Approach

1. **Pick an eye-tracking corpus** (Provo, Dundee, or GECO). Each gives per-word fixation durations on naturalistic text — a noisy but real measure of cognitive processing demand.

2. **Define the test.** For a held-out passage, predict per-region reading time (or surprisal-residualized reading time) from text features. Compare three feature sources:
   - sentence-transformers embedding (semantic baseline)
   - LLaMA 3.2-3B last hidden state (LLM baseline)
   - TRIBE brain vector, mean-pooled (Cognix)

3. **Evaluation.** R² (or AUC for above-/below-median fixations) on held-out subjects. The question is binary: does Cognix beat *both* baselines, or not?

4. **Decision gate.**
   - If Cognix wins → structural work (region pooling, distilled model) is justified.
   - If Cognix loses → the divergence is real but not behaviorally useful. Re-examine before building further.

Uses cached pooled vectors and a few hundred MB of public eye-tracking data. CPU only. Notebook: TBD.

## Phase 4: Region-specific pooling (deferred)

Only run after Phase 3 validates. Mean pooling drowns localized signals (emotional arousal Brain−LLaMA = −0.070 is the clearest case). If Phase 3 shows Cognix has downstream value, region pooling becomes the architectural choice for Phase 5: structured per-region embedding vs single vector.

When run, must include: per-region mean-centering (Round 2 showed this is essential), discrimination AUC vs random_baseline (not just argmax), size-matched random-vertex controls, and a label-shuffle null. Note that the amygdala — central to emotional arousal — is subcortical and not on the fsaverage5 surface.

## Phase 5: Distilled cognitive embedding model

Train a projection head (MLP) on frozen LLaMA features using TRIBE brain similarities as supervision (contrastive learning). Architecture depends on Phase 4:
- If region pooling works: per-region heads → structured embedding
- If not: single MLP → 512-d cognitive embedding

Train on ~1.7M pairs from cached vectors (CPU, minutes). Re-evaluate on the Phase 3 downstream task.

## Phase 6: Applications
Cognitive readability scoring, cross-topic similarity, AI alignment evaluation, multimodal extension, knowledge distillation to standalone small transformer.
