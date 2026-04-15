
# System Design Doc: Cognix (Cognitive Similarity Embeddings)

## 1) Project Summary

Build a **brain-grounded embedding system** that maps text (v1) into a fixed-length vector where similarity reflects **predicted similarity of human brain processing**, rather than standard semantic similarity.

```text
input text -> TRIBE v2 -> predicted brain-response tensor (T x 20,484) -> temporal pooling -> projection head -> embedding (512-d) -> cosine similarity
```

This is **not** a brain-decoding system (fMRI -> content) and **not** a CLIP clone. It builds a **new embedding space derived from predicted brain responses**.

---

## 2) Why this project exists

Standard embeddings capture semantic similarity. They do **not** capture:
- cognitive load (how hard is this to process?)
- affective processing (what emotions does this trigger neurally?)
- sensorimotor grounding (does this evoke physical sensation?)
- "how similarly two inputs are processed by the human brain"

TRIBE v2 (Meta, March 2026) makes this feasible. It predicts high-resolution fMRI brain responses (~20,484 cortical vertices) from video, audio, and text, trained on 1,000+ hours of fMRI data from 720+ subjects.

---

## 3) Links

### TRIBE v2
- Blog: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
- Paper: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- Code: https://github.com/facebookresearch/tribev2

### Related work
- CLIP: https://arxiv.org/abs/2103.00020
- BrainCLIP: https://arxiv.org/abs/2302.12971
- MindEye2: https://arxiv.org/abs/2403.11207
- RSA (Kriegeskorte et al. 2008): https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full

---

## 4) How TRIBE v2 works inside

### Architecture

```text
Stage 1 (feature extraction):
  text  -> LLaMA 3.2-3B      -> language features
  video -> V-JEPA2-Giant      -> visual features
  audio -> Wav2Vec-BERT 2.0   -> audio features

Stage 2 (fusion):
  All features compressed to D=384, concatenated -> D=1152
  -> 8-layer Transformer encoder (100-second context windows)

Stage 3 (brain prediction):
  Transformer outputs -> decimated to fMRI frequency
  -> Subject Block -> predicted cortical activity (~20,484 vertices)
```

### Text tokenization (important quirk)

TRIBE does not process text like a normal NLP model:

```text
text -> gTTS (text-to-speech) -> audio -> WhisperX -> word-level timestamps
timestamps + LLaMA 3.2-3B -> features at 2 Hz -> brain transformer -> voxel predictions
```

The TTS roundtrip exists because TRIBE was trained on naturalistic stimuli with inherent temporal structure. Text needs artificial temporal structure via TTS to fit this framework.

**Implication:** Short texts (under ~10 words) produce very few time steps (T=1-3). Paragraph-length inputs work better.

### Output

```python
preds, segments = model.predict(events=events_df)
brain_output = np.asarray(preds)
# shape: (T, 20484)
# T = time steps at 1 Hz (one per second of TTS audio)
# 20,484 = fsaverage5 cortical mesh
#   Left hemisphere:  indices 0-10,241
#   Right hemisphere: indices 10,242-20,483
```

### What this means for Cognix

TRIBE feeds text into LLaMA 3.2, extracts its internal representations, then maps them to brain activity via a learned nonlinear transformation supervised by real fMRI data. The brain prediction is therefore a **transformation of LLaMA features** — it cannot capture cognitive properties LLaMA doesn't encode. See Section 8 (risks).

---

## 5) How Cognix differs from related work

| System | Direction | What it captures |
|--------|-----------|-----------------|
| Sentence-transformers | text -> embedding | Semantic meaning |
| CLIP | image/text -> shared space | Cross-modal semantic alignment |
| BrainCLIP | real fMRI -> CLIP space | Brain decoding ("what were they looking at?") |
| MindEye2 | real fMRI -> image | Image reconstruction from brain scans |
| **Cognix** | text -> predicted brain response -> new space | Cognitive processing similarity |

BrainCLIP/MindEye2 go **brain -> content** (decoding). Cognix goes **content -> brain** (encoding) and builds a new similarity space. No fMRI scanner needed at inference.

---

## 6) Validation experiment — COMPLETED

### Round 1 results (100 pairs, 10 categories, April 2026)

**Pearson r = 0.24 (p = 0.017). Spearman r = 0.32 (p = 0.0015).**

Brain similarity and semantic similarity are barely correlated.

### Per-category results

| Category | N | Semantic sim | Brain sim | Divergence |
|----------|---|-------------|-----------|------------|
| Syntactic surprise | 5 | 0.090 | 0.872 | **+0.782** |
| Cognitive load | 5 | 0.087 | 0.845 | **+0.758** |
| Spatial scene | 5 | 0.329 | 0.944 | **+0.615** |
| Narrative suspense | 5 | 0.237 | 0.829 | **+0.592** |
| Theory of mind | 5 | 0.260 | 0.824 | **+0.564** |
| Sensorimotor | 5 | 0.396 | 0.942 | **+0.547** |
| Emotional arousal | 5 | 0.285 | 0.735 | **+0.450** |
| Control (diff processing) | 15 | 0.322 | 0.587 | +0.265 |
| Paraphrase | 24 | 0.812 | 0.907 | +0.095 |
| Unrelated | 23 | 0.017 | 0.824 | +0.807 |

### What went right

1. **Every divergence category diverges as predicted.** All 7 categories (cognitive load, syntactic surprise, spatial, sensorimotor, emotion, suspense, theory of mind) show high brain similarity with low semantic similarity.

2. **Control category has the lowest brain similarity (0.587).** Same topic, different complexity (e.g., "Water boils when you heat it" vs. a dense paragraph about nucleation). The brain space genuinely distinguishes processing demands within the same topic. This is the strongest single piece of evidence.

3. **Sensorimotor and spatial categories are very consistent.** Std of 0.025 and 0.018 — tight, repeatable signal.

### Red flags from Round 1

1. **High baseline brain similarity (BIGGEST CONCERN).** 81% of all pairs have brain sim > 0.7. Even completely unrelated texts average 0.824. The mean-pooled vectors are dominated by a shared "the brain is processing language" baseline. The useful signal is in the relative differences above this high floor — spatial scenes at 0.944 vs. controls at 0.587 — but the dynamic range is compressed.

2. **The r=0.24 needs careful interpretation.** Low correlation could mean "brain space captures something genuinely different" (good) OR "brain space doesn't discriminate well and returns ~0.8 for everything while semantic space actually differentiates" (bad). The truth is a mix of both. The semantic space uses its full range (-0.07 to 0.93). The brain space crams 81% of values into the 0.7-1.0 range.

3. **Unrelated pairs should be low in brain space — they're not.** We expected both scores to be low for unrelated pairs. Semantic sim is near zero (correct), but brain sim is 0.824. This means the brain space can't cleanly distinguish "cognitively similar" from "just any two texts." The divergence categories score higher than unrelated, but the margin is thinner than the raw numbers suggest.

4. **Only 5 pairs per divergence category.** Not enough for per-category statistical significance. The overall pattern holds, but any single category result could be noise.

5. **3 texts failed TRIBE inference.** All short sentences. TRIBE's TTS pipeline can't handle very short inputs reliably.

### Decision

**Proceed to Round 2 (1,000 pairs).** Scale up the same method — don't change variables. Confirm the signal is real at scale, THEN experiment with fingerprinting methods.

---

## 7) Concrete divergence examples

These examples were validated in Round 1. All show the predicted pattern: low semantic similarity, high brain similarity.

### Cognitive load (sem=0.087, brain=0.845)
Dense legal text vs. dense mathematical proof — different topics, same heavy working memory demand.

### Emotional arousal (sem=0.285, brain=0.735)
War scene vs. medical diagnosis — different content, same amygdala/insula activation pattern.

### Sensorimotor (sem=0.396, brain=0.942)
Kicking a ball vs. stomping a brake — different scenarios, same motor cortex activation for leg/foot.

### Spatial scene (sem=0.329, brain=0.944)
Vast desert vs. open ocean — different domains, same spatial scene processing.

### Syntactic surprise (sem=0.090, brain=0.872)
Garden-path sentences from different topics — same parse-fail-reparse brain pattern.

### Narrative suspense (sem=0.237, brain=0.829)
Thriller vs. medical emergency — different genres, same anticipatory processing trajectory.

### Theory of mind (sem=0.260, brain=0.824)
Reading deception vs. child hiding candy — different contexts, same mentalizing network activation.

---

## 8) Risks (updated post-validation)

### 1. High baseline brain similarity (HIGH — confirmed in Round 1)
Mean-pooled whole-brain vectors are dominated by shared language-processing activation. This compresses the useful dynamic range and makes it hard to distinguish "cognitively similar" from "any two texts."

**Potential mitigations (to test after Round 2):**
- Mean-centering: subtract the average brain vector across a corpus
- MVP z-scoring: z-score across vertices per time point to emphasize relative patterns
- Region-specific pooling: use only relevant brain regions instead of all 20,484 vertices
- Variance pooling: use activation variance instead of mean
- For multimodal: separate baseline vectors per modality (text, audio, video)

### 2. The LLaMA ceiling (MEDIUM — not yet tested)
TRIBE predicts brain responses from LLaMA 3.2 features. If raw LLaMA embeddings produce the same divergence pattern as brain vectors, then TRIBE's brain mapping adds nothing. Need to test LLaMA embeddings directly as a baseline.

### 3. Small sample sizes (MEDIUM — Round 2 addresses this)
5 pairs per divergence category is insufficient. Need 25+ per category for statistical robustness.

### 4. Short text failures (LOW — fixable)
3/175 texts failed TRIBE inference. Use paragraph-length inputs minimum.

### 5. Collapse to generic semantics (LOW — disconfirmed by r=0.24)
The original primary risk. Validation shows brain space IS different from semantic space. Downgraded.

---

## 9) Pipeline

### Step 1: Text input
Paragraph-length passages minimum. Short sentences produce too few TRIBE time steps.

### Step 2: TRIBE v2 inference
```python
events_df = model.get_events_dataframe(text_path="input.txt")
preds, segments = model.predict(events=events_df)
brain_output = np.asarray(preds)  # (T, 20484)
```

### Step 3: Temporal pooling
```python
pooled = brain_output.mean(axis=0)  # (20484,)
```

**Scientific precedent:** Standard in neuroimaging (Huth et al. 2016, MVPA, Brain-Score). Loses temporal dynamics but produces fixed-size vectors.

### Step 4: Projection head (post-Round 2)
```python
projection_head = nn.Sequential(
    nn.Linear(20484, 1024),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, 512),
)
embedding = projection_head(pooled)  # (512,)
```

### Step 5: Contrastive training (post-Round 2)
Train the projection head on cached brain vectors with contrastive loss. Positive pairs (paraphrases) -> close. Negative pairs (unrelated) -> far.

### Caching strategy
TRIBE inference is slow (~38 sec/text on A100). Cache pooled vectors to disk. Train the projection head entirely on cached outputs.

---

## 10) Round 2 plan

**Goal:** Confirm Round 1 signal at scale. Same method, more data.

- 1,000 pairs, 25+ per divergence category
- Longer texts (paragraph-length minimum)
- Same pipeline: mean-pooled TRIBE vectors, cosine similarity, compare to sentence-transformer
- Use L4 GPU instead of A100 to save compute units (~2-3 units/hr vs ~7-8)
- Add LLaMA embedding baseline (tests whether brain mapping adds value beyond LLaMA itself)
- Add random-pair baseline (expected cosine similarity between arbitrary brain vectors — quantifies the high baseline)

---

## 11) Alternative fingerprinting methods (to explore AFTER Round 2)

These are established methods from neuroimaging for creating brain fingerprints. Each captures different aspects of the brain response. Test after Round 2 confirms the signal at scale.

**Pooling variants:**
- Mean-centering (subtract corpus-average vector to remove shared baseline)
- MVP z-scoring (z-score across vertices per time point — emphasizes relative activation patterns)
- Max pooling (keep peak activations)
- Variance pooling (how much does activation fluctuate — captures processing difficulty)

**Region-specific analysis:**
- Emotional regions only (amygdala-adjacent, insula)
- Motor cortex only
- Prefrontal cortex only (working memory, executive function)
- Language network only (left temporal/frontal)

**Alternative similarity metrics:**
- Correlation distance (mean-centered cosine — may fix the high baseline issue)
- CKA (Centered Kernel Alignment — compares representational geometry)

**More advanced methods:**
- Beta-series extraction (fit a GLM per stimulus instead of raw averaging)
- t-statistic maps (where did the brain respond significantly, not just how much)
- Connectivity fingerprinting (Finn et al. 2015 — inter-region correlation patterns during processing)
- Voxel reliability weighting (weight vertices by how reliably TRIBE predicts them)

---

## 12) Compute

- TRIBE v2 needs 40GB+ VRAM for text (LLaMA 3.2 + brain transformer)
- ~38 seconds per text on A100, bottleneck is TTS + WhisperX + LLaMA, not GPU compute
- L4 (24GB) may work for text-only and costs 3x fewer Colab compute units
- Cache brain vectors aggressively — 175 texts took ~2 hours on A100

---

## 13) HuggingFace release (post-training)

**Upload:** projection model weights, inference pipeline, config, precomputed demo embeddings, model card with divergence examples.

**Do not upload:** TRIBE v2 weights (users install separately per Meta's license).

```python
from cognix import CognitiveEmbedder
model = CognitiveEmbedder.from_pretrained("davidgershony/cognix-v1")
embedding = model.encode("The desert stretched flat and empty to the horizon.")
```

---

## 14) Licensing

TRIBE v2: **CC BY-NC 4.0**. Non-commercial use only. Cognix inherits this constraint. README must state dependency on TRIBE v2 with attribution to Meta.

---

## 15) Limitations

- TRIBE predicts coarse fMRI-scale responses (~20,484 vertices), not individual neurons
- Predictions are population-level (averaged across subjects), not personalized
- Brain predictions are a transformation of LLaMA features — cannot capture properties LLaMA doesn't encode
- Mean-pooled vectors have high baseline similarity (~0.82), compressing useful dynamic range
- Not a direct measure of reward, happiness, or preference
- TRIBE inference is slow (~38 sec/text) and requires GPU
- Non-commercial license only
