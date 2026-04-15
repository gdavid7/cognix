
# System Design Doc: Cognix (Cognitive Similarity Embeddings)

## 1) Project Summary

Build a **cognitive measurement layer** that maps any content into interpretable brain-derived scores across multiple cognitive dimensions.

Not a single embedding. A multi-axis cognitive profile:

```json
{
  "cognitive_load": 0.82,
  "emotional_intensity": 0.34,
  "motor_engagement": 0.91,
  "spatial_processing": 0.15,
  "overall_brain_fingerprint": [512-d vector]
}
```

Built on [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2), which predicts ~20,484 cortical vertices of fMRI brain response from text, audio, and video. The vertices map to a standard brain atlas (fsaverage5), so each vertex belongs to a known brain region. We exploit this to extract region-specific cognitive axes, not just a single opaque embedding.

This is **not** a brain-decoding system (fMRI -> content), **not** a CLIP clone, and **not** "just a better sentence-transformer." It measures something fundamentally different: **how the brain processes content**, not what the content means.

---

## 2) Why this project exists

Standard embeddings tell you what content is **about**. They do not tell you:
- How cognitively demanding it is to process
- What emotions it triggers neurally
- Whether it evokes physical/motor sensation
- How similarly the brain would handle two different pieces of content

TRIBE v2 (Meta, March 2026) predicts high-resolution fMRI brain responses from video, audio, and text, trained on 1,000+ hours of real fMRI data from 720+ subjects. Because its output maps to a standard brain atlas, we can decompose the prediction into region-specific signals — not just one vector, but separate measurements for cognitive load, emotion, motor engagement, spatial processing, and more.

---

## 3) Roadmap

### Completed
- **Round 1 validation (100 pairs):** Pearson r=0.24, p=0.017. Brain similarity and semantic similarity are barely correlated. All 7 divergence categories work as predicted. See Section 6 for full results.

### Phase 2: Confirm at scale (CURRENT)
- 1,000 pairs, 25+ per divergence category, paragraph-length minimum
- Same method as Round 1 (mean-pooled whole-brain TRIBE vectors, cosine sim vs. sentence-transformer)
- Add LLaMA embedding baseline (tests whether TRIBE's brain mapping adds value beyond LLaMA itself)
- Add random-pair baseline (quantifies the high baseline brain similarity)
- Cache **raw tensors** (T, 20484) to Google Drive — needed for all future experiments
- Use A100 GPU, overnight run

### Phase 3: Region-based decomposition
This is the key step that transforms Cognix from "a single embedding" into "a cognitive measurement layer."

TRIBE's 20,484 vertices sit on the fsaverage5 cortical mesh. Standard brain atlases (e.g., Desikan-Killiany, Schaefer) assign every vertex to a named brain region. By indexing into these regions:

```python
prefrontal_indices = atlas.get_vertices("prefrontal_cortex")
limbic_indices = atlas.get_vertices("amygdala", "insula", "anterior_cingulate")
motor_indices = atlas.get_vertices("precentral_gyrus", "supplementary_motor")
spatial_indices = atlas.get_vertices("parahippocampal", "retrosplenial")

cognitive_load = pooled[prefrontal_indices].mean()
emotion = pooled[limbic_indices].mean()
motor = pooled[motor_indices].mean()
spatial = pooled[spatial_indices].mean()
```

Each text gets a multi-axis cognitive profile. Similarity becomes task-specific: compare texts by emotional profile, by cognitive demand, or by the full brain fingerprint.

**Validation:** Check that the "emotion axis" actually scores higher on emotional arousal pairs than on cognitive load pairs. Check that the "motor axis" scores higher on sensorimotor pairs. If the axes align with the pair categories, the decomposition works.

**Assumption to verify:** TRIBE's vertex predictions are spatially meaningful at the region level. Plausible (it was trained on real fMRI mapped to fsaverage5) but needs testing.

### Phase 4: Fingerprinting experiments
Test alternative methods on the cached raw tensors (no additional GPU needed):
- Mean-centering, z-scoring, max pooling, variance pooling
- Per-region similarity instead of whole-brain similarity
- Correlation distance, CKA, RSA
- See Section 11 for full list with references

### Phase 5: Build and train the model
- Train projection head(s) on cached brain vectors with contrastive loss
- Options: single 512-d embedding, OR separate per-axis projections, OR multi-task head
- Evaluate: retrieval metrics, clustering, per-axis interpretability
- Package and release on HuggingFace

### Phase 6: Scale and extend
- Knowledge distillation: train a small CPU-friendly student model that predicts the cognitive profile directly from text, removing the GPU/TRIBE requirement at inference
- Multimodal support (video, audio) with per-modality baseline vectors — TRIBE already supports these modalities, so cross-modal comparison in brain space becomes possible
- Cross-modal retrieval: "find a podcast that engages the brain like this YouTube video"

### End state

```python
from cognix import CognitiveProfiler

profiler = CognitiveProfiler.from_pretrained("davidgershony/cognix-v1")

profile = profiler.analyze("The desert stretched flat and empty to the horizon.")
# {
#   "cognitive_load": 0.23,
#   "emotional_intensity": 0.12,
#   "motor_engagement": 0.05,
#   "spatial_processing": 0.89,
#   "narrative_suspense": 0.08,
#   "embedding": [512-d vector]
# }

# Compare two texts along a specific axis
sim = profiler.compare(text_a, text_b, axis="emotion")

# Or compare full brain fingerprints
sim = profiler.compare(text_a, text_b, axis="all")
```

---

## 4) Links

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

## 5) How TRIBE v2 works inside

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

1. **Every divergence category diverges as predicted.** All 7 categories show high brain similarity with low semantic similarity.
2. **Control category has the lowest brain similarity (0.587).** The brain genuinely distinguishes processing demands within the same topic.
3. **Sensorimotor and spatial categories are very consistent.** Std of 0.025 and 0.018 — tight, repeatable signal.

### Red flags from Round 1

1. **High baseline brain similarity (BIGGEST CONCERN).** 81% of pairs have brain sim > 0.7. Even unrelated texts average 0.824. Mean-pooled vectors are dominated by a shared "processing language" baseline. Useful signal is in relative differences above this floor.
2. **The r=0.24 needs careful interpretation.** Partly "brain captures something different" (good), partly "brain returns ~0.8 for everything" (bad). Semantic space uses full range (-0.07 to 0.93). Brain space crams 81% into 0.7-1.0.
3. **Unrelated pairs should be low in brain space — they're not.** Semantic sim near zero (correct), brain sim 0.824 (wrong). Brain space can't cleanly separate "cognitively similar" from "any two texts."
4. **Only 5 pairs per divergence category.** Not statistically robust per category. Round 2 needs 25+.
5. **3 texts failed TRIBE inference.** All short sentences.

---

## 7) Concrete divergence examples (validated in Round 1)

### Cognitive load (sem=0.087, brain=0.845)

| Text A | Text B |
|--------|--------|
| "The party of the first part shall indemnify and hold harmless the party of the second part against any and all claims, damages, losses, costs, and expenses arising out of or in connection with any breach of this agreement." | "Given that the eigenvalues of a Hermitian matrix are real, and that eigenvectors corresponding to distinct eigenvalues are orthogonal, we can construct an orthonormal basis for the space by applying the Gram-Schmidt procedure to each eigenspace." |

Different topics (law vs. math), same heavy working memory demand and sustained prefrontal activation.

### Emotional arousal (sem=0.285, brain=0.735)

| Text A | Text B |
|--------|--------|
| "The soldier held his dying friend in his arms and screamed for a medic, but no one came. The blood soaked through his uniform as the light faded from his friend's eyes." | "She sat in the doctor's office and heard the words she had been dreading. Stage four. Inoperable. She drove home in silence and sat in the driveway for an hour before she could walk inside." |

Different content (war vs. medical), same amygdala/insula/anterior cingulate activation.

### Sensorimotor (sem=0.396, brain=0.942)

| Text A | Text B |
|--------|--------|
| "She kicked the ball hard across the wet grass and felt the impact run up through her shin." | "He stomped on the brake pedal with everything he had, his whole leg locking as the car skidded forward." |

Different scenarios (sports vs. driving), same motor cortex activation for leg/foot actions.

### Spatial scene (sem=0.329, brain=0.944)

| Text A | Text B |
|--------|--------|
| "The desert stretched flat and empty to the horizon in every direction. There was no shade, no water, no movement. Just sand and sky." | "Nothing but open ocean in every direction. No land in sight. The boat was a speck on an endless grey surface under an endless grey sky." |

Different domains (desert vs. ocean), same spatial scene processing.

### Syntactic surprise (sem=0.090, brain=0.872)

| Text A | Text B |
|--------|--------|
| "The complex houses married and single soldiers and their families." | "The rat the cat the dog chased killed ate the malt." |

Different topics, same garden-path reanalysis pattern.

### Narrative suspense (sem=0.237, brain=0.829)

| Text A | Text B |
|--------|--------|
| "He turned the corner. The hallway was empty. Then he heard the lock click shut behind him. He reached for the handle. It didn't move." | "The surgeon paused. The monitor flatlined. She looked at the wound and then she saw it: a second bleeder, deep, pulsing. She had thirty seconds." |

Different genres (thriller vs. medical), same anticipatory processing trajectory.

### Theory of mind (sem=0.260, brain=0.824)

| Text A | Text B |
|--------|--------|
| "He said he was fine, but she could tell from the way he gripped the steering wheel that he wasn't. She decided not to push it." | "The child hid the candy under the pillow, not realizing his sister had been watching from the hallway the entire time." |

Different contexts, same mentalizing network activation.

### Control: same topic, different processing (sem=0.322, brain=0.587 — LOWEST)

| Text A | Text B |
|--------|--------|
| "Water boils when you heat it enough. It turns into steam." | "The nucleation of vapor bubbles in superheated liquid water is governed by the interplay between the Laplace pressure differential across the curved liquid-vapor interface, the degree of metastable superheat relative to the saturation temperature at ambient pressure, and the availability of heterogeneous nucleation sites." |

Same topic, but the brain processes the simple version completely differently from the dense version. Strongest evidence that brain space captures processing demands, not just topic.

---

## 8) Risks (updated post-validation)

1. **High baseline brain similarity (HIGH)** — Mean-pooled whole-brain vectors are dominated by shared language-processing activation. Mitigations: mean-centering, z-scoring, region-specific pooling, variance pooling. For multimodal: separate baselines per modality.
2. **The LLaMA ceiling (MEDIUM)** — If raw LLaMA embeddings show the same divergence, TRIBE's brain mapping adds nothing. Need to test.
3. **Small sample sizes (MEDIUM)** — 5 pairs per category. Round 2 fixes this.
4. **Short text failures (LOW)** — Use paragraph-length minimum.
5. **Collapse to generic semantics (LOW)** — Disconfirmed by r=0.24.

---

## 9) Pipeline

```python
# Step 1: TRIBE inference
events_df = model.get_events_dataframe(text_path="input.txt")
preds, segments = model.predict(events=events_df)
brain_output = np.asarray(preds)  # (T, 20484)

# Step 2: Temporal mean pooling
pooled = brain_output.mean(axis=0)  # (20484,)

# Step 3: Projection head (post-Round 2)
embedding = projection_head(pooled)  # (512,)

# Step 4: Similarity
sim = cosine_similarity(embedding_a, embedding_b)
```

Scientific precedent for mean pooling: [Huth et al. 2016](https://www.nature.com/articles/nature17637), [MVPA](https://pmc.ncbi.nlm.nih.gov/articles/PMC3389290/), [Brain-Score](https://www.biorxiv.org/content/10.1101/407007v2.full). Cache pooled vectors aggressively — TRIBE takes ~38 sec/text on A100.

---

## 10) How Cognix differs from related work

| System | Direction | What it captures |
|--------|-----------|-----------------|
| Sentence-transformers | text -> embedding | Semantic meaning |
| CLIP | image/text -> shared space | Cross-modal semantic alignment |
| BrainCLIP | real fMRI -> CLIP space | Brain decoding |
| MindEye2 | real fMRI -> image | Image reconstruction |
| **Cognix** | text -> predicted brain response -> new space | Cognitive processing similarity |

BrainCLIP/MindEye2 go **brain -> content** (decoding). Cognix goes **content -> brain** (encoding). No fMRI scanner needed.

---

## 11) Alternative fingerprinting methods (to explore after Round 2)

### Pooling variants
- **Mean-centering** — subtract corpus-average vector. May fix the high-baseline problem directly.
- **MVP z-scoring** — z-score across vertices per time point. Core technique in [multi-voxel pattern analysis (Haxby et al. 2001)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3389290/).
- **Max pooling** — keep peak activations instead of averaging.
- **Variance pooling** — captures processing difficulty and cognitive effort.

### Region-specific analysis
- Emotional regions (amygdala-adjacent, insula), motor cortex, prefrontal cortex, language network (left temporal/frontal)

### Alternative similarity metrics
- **Correlation distance** — mean-centered cosine
- **CKA** — compares representational geometry. [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414)
- **RSA** — compare full dissimilarity matrices. [Kriegeskorte et al. 2008](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full)

### Advanced neuroimaging methods
- **Beta-series extraction** — GLM per stimulus. [Rissman et al. 2004](https://pubmed.ncbi.nlm.nih.gov/15488425/)
- **Connectivity fingerprinting** — inter-region correlation patterns. [Finn et al. 2015](https://www.nature.com/articles/nn.4135)
- **Voxel reliability weighting** — weight vertices by TRIBE's prediction reliability.

---

## 12) Applications and research directions

**Applications:** Cognitive readability scoring (beyond Flesch-Kincaid), brain-grounded content recommendation, cross-topic emotional arousal detection, adaptive learning material matching, AI alignment benchmarking via [Brain-Score](https://www.biorxiv.org/content/10.1101/407007v2.full) extension to language.

**Research:** Region-specific embeddings (emotional vs. motor vs. prefrontal), cross-modal cognitive similarity (video + text via TRIBE multimodal), validation against real fMRI datasets, individual brain profile adaptation, temporal dynamics as similarity features, knowledge distillation to CPU-friendly student models.

---

## 13) Compute

- TRIBE v2 needs 40GB+ VRAM. ~38 sec/text on A100, bottleneck is TTS + WhisperX + LLaMA.
- L4 (24GB) may work for text-only, costs 3x fewer Colab compute units.
- Cache brain vectors aggressively — 175 texts took ~2 hours on A100.

---

## 14) HuggingFace release (post-training)

**Upload:** projection model weights, inference pipeline, config, precomputed demo embeddings, model card with divergence examples.

```python
from cognix import CognitiveEmbedder
model = CognitiveEmbedder.from_pretrained("davidgershony/cognix-v1")
embedding = model.encode("The desert stretched flat and empty to the horizon.")
```

---

## 15) Licensing

TRIBE v2: **CC BY-NC 4.0**. Non-commercial use only. Cognix inherits this constraint.

---

## 16) Limitations

- TRIBE predicts coarse fMRI-scale responses (~20,484 vertices), not individual neurons
- Predictions are population-level (averaged across subjects), not personalized
- Brain predictions are a transformation of LLaMA features — cannot capture properties LLaMA doesn't encode
- Mean-pooled vectors have high baseline similarity (~0.82), compressing useful dynamic range
- Not a direct measure of reward, happiness, or preference
- TRIBE inference is slow (~38 sec/text) and requires GPU
- Non-commercial license (CC BY-NC) inherited from TRIBE v2
