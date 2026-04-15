
# Cognix: Brain-Grounded Cognitive Fingerprints

## 1) What this project is

Research into the optimal way to turn predicted brain response tensors into cognitive fingerprints — compact representations that capture how the brain processes content, not just what the content means.

Core pipeline:

```
input -> TRIBE v2 -> brain tensor (T, 20484) -> [fingerprinting method] -> cognitive fingerprint -> downstream tasks
```

The fingerprinting method is the research question. Mean pooling is the baseline. Region-based decomposition, z-scoring, variance pooling, and other established neuroimaging techniques are candidates. The goal is to find which method(s) produce fingerprints that are:

1. **Meaningfully different** from standard semantic embeddings
2. **Derived from the brain mapping**, not just LLaMA features passed through (see Section 7)
3. **Useful** for downstream tasks like similarity, retrieval, and cognitive profiling

This is research, not a finished product. Some of this may not work. The exciting part is that Round 1 validation (r=0.24, p=0.017) suggests the brain tensor contains signal that standard embeddings miss — the open question is how to extract it.

---

## 2) Why this project exists

Standard embeddings capture semantic similarity — what content is **about**. They do not capture:
- Cognitive load (how demanding is this to process?)
- Emotional arousal (what does this trigger neurally?)
- Sensorimotor grounding (does this evoke physical sensation?)
- Processing similarity ("would the brain handle these two inputs the same way?")

TRIBE v2 (Meta, March 2026) makes this feasible. It predicts ~20,484 cortical vertices of fMRI brain response from text, audio, and video, trained on 1,000+ hours of real fMRI from 720+ subjects. The output maps to a standard brain atlas (fsaverage5), meaning each vertex belongs to a known brain region — in principle enabling region-specific signal extraction. Whether this works in practice is one of the things we need to test.

---

## 3) Core scientific risks

These are ordered by severity. The project's value depends on addressing all three.

### Risk 1: The LLaMA ceiling (CRITICAL — untested)

TRIBE v2's text pathway feeds text through **LLaMA 3.2-3B**, then maps LLaMA's features to brain activity via a learned transformation. The brain tensor is therefore a function of LLaMA features. If the fingerprint we extract doesn't capture anything beyond what raw LLaMA embeddings already contain, then the entire brain-mapping step adds no value — we're just re-encoding LLaMA in a noisier space.

**This is the single most important thing to test.** Round 2 includes a direct LLaMA embedding baseline. If LLaMA embeddings show the same divergence pattern as brain fingerprints, the project's core claim is undermined.

Why it might still work: TRIBE's mapping is an 8-layer Transformer trained on real fMRI data from 720+ subjects. This is a significant nonlinear transformation supervised by neuroscience data that LLaMA never saw. The mapping re-weights dimensions based on what the brain cares about — amplifying sensorimotor, affective, and attentional features while suppressing syntactic and distributional features that LLaMA weights equally. But this must be demonstrated empirically, not assumed.

### Risk 2: High baseline brain similarity (HIGH — confirmed in Round 1)

Mean-pooled whole-brain vectors have a cosine similarity of ~0.82 even for completely unrelated texts. The useful signal lives in a compressed range above this floor. **Baseline removal is required before any similarity computation or axis extraction** — it's not an optimization, it's a prerequisite. Without it, everything looks similar.

Candidate solutions (to test in Phase 4):
- Mean-centering: subtract the corpus-average brain vector
- Z-scoring across vertices per time point
- Region-specific pooling (avoids averaging signal with 20,000 irrelevant vertices)
- Correlation distance instead of cosine similarity

### Risk 3: Region validity (MEDIUM — untested)

The multi-axis cognitive profile idea (prefrontal = load, limbic = emotion, motor cortex = motor engagement) assumes TRIBE's vertex predictions are spatially meaningful at the region level. TRIBE was trained on real fMRI mapped to fsaverage5, so this is plausible — but "plausible" is not "verified."

**Region validity test:** Run TRIBE on texts with known cognitive properties (our divergence pairs). Check whether the prefrontal vertices actually activate more for cognitive load pairs, whether limbic vertices activate more for emotional pairs, etc. If the spatial signal isn't there, region-based decomposition won't work and we fall back to whole-brain methods with baseline removal.

---

## 4) Roadmap

### Completed
- **Round 1 validation (100 pairs):** Pearson r=0.24, p=0.017. Brain similarity and semantic similarity are barely correlated. All 7 divergence categories diverge as predicted. See Section 6.

### Phase 2: Confirm at scale (CURRENT)
- 1,000 pairs, 25+ per divergence category, paragraph-length minimum
- Same method as Round 1 (mean-pooled whole-brain TRIBE vectors)
- **LLaMA embedding baseline** — the critical test (Risk 1)
- Random-pair baseline — quantifies the high baseline (Risk 2)
- Cache **raw tensors** (T, 20484) to Google Drive for all future experiments
- Adversarial pairs to prevent overfitting

### Phase 3: Baseline removal
Address Risk 2. Required before any downstream work.
- Test mean-centering, z-scoring, correlation distance on cached tensors
- Establish which method produces the cleanest separation between related and unrelated pairs
- This step is not optional — it's a prerequisite for everything that follows

### Phase 4: Region validity test
Address Risk 3. Determine whether multi-axis decomposition is feasible.
- Map fsaverage5 vertices to brain regions using a standard atlas (Desikan-Killiany or Schaefer)
- Check: do prefrontal vertices respond more to cognitive load pairs? Do limbic vertices respond more to emotional pairs?
- If yes: proceed to multi-axis fingerprinting
- If no: the spatial structure isn't reliable at the region level — fall back to whole-brain methods with learned projections

### Phase 5: Fingerprinting experiments
Find the optimal method to turn brain tensors into fingerprints.
- Whole-brain methods: mean-centering + cosine, z-scored pooling, variance pooling, max pooling
- Region-based methods: per-region scores, region-weighted pooling
- Learned methods: MLP projection, contrastive training, multi-task heads
- Evaluation: which method produces the most useful fingerprint for similarity, retrieval, and cognitive profiling?
- See Section 11 for full list with references

### Phase 6: Package and release
- Release the fingerprinting pipeline + trained projection (if applicable) on HuggingFace
- Include precomputed fingerprints for demo datasets
- Model card with honest evaluation: what works, what doesn't, what the LLaMA baseline showed

### Future
- Knowledge distillation to CPU-friendly model
- Multimodal support (video, audio) with per-modality baselines
- Validation against real fMRI datasets

---

## 5) Pipeline (explicit)

```
input text
  -> TRIBE v2
  -> raw brain tensor (T, 20484)
  -> temporal pooling (mean, max, or variance over T)
  -> baseline removal (subtract corpus mean, z-score, or correlation distance)
  -> brain fingerprint (20484-d, or region-decomposed)
  -> [optional] learned projection (20484 -> 512-d embedding)
  -> downstream tasks:
       - pairwise similarity (cosine of fingerprints)
       - cognitive axis scores (if region validity passes)
       - retrieval, clustering, classification
```

**Key point:** Baseline removal is an explicit required step in the pipeline, not a post-hoc fix. Raw mean-pooled vectors are not usable for similarity due to the ~0.82 floor.

**Where the embedding fits:** The 512-d learned projection is one possible downstream output of the fingerprint, useful for efficient storage and retrieval. It's not the core representation — the brain fingerprint is. The embedding compresses the fingerprint for practical use.

---

## 6) How TRIBE v2 works inside

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
  -> Subject Block -> predicted cortical activity (~20,484 vertices on fsaverage5)
```

### Text tokenization quirk

```text
text -> gTTS (text-to-speech) -> audio -> WhisperX -> word-level timestamps
timestamps + LLaMA 3.2-3B -> features at 2 Hz -> brain transformer -> voxel predictions
```

Short texts (under ~10 words) produce very few time steps. Paragraph-length inputs recommended.

### What this means for the LLaMA ceiling risk

The brain tensor is a learned nonlinear transformation of LLaMA features. The 8-layer Transformer and real-fMRI supervision make this a non-trivial mapping — but the input signal is still LLaMA. Any cognitive property that LLaMA doesn't encode cannot appear in the brain prediction. The LLaMA baseline test (Phase 2) directly measures how much the brain mapping reshapes the similarity geometry vs. passing through what LLaMA already represents.

---

## 7) Validation results — Round 1 (COMPLETED)

### Results (100 pairs, 10 categories, April 2026)

**Pearson r = 0.24 (p = 0.017). Spearman r = 0.32 (p = 0.0015).**

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

### What this tells us
- Brain similarity and semantic similarity are barely correlated — the brain tensor contains different information
- All 7 divergence categories diverge in the predicted direction
- Control category (same topic, different complexity) has the lowest brain similarity (0.587) — processing demands are captured
- **BUT:** unrelated pairs average 0.824 brain similarity, confirming the high baseline problem
- **AND:** the LLaMA baseline has not been tested — we don't yet know if this divergence comes from the brain mapping or from LLaMA features

### What we don't know yet
- Whether the divergence survives at 1,000 pairs (Round 2)
- Whether LLaMA embeddings show the same pattern (the critical test)
- Whether the spatial structure of the brain tensor is region-meaningful
- Whether baseline removal produces cleaner separation

---

## 8) Concrete divergence examples (validated in Round 1)

### Cognitive load (sem=0.087, brain=0.845)

| Text A | Text B |
|--------|--------|
| "The party of the first part shall indemnify and hold harmless the party of the second part against any and all claims, damages, losses, costs, and expenses arising out of or in connection with any breach of this agreement." | "Given that the eigenvalues of a Hermitian matrix are real, and that eigenvectors corresponding to distinct eigenvalues are orthogonal, we can construct an orthonormal basis for the space by applying the Gram-Schmidt procedure to each eigenspace." |

### Emotional arousal (sem=0.285, brain=0.735)

| Text A | Text B |
|--------|--------|
| "The soldier held his dying friend in his arms and screamed for a medic, but no one came. The blood soaked through his uniform as the light faded from his friend's eyes." | "She sat in the doctor's office and heard the words she had been dreading. Stage four. Inoperable. She drove home in silence and sat in the driveway for an hour before she could walk inside." |

### Sensorimotor (sem=0.396, brain=0.942)

| Text A | Text B |
|--------|--------|
| "She kicked the ball hard across the wet grass and felt the impact run up through her shin." | "He stomped on the brake pedal with everything he had, his whole leg locking as the car skidded forward." |

### Spatial scene (sem=0.329, brain=0.944)

| Text A | Text B |
|--------|--------|
| "The desert stretched flat and empty to the horizon in every direction. There was no shade, no water, no movement. Just sand and sky." | "Nothing but open ocean in every direction. No land in sight. The boat was a speck on an endless grey surface under an endless grey sky." |

### Syntactic surprise (sem=0.090, brain=0.872)

| Text A | Text B |
|--------|--------|
| "The complex houses married and single soldiers and their families." | "The rat the cat the dog chased killed ate the malt." |

### Narrative suspense (sem=0.237, brain=0.829)

| Text A | Text B |
|--------|--------|
| "He turned the corner. The hallway was empty. Then he heard the lock click shut behind him. He reached for the handle. It didn't move." | "The surgeon paused. The monitor flatlined. She looked at the wound and then she saw it: a second bleeder, deep, pulsing. She had thirty seconds." |

### Theory of mind (sem=0.260, brain=0.824)

| Text A | Text B |
|--------|--------|
| "He said he was fine, but she could tell from the way he gripped the steering wheel that he wasn't. She decided not to push it." | "The child hid the candy under the pillow, not realizing his sister had been watching from the hallway the entire time." |

### Control: same topic, different processing (sem=0.322, brain=0.587)

| Text A | Text B |
|--------|--------|
| "Water boils when you heat it enough. It turns into steam." | "The nucleation of vapor bubbles in superheated liquid water is governed by the interplay between the Laplace pressure differential across the curved liquid-vapor interface, the degree of metastable superheat relative to the saturation temperature at ambient pressure, and the availability of heterogeneous nucleation sites." |

---

## 9) How Cognix differs from related work

| System | Direction | What it captures |
|--------|-----------|-----------------|
| Sentence-transformers | text -> embedding | Semantic meaning |
| CLIP | image/text -> shared space | Cross-modal semantic alignment |
| BrainCLIP | real fMRI -> CLIP space | Brain decoding |
| MindEye2 | real fMRI -> image | Image reconstruction |
| **Cognix** | text -> predicted brain response -> fingerprint | Cognitive processing characteristics |

BrainCLIP/MindEye2 go **brain -> content** (decoding). Cognix goes **content -> brain** (encoding) and uses the brain tensor as a feature space for fingerprinting. No fMRI scanner needed.

---

## 10) Alternative fingerprinting methods (Phase 5)

### Pooling variants
- **Mean-centering** — subtract corpus-average vector. [Huth et al. 2016](https://www.nature.com/articles/nature17637)
- **MVP z-scoring** — z-score across vertices per time point. [Haxby et al. 2001](https://pmc.ncbi.nlm.nih.gov/articles/PMC3389290/)
- **Max pooling** — keep peak activations
- **Variance pooling** — captures processing difficulty

### Region-specific (contingent on Phase 4 region validity test)
- Per-region scores using standard brain atlases (Desikan-Killiany, Schaefer)
- Region-weighted pooling

### Similarity metrics
- **Correlation distance** — mean-centered cosine
- **CKA** — [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414)
- **RSA** — [Kriegeskorte et al. 2008](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full)

### Advanced
- **Beta-series extraction** — [Rissman et al. 2004](https://pubmed.ncbi.nlm.nih.gov/15488425/)
- **Connectivity fingerprinting** — [Finn et al. 2015](https://www.nature.com/articles/nn.4135)
- **Voxel reliability weighting**

---

## 11) Applications and research directions

**If the fingerprinting works:** Cognitive readability scoring, brain-grounded content recommendation, cross-topic emotional arousal detection, adaptive learning material matching, AI alignment benchmarking via [Brain-Score](https://www.biorxiv.org/content/10.1101/407007v2.full) extension to language.

**Research:** Region-specific signal extraction, cross-modal cognitive similarity (video + text via TRIBE multimodal), validation against real fMRI datasets, temporal dynamics as fingerprint features, knowledge distillation.

**If region decomposition works:** Multi-axis cognitive profiles — cognitive load score, emotional intensity score, motor engagement score per text. This is speculative until Phase 4 validates the spatial structure.

---

## 12) Links

### TRIBE v2
- Blog: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
- Paper: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- Code: https://github.com/facebookresearch/tribev2

### Related work
- CLIP: https://arxiv.org/abs/2103.00020
- BrainCLIP: https://arxiv.org/abs/2302.12971
- MindEye2: https://arxiv.org/abs/2403.11207

---

## 13) Compute

- TRIBE v2 needs 40GB+ VRAM. ~38 sec/text on A100.
- Cache raw tensors aggressively — 175 texts took ~2 hours on A100.
- All fingerprinting experiments run on cached data (no GPU needed after caching).

---

## 14) Licensing

TRIBE v2: **CC BY-NC 4.0**. Non-commercial use only. Cognix inherits this constraint.

---

## 15) Limitations

- Brain predictions are a transformation of LLaMA 3.2 features — cannot capture properties LLaMA doesn't encode (the LLaMA ceiling)
- Mean-pooled vectors have ~0.82 baseline similarity — baseline removal is required, not optional
- Region-based decomposition assumes spatial validity of TRIBE predictions — unverified
- TRIBE predicts coarse fMRI-scale responses (~20,484 vertices), not individual neurons
- Predictions are population-level (averaged across subjects), not personalized
- Multi-axis cognitive profiles are speculative until region validity is confirmed
- TRIBE inference is slow (~38 sec/text) and requires GPU
- Non-commercial license (CC BY-NC) inherited from TRIBE v2
