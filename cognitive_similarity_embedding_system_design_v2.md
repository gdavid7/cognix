
# Cognix: Brain-Grounded Cognitive Similarity

## 1. What this is

Cognix explores whether predicted brain responses can measure something about text that standard embeddings miss. Standard embeddings capture semantic similarity — what content is *about*. Cognix asks whether TRIBE v2's brain predictions capture **cognitive similarity** — how similarly the brain *processes* two inputs, regardless of topic.

Core pipeline:

```
text → TRIBE v2 → brain tensor (T, 20484) → mean pooling → (20484,) → cosine similarity
```

This is research, not a finished system. The central question is whether the brain mapping step reshapes the representation space in a useful way — or whether it's just re-encoding what LLaMA already knows.

---

## 2. The hypothesis

Two texts can be semantically unrelated but cognitively similar. A dense legal clause and a dense math proof are about different things, but both demand heavy cognitive load. A war scene and a cancer diagnosis evoke different content but similar emotional processing. Standard embeddings miss this because they encode *meaning*, not *processing demands*.

**If** TRIBE v2's brain predictions capture processing characteristics that the underlying LLaMA model does not already represent, **then** brain-predicted similarity should diverge from both semantic similarity and raw LLaMA similarity in predictable, category-specific ways.

The "if" is the part we're testing.

---

## 3. How TRIBE v2 works

Understanding the architecture is essential for interpreting results, because it determines what the brain predictions can and cannot capture.

### Architecture

```
Stage 1 — Feature extraction:
  text  → LLaMA 3.2-3B      → language features
  video → V-JEPA2-Giant      → visual features
  audio → Wav2Vec-BERT 2.0   → audio features

Stage 2 — Fusion:
  All features → compressed to D=384, concatenated → D=1152
  → 8-layer Transformer encoder (100-second context windows)

Stage 3 — Brain projection:
  Transformer outputs → decimated to fMRI frequency
  → Subject Block → predicted cortical activity (~20,484 vertices on fsaverage5)
```

### Text tokenization path

```
text → gTTS (text-to-speech) → audio → WhisperX → word-level timestamps
timestamps + LLaMA 3.2-3B → features at 2 Hz → brain transformer → vertex predictions
```

Short texts (under ~10 words) produce very few time steps. Paragraph-length inputs recommended.

### Why this matters for the LLaMA question

The brain tensor is a learned nonlinear transformation of LLaMA features. The 8-layer Transformer and real-fMRI supervision (1,000+ hours from 720+ subjects) make this non-trivial — but the input signal is still LLaMA. Two things follow:

1. Any cognitive property that LLaMA doesn't encode *at all* cannot appear in the brain prediction.
2. But the brain mapping could *amplify* cognitive dimensions that LLaMA encodes weakly and *suppress* semantic dimensions that LLaMA weights heavily.

Whether (2) actually happens is an empirical question. That's what the LLaMA baseline experiment tests.

---

## 4. What we've found so far (Round 1)

100 pairs across 10 categories. Each pair scored on two axes: semantic similarity (sentence-transformers) and brain similarity (mean-pooled TRIBE v2 cosine).

**Pearson r = 0.24 (p = 0.017). Spearman r = 0.32 (p = 0.002).**

Brain similarity and semantic similarity are barely correlated.

| Category | N | Semantic sim | Brain sim | Divergence |
|----------|---|-------------|-----------|------------|
| Syntactic surprise* | 5 | 0.090 | 0.872 | **+0.782** |
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

- All 7 divergence categories diverge in the predicted direction — brain says "similar" where semantics says "not similar"
- The control category (same topic, different complexity) has the *lowest* brain similarity (0.587), meaning the brain representation distinguishes processing demands even within the same topic
- Paraphrases score high on both metrics, as expected

*\*Round 1 syntactic surprise pairs were short garden-path sentences (<10 words). These are unreliable in TRIBE's TTS-based pipeline. Replaced with paragraph-length syntactic complexity pairs in Round 2.*

### What this doesn't tell us

- Whether the divergence comes from the brain mapping or from LLaMA features (untested)
- Whether the results hold at scale (5 pairs per category is too few)
- Whether the 0.82 baseline for unrelated pairs is obscuring the signal

---

## 5. Concrete examples

### Cognitive load (semantic = 0.09, brain = 0.85)

| Text A | Text B |
|--------|--------|
| "The party of the first part shall indemnify and hold harmless the party of the second part against any and all claims, damages, losses, costs, and expenses arising out of or in connection with any breach of this agreement." | "Given that the eigenvalues of a Hermitian matrix are real, and that eigenvectors corresponding to distinct eigenvalues are orthogonal, we can construct an orthonormal basis for the space by applying the Gram-Schmidt procedure to each eigenspace." |

Different topics. Both demand heavy processing. Brain sees them as similar.

### Emotional arousal (semantic = 0.29, brain = 0.74)

| Text A | Text B |
|--------|--------|
| "The soldier held his dying friend in his arms and screamed for a medic, but no one came." | "She sat in the doctor's office and heard the words she had been dreading. Stage four. Inoperable." |

### Sensorimotor (semantic = 0.40, brain = 0.94)

| Text A | Text B |
|--------|--------|
| "She kicked the ball hard across the wet grass and felt the impact run up through her shin." | "He stomped on the brake pedal with everything he had, his whole leg locking as the car skidded forward." |

### Spatial scene (semantic = 0.33, brain = 0.94)

| Text A | Text B |
|--------|--------|
| "The desert stretched flat and empty to the horizon in every direction. There was no shade, no water, no movement." | "Nothing but open ocean in every direction. No land in sight. The boat was a speck on an endless grey surface." |

### Syntactic surprise (semantic = 0.09, brain = 0.87)

| Text A | Text B |
|--------|--------|
| "The complex houses married and single soldiers and their families." | "The rat the cat the dog chased killed ate the malt." |

### Narrative suspense (semantic = 0.24, brain = 0.83)

| Text A | Text B |
|--------|--------|
| "He turned the corner. The hallway was empty. Then he heard the lock click shut behind him." | "The surgeon paused. The monitor flatlined. She looked at the wound and then she saw it: a second bleeder, deep, pulsing." |

### Theory of mind (semantic = 0.26, brain = 0.82)

| Text A | Text B |
|--------|--------|
| "He said he was fine, but she could tell from the way he gripped the steering wheel that he wasn't." | "The child hid the candy under the pillow, not realizing his sister had been watching from the hallway." |

### Control: same topic, different processing (semantic = 0.32, brain = 0.59)

| Text A | Text B |
|--------|--------|
| "Water boils when you heat it enough. It turns into steam." | "The nucleation of vapor bubbles in superheated liquid water is governed by the interplay between the Laplace pressure differential across the curved liquid-vapor interface..." |

Same topic, different cognitive demands. Brain sees them as *dissimilar* — it distinguishes the processing difference.

---

## 6. Round 2 results (April 2026)

923 pairs (243 handcrafted, 180 STS-B, 500 Wikipedia random), 1,835 unique texts. All three experiments completed.

### Experiment 1: LLaMA baseline — PASSED

| Comparison | Pearson r | Spearman r |
|---|---|---|
| Brain vs Semantic | 0.43 | 0.50 |
| Brain vs LLaMA | 0.44 | 0.54 |
| LLaMA vs Semantic | 0.83 | 0.79 |

LLaMA and sentence-transformers largely agree (r=0.83). Brain vectors diverge from both (r≈0.43-0.44). **The brain mapping substantially reshapes the similarity geometry beyond what LLaMA encodes.** The 8-layer Transformer trained on real fMRI data amplifies cognitive dimensions that LLaMA encodes weakly.

### Experiment 2: Scale — CONFIRMED

Divergence holds at 10× more pairs. All 7 divergence categories show statistically significant brain vs semantic divergence (p < 0.001 for every category). Permutation test p=0.0000.

Per-category results (handcrafted pairs):

| Category | N | Sem | LLaMA | Brain | Brain−LLaMA |
|---|---|---|---|---|---|
| cognitive_load | 25 | 0.050 | 0.509 | 0.833 | **+0.324** |
| control (diff processing) | 30 | 0.392 | 0.642 | 0.776 | +0.134 |
| spatial_scene | 20 | 0.383 | 0.870 | 0.949 | +0.079 |
| sensorimotor | 20 | 0.428 | 0.861 | 0.905 | +0.044 |
| syntactic_complexity | 20 | 0.201 | 0.870 | 0.909 | +0.039 |
| narrative_suspense | 20 | 0.250 | 0.794 | 0.824 | +0.030 |
| theory_of_mind | 20 | 0.253 | 0.831 | 0.838 | +0.007 |
| emotional_arousal | 20 | 0.297 | 0.819 | 0.749 | **−0.070** |
| paraphrase | 30 | 0.853 | 0.955 | 0.954 | −0.001 |
| unrelated | 30 | −0.009 | 0.623 | 0.793 | +0.171 |

Cognitive load shows the largest brain vs LLaMA divergence (+0.324). Emotional arousal is the only category where brain sim < LLaMA sim (−0.070), supporting the hypothesis that mean pooling drowns limbic signal in 20,484 vertices.

### Experiment 3: Baseline removal — EFFECTIVE

Mean-centering compresses the random-pair floor from 0.812 to 0.264. Centered brain vs semantic: r=0.55 (higher than raw r=0.43 because mean-centering removes the shared language-processing baseline that inflated all brain similarities).

### Robustness checks

- Text length vs brain sim: r=−0.31 (moderate, not a confound)
- Permutation test: p=0.0000
- Brain vs semantic per source: handcrafted r=0.40, STS-B r=0.26, Wikipedia r=0.15
- Zero inference failures (923/923 pairs, 1835/1835 texts)

### Key insight

Brain vs semantic r went from 0.24 (Round 1, 100 pairs) to 0.43 (Round 2, 923 pairs). The correlation is higher at scale than Round 1 suggested. This doesn't undermine the finding — r=0.43 still indicates substantial divergence — but Round 1's r=0.24 was likely deflated by small sample size and the short garden-path sentences that were replaced.

---

## 7. Risks

### The LLaMA ceiling — RESOLVED

Brain vs LLaMA r=0.44. The brain mapping substantially reshapes the geometry. TRIBE adds value beyond its input model.

### High baseline similarity — RESOLVED

Mean-centering drops the random-pair floor from 0.812 to 0.264. The dynamic range is now usable.

### Pair design circularity (acknowledged)

The divergence pairs were designed to diverge — texts chosen for low semantic similarity but similar cognitive properties. A skeptic could ask: "How do you know these pairs are cognitively similar independent of TRIBE?" The answer is that the 7 categories correspond to established cognitive neuroscience constructs (cognitive load, sensorimotor processing, emotional arousal, etc.) supported by decades of fMRI literature. The STS-B and Wikipedia-random pairs provide external validation that isn't hand-designed.

### Emotional arousal underperformance (new)

Emotional arousal is the only category where brain sim < LLaMA sim. Hypothesis: emotional processing is concentrated in ~500 limbic vertices, and mean pooling over 20,484 vertices drowns the signal. Phase 3 region-specific pooling tests this.

---

## 8. Next phases

The LLaMA baseline passed and divergence holds at scale. **What's still unproven is whether the divergence is useful.** Round 2's evidence is internal (hand-designed pairs, derived metrics). The next phase grounds the project against an external behavioral signal before any structural work.

### Phase 3: Downstream validation on eye-tracking (next)

Cognix says brain vectors capture cognitive processing characteristics that LLaMA doesn't. The honest test is whether they predict an external measure of cognitive processing — per-word reading time on naturalistic text.

Steps:
- Pick an eye-tracking corpus: Provo, Dundee, or GECO. All public, all have per-word fixation durations from many subjects on natural reading.
- For each fixation point, build a small text window around it. Compute three feature representations of the window: sentence-transformer embedding, LLaMA 3.2-3B mean-pooled hidden state, TRIBE pooled brain vector (cached).
- Predict reading time (or surprisal-residualized reading time) from each feature source. Hold out subjects.
- Report R² (or AUC for above-/below-median) for all three feature sources side by side.

Decision gate:
- If Cognix beats both baselines → divergence is behaviorally useful. Phase 4+ is justified.
- If Cognix loses or ties → the divergence is real but not useful for predicting human behavior. Stop and re-examine before building a distilled model on a signal that doesn't translate.

Cost: cached vectors + a few hundred MB of public eye-tracking data. CPU only. ~1 day of work.

### Phase 4: Region-specific pooling (deferred)

Only if Phase 3 validates. Mean pooling averages 20,484 vertices equally and drowns localized signals — emotional arousal (Brain−LLaMA = −0.070) is the clearest case. Whether region pooling rescues it determines the Phase 5 architecture: structured per-region embedding vs. single vector.

When run, must include:
- Per-region mean-centering. Round 2 showed mean-centering compresses random-pair similarity from 0.812 to 0.264 and lifts brain-vs-semantic correlation from 0.43 to 0.55. Each region has its own baseline; raw cosine on sub-vectors will be dominated by it.
- Discrimination AUC vs. random_baseline pairs, not argmax of mean similarity. Higher mean sim doesn't mean a region "captures" a dimension; it may just be uniformly active.
- Size-matched random-vertex-subset controls. Different regions have different vertex counts, which changes cosine baseline distributions.
- Label-shuffle null over the region→category mapping. With 6 regions and 4 categories, ~0.67 matches happen by chance.
- Honest acknowledgment that the amygdala — central to emotional arousal — is subcortical and not on the fsaverage5 surface. Limbic pooling may fail for that reason alone.

Cost: cached pooled vectors `(20484,)` only. No GPU.

### Phase 5: Distilled cognitive embedding model (1–2 weeks)

TRIBE is too slow for real use (~38 sec/text, 40GB GPU). The goal is a small, fast model that approximates TRIBE's similarity geometry:

```
text → frozen LLaMA 3.2 → learned projection head → 512-d cognitive embedding
```

The projection head (a small MLP or transformer) is trained via contrastive learning using TRIBE's brain similarities as supervision: pairs that TRIBE says are cognitively similar should be close in the embedding space, pairs it says are different should be far apart.

Training data: ~1.7 million possible pairings from 1,835 cached texts, each with precomputed brain similarity. Training runs on CPU in minutes (small MLP on cached vectors).

Evaluation: re-run the Phase 3 eye-tracking test on the distilled embeddings. The bar is to match TRIBE's downstream performance at a fraction of the cost.

If Phase 4 region pooling helps, the projection head trains on region-decomposed features — a region-aware cognitive embedding with interpretable per-region scores.

### Phase 6: Applications and extensions

These build on a working, validated embedding model. Each requires its own validation.

- Cognitive readability scoring — quantify how demanding a text is to process
- Cross-topic similarity based on processing demands rather than meaning
- AI alignment evaluation via comparison to brain-predicted representations
- Multimodal cognitive similarity using TRIBE's video/audio pathways
- Knowledge distillation from the LLaMA-based model to a smaller standalone transformer

---

## 9. Related work

| System | Direction | What it captures |
|--------|-----------|-----------------|
| Sentence-transformers | text → embedding | Semantic meaning |
| CLIP | image/text → shared space | Cross-modal semantic alignment |
| BrainCLIP | real fMRI → CLIP space | Brain decoding |
| MindEye2 | real fMRI → image | Image reconstruction |
| Brain-Score | model → brain predictivity score | How brain-like a model is |
| **Cognix** | text → predicted brain response → similarity | Cognitive processing similarity |

BrainCLIP/MindEye2 go **brain → content** (decoding). Cognix goes **content → brain** (encoding) and uses the brain tensor as a representation space. No fMRI scanner needed.

Brain-Score asks "how well does this model predict the brain?" Cognix asks "can brain predictions be used as a useful feature space?"

### Key related papers
- Huth et al. (2016) — semantic maps from fMRI, region-specific language representation
- Caucheteux & King (2022) — LLMs partially converge with brain activity in NLP
- Schrimpf et al. (2021) — neural architecture of language, brain predictivity benchmarks
- Kriegeskorte et al. (2008) — representational similarity analysis (RSA)

---

## 10. Fingerprinting methods to explore (after Round 2)

Only pursue these if the LLaMA baseline confirms the brain mapping adds value.

**Pooling:** mean-centering, z-scoring across vertices, variance pooling, max pooling

**Region-specific:** map vertices to brain regions via Desikan-Killiany or Schaefer atlas, extract per-region scores

**Similarity metrics:** correlation distance (mean-centered cosine), CKA, RSA

**Learned:** MLP projection (20484 → 512), contrastive training, multi-task heads

---

## 11. Technical details

### Compute
- TRIBE v2 needs 40GB+ VRAM. ~38 sec/text on A100
- 1,835 unique texts ≈ 19.4 hours on A100
- Cache raw tensors to Google Drive for all future experiments
- All fingerprinting experiments run on cached data (no GPU needed after caching)

### Licensing
TRIBE v2: **CC BY-NC 4.0**. Non-commercial use only. Cognix inherits this constraint.

### Limitations
- Brain predictions are a transformation of LLaMA 3.2 features — cannot capture properties LLaMA doesn't encode
- Mean-pooled vectors have ~0.82 baseline similarity — baseline removal required
- Predictions are population-level (averaged across subjects), not personalized
- TRIBE predicts coarse fMRI-scale responses (~20,484 vertices), not individual neurons
- TRIBE inference is slow (~38 sec/text) and requires 40GB+ GPU
- Non-commercial license inherited from TRIBE v2

### Links

**TRIBE v2:**
- Blog: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
- Paper: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- Code: https://github.com/facebookresearch/tribev2

**Related work:**
- CLIP: https://arxiv.org/abs/2103.00020
- BrainCLIP: https://arxiv.org/abs/2302.12971
- MindEye2: https://arxiv.org/abs/2403.11207
