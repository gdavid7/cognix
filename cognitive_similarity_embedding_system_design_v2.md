
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

## 6. What we need to test next (Round 2)

Round 2 is the real validation. Three experiments, in order of importance:

### Experiment 1: LLaMA baseline (critical)

Extract LLaMA 3.2-3B embeddings (last hidden state, mean-pooled across tokens) for every text. Compute pairwise cosine similarity. Compare three numbers per pair:

| Metric | Source |
|--------|--------|
| Semantic sim | sentence-transformers (all-MiniLM-L6-v2) |
| LLaMA sim | LLaMA 3.2-3B last hidden state, mean-pooled |
| Brain sim | TRIBE v2 brain predictions, mean-pooled |

**If brain sim ≠ LLaMA sim:** The brain mapping reshapes the geometry. TRIBE adds value beyond its input model. Proceed.

**If brain sim ≈ LLaMA sim:** The brain mapping is cosmetic. The divergence from semantic similarity is just "LLaMA ≠ sentence-transformers," which is unsurprising. Rethink the approach.

This is the single most important experiment.

### Experiment 2: Scale (important)

Does the r ≈ 0.24 divergence hold with 943 pairs (20–25 per divergence category)? Round 1 had 5 per category — not enough for per-category statistical significance.

Dataset breakdown:
- 243 handcrafted divergence pairs (7 categories + controls + adversarial)
- 180 STS-B pairs (filtered to both texts >= 15 words for TRIBE compatibility)
- 500 random Wikipedia pairs (baseline)

Data fixes from Round 1:
- Syntactic surprise (short garden-path sentences) replaced with syntactic complexity (paragraph-length texts with nested relative clauses and complex subordination, ordinary vocabulary)
- Control pairs: simple texts extended to paragraph length to eliminate length confound
- STS-B: filtered out pairs with texts under 15 words (TRIBE needs paragraph-length input)

### Experiment 3: Baseline removal (important)

Mean-center all brain vectors (subtract the corpus-average vector) before computing cosine similarity. This should:
- Compress the 0.82 floor toward 0
- Reveal the true dynamic range of brain similarity
- Make the divergence categories interpretable without the "everything is 0.8+" problem

---

## 7. Risks

### The LLaMA ceiling (untested, could be fatal)

TRIBE's text pathway feeds text through LLaMA 3.2-3B, then maps LLaMA's features to brain activity. If the fingerprint doesn't capture anything beyond what LLaMA already contains, the brain-mapping step adds no value.

Why it might still work: TRIBE's mapping is an 8-layer Transformer trained on real fMRI from 720+ subjects. This is a significant nonlinear transformation supervised by neuroscience data. It should re-weight dimensions based on what the brain cares about — amplifying sensorimotor, affective, and attentional features while suppressing distributional features. But "should" is not "does."

### High baseline similarity (confirmed, addressable)

Mean-pooled whole-brain vectors produce ~0.82 cosine similarity even for unrelated texts. The useful signal lives in a compressed range above this floor. Mean-centering is the standard fix and should work, but it hasn't been tested on this data yet.

### Pair design circularity (acknowledged)

The divergence pairs were designed to diverge — texts chosen for low semantic similarity but similar cognitive properties. A skeptic could ask: "How do you know these pairs are cognitively similar independent of TRIBE?" The answer is that the 7 categories correspond to established cognitive neuroscience constructs (cognitive load, sensorimotor processing, emotional arousal, etc.) supported by decades of fMRI literature. The STS-B and Wikipedia-random pairs provide external validation that isn't hand-designed.

---

## 8. If the experiments pass

If brain sim ≠ LLaMA sim and the divergence holds at scale, several directions open up:

**Immediate next steps:**
- Baseline removal and alternative pooling methods (variance, max, region-specific)
- Region validity test: do prefrontal vertices respond more to cognitive load? Limbic to emotion?
- Learned projection head (20484 → 512-d) for efficient storage and retrieval

**Longer-term possibilities:**
- Cognitive readability scoring — quantify how demanding a text is to process
- Cross-topic similarity based on processing demands rather than meaning
- Cognitively-targeted advertising — if a user engages with an ad, serve them ads that are cognitively similar (same emotional intensity, same level of narrative tension, same sensorimotor engagement) rather than just topically similar. A user who clicks on a visceral rock-climbing ad might respond to a car chase ad, not another climbing ad.
- AI alignment evaluation via comparison to brain-predicted representations
- Knowledge distillation to a CPU-friendly model
- Multimodal cognitive similarity using TRIBE's video/audio pathways

These are research directions, not product promises. Each would require its own validation.

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
- 1,853 unique texts ≈ 19.5 hours on A100
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
