
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

## 7) Concrete divergence examples (validated in Round 1)

All examples below show the predicted pattern: low semantic similarity, high brain similarity.

### Cognitive load (sem=0.087, brain=0.845)

| Text A | Text B |
|--------|--------|
| "The party of the first part shall indemnify and hold harmless the party of the second part against any and all claims, damages, losses, costs, and expenses arising out of or in connection with any breach of this agreement." | "Given that the eigenvalues of a Hermitian matrix are real, and that eigenvectors corresponding to distinct eigenvalues are orthogonal, we can construct an orthonormal basis for the space by applying the Gram-Schmidt procedure to each eigenspace." |

Different topics (law vs. math), same heavy working memory demand and sustained prefrontal activation.

### Emotional arousal (sem=0.285, brain=0.735)

| Text A | Text B |
|--------|--------|
| "The soldier held his dying friend in his arms and screamed for a medic, but no one came. The blood soaked through his uniform as the light faded from his friend's eyes." | "She sat in the doctor's office and heard the words she had been dreading. Stage four. Inoperable. She drove home in silence and sat in the driveway for an hour before she could walk inside." |

Different content (war vs. medical), same amygdala/insula/anterior cingulate activation for emotional distress.

### Sensorimotor (sem=0.396, brain=0.942)

| Text A | Text B |
|--------|--------|
| "She kicked the ball hard across the wet grass and felt the impact run up through her shin." | "He stomped on the brake pedal with everything he had, his whole leg locking as the car skidded forward." |

Different scenarios (sports vs. driving), same motor cortex activation for leg/foot actions. The brain simulates the physical action while reading.

### Spatial scene (sem=0.329, brain=0.944)

| Text A | Text B |
|--------|--------|
| "The desert stretched flat and empty to the horizon in every direction. There was no shade, no water, no movement. Just sand and sky." | "Nothing but open ocean in every direction. No land in sight. The boat was a speck on an endless grey surface under an endless grey sky." |

Different domains (desert vs. ocean), same spatial scene processing in parahippocampal place area and retrosplenial cortex.

### Syntactic surprise (sem=0.090, brain=0.872)

| Text A | Text B |
|--------|--------|
| "The complex houses married and single soldiers and their families." | "The rat the cat the dog chased killed ate the malt." |

Completely different topics, same garden-path reanalysis — the brain parses, fails, and reparses, producing a spike in left inferior frontal gyrus.

### Narrative suspense (sem=0.237, brain=0.829)

| Text A | Text B |
|--------|--------|
| "He turned the corner. The hallway was empty. Then he heard the lock click shut behind him. He reached for the handle. It didn't move." | "The surgeon paused. The monitor flatlined. She looked at the wound and then she saw it: a second bleeder, deep, pulsing. She had thirty seconds." |

Different genres (thriller vs. medical), same anticipatory processing trajectory — building tension, prediction error, dopaminergic surprise.

### Theory of mind (sem=0.260, brain=0.824)

| Text A | Text B |
|--------|--------|
| "He said he was fine, but she could tell from the way he gripped the steering wheel that he wasn't. She decided not to push it." | "The child hid the candy under the pillow, not realizing his sister had been watching from the hallway the entire time." |

Different contexts (adult emotional deception vs. child behavior), same mentalizing network activation — reasoning about others' beliefs, intentions, and hidden states via medial prefrontal cortex and temporoparietal junction.

### Control: same topic, different processing (sem=0.322, brain=0.587 — LOWEST)

| Text A | Text B |
|--------|--------|
| "Water boils when you heat it enough. It turns into steam." | "The nucleation of vapor bubbles in superheated liquid water is governed by the interplay between the Laplace pressure differential across the curved liquid-vapor interface, the degree of metastable superheat relative to the saturation temperature at ambient pressure, and the availability of heterogeneous nucleation sites." |

Same topic, but the brain processes the simple version completely differently from the dense version. This is the strongest evidence that brain space captures processing demands, not just topic.

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

Established methods from neuroimaging for creating brain fingerprints. Each captures different aspects of the brain response. Test after Round 2 confirms the signal at scale.

### Pooling variants
- **Mean-centering** — subtract corpus-average vector to remove shared "processing language" baseline. May fix the high-baseline problem directly.
- **MVP z-scoring** — z-score across vertices per time point, emphasizing relative activation patterns over absolute levels. Core technique in [multi-voxel pattern analysis (Haxby et al. 2001)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3389290/).
- **Max pooling** — keep peak activations instead of averaging. Preserves strongest regional responses.
- **Variance pooling** — how much does activation fluctuate over time? Captures processing difficulty and cognitive effort.

### Region-specific analysis
- Emotional regions only (amygdala-adjacent, insula)
- Motor cortex only (for sensorimotor pairs)
- Prefrontal cortex only (working memory, executive function, cognitive load)
- Language network only (left temporal/frontal)

### Alternative similarity metrics
- **Correlation distance** — mean-centered cosine, may fix the high baseline issue
- **CKA (Centered Kernel Alignment)** — compares representational geometry between systems rather than individual pairs. Standard in DNN-to-brain comparison. [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414)
- **RSA (Representational Similarity Analysis)** — compare full dissimilarity matrices rather than individual pairs. The foundational method for relating brain and model representations. [Kriegeskorte et al. 2008](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full)

### Advanced methods from neuroimaging
- **Beta-series extraction** — fit a GLM to estimate one activation weight per vertex per stimulus, instead of raw time-series averaging. More principled because it accounts for the hemodynamic response shape. [Rissman et al. 2004](https://pubmed.ncbi.nlm.nih.gov/15488425/)
- **t-statistic maps** — instead of raw activation, compute where the brain responded *significantly* vs. baseline. Gives "where did the brain care?" rather than "how much did each vertex fire."
- **Connectivity fingerprinting** — compute correlations *between* brain regions during processing. Two texts might activate the same regions at different levels but show the same inter-region communication pattern. This is how individual brain fingerprinting was demonstrated. [Finn et al. 2015, Nature Neuroscience](https://www.nature.com/articles/nn.4135)
- **Voxel reliability weighting** — weight vertices by how reliably TRIBE predicts them, instead of treating all 20,484 equally. Noisy vertices get downweighted.

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

## 15) Real-world applications

### Content accessibility and readability
Current readability scores (Flesch-Kincaid, etc.) use surface features like word length and sentence length. Cognix could measure actual predicted cognitive processing demand. A patient information leaflet that triggers the same brain pattern as a dense legal contract is probably too complex for its audience — even if its Flesch-Kincaid score looks fine. Accessible writing tools could flag cognitively demanding passages regardless of surface-level simplicity.

### Education and adaptive learning
Two explanations of the same concept can have identical semantic content but wildly different processing demands. Cognix could match learning materials to a student's cognitive level by comparing the brain fingerprint of candidate materials against materials the student has successfully engaged with. This goes beyond topic matching (which semantic embeddings already do) to matching *how the brain handles the material*.

### Search and retrieval
Standard search finds documents about the same topic. Cognix could find documents that are *processed similarly* regardless of topic: "find me articles that are as cognitively demanding as this legal brief" or "find video clips with a similar attentional profile to this one." This is a new axis for retrieval that doesn't exist in any current search system.

### Mental health and content moderation
Emotional arousal pairs (war scenes vs. medical diagnoses) trigger similar brain distress patterns despite different topics. Keyword filters miss this cross-domain emotional similarity. Brain-grounded embeddings could identify content with high emotional arousal signatures without relying on topic-specific keyword lists, enabling more nuanced content warnings and moderation.

### Recommendation systems
Current recommenders suggest "more of the same topic." Brain-grounded embeddings could recommend based on cognitive engagement patterns: "You were deeply engaged by this thriller — here's a medical drama with a similar suspense-processing profile" instead of "here's another thriller." This captures the *experience* of consuming content, not just its subject.

### AI alignment and interpretability
Compare how language models represent text vs. how the human brain processes it. Cognix embeddings could serve as a human-grounded reference point for evaluating whether model representations align with human cognition. [Caucheteux & King (2022)](https://www.nature.com/articles/s42003-022-03036-1) showed that LLMs and brains partially converge — Cognix could quantify exactly where they diverge and what that means.

---

## 16) Research directions

### Investigating what brain space captures that semantic space doesn't
The Round 1 results show divergence, but we don't yet know precisely *what* the brain dimensions encode. Systematic probing — varying one cognitive property at a time (e.g., holding topic constant while varying emotional intensity, or holding complexity constant while varying sensorimotor content) — could map which brain dimensions correspond to which cognitive properties.

### Region-specific cognitive embeddings
Instead of one embedding from the whole brain, build separate embeddings from distinct cortical networks: an "emotional embedding" from limbic regions, a "motor embedding" from sensorimotor cortex, a "complexity embedding" from prefrontal cortex. These could be used independently or combined. This would let you query: "find text that *feels* similar" (emotional embedding) vs. "find text that's *similarly demanding*" (prefrontal embedding).

### Cross-modal cognitive similarity
TRIBE v2 supports video and audio. Do a thriller movie clip and a thriller novel excerpt produce similar brain fingerprints? If so, Cognix enables cross-modal retrieval: "find a podcast that engages the brain like this YouTube video." This is impossible with current embedding systems because CLIP-style models only capture semantic cross-modal alignment, not processing-level similarity.

### Validating against real fMRI data
TRIBE v2 produces *predicted* brain responses. How well do Cognix similarity rankings match similarity rankings computed from *real* fMRI data? Datasets like the Narratives collection (Nastase et al. 2021) have multiple subjects listening to the same stories — computing representational similarity from real fMRI and comparing to Cognix's predictions would ground-truth the entire approach.

### Individual differences in cognitive similarity
TRIBE v2 predicts population-average brain responses. But people differ — a lawyer might process legal text effortlessly while struggling with math, reversing the "cognitive load" similarity for those specific texts. Adapting Cognix to individual brain profiles (using TRIBE's subject-specific prediction capability) could create personalized cognitive embeddings.

### Temporal dynamics as a feature
Mean pooling discards temporal information. The *trajectory* of brain activation over time — how processing unfolds — may carry distinct similarity information. Two texts might produce similar average activation but very different temporal patterns (one builds slowly, the other spikes immediately). Preserving this temporal structure could capture suspense, surprise, and narrative arc as similarity dimensions.

### Brain-grounded evaluation of LLMs
Use Cognix as a benchmark: given two LLMs, which one's internal representations are more aligned with human brain processing patterns? This extends [Brain-Score (Schrimpf et al.)](https://www.biorxiv.org/content/10.1101/407007v2.full) from vision to language. An LLM whose representations better predict brain responses might be one that "understands" language more like humans do — with implications for safety and alignment.

### Cognitive complexity scoring
Instead of comparing pairs, extract a single scalar per text: how much total brain activation does this text produce? The L2 norm or variance of the brain vector could serve as a "cognitive complexity score" — a brain-grounded readability metric. This is simpler than the full embedding system and could be a standalone tool.

---

## 17) Limitations

- TRIBE predicts coarse fMRI-scale responses (~20,484 vertices), not individual neurons
- Predictions are population-level (averaged across subjects), not personalized
- Brain predictions are a transformation of LLaMA features — cannot capture properties LLaMA doesn't encode
- Mean-pooled vectors have high baseline similarity (~0.82), compressing useful dynamic range
- Not a direct measure of reward, happiness, or preference — any claims must be validated
- TRIBE inference is slow (~38 sec/text) and requires GPU
- Non-commercial license (CC BY-NC) inherited from TRIBE v2
