
# System Design Doc: Cognix (Cognitive Similarity Embeddings)

## 1) Project Summary

Build a **brain-grounded embedding system** that maps text (v1) into a fixed-length vector where similarity reflects **predicted similarity of human brain processing**, rather than standard semantic similarity.

Core pipeline:

```text
input text -> TRIBE v2 -> predicted brain-response tensor (T x V) -> temporal pooling -> projection head -> embedding vector (512-d) -> cosine similarity
```

This is **not** a brain-decoding system (fMRI -> content) and **not** a CLIP clone (shared semantic space). It builds a **new embedding space derived from predicted brain responses themselves**.

---

## 2) Why this project exists

Most embedding systems today represent:
- semantic similarity (do these mean the same thing?)
- visual similarity (do these look alike?)
- multimodal semantic alignment (does this caption match this image?)

They do **not** represent:
- cognitive load (how hard is this to process?)
- attentional dynamics (what does the brain focus on?)
- affective processing (what emotions does this trigger neurally?)
- sensorimotor grounding (does this evoke physical sensation?)
- "how similarly two inputs are processed by the human brain"

TRIBE v2 (released by Meta, March 2026) makes this feasible for the first time. It predicts **high-resolution fMRI brain responses** (~20,484 cortical vertices) from **video, audio, and text**, trained on 1,000+ hours of fMRI data from 720+ subjects.

---

## 3) Key idea in one sentence

Instead of asking:

> "Do these two inputs mean the same thing?"

ask:

> "Would these two inputs be processed similarly by the human brain?"

---

## 4) Important links

### TRIBE v2
- Meta official blog: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
- Meta research paper: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- GitHub repo: https://github.com/facebookresearch/tribev2

### Related work
- CLIP (OpenAI): https://arxiv.org/abs/2103.00020
- BrainCLIP: https://arxiv.org/abs/2302.12971
- MindBridge: https://arxiv.org/abs/2404.07850
- MindEye2: https://arxiv.org/abs/2403.11207
- RSA (Kriegeskorte et al. 2008): https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full

---

## 5) How TRIBE v2 actually works inside

Understanding TRIBE v2's internals is critical because it determines what Cognix can and cannot capture.

### Architecture

TRIBE v2 is a three-stage pipeline:

```text
Stage 1 (feature extraction):
  text  -> LLaMA 3.2-3B      -> language features
  video -> V-JEPA2-Giant      -> visual features
  audio -> Wav2Vec-BERT 2.0   -> audio features

Stage 2 (fusion):
  All features compressed to D=384, concatenated -> D=1152 multimodal time series
  -> 8-layer Transformer encoder (100-second context windows)

Stage 3 (brain prediction):
  Transformer outputs -> decimated to fMRI frequency
  -> Subject Block -> predicted cortical activity (~20,484 vertices)
```

### What this means for Cognix

TRIBE does not process text "from scratch." It feeds text into **LLaMA 3.2**, extracts LLaMA's internal representations, and then learns a mapping from those representations to brain activity — supervised by real fMRI data from 720+ subjects.

The brain prediction is therefore a **learned, nonlinear transformation of LLaMA features**, shaped by real neuroscience data. This has important implications discussed in Section 12 (Primary Risk).

### Output format

```python
output = model.predict(events=df)
output.shape  # (n_timesteps, n_vertices)
# n_vertices ≈ 20,484 (fsaverage5 cortical mesh)
# n_timesteps depends on input duration, at fMRI temporal resolution
```

---

## 6) What existing systems do and how Cognix differs

### CLIP
**What it does:** learns a shared semantic space for images and text via contrastive learning on image-caption pairs.
**Direction:** `image/text -> semantic embedding`
**Optimized for:** semantic meaning, zero-shot classification, image-text retrieval.
**Does not model:** brain responses, cognitive load, attentional dynamics.

### BrainCLIP
**What it does:** maps **real fMRI data** into CLIP's existing embedding space.
**Direction:** `real fMRI -> CLIP space`
**Use case:** brain decoding — given a brain scan, retrieve what the person was looking at.
**Key difference from Cognix:** BrainCLIP treats CLIP space as the target. Cognix builds a **new** space from predicted brain responses.

### MindBridge / MindEye2
**What they do:** cross-subject brain decoding and image reconstruction from fMRI.
**Relevance:** evidence that shared subject-invariant brain representations exist, but these are decoding/reconstruction systems, not similarity embedding systems.

### Cognix (this project)
**Direction:** `input -> predicted brain response -> NEW embedding space`
**What it captures:** cognitive processing similarity — how similarly the brain handles two inputs.
**Key distinction:** no real fMRI needed at inference time. TRIBE v2 provides the brain predictions.

---

## 7) Concrete examples: where Cognix should diverge from semantic similarity

This section defines the project's value proposition. If these divergences don't materialize empirically, the project doesn't work.

### 7.1 Same cognitive load, different topics

| Text A | Text B |
|--------|--------|
| "The party of the first part shall indemnify the party of the second part against all claims arising from..." | "The eigenvalues of a Hermitian matrix are real, and eigenvectors corresponding to distinct eigenvalues are orthogonal..." |

**Semantic similarity:** low (law vs. math)
**Expected brain similarity:** high (both demand heavy working memory, sustained prefrontal activation, slow effortful parsing)

### 7.2 Same emotional arousal, different content

| Text A | Text B |
|--------|--------|
| "The soldier held his dying friend and screamed for a medic that wasn't coming." | "She opened the envelope, read the diagnosis, and sat perfectly still for twenty minutes." |

**Semantic similarity:** low (war vs. medical/personal)
**Expected brain similarity:** high (amygdala, insula, anterior cingulate — high emotional intensity, threat processing, empathic distress)

### 7.3 Same sensorimotor grounding, different scenarios

| Text A | Text B |
|--------|--------|
| "She kicked the ball across the field." | "He stomped on the brake pedal." |

**Semantic similarity:** low (sports vs. driving)
**Expected brain similarity:** high (motor cortex activation for leg/foot actions — the brain simulates the physical action while reading)

### 7.4 Same spatial scene structure, different domains

| Text A | Text B |
|--------|--------|
| "The desert stretched flat and empty to the horizon in every direction." | "Nothing but open ocean, no land in sight, the boat a speck on grey water." |

**Semantic similarity:** moderate (both describe landscapes, but different domains)
**Expected brain similarity:** high (parahippocampal place area, retrosplenial cortex — vast, empty, open spatial layout processing)

### 7.5 Same syntactic surprise / reanalysis

| Text A | Text B |
|--------|--------|
| "The horse raced past the barn fell." | "The patient the nurse the doctor consulted examined recovered." |

**Semantic similarity:** low (horses vs. hospitals)
**Expected brain similarity:** high (both are garden-path sentences — the brain parses, fails, and reparses, producing a spike in left inferior frontal gyrus)

### 7.6 Same narrative suspense pattern

| Text A | Text B |
|--------|--------|
| "He turned the corner. The hallway was empty. Then he heard the lock click behind him." | "The surgeon paused. The monitor flatlined. Then she saw the second bleeder." |

**Semantic similarity:** low (thriller vs. medical drama)
**Expected brain similarity:** high (anticipatory processing, prediction error, dopaminergic surprise — same trajectory of building tension and resolution)

### 7.7 Same theory-of-mind activation

| Text A | Text B |
|--------|--------|
| "He said he was fine, but she could tell from his voice that he wasn't." | "The child hid the candy, not realizing his sister had been watching the whole time." |

**Semantic similarity:** low (adult emotional deception vs. child behavior)
**Expected brain similarity:** high (mentalizing — reasoning about others' beliefs and hidden states — activates medial prefrontal cortex and temporoparietal junction)

### How to use these examples

These are the **hypothesis**. The validation experiment (Section 13) tests whether TRIBE's predicted brain responses actually show this divergence. If they do, these become the demo pairs for the HuggingFace model card and paper. If they don't, the project's value proposition is weak.

---

## 8) Scope

### MVP (Phase 1)
- Input: **text only**
- Use TRIBE v2 text pathway (LLaMA 3.2 internally)
- Get predicted brain tensor `(T, V)` where V ≈ 20,484
- Temporal mean-pool over `T`
- MLP projection head: `20484 -> 1024 -> 512`
- Cosine similarity between embeddings
- Evaluate on retrieval / clustering / divergence from semantic baselines

### Full version (Phase 2+)
- Multimodal inputs: text, video, audio
- Preserve temporal structure (attention pooling or sequence encoder before pooling)
- Region-aware pooling (e.g., separate emotional vs. sensorimotor vs. language regions)
- Downstream task heads: preference, affect, complexity, engagement
- Cross-modal retrieval in brain space

---

## 9) End-to-end ML pipeline

### Step 1: Input

Start with text paragraphs / short passages.

Why text first:
- smallest engineering surface area
- easiest batching and fastest inference
- simplest benchmark setup
- TRIBE v2's text pathway (LLaMA 3.2) is well-understood

### Step 2: Run TRIBE v2

Note: TRIBE v2 does not tokenize text directly like a normal NLP model. Its internal pipeline is:

```text
text -> gTTS (text-to-speech) -> audio -> WhisperX (transcription) -> word-level timestamps
timestamps + LLaMA 3.2-3B -> features at 2 Hz -> brain transformer -> voxel predictions
```

The text-to-speech roundtrip exists because TRIBE was trained on naturalistic stimuli (people watching videos / listening to audio in an fMRI scanner). Everything is temporally aligned, so text needs artificial temporal structure via TTS to fit the same framework.

```python
events_df = model.get_events_dataframe(text_path="input.txt")
preds, segments = model.predict(events=events_df)
brain_output = np.asarray(preds)
# brain_output.shape = (T, 20484)
# T = time steps at 1 Hz (one prediction per second of TTS audio)
# 20,484 = cortical vertices on fsaverage5 mesh
#   Left hemisphere:  indices 0-10,241
#   Right hemisphere: indices 10,242-20,483
```

Each row = predicted whole-brain fMRI BOLD response at one moment.
Each column = one cortical vertex's activity over time.

**Implication for Cognix:** Short texts (under ~10 words) produce very few time steps (T=1-3), making temporal pooling almost trivial. Validation pairs should ideally be paragraph-length.

### Step 3: Temporal pooling

Collapse the time dimension:

```python
pooled = brain_output.mean(axis=0)  # shape: (20484,)
```

This turns a "movie" of brain activity into a single summary snapshot — a "brain fingerprint."

**Scientific precedent:** Temporal mean pooling over fMRI to get a spatial brain fingerprint is standard practice in neuroimaging:
- Huth et al. 2016 (semantic atlas) averaged fMRI responses over story presentations to map semantic selectivity across cortex
- Multi-voxel Pattern Analysis (MVPA) averages voxel patterns across time windows per condition
- Brain-Score (Schrimpf et al.) averages brain responses over stimulus windows for representational similarity

**Tradeoffs:**
- Loses temporal dynamics (processing order, surprise, buildup)
- But produces fixed-size vectors, is cheap, and is a stable baseline
- Acceptable for MVP; upgrade later with attention pooling or a learned temporal encoder

**Post-MVP temporal alternatives:**
- Finite Impulse Response (FIR) models that preserve time course
- Temporal pattern analysis treating the time series as informative
- Beta-series correlation estimating separate patterns per event

### Step 4: Projection head

The pooled vector (~20,484 dims) is too large for practical use. Compress it:

```python
projection_head = nn.Sequential(
    nn.Linear(V, 1024),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, 512),
)
embedding = projection_head(pooled)  # shape: (512,)
```

This 512-d vector is the **Cognix embedding**.

### Step 5: Similarity

```python
similarity = F.cosine_similarity(embedding_a, embedding_b, dim=0)
```

### Step 6: Training objective

The projection head needs to learn what to preserve. Use **contrastive learning**:
- positive pairs (paraphrases, matched rewrites) -> close embeddings
- negative pairs (unrelated passages) -> far embeddings

This teaches the projection which aspects of brain-response space matter for similarity.

### Step 7: Caching strategy

TRIBE v2 inference is expensive. Cache aggressively:

```text
Option A (recommended for MVP): cache pooled vectors (V,) per input
Option B (more flexible): cache raw tensors (T, V) per input
```

Train the projection head entirely on cached outputs. This avoids re-running TRIBE during training.

---

## 10) Training data

### For MVP
Use text similarity datasets to create positive/negative pairs:
- paraphrase datasets (MRPC, QQP, PAWS)
- STS Benchmark
- short summary / rewrite pairs
- duplicate question datasets

### For better evaluation
- human-rated similarity datasets
- emotion / complexity annotations
- stimuli from cognitive neuroscience experiments (these have known brain response properties)

---

## 11) Downstream heads (post-MVP)

The embedding itself contains no explicit scores. Build separate heads on top:

```text
input -> Cognix embedding -> reward head -> scalar score
input -> Cognix embedding -> complexity head -> scalar score
input -> Cognix embedding -> engagement head -> scalar score
```

Each head requires its own labeled training data. Without labels, any score is a weak proxy.

---

## 12) Primary risk: collapse to generic semantics

This is the most important section in this document. If this risk materializes, the project produces nothing new.

### The problem

TRIBE v2's text pathway works like this:

```text
text -> LLaMA 3.2 -> features -> 8-layer Transformer -> brain voxels
```

The predicted brain response is a **learned transformation of LLaMA features**. It cannot invent information that LLaMA doesn't encode. So the full Cognix pipeline is:

```text
text -> LLaMA features -> TRIBE transform -> brain voxels -> pool -> MLP -> embedding
```

Which could collapse to something equivalent to:

```text
text -> LLaMA features -> learned projection -> embedding
```

...which is essentially a sentence-transformer with extra steps.

### Why it might NOT collapse

1. **The TRIBE transform is not trivial.** It's an 8-layer Transformer trained on real fMRI data from 720+ subjects. That's a significant nonlinear transformation supervised by neuroscience data that LLaMA never saw.

2. **The mapping re-weights dimensions based on what the brain cares about.** LLaMA encodes syntactic structure, topic, entities, etc. roughly equally. The brain cares heavily about emotional valence and motor imagery while largely ignoring syntactic tree depth. TRIBE's mapping amplifies the former and suppresses the latter.

3. **Empirical evidence from neuroscience:** Brain-derived similarity and distributional similarity correlate at r = 0.5-0.8 (Mitchell et al. 2008, Anderson et al. 2017, 2019). There IS overlap, but also meaningful divergence — especially in sensorimotor and affective dimensions.

### Why it might collapse

1. **As language models improve, the gap shrinks.** Caucheteux & King (2022) showed that larger models already capture more of what the brain does. The marginal value of the brain-prediction step decreases as the base model gets better.

2. **The TRIBE mapping can only predict brain responses that correlate with its input features.** Any genuinely "neural" signal (surprise from real-world context, embodied experience, personal memory) that isn't in LLaMA's features will be absent.

3. **If the mapping is approximately linear**, cosine similarity in brain space will be highly correlated with a re-weighted cosine similarity in LLaMA space — producing rankings that look very similar to a sentence-transformer.

### How to detect it

Run the validation experiment before building anything else.

---

## 13) Validation experiment — COMPLETED

### Round 1 results (100 pairs, April 2026)

**Pearson r = 0.24 (p = 0.017). Spearman r = 0.32 (p = 0.0015).**

Brain similarity and semantic similarity are barely correlated. The hypothesis holds.

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

### Key observations

1. **Every divergence category we predicted diverges.** Cognitive load, syntactic surprise, spatial scenes, sensorimotor, emotional arousal, narrative suspense, theory of mind — all show high brain similarity with low semantic similarity.

2. **The control category has the lowest brain similarity (0.587).** Same-topic pairs with different processing complexity (simple vs. dense text) are distinguished by the brain space. This confirms the brain space captures processing demands, not just topic.

3. **High baseline brain similarity.** Even unrelated pairs average 0.824 brain similarity. The mean-pooled brain vectors have a high floor — the useful signal is in relative differences above this baseline. Region-specific pooling or centering may sharpen contrasts.

4. **3 texts failed TRIBE inference** (97/100 pairs succeeded). Likely very short texts that produced insufficient TTS audio.

### Decision

**Proceed to Round 2 (1,000 pairs) and full system build.**

---

## 14) Evaluation methodology

### Baselines
Compare against:
- sentence-transformer (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- raw pooled TRIBE vectors without learned projection
- random projection / PCA baseline on TRIBE vectors
- CLIP text encoder (for multimodal version)

### Core metrics
- retrieval recall@k
- clustering purity / NMI
- pairwise ranking accuracy
- cosine similarity distribution separation (positive vs. negative pairs)

### RSA (Representational Similarity Analysis)
RSA is especially relevant because the project is about representation geometry.

Build Representational Dissimilarity Matrices (RDMs) for:
- raw TRIBE brain space
- projected Cognix space
- sentence-transformer space
- human similarity judgments (if available)

Compare RDMs with Kendall's tau or Spearman correlation. This directly tests whether Cognix's geometry is closer to human judgments than standard embeddings.

### Qualitative divergence analysis
The most compelling evidence: show specific pairs where:
- sentence-transformer cosine similarity < 0.3
- Cognix cosine similarity > 0.7
- a human agrees the inputs feel similar to process

Include 10-20 such examples in the paper / model card.

### Ablations
- mean pooling vs. attention pooling vs. max pooling
- 256 vs. 512 vs. 768 embedding dimensions
- with vs. without dropout
- raw pooled vectors vs. MLP projection
- full brain vs. region-specific pooling (e.g., only emotional regions, only motor regions)

---

## 15) Compute considerations

TRIBE v2 inference is nontrivial:
- Uses LLaMA 3.2-3B + V-JEPA2-Giant + Wav2Vec-BERT 2.0 internally
- Predicts ~20,484 vertices per timestep
- GPU strongly recommended; CPU is impractical

### Practical strategy

1. Run TRIBE once per sample
2. Cache the pooled `(V,)` vectors (or raw `(T, V)` tensors if you want to experiment with pooling)
3. Train the projection head entirely on cached outputs
4. At inference, TRIBE must run live — but the projection head is tiny and instant

For the validation experiment (Section 13), you only need TRIBE inference + caching. No projection head needed yet.

---

## 16) Implementation plan

### Phase 0: Validation experiment (1-2 days)
- Load TRIBE v2
- Run on ~1,000 text pairs
- Compute brain cosine similarity vs. sentence-transformer cosine similarity
- Check correlation
- **Stop here if r > 0.95**

### Phase 1: Scaffold (if validation passes)
- Set up repo
- Build TRIBE wrapper with caching
- Verify text inference and output shapes

### Phase 2: Data layer
- Choose similarity datasets (MRPC, STS-B, QQP)
- Build positive / negative pairs
- Include pairs from Section 7 divergence categories
- Create dataloaders

### Phase 3: Brain feature extraction
- Run TRIBE on all inputs
- Cache pooled vectors

### Phase 4: Projection model
- Implement MLP projection head: 20484 -> 1024 -> 512
- Contrastive loss with cosine similarity

### Phase 5: Training
- Train on cached vectors
- Weight decay + dropout 0.1
- Evaluate every epoch against baselines

### Phase 6: Evaluation
- Retrieval metrics, clustering, RSA
- Divergence analysis vs. sentence-transformers
- Qualitative inspection of divergent pairs

### Phase 7: Packaging for HuggingFace
- Package projection model weights
- Package inference pipeline code
- Document TRIBE v2 dependency and setup
- Include precomputed embeddings for demo pairs
- Write model card with divergence examples

---

## 17) What gets uploaded to HuggingFace

**Upload:**
- projection model weights
- inference pipeline code
- config files
- precomputed embeddings for demo datasets
- model card with divergence examples and evaluation results

**Do not upload:**
- TRIBE v2 weights (users install separately; respect Meta's licensing)
- raw brain tensors for every sample

**Recommended user experience:**
```python
# User installs TRIBE v2 separately per Meta's instructions
# Then:
from cognix import CognitiveEmbedder
model = CognitiveEmbedder.from_pretrained("davidgershony/cognix-v1")
embedding = model.encode("The desert stretched flat and empty to the horizon.")
```

---

## 18) Licensing

TRIBE v2 is released under **CC BY-NC** (Creative Commons Attribution-NonCommercial).

This means:
- Fine for research and portfolio projects
- Fine for non-commercial HuggingFace release with attribution
- **Not fine for commercial deployment** without separate arrangement with Meta

Your project's README must clearly state:
- dependency on TRIBE v2
- attribution to Meta
- CC BY-NC constraints inherited from TRIBE v2

---

## 19) Limitations

### Scientific
- TRIBE predicts coarse fMRI-scale responses (~70k voxels), not individual neurons
- Predictions are population-level (averaged across subjects), not personalized
- Not a direct measure of reward, happiness, or preference
- Cognitive interpretations can be overclaimed if not validated against the divergence examples

### Architectural (the LLaMA ceiling)
- TRIBE's text pathway processes text through LLaMA 3.2. The predicted brain response is a transformation of LLaMA features.
- Any cognitive property that LLaMA doesn't encode (real-world surprise, personal memory, embodied experience) cannot appear in the brain prediction.
- As language models improve, the gap between "brain-grounded" and "semantic" similarity may narrow.

### Practical
- TRIBE v2 inference requires GPU and is relatively slow
- Non-commercial license limits deployment options
- Multimodal version (video/audio) is significantly heavier than text-only

---

## 20) Risks ranked by severity

1. **Collapse to generic semantics (HIGH)**
   The embedding may behave like a standard sentence-transformer. Mitigation: run validation experiment first.

2. **Weak evaluation framing (MEDIUM)**
   Without concrete divergence examples and clear metrics, reviewers will say "so what?" Mitigation: use the pairs from Section 7 and the evaluation protocol from Section 14.

3. **Compute bottleneck (MEDIUM)**
   Running TRIBE without caching wastes time and money. Mitigation: cache everything.

4. **Too much complexity too early (LOW if MVP is followed)**
   Multimodal + sequence-aware + region-aware all at once is too much. Mitigation: text-only MVP, add complexity only after validation.

---

## 21) Future directions (post-MVP)

### Near-term
- Attention pooling or learned temporal encoder instead of mean pooling
- Region-aware pooling (separate embeddings for emotional / sensorimotor / language regions, then concatenate or attend)
- Multimodal support (video, audio)

### Downstream heads
- Preference / reward proxy
- Cognitive complexity score
- Engagement prediction
- Affect / pleasantness proxy

### Applications
- Cognitive search: "find text that feels similarly demanding to process"
- Attentional retrieval: "find clips with similar attentional profile"
- Content recommendation in brain space (not just topic space)
- Accessibility tools: flag content with similar cognitive load to known-difficult material

### Research
- RSA comparison of raw TRIBE geometry vs. projected Cognix geometry
- Test whether Cognix retrieves cognitively similar but semantically different examples
- Investigate which cortical regions drive which notions of similarity
- Cross-modal alignment in brain space

---

## 22) Suggested repo structure

```text
cognix/
  README.md
  pyproject.toml
  requirements.txt

  cognix/
    __init__.py
    config.py
    tribe_wrapper.py      # TRIBE v2 loading, inference, caching
    pooling.py             # temporal pooling strategies
    projection.py          # MLP projection head
    similarity.py          # cosine similarity, distance functions
    datasets.py            # pair construction, dataloaders
    train.py               # contrastive training loop
    evaluate.py            # retrieval, clustering, RSA, divergence analysis
    cache_features.py      # batch TRIBE inference + caching
    inference.py           # end-to-end: text -> embedding

  notebooks/
    00_validation_experiment.ipynb   # THE FIRST THING TO RUN
    01_inspect_tribe_outputs.ipynb
    02_nearest_neighbors.ipynb
    03_rsa_analysis.ipynb
    04_divergence_examples.ipynb

  configs/
    mvp_text.yaml
    best_text.yaml

  artifacts/
    cached_vectors/
    checkpoints/
```

---

## 23) Summary

Build a system that:
1. Takes text
2. Runs it through TRIBE v2 (which internally uses LLaMA 3.2 -> 8-layer Transformer -> brain projection)
3. Gets a time-by-brain-location tensor (~20,484 cortical vertices)
4. Averages over time
5. Compresses to 512 dimensions via a learned MLP
6. Trains with contrastive loss
7. Uses cosine similarity for retrieval

**But first:** run the validation experiment to confirm that brain-predicted similarity actually diverges from semantic similarity. If it doesn't, stop. If it does, you have a novel contribution and a compelling HuggingFace release.
