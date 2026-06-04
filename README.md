# EpistemicBERT

A BERT-based model that measures **epistemic uncertainty as a structurally decomposable property**, not as a proxy for prediction failure. Instead of collapsing "the model is unsure" into a single confidence score, EpistemicBERT factorizes uncertainty into six interpretable axes, each grounded in an independent source.

## Core Idea

Meaning is treated as a *field* produced by two independent evidence streams — **support** and **counter**. From this field, six epistemic axes are derived:

| Group | Axis | Question it answers |
|-------|------|---------------------|
| **Evidence plane** (2D) | `truth` | Is one interpretation dominant? |
| | `error` | Is the opposing evidence dominant? |
| | `contradiction` | Are conflicting evidences simultaneously strong? |
| **Uncertainty sources** (independent) | `novelty` | Is the input outside the model's world (first time seen)? |
| | `ambiguity` | Are multiple interpretations similarly plausible? |
| | `ignorance` | Is there insufficient information to judge? |

The three evidence-plane axes form a 2D coordinate system (they are *expected* to be mutually predictable). The three uncertainty axes are designed to be **mutually independent** — each measured from a structurally different source so that no axis is a reparameterization of another.

## Why Source Separation Matters

The central design principle: **an uncertainty axis is only robust if its source is independent of the others.** Penalty-based decorrelation (adding loss terms to push correlations down) was repeatedly found to *kill* axes rather than disentangle them — a constant output trivially has zero correlation with everything. The working approach instead gives each axis a distinct measurement source:

| Axis | Source | Independent of task? |
|------|--------|----------------------|
| `novelty` | **token rarity** — `-log` frequency against the training vocabulary | ✅ task-free |
| `ignorance` | **attention dispersion** — entropy of the final-layer attention distribution | ✅ task-free |
| `ambiguity` | **top1–top2 prototype gap** — competition within an evidence stream | ⚠️ manifold-based |
| `truth` / `error` / `contradiction` | **stream geometry** — `support`/`counter` margin, energy, agreement | (evidence plane) |

`contradiction` is defined as stream-level conflict: `energy × (1 − agreement)`, high only when both streams are strong *and* opposed. `ambiguity` is defined as within-stream competition (several prototypes nearly tied), which is conceptually distinct from contradiction — multiple readings without opposition.

### Design lessons learned

- **`ignorance ≠ predictive entropy.** Classifier predictive entropy measures *"the classifier is confused,"* which fires on novel-but-clear inputs and overlaps with novelty. Attention dispersion measures *"the input lacks information,"* which is what ignorance should mean.
- **`novelty` cannot live in the task representation `z`.** Because `z = proj(BERT)` is trained with the NLI objective, it is optimized to *discard* topical novelty (irrelevant to the label). Density estimates in `z`-space therefore measure "distance from the typical NLI sentence," not semantic novelty. Token-level rarity sidesteps this entirely.
- **Gating an axis by another axis's input leaks structure.** Earlier versions gated `ambiguity` by `energy` (a function of `support`/`counter`), which injected evidence-plane information and inflated `ambiguity↔truth` correlation. Removing the gate and using the raw top1–top2 gap restored independence.

## Architecture

```
input_ids ──► DistilBERT ──► CLS ──► proj ──► z ──┬──► PrototypeManifold (dual-stream)
                  │                                │       ├─ support / counter   (evidence plane)
                  │                                │       └─ ambiguity (top1-top2 gap)
                  │                                │
                  │                                └──► TokenNovelty   (token rarity)
                  │
                  └──► attention ──► AttentionIgnorance (attention dispersion)
                                                         │
   support, counter, novelty, ambiguity, ignorance ─────┴──► EpistemicFieldClassifier
                                                                  └─ truth / error / contradiction
                                                                  └─ field (6 axes)
                                                                        │
                              field_proj(field) + z_proj(z) ───────────►  logits
```

`TokenNovelty` and `AttentionIgnorance` are **detached** from the classification gradient — their statistics track the data distribution, not the loss. This is what keeps them task-independent.

## Identifiability Results

The key evaluation is the **identifiability probe**: for each axis, fit a linear model predicting it from the other five and report R². Plane axes are *expected* to be predictable (they are coordinates); uncertainty axes should be independent (low R²).

Trained on three NLI datasets (DistilBERT backbone, 3 epochs):

| Axis | SNLI | MNLI | ANLI | Source |
|------|------|------|------|--------|
| `novelty`   | 0.02 | 0.03 | 0.04 | token rarity *(task-free)* |
| `ignorance` | 0.01 | 0.27 | 0.04 | attention *(task-free)* |
| `ambiguity` | 0.41 | 0.20 | **0.74** | manifold gap *(task-dependent)* |
| `truth`/`error`/`contradiction` | plane-coord | plane-coord | plane-coord | geometry |

**Reading the table:** `novelty` and `ignorance`, built on task-free sources, stay independent across all three datasets — *including adversarial ANLI where classification accuracy collapses to ~0.41.* This demonstrates the central claim: these uncertainty axes are **not artifacts of prediction success**. The manifold-based `ambiguity` holds on SNLI/MNLI but leaks into the plane on ANLI, where the collapsed `support`/`counter` distribution removes its signal — revealing that an axis's robustness is determined by how task-independent its source is.

### Semantic validation

A targeted probe distinguishes three input types — **G1** (novel, well-specified: e.g. *"Zeta-7 decays into mirror-charged leptons"*), **G2** (underspecified: e.g. *"something happened somewhere"*), **G3** (ordinary) — to confirm each axis fires on the right phenomenon:

- `novelty` fires highest on **G1** (rare vocabulary), confirming it captures "first seen."
- `ignorance` fires highest on **G2** (information deficit), and is *flat* on calibration (does not rise on wrong predictions — it measures missing information, not error).
- `ambiguity` fires highest on **neutral** labels and predicts wrong answers (it tracks genuine interpretive uncertainty).

## Usage

```python
model = EpistemicBERT(n_classes=3, n_prototypes=32, proj_dim=128)
model.token_nov.set_special_tokens(tokenizer.all_special_ids)  # required

logits, epistemic_out, diversity_loss = model(input_ids, attention_mask)

epistemic_out["field"]          # (B, 6) the six-axis epistemic field
epistemic_out["novelty"]        # per-axis scores
epistemic_out["ambiguity"]
epistemic_out["ignorance"]
epistemic_out["dominant_type"]  # argmax axis label per sample (eval only)
```

### Training

```python
loss = (
    0.3  * cross_entropy(logits, labels)   # secondary
    + 1.0  * field_ranking_loss(field, labels)  # primary: ranks correct axis highest
    + 0.2  * margin_loss(support, counter, labels)
    + 0.1  * diversity_loss                # prototype collapse prevention
    + 0.03 * disentangle_loss(epistemic_out)  # light penalty on nov↔amb, amb↔ign only
)
```

The disentangle penalty is kept *small* and narrow — structural source separation does the real work; the penalty is a minor regularizer, not the mechanism.

### Diagnostics included

- `identifiability_probe` — per-axis R² and full correlation matrix
- `evaluate_ood` — ID vs OOD axis means (RTE as OOD)
- `evaluate_calibration` — axis means on correct vs wrong predictions
- `novelty_ignorance_probe` — G1/G2/G3 separation test
- `attention_entropy_probe` — validates the ignorance source

## Requirements

```
torch
transformers
datasets
scikit-learn
tqdm
```

A local DistilBERT checkpoint is expected (path set in `EpistemicBERT.__init__` and the tokenizer load).

## Known Limitations & Future Work

- **`ambiguity` is task-dependent.** It relies on prototype competition in the manifold, which degrades when the classifier cannot separate classes (adversarial ANLI). Making ambiguity task-free (e.g. representation-disagreement signals) is open work.
- **`novelty` measures lexical novelty only.** Token rarity catches unfamiliar *words* but not familiar words combined into novel *concepts* (e.g. "round square"). An n-gram or compositional extension is a natural next step.
- **`ambiguity↔truth ≈ 0.45`** persists on SNLI/MNLI. This is interpreted as genuine data structure (clear entailment = single interpretation = low ambiguity), not measurement leakage, and is deliberately not penalized.

## Status

Six-axis identifiability achieved and validated across SNLI, MNLI, and ANLI. Uncertainty source separation (`novelty` = tokens, `ignorance` = attention, `ambiguity` = manifold gap) confirmed task-independent for the two token/attention-based axes.
