# EpistemicBERT

A BERT-based model that measures **epistemic uncertainty as a structurally decomposable property**, not as a proxy for prediction failure. Instead of collapsing “the model is unsure” into a single confidence score, EpistemicBERT factorizes uncertainty into six interpretable axes, each grounded in an independent source.

> **v2 (current).** All three uncertainty axes are now sourced from task-independent signals (token / layer / attention) and stay identifiable across SNLI, MNLI, **and adversarial ANLI** — even where classification accuracy collapses to ~0.41. This resolves v1’s one remaining weakness, where the manifold-based `ambiguity` leaked into the evidence plane on ANLI.

## Core Idea

Meaning is treated as a *field* produced by two independent evidence streams — **support** and **counter**. From this field, six epistemic axes are derived:

|Group                                |Axis           |Question it answers                                      |
|-------------------------------------|---------------|---------------------------------------------------------|
|**Evidence plane** (2D)              |`truth`        |Is one interpretation dominant?                          |
|                                     |`error`        |Is the opposing evidence dominant?                       |
|                                     |`contradiction`|Are conflicting evidences simultaneously strong?         |
|**Uncertainty sources** (independent)|`novelty`      |Is the input outside the model’s world (first time seen)?|
|                                     |`ambiguity`    |Are multiple interpretations similarly plausible?        |
|                                     |`ignorance`    |Is there insufficient information to judge?              |

The three evidence-plane axes form a 2D coordinate system (they are *expected* to be mutually predictable). The three uncertainty axes are designed to be **mutually independent** — each measured from a structurally different source so that no axis is a reparameterization of another.

## Why Source Separation Matters

The central design principle: **an uncertainty axis is only robust if its source is independent of the others — and of the task itself.** Penalty-based decorrelation (adding loss terms to push correlations down) was repeatedly found to *kill* axes rather than disentangle them — a constant output trivially has zero correlation with everything. The working approach instead gives each axis a distinct, task-free measurement source:

|Axis                               |Source                                                                               |Task-independent?            |
|-----------------------------------|-------------------------------------------------------------------------------------|-----------------------------|
|`novelty`                          |**token rarity** — `-log` frequency against the training vocabulary                  |yes — token statistics       |
|`ambiguity`                        |**layer disagreement** — directional variance of `[CLS]` across the final BERT layers|yes — representation geometry|
|`ignorance`                        |**attention dispersion** — entropy of the final-layer attention distribution         |yes — attention structure    |
|`truth` / `error` / `contradiction`|**stream geometry** — `support`/`counter` margin, energy, agreement                  |(evidence plane)             |

`contradiction` is defined as stream-level conflict: `energy x (1 - agreement)`, high only when both streams are strong *and* opposed. `ambiguity` is defined as non-convergence of interpretation — several BERT layers pointing `[CLS]` in different directions — which is conceptually distinct from contradiction (multiple readings *without* opposition).

### v1 -> v2: making ambiguity task-free

In v1, `ambiguity` was the top1-top2 prototype gap inside the dual-stream manifold. Because the manifold is built from the classification signal, `ambiguity` degraded when the classifier could not separate classes:

```
ambiguity identifiability (R2, lower = more independent)
                    SNLI    MNLI    ANLI
v1 (manifold gap)   0.41    0.20    0.74   <- collapses on adversarial ANLI
v2 (layer disagree) 0.42    0.25    0.16   <- holds across all three
```

Layer-to-layer representation disagreement is *orthogonal to classification success*: even when the final layer cannot decide a label (ANLI, acc ~ 0.41), whether the intermediate layers converge is a separate signal. Moving `ambiguity` to this source kept it independent on ANLI without touching SNLI/MNLI behavior.

A side effect: with `ambiguity` removed from the manifold, `contradiction` now owns the stream geometry outright and separates much more sharply by label (e.g. on SNLI, `contradiction` reaches ~0.93 on contradiction-labeled inputs vs ~0.33 elsewhere, with `truth<->contradiction` correlation near 0).

### Design lessons learned

- **`ignorance` != predictive entropy.** Classifier predictive entropy measures *“the classifier is confused,”* which fires on novel-but-clear inputs and overlaps with novelty. Attention dispersion measures *“the input lacks information,”* which is what ignorance should mean.
- **`novelty` cannot live in the task representation `z`.** Because `z = proj(BERT)` is trained with the NLI objective, it is optimized to *discard* topical novelty (irrelevant to the label). Density estimates in `z`-space therefore measure “distance from the typical NLI sentence,” not semantic novelty. Token-level rarity sidesteps this entirely.
- **`ambiguity` cannot live in the manifold.** Same root cause — the manifold is a function of the classification signal, so a manifold-based axis inherits task dependence and breaks under distribution shift / adversarial inputs. Layer disagreement is the task-free replacement.
- **Bigger backbones do not fix task-dependent sources.** Swapping DistilBERT for RoBERTa-base *raised* `ambiguity`’s ANLI leakage (0.74 -> 0.83) despite higher train accuracy — confirming the leak was a source-design problem, not a capacity problem.

## Architecture

```
input_ids -> DistilBERT -> CLS -> proj -> z --+--> PrototypeManifold (dual-stream)
               |  |                           |       support / counter   (evidence plane)
               |  |                           |
               |  |                           +--> TokenNovelty   (token rarity)
               |  |
               |  +--> hidden_states -> LayerAmbiguity (layer disagreement)
               |
               +--> attention -> AttentionIgnorance (attention dispersion)
                                                     |
  support, counter, novelty, ambiguity, ignorance --+--> EpistemicFieldClassifier
                                                            truth / error / contradiction
                                                            field (6 axes)
                                                                  |
                       field_proj(field) + z_proj(z) ------------>  logits
```

`TokenNovelty`, `LayerAmbiguity`, and `AttentionIgnorance` are all **detached** from the classification gradient — their statistics track the data distribution, not the loss. This is what keeps them task-independent.

## Identifiability Results (v2)

For each axis, fit a linear model predicting it from the other five and report R2. Plane axes are *expected* to be predictable (coordinates); uncertainty axes should be independent (low R2).

|Axis                           |SNLI       |MNLI       |ANLI       |Source              |
|-------------------------------|-----------|-----------|-----------|--------------------|
|`novelty`                      |0.03       |0.06       |0.10       |token rarity        |
|`ambiguity`                    |0.42       |0.25       |0.16       |layer disagreement  |
|`ignorance`                    |0.01       |0.08       |0.08       |attention dispersion|
|`truth`/`error`/`contradiction`|plane-coord|plane-coord|plane-coord|geometry            |

**The key result is the ANLI column.** ANLI is adversarially constructed; DistilBERT validation accuracy there is ~0.41 (barely above chance). All three uncertainty axes nonetheless remain independent — demonstrating the central claim that these axes are **not artifacts of prediction success**.

### Semantic validation

A targeted probe distinguishes three input types — **G1** (novel, well-specified, e.g. *“Zeta-7 decays into mirror-charged leptons”*), **G2** (underspecified, e.g. *“something happened somewhere”*), **G3** (ordinary) — to confirm each axis fires on the right phenomenon:

- `novelty` fires highest on **G1** (rare vocabulary) and is positive on OOD — captures “first seen.”
- `ignorance` fires highest on **G2** (information deficit) and is *flat* on calibration (does not rise on wrong predictions — it measures missing information, not error).
- `ambiguity` fires highest on **neutral** labels and on **G2** — tracks genuine interpretive non-convergence.

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

`EpistemicBERT.__init__` requires `output_attentions=True` and `output_hidden_states=True` on the backbone config (set internally).

### Training

```python
loss = (
    0.3  * cross_entropy(logits, labels)        # secondary
    + 1.0  * field_ranking_loss(field, labels)  # primary: ranks correct axis highest
    + 0.2  * margin_loss(support, counter, labels)
    + 0.1  * diversity_loss                      # prototype collapse prevention
    + 0.03 * disentangle_loss(epistemic_out)     # light penalty, nov<->amb / amb<->ign only
)
```

The disentangle penalty is kept *small* and narrow — structural source separation does the real work; the penalty is a minor regularizer, not the mechanism.

### Diagnostics included

- `identifiability_probe` — per-axis R2 and full correlation matrix
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

Backbone defaults to DistilBERT; set the checkpoint path/name in `EpistemicBERT.__init__` and the tokenizer load. RoBERTa was tested for diagnosis (see lessons above) but DistilBERT is the v2 default.

## Known Limitations & Future Work

- **Attention-source normalization is dataset/backbone-specific.** The EMA standardization in `AttentionIgnorance` can saturate (sigmoid -> ~1 for all inputs) when attention-entropy scale shifts, e.g. under RoBERTa or ANLI. Identifiability still passes, but the normalization parameters need re-tuning per setup.
- **Plane structure loosens slightly in v2.** With `ambiguity` moved out of the manifold, `truth` R2 runs ~0.62-0.79 (vs ~0.88-0.99 in v1). Still plane-coord, but the evidence plane is marginally less tightly coupled — a side effect of the source change, not a defect.
- **`novelty` measures lexical novelty only.** Token rarity catches unfamiliar *words* but not familiar words combined into novel *concepts* (e.g. “round square”). An n-gram / compositional extension is a natural next step.
- **`ambiguity<->truth ~ 0.3-0.4` and `ambiguity<->contradiction ~ 0.3-0.55`** persist on SNLI/MNLI (near 0 on ANLI). Interpreted as genuine data structure (clear entailment = single interpretation; contradictory inputs are often also interpretively split), not measurement leakage, and deliberately not penalized.

### Roadmap -> v3: downstream utility

v1 and v2 establish that the six axes are *separable and task-robust*. v3 should show they are *useful* — that decomposed uncertainty beats a single confidence score:

- **Selective prediction.** Reject by `ignorance`/`ambiguity` and compare risk-coverage curves against single-entropy rejection.
- **OOD detection.** Benchmark `novelty` (token-rarity) against standard baselines (MSP, Mahalanobis).
- **Failure typing.** Show that high-`ambiguity` errors and high-`ignorance` errors are genuinely different failure modes, with examples.

## Status

- **v1** — six-axis identifiability established; `ambiguity` (manifold) task-dependent, leaks on ANLI.
- **v2 (current)** — `ambiguity` moved to layer disagreement; all three uncertainty axes task-free and identifiable across SNLI, MNLI, and adversarial ANLI.