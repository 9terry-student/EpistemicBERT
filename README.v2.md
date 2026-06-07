# EpistemicBERT

A BERT-based model that treats **epistemic uncertainty as a structurally decomposable property of the input**, not as a proxy for prediction failure. Instead of collapsing "the model is unsure" into a single confidence score, EpistemicBERT factorizes the epistemic state into six interpretable axes, each grounded in an independent measurement source.

> **FEVER stage (current).** Migrated from NLI (SNLI/MNLI/ANLI) to FEVER fact-verification on the VitaminC dataset. The label correspondence holds across 3 seeds — `SUPPORTS → truth`, `REFUTES → error`, `NEI → ignorance` — and the main result is **failure typing**: confident errors that are indistinguishable under a single confidence score split into reproducible types under the six-axis decomposition. See "Main Result" below.

## Core Idea

Meaning is treated as a *field* produced by two evidence streams — **support** and **counter** — read off a claim/evidence pair. From this field, six epistemic axes are derived:

| Group | Axis | Question it answers |
|-------|------|---------------------|
| **Evidence plane** (2D) | `truth` | Does the evidence support the claim? |
| | `error` | Does the evidence refute the claim? |
| | `contradiction` | Are support and counter simultaneously strong? |
| **Uncertainty sources** (independent) | `novelty` | Is the input outside the model's world (first seen)? |
| | `ambiguity` | Are multiple interpretations similarly plausible? |
| | `ignorance` | Is there insufficient/misaligned evidence to judge? |

The three evidence-plane axes form a 2D coordinate system (they are *expected* to be mutually predictable). The three uncertainty axes are designed to be **mutually independent** and, crucially, **independent of the task** — each measured from a structurally different source so that no axis is a reparameterization of another, and so that none inherits the classification signal.

## Why Source Location Matters

The central design principle, established repeatedly across the NLI stage and confirmed again here: **an uncertainty axis is only robust if its measurement source lies outside the classification pathway.** Penalty-based decorrelation was repeatedly found to *kill* axes rather than disentangle them (a constant output trivially has zero correlation with everything). The working approach gives each axis a distinct source that the classification gradient does not flow through:

| Axis | Source | Outside the classification pathway? |
|------|--------|-------------------------------------|
| `novelty` | **token rarity** — `-log` frequency against the running training vocabulary | yes — token statistics |
| `ignorance` | **cross-attention alignment** — claim→evidence attention mass, split at `[SEP]` | yes — attention structure |
| `ambiguity` | **layer disagreement** — directional variance of `[CLS]` across the final layers | partial — `[CLS]` carries the classification signal (see Limitations) |
| `truth` / `error` / `contradiction` | **stream geometry** — `support`/`counter` margin, energy, agreement | (evidence plane, learned) |

`novelty`, `ambiguity`, and `ignorance` are all **detached** from the classification gradient — their statistics track the data distribution, not the loss. `truth`/`error`/`contradiction` are *not* detached; they are the learned judgment plane.

## Migration: NLI to FEVER

NLI training structurally requires accepting the premise as true, so more training worsens confident hallucination rather than fixing it — the model cannot *doubt* a premise. FEVER verification does the opposite: it checks a claim against evidence. This motivated the migration. Two changes were required:

1. **Cross-attention ignorance.** The ignorance source measures how much the claim tokens actually attend to the evidence tokens (split at `[SEP]`); low alignment = high ignorance. This is query-conditional and dataset-invariant, preferred over PMI (dataset-dependent) or IDF (duplicates novelty).

2. **Loss label-to-axis mapping.** The original loss mapped labels through NLI's axis assignment (`label 0 -> truth`, `label 2 -> contradiction`). FEVER labels are permuted (`SUPPORTS=0, REFUTES=1, NEI=2`), so `REFUTES` was being trained toward the wrong axis and `NEI` toward contradiction — which is why `error -> REFUTES` initially failed. Remapping `field_ranking_loss` and `margin_loss` to FEVER semantics (`SUPPORTS -> truth`, `REFUTES -> error` + `counter > support`, `NEI -> ignorance` + both streams low) resolved it **without** touching the encoder. This isolated the cause: the failure was a stale label map, not a structural inability of the merged `[CLS]` to separate `SUPPORTS`/`REFUTES`.

### Label correspondence (3 seeds: 42, 7, 123)

| Mapping | seed 42 | seed 7 | seed 123 |
|---------|---------|--------|----------|
| `truth -> SUPPORTS` | yes | yes | yes |
| `error -> REFUTES` | yes | yes | yes |
| `ignorance -> NEI` | yes | yes | yes |

Support/counter by label (seed 42, epoch 3) confirm the direction is genuine, not a coordinate artifact: `REFUTES` shows `counter=0.731 > support=0.471` (under the old loss it was `support=0.624 > counter=0.592`). The merged `[CLS]` carried enough directional information; the old loss had simply trained `REFUTES` toward neutrality.

## Main Result: Failure Typing

A single confidence score collapses all confident errors into one bucket. The six-axis decomposition splits them into **reproducible types that confidence cannot see**.

Taking confident wrong predictions (`conf > 0.95`) and assigning each to the highest-firing uncertainty axis (or `metacognitive` if all three stay at the correct-prediction baseline):

| Type | seed 42 | seed 7 | seed 123 | reading |
|------|---------|--------|----------|---------|
| `ignorance` | 43 | 33 | 62 | evidence misaligned with claim |
| `ambiguity` | 8 | 1 | 20 | interpretation genuinely split |
| `novelty` | 5 | 4 | 19 | unfamiliar vocabulary |
| `metacognitive` | 7 | 6 | 11 | input clean, judgment simply wrong |
| **confident-wrong total** | 63 | 44 | 112 | |

**The `ignorance` type dominates and reproduces** — 55–75% of confident errors across all three seeds, with the `ignorance` axis itself independent in every seed (R2 = 0.19 / 0.21 / 0.19).

**Confidence cannot separate these types.** All types sit at `conf ~ 0.97-0.99` and `entropy ~ 0.07-0.13`. Under a single score they are one undifferentiated bucket; the six-axis field is what tells them apart.

Representative cases (claim || evidence):

- **ignorance** — *"In the last EFL Cup, Manchester defeated Kepa and Sarri"* || evidence about a different match entirely. The claim and evidence do not engage; `ign ~ 0.75`.
- **ambiguity** — *"The Equalizer 2's rating is over 49%, with more than 96 reviews"* vs *"under 50%, with less than 97 reviews"*. Boundary values where `SUPPORTS`/`REFUTES` is genuinely contestable; `amb ~ 0.43`, both readings flagged.
- **metacognitive** — *"Thon Maker's uncle was the black panther"* || evidence saying *"his uncle, a local administrator"* (roles swapped). Familiar words, aligned evidence (`ign ~ 0.40`), but the directional judgment is wrong. No uncertainty axis fires — correctly, because the deficit is not in the input.

This reframes the high silent-wrong rate (0.61, see below): confident errors are not one phenomenon. Each type implies a *different* remedy — `ignorance` wants more/better evidence, `ambiguity` wants the boundary disambiguated, `metacognitive` wants re-reasoning. A single confidence score cannot prescribe any of these because it cannot tell them apart.

## Identifiability (3 seeds)

For each axis, fit a linear model predicting it from the other five; report R2. Plane axes are *expected* to be predictable (coordinates); uncertainty axes should be independent (low R2).

| Axis | seed 42 | seed 7 | seed 123 | verdict |
|------|---------|--------|----------|---------|
| `novelty` | 0.28 | 0.33 | 0.24 | independent, all seeds |
| `ignorance` | 0.19 | 0.21 | 0.19 | independent, all seeds (rock-solid) |
| `ambiguity` | 0.13 | **0.76** | 0.25 | **conditional — leaks in seed 7** |
| `truth` / `error` / `contradiction` | plane-coord | plane-coord | plane-coord | geometry |

`novelty` and `ignorance` hold across every seed. `ignorance` in particular is nearly invariant (0.19/0.21/0.19). The OOD experiment (RTE) confirms direction: `ignorance` rises **+0.389** ID->OOD, the largest move of any axis — evidence-deficit detection survives distribution shift.

## Mechanism Evidence: source location decides robustness

The `ambiguity` leak in seed 7 is not a bug to hide — it is a **positive control** for the central thesis. The three uncertainty sources differ in one variable: whether their source lies inside or outside the classification pathway.

```
source           location              independent across 3 seeds?
novelty          token rarity          yes  (outside the pathway)
ignorance        cross-attention       yes  (outside the pathway)
ambiguity        [CLS] layer-disagree  no   leaks in seed 7 (inside the pathway)
```

In seed 7, the *raw* layer-disagreement signal — before any normalization — already correlates 0.74 with `error` and 0.80 with `contradiction`. Normalization passes this through; it does not create it. Because `[CLS]` is the channel the classification signal flows through, any axis read from it can, depending on which minimum training converges to, absorb the judgment signal and bleed into the evidence plane. The token- and attention-based sources, which sit off that channel, never do.

This is the same lesson the NLI stage taught three times: `novelty` could not live in `z`; `ignorance` could not be predictive entropy; **`ambiguity` cannot live in `[CLS]`.** Source location, not capacity or penalty strength, determines whether an axis stays task-free.

## What FEVER does *not* have: contradiction

`contradiction` has **no FEVER label correspondence** — it is flat across all three labels (~0.12-0.23, peaking weakly at `SUPPORTS`, not `REFUTES`). This is expected, not broken: the FEVER label set (`SUPPORTS`/`REFUTES`/`NEI`) has no slot for "evidence-internal conflict," unlike NLI's contradiction label. `REFUTES` ("evidence refutes the claim") is owned by `error`, a distinct concept from contradiction ("two evidences conflict"). The axis remains as a plane coordinate, reusable for future multi-evidence data. The presence of a label correspondence in NLI and its absence in FEVER is itself a finding about the difference between entailment and verification tasks.

## Architecture

```
input_ids --> DistilBERT --> CLS --> proj --> z --+--> PrototypeManifold (dual-stream)
                  |  |                            |       support / counter   (evidence plane)
                  |  |                            |
                  |  |                            +--> TokenNovelty   (token rarity)
                  |  |
                  |  +--> hidden_states --> LayerAmbiguity (layer disagreement)
                  |
                  +--> attention --> AttentionIgnorance (cross-attention claim->evidence)
                                                         |
   support, counter, novelty, ambiguity, ignorance -----+--> EpistemicFieldClassifier
                                                                  truth / error / contradiction
                                                                  field (6 axes)
                                                                        |
                              field_proj(field) + z_proj(z) -----------> logits
```

`EpistemicBERT.__init__` requires `output_attentions=True` and `output_hidden_states=True` on the backbone (set internally). Backbone defaults to `distilbert-base-uncased`.

## Usage

```python
model = EpistemicBERT(n_classes=3, proj_dim=128)
model.token_nov.set_special_tokens(tokenizer.all_special_ids)  # required
model.attn_ign.sep_id = tokenizer.sep_token_id                  # required

logits, epistemic_out, _ = model(input_ids, attention_mask)

epistemic_out["field"]          # (B, 6) the six-axis epistemic field
epistemic_out["ignorance"]      # per-axis scores
epistemic_out["support_raw"]    # stream values (used by margin_loss)
epistemic_out["dominant_type"]  # argmax axis label per sample (eval only)
```

### Training

```python
loss = (
    0.3 * cross_entropy(logits, labels)            # secondary
    + 1.0 * field_ranking_loss(field, labels)      # primary: ranks correct axis highest
    + 0.2 * margin_loss(support, counter, labels)  # SUPPORTS: s>c, REFUTES: c>s, NEI: both low
    + 0.03 * disentangle_loss(epistemic_out)       # light regularizer, nov-amb / amb-ign
)
```

FEVER label-to-axis map (the fix that made `error -> REFUTES` work):
`field_ranking_loss` ranks `truth`(0) for `SUPPORTS`, `error`(1) for `REFUTES`, `ignorance`(5) for `NEI`. `margin_loss` pushes `support > counter` for `SUPPORTS`, `counter > support` for `REFUTES`, and both streams low for `NEI`.

### Diagnostics included

- `evaluate_fever_axes` — axis-to-FEVER-label mapping check
- `identifiability_probe` — per-axis R2 and full correlation matrix
- `failure_typing` — confident-wrong type separation (the main result)
- `diagnose_ambiguity` — raw-disp vs normalized correlation, to locate a leak's source
- `evaluate_ood` — ID vs OOD axis means (RTE as OOD)
- `evaluate_confident_wrong` — axis means on correct / wrong / high-confidence-wrong
- `novelty_ignorance_probe`, `attention_entropy_probe` — semantic source validation

## Requirements

```
torch
transformers
datasets        # tals/vitaminc, glue (RTE for OOD)
scikit-learn
tqdm
```

## Known Limitations & Future Work

- **`ambiguity` is only conditionally task-free.** Its `[CLS]` layer-disagreement source lies inside the classification pathway and leaks into the evidence plane in some seeds (R2 0.76 in seed 7 vs 0.13/0.25 elsewhere). Six axes are retained, but `ambiguity`'s robustness is claimed weakly, not as established. Replacing it with an off-pathway source is open work (the natural next step, deliberately deferred).
- **Confident hallucination is structurally outside the six-axis frame.** The `metacognitive` type — clean input, familiar vocabulary, aligned evidence, but wrong judgment — has no uncertainty axis that should fire, because the deficit is in the model's own verdict, not in the input. Across `conf > 0.95` errors, all uncertainty deltas are ~0 (`ignorance +0.008`, `ambiguity -0.046`). This is the boundary between *measurement* (what these axes do) and *action* (re-questioning a verdict), and it is the same across NLI and FEVER because "clean input + wrong verdict" is task-agnostic. Crossing it requires an intervention layer, not a better measurement source — explicitly out of scope here.
- **Selective prediction loses to a single-entropy baseline by design** (best baseline AURC 0.199 vs `error` 0.298). The uncertainty axes are deliberately confidence-independent; they do not rank "likely wrong," so they should *not* win risk-coverage. The same property drives the 0.61 silent-wrong rate. This is a property of the framing, not a defect — the model is evaluated on what it *encodes* (identifiability, transfer, failure typing), not on confidence ranking.
- **`novelty` measures lexical novelty only.** Token rarity catches unfamiliar words, not familiar words combined into novel concepts. An n-gram / compositional extension is a natural next step.

## Status

- **NLI stage** — six-axis identifiability established across SNLI/MNLI/ANLI; uncertainty sources moved off the classification pathway (token / layer / attention) and stay independent even where accuracy collapses (~0.41 on ANLI).
- **FEVER stage (current)** — label mapping `SUPPORTS->truth / REFUTES->error / NEI->ignorance` reproduces across 3 seeds; cross-attention ignorance NEI-aligned and OOD-robust (delta +0.39); **failure typing** separates confident errors into reproducible types invisible to a single confidence score; `ambiguity` found to be conditionally task-free (`[CLS]` source), `contradiction` found to have no FEVER label correspondence.
