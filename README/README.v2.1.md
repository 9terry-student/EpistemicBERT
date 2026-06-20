# EpistemicBERT

A BERT-based model that treats **epistemic uncertainty as a structurally decomposable
property of the input**, not as a proxy for prediction failure. Instead of collapsing
"the model is unsure" into a single confidence score, EpistemicBERT factorizes the
epistemic state into six interpretable axes, each grounded in an independent
measurement source.

> **v2.1 (current). Human-evaluation stage (VitaminC).** v2's apparent success — the six
> axes "typing" confident errors (`ignorance` dominating, 55–75% of cases) — turned out to
> be **gold-label leakage**: re-evaluating against gold-independent human judgments of
> evidence-sufficiency collapsed the apparent signal from AUROC 0.718 to 0.376–0.585 (near
> chance in the well-powered stratum). The project has been reframed around this finding:
> not "the model fails," but **"standard UQ evaluation on fact-verification conflates two
> different constructs (NLI entailment vs. human evidence-sufficiency), and confounds the
> second with leakage from the first."** See "Human-Evaluation Stage" below for the full
> arc.
>
> **v2 → v2.1 changelog:** v2 established axis identifiability across SNLI/MNLI/ANLI
> (unchanged, still valid — see "NLI Stage" below). v2.1 adds the full VitaminC
> human-relabeling study: the leakage diagnosis, blind-protocol codebook, axis-level and
> ensemble-level failure mechanisms, two v3-prototype frame-check attempts (spaCy/GLiNER),
> and an open question about gold-annotator page-context access. Status: test-retest κ
> pending before the resulting paper is submission-ready.

## Core Idea

Meaning is treated as a *field* produced by two evidence streams — **support** and
**counter** — read off a claim/evidence pair. From this field, six epistemic axes are
derived:

| Group | Axis | Question it answers |
|-------|------|---------------------|
| **Evidence plane** (2D) | `truth` | Does the evidence support the claim? |
| | `error` | Does the evidence refute the claim? |
| | `contradiction` | Are support and counter simultaneously strong? |
| **Uncertainty sources** (independent) | `novelty` | Is the input outside the model's world (first seen)? |
| | `ambiguity` | Are multiple interpretations similarly plausible? |
| | `ignorance` | Is there insufficient/misaligned evidence to judge? |

The three evidence-plane axes form a 2D coordinate system (they are *expected* to be
mutually predictable). The three uncertainty axes are designed to be **mutually
independent** and, crucially, **independent of the task** — each measured from a
structurally different source so that no axis is a reparameterization of another, and so
that none inherits the classification signal.

## Why Source Location Matters

The central design principle, established across the NLI stage and confirmed again
later: **an uncertainty axis is only robust if its measurement source lies outside the
classification pathway.** Penalty-based decorrelation was repeatedly found to *kill* axes
rather than disentangle them. The working approach gives each axis a distinct source the
classification gradient does not flow through:

| Axis | Source | Outside the classification pathway? |
|------|--------|-------------------------------------|
| `novelty` | **token rarity** — `-log` frequency against the running training vocabulary | yes |
| `ignorance` | **cross-attention alignment** — claim→evidence attention mass, split at `[SEP]` | yes |
| `ambiguity` | **layer disagreement** — directional variance of `[CLS]` across final layers | yes |
| `truth`/`error`/`contradiction` | **stream geometry** — support/counter margin, energy, agreement | (learned plane) |

All three uncertainty axes are **detached** from the classification gradient.

## NLI Stage (SNLI / MNLI / ANLI): identifiability

Before fact-verification, the axes were validated for **identifiability** — fit a linear
model predicting each axis from the other five; low R2 means the axis carries
non-redundant information.

| Axis | SNLI | MNLI | ANLI | Source |
|------|------|------|------|--------|
| `novelty` | 0.03 | 0.06 | 0.10 | token rarity |
| `ambiguity` | 0.42 | 0.25 | 0.16 | layer disagreement |
| `ignorance` | 0.01 | 0.08 | 0.08 | attention dispersion |

**The ANLI column is the key result.** ANLI is adversarially constructed; accuracy there
collapses to ~0.41 (barely above chance). All three axes remain independent regardless —
demonstrating the axes are not artifacts of classification success. (An earlier
manifold-based `ambiguity` leaked badly on ANLI, R2=0.74; moving the source to
layer-disagreement fixed it to 0.16. A larger backbone, RoBERTa, *raised* the old leak to
0.83, ruling out capacity as the explanation — it was a source-design problem.)

A semantic probe additionally found `ignorance` fires on information deficit and stays
**flat on calibration** (does not rise on wrong predictions — it measures missing
information, not error), and rises **+0.389** under OOD (RTE vs. ID) — the largest move
of any axis. This independence/OOD-robustness result is later load-bearing: it shows the
axes are well-designed, which sharpens (rather than excuses) the negative finding below.

## Human-Evaluation Stage (VitaminC): does UQ track human judgment?

**The question.** Fact-verification gold labels (SUPPORTS/REFUTES/NEI) encode an
NLI-style entailment relation. This project asks a different question: does a human,
given only the claim and evidence, find the evidence **sufficient to decide** the claim
at all? These can diverge — gold can mark a frame-mismatched pair "REFUTES" by reading
surface content conflict, while a human notices the evidence doesn't even refer to the
same entity as the claim.

**The arc:**

1. **Apparent success, then leakage.** Confident errors (`conf > 0.95`, wrong vs. gold;
   n=219 across 3 seeds) were first scored against gold-visible labels: `error` AUROC
   0.718, looked like real diagnostic signal.
2. **Blind relabeling breaks it.** Re-annotating the same 219 cases with gold and model
   prediction hidden, using a frame-based codebook (Step 1: do claim/evidence share an
   anchor? Step 2: any predicate/commitment/target mismatch despite anchor? Step 3:
   decisive or not?) yields type counts I=123/M=51/A=45. Cramér's V between gold and
   this human type is **0.264** (bias-corrected) — gold explains almost none of the
   human judgment (gold-conditioned majority accuracy 60.3% vs. 56.2% base rate, +4.1pp).
   Under the old gold-visible labels, NEI→I was 0.98 and non-NEI→M was 0.99 — i.e. human
   labels were a near-restatement of gold until the protocol went blind.
3. **The leakage explains the "success."** Re-scoring the *same* signal against
   gold-independent human labels: `error` AUROC drops 0.718 → 0.376; in the
   well-powered gold=REFUTES stratum (n=113), `ignorance` AUROC = 0.515 — chance.
   Apparent signal under small/imbalanced strata (SUPPORTS, NEI) was shown via bootstrap
   CI + raw-distribution inspection to be sample-size artifacts, not real.
4. **Axis-level diagnosis: two distinct failure mechanisms, not one.** Extracting raw
   per-id axis scores (never previously saved; required a fresh forward pass) showed:
   - `novelty`: std=0.048 (smallest of the three), AUROC=0.484 — fails because its
     construct (lexical rarity vs. training vocabulary) doesn't apply when confident
     errors are drawn in-domain from the same VitaminC lexical distribution. Not a design
     flaw — novelty's construct was already validated correctly on RTE-as-OOD (+0.389,
     above). Wrong evaluation stage, not a broken axis.
   - `ambiguity` (std=0.181, AUROC=0.555) and `ignorance` (std=0.185, AUROC=0.585): both
     have ~4x novelty's variance, so insufficient variance does *not* explain their
     failure — they vary plenty, but the variation is orthogonal to claim/evidence
     referent-matching (frame). A clean, well-formed sentence about the wrong entity is
     neither informationally sparse (ignorance's construct) nor representationally
     non-convergent (ambiguity's construct) — frame sits outside what either axis
     measures.
5. **Ensemble does not fix it: bias, not variance.** All three seeds (42/7/123) share the
   same architecture/six-axis design/gold supervision; 3-seed exact-prediction-agreement
   is **highest** on human-Insufficient cases (0.935, vs. 0.889 for A and 0.784 for M) —
   the signature of a shared structural blind spot, not seed-dependent noise. Averaging
   models that agree 93.5% of the time on exactly the cases that matter cancels nothing.
6. **v3 prototyping: fixed-schema entity matching is insufficient (not proven impossible).**
   Tried two independent frame-check prototypes (claim/evidence entity-overlap as a
   referent-matching signal): spaCy NER (8-label subset) and GLiNER zero-shot NER (8
   custom labels). Both showed the same coverage-failure mechanism — when an input's
   relevant entity type falls outside the chosen schema (e.g., "Treaty on the
   Prohibition of Nuclear Weapons" — no LAW/document-title label in either schema), both
   sides return zero entities, "no overlap" is returned by construction, and
   `frame_uncertain` is inflated regardless of whether frame genuinely mismatches.
   Resulting AUROC: spaCy 0.545 (0.573 excluding 5 confirmed data-corruption cases —
   VitaminC's own Wikipedia lead-sentence subject omission, verified against the
   original `tals/vitaminc` data); GLiNER 0.488 (0.521 excluding corruption) — *worse*,
   diagnosed to 13.7% empty-entity-extraction rate in M cases (17x the I-case rate),
   tracing to the specific 8-label schema's blind spots. **This is evidence that two
   partial schemas are insufficient, not proof that fixed-schema extraction is
   fundamentally impossible** — the full available schema space (e.g., complete
   OntoNotes incl. LAW, or a substantially larger GLiNER label list) was not exhausted.
   Stopped here on diminishing-returns grounds, not because the question is resolved.

**Open question (unresolved, disclosed not assumed):** whether VitaminC's original gold
annotators saw Wikipedia `page`-title context (which would disambiguate subject-omitted
evidence) during labeling is not documented in the original paper (Schuster et al., NAACL
2021) or the public codebase, and could not be confirmed via available sources. FEVER
(VitaminC's predecessor) explicitly permitted this; VitaminC's own practice is
undocumented. Flagged as a limitation; mitigation (re-relabeling a subset with page-title
visible) proposed as future work.

**The throughline, stated once:** independence (ANLI) and reproducibility
(ensemble-resistant bias / planned test-retest κ) are *necessary* properties of a good
measurement instrument — but neither implies the instrument measures the *right*
construct. Well-designed, task-robust, OOD-sensitive axes can still fail to align with
human evidence-sufficiency. The contribution is the construct separation and the blind
protocol that makes it visible, not a claim that uncertainty quantification is broken.

## v3 Candidate Directions (schema-free alternatives)

1. **LLM-as-judge** for direct frame/entailment judgment — most likely to generalize
   across the open-domain entity space, but reintroduces the exact validity problem this
   project raises about LLM-based evaluation; would need its own small-sample
   human-agreement check before use, not default trust.
2. **Mention-span-based coreference** (fastcoref, AllenNLP coref) — treats all noun
   phrases as candidate mentions rather than only schema-tagged spans; likely catches
   both pronoun cases and out-of-schema titles without a label list.
3. **Retrieval-augmented grounding** against an external KB instead of a closed schema —
   not yet attempted.

## Requirements

```
torch
transformers
datasets        # tals/vitaminc, glue (RTE for OOD)
scikit-learn
pandas
spacy / gliner  # v3 prototyping only
tqdm
```

## Known Limitations & Future Work

- **Single annotator.** Test-retest κ (time-gapped, codebook-only re-labeling) is the
  current bottleneck before submission — it is the inference that rules out "the null
  result is just annotation noise" rather than construct mismatch. Inter-annotator
  agreement (second annotator, codebook-only, blind) is the planned extension.
- **Gold-annotator page-context access is undocumented** (see above) — disclosed, not
  resolved.
- **`novelty` measures lexical novelty only**, validated on OOD (RTE) but not on
  confident-errors drawn in-domain (wrong evaluation stage for this axis, not a defect).
- **v3 frame-check is at the prototyping stage**, not integrated into the model; current
  evidence rules out two fixed-schema approaches and ensembling as fixes, but does not
  yet validate a working replacement.

## Status (v2.1)

- **v1/v2 — NLI stage** — six-axis identifiability established across SNLI/MNLI/ANLI;
  uncertainty sources moved off the classification pathway and stay independent even
  where accuracy collapses (~0.41 on ANLI).
- **v2.1 — Human-evaluation stage (VitaminC, current)** — gold-leakage diagnosed and
  corrected via blind relabeling (n=219, Cramér's V=0.264); UQ-human alignment falsified
  in the well-powered stratum (ignorance AUROC 0.515, chance); axis-level and
  ensemble-level failure mechanisms diagnosed; two v3 frame-check prototypes attempted
  and found insufficient (not disproven in general). **Pending: test-retest κ** (the
  remaining load-bearing inference before the negative-result/evaluation-methodology
  paper is submission-ready).
