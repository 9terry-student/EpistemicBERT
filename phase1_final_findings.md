# Phase 1 Final Findings — Gold-Independent Re-evaluation

**Status:** Original hypothesis FALSIFIED. Pivoted to evaluation-methodology finding.
**Date:** June 2026
**Bottleneck for next phase:** human taxonomy reliability (test-retest κ) — NOT yet validated.

---

## The arc of this investigation (what happened, in order)

1. **Original claim:** EpistemicBERT's uncertainty axes (esp. ignorance via cross-attention)
   diagnose human-perceived epistemic failure types (I/A/M) in confident errors.

2. **First evaluation (old labels):** ignorance AUROC 0.449 (fail), but evidence-plane
   axes looked strong — error 0.718, evidence-plane(3) CV 0.695. Apparent success.

3. **Confounding suspicion → check:** within gold=REFUTES, error AUROC collapsed 0.718 → 0.198.
   Discovered old human labels were ~95% determined by gold (I≈NEI, M≈non-NEI).
   The "success" was gold-label leakage, not genuine diagnosis.

4. **Blind re-labeling:** re-annotated all 219 items with gold/pred HIDDEN, using a
   2-axis raw scheme (sufficiency × difficulty) → derived I/A/M, frame-based decision rules.

5. **New labels are genuinely gold-independent:**
   - Distribution after blind relabeling + 7 rule-based corrections: **I=123, M=51, A=45**
     (n=219; was I=48/M=154/A=2 under gold-visible labeling)
   - **Cramér's V (gold × derived_type) = 0.264** (bias-corrected; 0.279 uncorrected) —
     recomputed directly from `relabel_final_for_analysis.csv` 2026-06-20, superseding an
     earlier draft value of 0.263 that had been misattributed to a stale intermediate
     label count (I=119/M=58/A=42, pre-correction). The two values are numerically close
     by coincidence; this entry is now the verified, final number.
   - gold-conditioned majority-rule accuracy = 0.603 vs base-rate 0.562 → **+0.041** over
     base rate (supersedes earlier "+0.046", same correction)
   - Crosstab (gold × human_type): NEI 21/33/5 (A/I/M), REFUTES 18/83/30, SUPPORTS 6/7/16

6. **Re-evaluation with clean labels → original hypothesis FALSIFIED:**
   All model-internal signals fail to separate human I/M (gold-independent).

---

## Final numbers (new gold-independent labels, n=219; I/M subset for AUROC)

### Overall AUROC (I vs M)
| signal | AUROC | note |
|--------|-------|------|
| truth | 0.394 | was 0.718 under leakage |
| error | 0.376 | was 0.718 — leakage gone |
| contradiction | 0.356 | |
| novelty | 0.545 | |
| ambiguity | 0.459 | |
| ignorance | 0.572 | highest, still ~chance |
| evidential_u | 0.276 | |
| mc_var | 0.385 | |
| mc_ent | 0.339 | |

**All near or below 0.5. No method separates human evidence-sufficiency judgment.**

### Per-gold AUROC (controlling for gold)
| gold | n (I/M) | ignorance | error | truth | novelty |
|------|---------|-----------|-------|-------|---------|
| REFUTES | 83/31 | 0.504 | 0.320 | 0.338 | 0.607 |
| SUPPORTS | 7/17 | 0.773 | 0.555 | 0.437 | 0.605 |
| NEI | 29/10 | 0.745 | 0.484 | 0.441 | 0.383 |

**CRITICAL CAVEAT:** ignorance looks high in SUPPORTS (0.773) and NEI (0.745),
but those cells are tiny (I=7, M=10 respectively) → unstable, likely noise.
The only well-powered cell is REFUTES (I=83/M=31), where ignorance = 0.504 (chance).
**Do NOT over-read the small-sample 0.77/0.74 — they are not evidence of diagnosis.**

### ours (6-axis auto_type) vs human
Accuracy 0.379, Macro-F1 0.246, Cohen κ = −0.039 (below chance).
Model predicts "ignorance" for most cases (138/219); human distribution is I/M/A spread.
→ model's "ignorance" measures something different from human "insufficient evidence."

---

## What is FALSIFIED
- "Uncertainty axes detect human epistemic failure types" — NOT supported.
- "ignorance axis diagnoses evidence deficit" — NOT supported (0.572 overall, 0.504 in REFUTES).
- "evidence-plane separates failure types" — was gold leakage (0.718 → 0.376 on clean labels).

## What SURVIVES (the real contribution)
1. **gold ≠ human epistemic judgment.** Human evidence-sufficiency judgments form a
   structure distinct from dataset gold labels (Cramér's V 0.263). Confident errors that
   are gold=REFUTES are often human-I (83/131), because the contradiction doesn't engage
   the claim's core predicate (frame mismatch: execution≠distribution, planned≠actualized,
   role≠event, write≠produce, etc.).
2. **UQ–human gap.** In this evaluation setup and 219-item subset, no existing uncertainty
   signal (6-axis, evidential DL, MC dropout) aligns with human evidence-sufficiency judgment.
3. **Methodological:** standard UQ evaluation on fact-verification is contaminated by gold
   leakage; blind gold-independent relabeling separates genuine signal from artifact.

## Claim phrasing discipline
- NOT "no method can ever do this."
- YES "in this evaluation setting and this 219-item human-annotated subset, existing
  uncertainty signals show no meaningful alignment with human evidence-sufficiency judgment."

---

## Candidate paper reframing
~~EpistemicBERT: Detecting epistemic failures via uncertainty decomposition~~ (dead)
→ **"Uncertainty Quantification Does Not Capture Human Evidence-Sufficiency Judgments:
   A Gold-Independent Evaluation"** (negative result + evaluation methodology)

Structure:
1. Standard UQ evaluation on fact-verification contains gold confounding.
2. Blind human relabeling (sufficiency × difficulty, frame-based) separates it.
3. Re-evaluate existing uncertainty methods on the clean labels.
4. All fail to align (overall AUROC 0.28–0.57; controlled-gold confirms).
5. Conclusion: model confidence/uncertainty ≠ human evidence-adequacy assessment.

---

## THE BOTTLENECK (next phase, do NOT skip)
A negative result is only valid if the measurement target is reliable. Must prove the
human taxonomy is stable BEFORE the UQ-gap claim holds:

1. **Test-retest κ (self-consistency)** — re-label a subset (~50 items) AFTER a time gap
   (days/week, NOT today — recall would inflate κ artificially). Need κ reasonably high
   (>0.6) to claim labels are stable.
2. **Label-consistency audit** — rules evolved DURING annotation (browser/PC predicate
   mismatch, Patriots/Bulldogs background-knowledge→A, write/produce→I). Verify early and
   late cases were labeled by the SAME final criteria. Mirror-pair check done partially;
   extend it.
3. **(Ideal) second annotator** — give codebook to one other person, label 30–50 items,
   compute inter-annotator κ. Single-annotator + strong test-retest may suffice for workshop.

If κ high → "UQ–human gap" is a real finding.
If κ low → taxonomy needs refinement (rules not yet stable), redo before any claim.

---

## Codebook (frame-based, as converged)
FRAME exists = shared semantic anchor (entity OR predicate-family OR event-domain)
  AND no mismatch in: predicate-type (execution/distribution, place/origin, collab/membership),
  commitment (planned/actualized, role/event), or target/time/event.
- sufficiency=0 (frame mismatch OR missing core predicate) → I
- sufficiency=1, difficulty=0 (frame ok, partial/non-decisive) → A
- sufficiency=1, difficulty=1 (frame ok, decisive support OR contradiction) → M
- O = broken input / dataset-label suspect (judge after attaching gold)
Background-knowledge rule: if claim's identifier (league/series/school) must be INFERRED
because evidence doesn't state it → A (regardless of inference strength).

## Data artifacts
- relabel_final_for_analysis.csv — blind labels + gold, I/A/M derived (n=219)
- relabel_with_gold.csv — same, with sufficiency/difficulty columns
- baseline_scores.csv — evidential_u, mc_var, mc_ent per confident-error id (regenerate if session lost)
- evidential_seed42.pt — evidential baseline (val acc 0.6494; regenerate if session lost)

---

## ADDED MEMO — gold ≠ human: construct difference, not (just) dataset error

**Question raised:** gold labels look "wrong" in several cases (e.g. gold=NEI but
evidence clearly supports the claim — Blue Apple, Starship/Sistar, Hwang/Chuncheon).

**Key framing (do not overclaim "gold is wrong"):**
- VitaminC gold is itself human-annotated (crowdworkers, NLI-style: does evidence
  *entail* the claim?). It is NOT ground truth — it is another set of human labels
  under a different instruction.
- Our labels answer a different question: can a human *judge the claim's truth from
  this evidence* (evidence sufficiency)? 
- These are likely **different constructs** (NLI entailment vs evidence adequacy),
  which is why they diverge systematically (Cramér's V 0.263). This is the project's
  core finding, framed as construct divergence — NOT "the dataset is buggy."

**Candidate paper message:**
"Fact-verification gold (NLI entailment) and human evidence-sufficiency judgments are
distinct constructs that diverge systematically; models trained on gold capture the
former, and no uncertainty signal captures the latter."

**MUST verify before claiming this (do at test-retest, with gold visible):**
Take the gold=NEI ∩ human=M cells (n=10) and classify each as:
  (a) gold genuinely wrong / suspect → reassign O
  (b) sufficient for a human but not NLI-entailed → genuine construct difference
  (c) my own labeling error → fix
The split among (a)/(b)/(c) determines whether "gold ≠ human" is a real construct
finding or partly label noise. Do NOT pre-judge these now (would contaminate test-retest).
gold=NEI∩M ids to inspect: 110, 86, 4, 134, 138, 214, 142, 189, 5, 197.

**Consistency-test candidates (include in test-retest subset; do NOT fix now):**
None outstanding — the four I initially flagged (4, 5, 62, 107) were re-confirmed
CORRECT by the annotator on review (under=strict<, over=strict>; 62 lacks film name;
107 lacks year). These are good evidence the codebook rules are being applied
consistently. Keep them in the retest subset anyway as positive controls.

---

## ADDED — Label consistency audit (run before test-retest, memory-independent)

Audited all 219 items for internal consistency (does NOT use recall, so valid now).

**[A] Exact-duplicate (claim+evidence identical) pairs: 0 inconsistencies.** Clean.

**[B] Same-claim, different-evidence pairs:** 2 found, BOTH justified by evidence
differences (correct to label differently):
- Adrian leg: id 57 evidence "with his legs" (plural→A) vs 213/58 "with his right
  leg" (named→M). Different evidence ⇒ different label is correct.
- Derrick Henry: id 39 evidence "assumed full control of Jaguars" (on-topic→A) vs
  92/38/182 "holds HS rushing record" (different predicate→I). Correct.

**[C] Mirror-pair families (value/name-swapped, same logical structure): 33 groups.**
31 fully consistent (single type each). 2 flagged:
- Adrian leg — explained above (evidence differs, not a true inconsistency).
- **L&T / yoyo (id 14=A, id 15=I) — GENUINE INCONSISTENCY.** Symmetric mirror:
    id 14: claim "L&T = alt name for Larsen&Toubro" / evidence "...known as yoyo"
    id 15: claim "Yoyo = alt name for Larsen&Toubro" / evidence "...known as L&T"
  Identical structure (claim's alt-name conflicts with evidence's stated alt-name),
  yet labeled A vs I. Must resolve to the SAME type. DO NOT fix now (would contaminate
  test-retest); include both in the retest subset as a consistency probe.

**Audit verdict:** label set is highly internally consistent (1 genuine mirror
inconsistency out of 219). This is a strong preliminary reliability signal, pending
the time-gapped test-retest κ.

---

## FINAL AUROC — corrected labels (I=123, M=51, A=45), n=219

Re-ran after 7 rule-based corrections. Result is essentially unchanged from the
pre-correction run → finding is ROBUST to label corrections.

### Overall AUROC (I vs M)
truth 0.382 | error 0.392 | contradiction 0.350 | novelty 0.525 | ambiguity 0.476 |
ignorance 0.562 | evidential_u 0.268 | mc_var 0.387 | mc_ent 0.352
→ All near or below 0.5. evidential_u inverts (I-type u LOWER than M-type:
  0.180 vs 0.260). No uncertainty signal separates human evidence-sufficiency.

### Per-gold AUROC (ignorance) — SAMPLE-SIZE CAVEAT IS DECISIVE
| gold | n (I/M) | ignorance AUROC | trustworthy? |
|------|---------|-----------------|--------------|
| REFUTES | 83/30 | 0.515 | YES (only well-powered cell) → ~chance |
| SUPPORTS | 7/16 | 0.777 | NO (I=7, unstable) |
| NEI | 33/5 | 0.812 | NO (M=5, noise; M shrank 10→5 after corrections) |

**The only adequately-powered cell (REFUTES, n=113) gives ignorance = 0.515 (chance).**
The high values in SUPPORTS (0.777) and NEI (0.812) are on 5–7 minority-class items.
CORRECTED INTERPRETATION (do not call these "noise" — that overclaims the negative
just as "signal" would overclaim the positive):
- With M=5 / I=7, the 95% CI on AUROC is extremely wide (~0.4–1.0); neither 0.5 nor
  0.8 can be ruled out. Status = UNDETERMINED, not "noise".
- Mildly suggestive that BOTH small cells point the SAME direction (high), and that
  REFUTES (where evidence IS present, so an "ignorance/evidence-deficit" signal should
  be weak) is the one at chance — this could be a structured pattern, not coincidence.
- This is an OPEN QUESTION requiring more NEI/SUPPORTS confident-error samples to
  resolve (achievable now: lower conf threshold 0.95→0.90, use full val, or more seeds).
  If 0.8 holds with larger n → real conditional signal; if it regresses to 0.5 → noise.
Honest framing: "adequately-powered cell at chance; small cells undetermined, flagged
for targeted follow-up." This is stronger AND more accurate than declaring noise.

### ours vs human
Accuracy 0.384, Macro-F1 0.246, Cohen κ = −0.060 (below chance). Model over-predicts
"ignorance" (138/219); human labels spread across I/A/M. Model's internal "ignorance"
≠ human "insufficient evidence."

### Bottom line (robust, corrected, gold-controlled)
No model-internal uncertainty signal (6-axis, evidential DL, MC dropout) aligns with
human evidence-sufficiency judgment in this 219-item subset. Holds after rule-based
label corrections and after controlling for gold. This is the paper's main (negative)
result.

---

## Bootstrap CI on all 18 gold×axis cells (corrected labels)

Ran 2000-iter bootstrap AUROC on every gold×axis cell to avoid cherry-picking.
★ = 95% CI excludes 0.5.

| gold | axis | n(I/M) | AUROC | 95% CI | ★ |
|------|------|--------|-------|--------|---|
| REFUTES | truth | 83/30 | 0.320 | [0.217,0.441] | ★ (inverted) |
| REFUTES | error | 83/30 | 0.312 | [0.206,0.431] | ★ (inverted) |
| REFUTES | contradiction | 83/30 | 0.297 | [0.191,0.412] | ★ (inverted) |
| REFUTES | novelty | 83/30 | 0.597 | [0.476,0.713] | |
| REFUTES | ambiguity | 83/30 | 0.498 | [0.370,0.629] | |
| REFUTES | ignorance | 83/30 | 0.514 | [0.391,0.644] | |
| SUPPORTS | error | 7/16 | 0.535 | [0.259,0.803] | |
| SUPPORTS | ambiguity | 7/16 | 0.221 | [0.045,0.433] | ★ (inverted) |
| SUPPORTS | ignorance | 7/16 | 0.776 | [0.533,0.964] | ★ (forward) |
| NEI | error | 33/5 | 0.628 | [0.429,0.829] | |
| NEI | truth | 33/5 | 0.263 | [0.075,0.486] | ★ (inverted) |
| NEI | ignorance | 33/5 | 0.808 | [0.559,1.000] | ★ (forward) |

(other cells omitted = CI includes 0.5)

### Reading
- **REFUTES (n=113, well-powered):** all 3 uncertainty axes ~0.5 with CI including 0.5
  → genuinely random. evidence-plane (truth/error/contradiction) ★ but INVERTED (<0.5),
  a residual gold artifact, not diagnosis. Main negative result is solid here.
- **ignorance is the ONLY axis with a forward (>0.5) ★, and it occurs in BOTH small
  cells (SUPPORTS 0.776, NEI 0.808).** Two independent small cells, same axis, same
  direction, both CIs exclude 0.5 — more than pure chance would casually predict.
- Structural coherence: ignorance (evidence-deficit) separates in SUPPORTS/NEI but is
  random in REFUTES — consistent with "refutation context HAS evidence, so an
  evidence-deficit signal should be weak there."

### Status: OPEN, TESTABLE HYPOTHESIS (not noise, not confirmed signal)
"ignorance separates human I/M in SUPPORTS/NEI contexts but not REFUTES."
Caveats: CI lower bounds (0.533, 0.559) barely clear 0.5; n is tiny (M=16, M=5);
bootstrap CIs overconfident at small n.
→ FALSIFIABLE PREDICTION for the small-cell expansion experiment:
   if ignorance AUROC stays ~0.78–0.81 in SUPPORTS/NEI with larger n → conditional
   signal CONFIRMED (enriches the paper beyond pure-negative); if it regresses to 0.5
   → noise CONFIRMED (clean negative). Either outcome is publishable.
   Run AFTER test-retest (new cases need blind labeling; avoid contaminating retest).

---

## Small-cell stress test (no new labeling) — NEI hypothesis FALSIFIED, SUPPORTS undetermined

Stress-tested the "ignorance separates I/M in SUPPORTS/NEI" hypothesis WITHOUT new
labeling, by (1) widening the negative class I-vs-(A+M) = the sufficiency axis, and
(2) inspecting raw ignorance distributions.

### Split comparison (ignorance AUROC, bootstrap)
| gold | A: I vs M | B: I vs (A+M) = sufficiency | verdict |
|------|-----------|------------------------------|---------|
| REFUTES | 0.517 [0.382,0.657] | 0.475 [0.366,0.585] | random (well-powered) |
| SUPPORTS | 0.779 [0.527,0.967] | 0.775 [0.524,0.975] | HOLDS but I=7 only |
| NEI | 0.812 [0.588,1.000] | **0.601 [0.447,0.749]** | **COLLAPSED → 0.81 was M=5 fluke** |

### Raw ignorance distributions (the decisive evidence)
gold=NEI:
  M (n=5): [0.085, 0.103, 0.207, 0.293, 0.412]  ← all happen to be LOW
  A (n=21): spread 0.13–0.66, indistinguishable from I
  I (n=33): spread 0.11–0.75
→ The 0.81 came entirely from 5 M-items coincidentally being low. Including A (also
  "sufficient") erases it (0.60, CI includes 0.5). **NEI hypothesis FALSIFIED.**

gold=SUPPORTS:
  I (n=7): median ~0.60   [0.347–0.804]
  M (n=16): median ~0.40  [0.139–0.655]
  A (n=6): median ~0.49 (intermediate)
→ Genuine directional separation (I > M), survives widening (0.775). BUT I=7 only;
  CI [0.524,0.975] barely clears 0.5. UNDETERMINED — could still be a 7-item fluke.

### Updated status
- **NEI conditional-signal hypothesis: REJECTED** (was a 5-item artifact; killed by
  honest negative-class widening + raw inspection, NO new labeling needed).
- **SUPPORTS: weak undetermined signal** (directional, survives widening, but n=7).
  This is now the ONLY cell warranting sample expansion. Target narrowed: enrich
  gold=SUPPORTS ∩ human=I cases only. Temper expectations — 7→larger may regress to
  0.5 just as NEI did.
- **Main negative finding STRENGTHENED:** REFUTES (well-powered) random in both splits;
  NEI "signal" shown to be artifact. Only a single small, fragile SUPPORTS hint remains.

Methodological note: this is a clean example of resolving a small-cell ambiguity by
distribution-widening + raw inspection rather than declaring "noise" prematurely OR
over-reading the point estimate. The user correctly resisted both premature dismissal
and premature acceptance.

---

## NLI-stage (ANLI) result integrated — independence ≠ validity

The project has an earlier NLI stage (SNLI / MNLI / ANLI; see README v2) that, read
together with the VitaminC falsification, STRENGTHENS the negative finding rather than
contradicting it. The two stages measured DIFFERENT things:

### What ANLI established (identifiability — the axes are well-designed)
On adversarial ANLI, DistilBERT classification accuracy collapses to ~0.41 (3-class,
chance ≈ 0.33). Yet all three uncertainty axes stay independent (R² predicting each from
the other five):
| axis | SNLI | MNLI | ANLI | source |
|------|------|------|------|--------|
| novelty | 0.03 | 0.06 | 0.10 | token rarity |
| ambiguity (v2) | 0.42 | 0.25 | 0.16 | layer disagreement |
| ignorance | 0.01 | 0.08 | 0.08 | attention dispersion |
"ANLI column is the key result": axes remain independent even where prediction accuracy
is near-chance → the axes are NOT artifacts of classification success. (v1's manifold-
based ambiguity leaked on ANLI at R²=0.74; moving it to layer-disagreement fixed it to
0.16 — confirming source location, not capacity, governs independence. RoBERTa raised the
v1 leak to 0.83, ruling out a capacity explanation.)

### What VitaminC established (validity — the axes don't match human judgment)
On VitaminC confident-errors with gold-independent human labels, no axis (incl.
ignorance) separates human I/M: overall ignorance AUROC 0.562; in the well-powered
gold=REFUTES stratum 0.515 (chance). Gold-controlled, robust to 7 label corrections.

### Why these are consistent, not contradictory (the paper's spine)
The NLI stage's OWN semantic probe already predicted this: ignorance "fires highest on G2
(information deficit) and is FLAT on calibration — does not rise on wrong predictions; it
measures missing information, not error." And RTE-as-OOD: ignorance rises +0.389 ID→OOD.
So across BOTH stages the picture is one coherent thing:
- ignorance is INDEPENDENT (ANLI R²≈0.08) and task-robust,
- ignorance DETECTS information deficit / distribution shift (G2 probe; RTE +0.389),
- ignorance does NOT separate human evidence-sufficiency type (VitaminC AUROC ≈ 0.51).

**Thesis: structural independence and OOD-robustness do NOT imply alignment with the human
evidence-sufficiency construct.** A well-designed, task-free, distribution-sensitive
uncertainty axis can still measure a DIFFERENT construct than the human judgment of
"is this evidence sufficient to decide the claim." This is a stronger, more interesting
claim than "the axis is just noise" — the axis is demonstrably well-behaved, yet still
mis-aligned. It elevates the result from "model failure" to "construct divergence."

### Caveat for write-up
The ANLI/RTE numbers are from the NLI stage (README v2), a different dataset/label regime
than the VitaminC human-relabeling study. Present them as supporting evidence that the
axes are well-constructed (independence/OOD), NOT as evidence about human-judgment
alignment (which only the VitaminC study tests). Do not conflate the two regimes.

---

## Axis-level failure diagnosis: why each axis fails differently (2026-06-20)

Following up on §6/§7 ("independence ≠ validity"), we decomposed *why* each
uncertainty axis fails to separate human I/M, by extracting raw per-id axis scores
(novelty, ambiguity, ignorance) via a fresh forward pass on the 219 confident-errors
(seed 42 checkpoint) — these scores were never previously saved to any file; only the
argmax "auto_type" category existed in `cw_answer_key.csv`.

**Result (n=219, std/AUROC by axis):**
| axis | overall std | \|mean(I)-mean(M)\| | ratio | AUROC (I vs M) |
|------|---|---|---|---|
| novelty | 0.048 | 0.002 | 0.045 | 0.484 |
| ambiguity | 0.181 | 0.029 | 0.158 | 0.555 |
| ignorance | 0.185 | 0.061 | 0.330 | 0.585 |

**Two distinct failure mechanisms, not one:**

1. **Novelty fails because its construct doesn't apply on this evaluation stage —
   not a design flaw.** Novelty = token rarity relative to the training vocabulary.
   Confident-errors are drawn *in-domain* from VitaminC (same lexical distribution as
   train), so "is this lexically novel" is a question whose answer is structurally
   near-constant here (std=0.048, the smallest of the three) — this follows directly
   from the source definition, not from running the experiment. Novelty's construct
   (detecting unfamiliar input) was already validated on the correct stage: the RTE-as-OOD
   probe (NLI stage, README v2) showed novelty rising +0.389 ID→OOD, exactly as designed.
   **Conclusion: novelty is not broken; it was evaluated on the wrong stage (in-domain
   confident-errors) for what it measures (out-of-distribution novelty).** No fix needed;
   document the scope mismatch rather than the axis.

2. **Ambiguity and ignorance fail because their source doesn't encode frame —
   a design gap, confirmed empirically.** Both have substantially higher variance than
   novelty (std 0.18–0.19, ~4x novelty's), so "insufficient variance" does *not* explain
   their failure (unlike novelty) — they vary plenty, but the variation is decoupled from
   the human I/M distinction (AUROC 0.56–0.59, ratio 0.16–0.33). Mechanistically: ignorance
   (attention dispersion) measures how concentrated vs. spread attention is over the
   input — this is orthogonal to whether the evidence's entity/event matches the claim's.
   The Resident Evil case (evidence: "the film ... finishing 5th", no film named) is
   *not* low-information by dispersion's standard (concrete, specific detail) — its
   sufficiency problem is referential, not informational. Ambiguity (layer-disagreement)
   similarly tracks representational non-convergence, which is also orthogonal to
   reference resolution — a clearly-worded sentence about the wrong entity is not
   "ambiguous" in the layer-disagreement sense.

**Implication for v3 (future work, not a revision of the current negative finding):**
the current architecture has no component dedicated to checking whether claim and
evidence share a referent (frame). None of the six axes are designed to detect this —
it sits orthogonal to all of them. The proposed next step is a **coreference/entity-
linking-based frame-check module**, added as a *gate prior to* ignorance (mirroring the
codebook's own Step-1-before-Step-2/3 structure) — not a revision of ambiguity (already
validated as measuring layer-disagreement, a different and legitimate construct) and not
a revision of novelty (already validated on RTE; the in-domain confident-error stage is
simply the wrong place to evaluate it). This keeps the current paper's negative finding
intact (axes are well-behaved, mis-aligned with human sufficiency) while giving a concrete,
falsifiable next architecture step.

---

## Ensemble does not fix frame-blindness: bias, not variance (2026-06-20)

A natural question once ambiguity/ignorance were shown to be frame-blind (previous
section) is whether multi-seed ensembling could help, since the project already trains
three seeds (42, 7, 123). We tested this directly rather than reasoning from architecture
alone, because the two failure mechanisms (variance vs. bias) make opposite predictions
and only the data can say which applies.

**Method.** Loaded all three seed checkpoints, ran a fresh forward pass on the 219
confident-errors, and measured how often the three seeds' predicted labels agree exactly
(`all_agree`), stratified by human_type.

**Result:**
| human_type | 3-seed exact-agreement rate |
|---|---|
| I (mostly frame-mismatch) | **0.935** |
| A | 0.889 |
| M | 0.784 |

**Interpretation.** Agreement is *highest*, not lowest, in the I (frame-mismatch-heavy)
group. All three seeds share the same architecture, the same six-axis design (no
component checks claim/evidence referent identity), and the same gold supervision; with
no source of disagreement specific to frame-mismatch inputs, the three models converge on
the same prediction on these cases at a higher rate than on other types. This is the
signature of **bias** (a structural blind spot shared across seeds), not **variance**
(seed-dependent noise) — and ensembling (averaging/voting across seeds) only cancels
variance; it does not correct a bias all members share. **Conclusion: multi-seed
ensembling is not an effective remedy for frame-blindness.** This rules out ensembling as
an alternative to the proposed v3 direction (a dedicated coreference/entity-linking
frame-check module) — the fix has to add information no current seed has, not average
existing seeds together.

*Note on a discarded secondary check:* an initial attempt to also test whether the
seeds' agreed-upon prediction matches gold was structurally circular (confident-errors
are defined as cases where the seed's prediction ≠ gold, so "agreed prediction = gold"
is true by construction near 0% within this subset) and is not reported as a finding.

---

## Open question: did VitaminC's gold annotators see page-title context? (2026-06-20, unresolved)

While inspecting why `novelty`/`ignorance` showed unexpectedly low variance/AUROC, two confident-error
cases (id 4/5 Resident Evil; id 51 Super Mario; id 144 Contemporary R&B) turned out to have evidence
sentences with the grammatical subject omitted (e.g., "is a series of platform games created by
Nintendo..."). We verified directly against the original `tals/vitaminc` HuggingFace dataset
(validation split, exact claim match) that this is NOT an artifact of our extraction pipeline — the
subject-omitted evidence is present in VitaminC itself, for both labeled versions of the same
wiki_revision_id (e.g., id 51's SUPPORTS-labeled and NEI-labeled rows for the same revision both omit
"Super Mario"). This rules out the hypothesis that gold annotators saw a complete sentence while our
blind human relabeling saw a truncated one from a shared source text.

**However, a related and more consequential question remains genuinely unresolved: did VitaminC's gold
annotators have access to the Wikipedia `page` title (stored as a separate field in the dataset) as
context when making the SUPPORTS/REFUTES/NEI judgment, even though the stored `evidence` text omits the
subject?** This matters because FEVER (VitaminC's predecessor, sharing some annotation lineage) explicitly
permitted annotators to use the page title to resolve coreference ("The title of the page could also be
used as evidence to resolve co-reference, but this decision was not explicitly recorded" — Thorne et al.
2018). If VitaminC's annotators had equivalent access, gold judgments on subject-omitted evidence may have
been made with disambiguating context our blind human-relabeling protocol deliberately withheld (we showed
annotators only claim + evidence text, not `page`).

**We attempted to verify this through three available channels, all inconclusive:**
1. The original paper (Schuster et al., NAACL 2021) — read in full; no explicit statement of what was shown
   to annotators during the SUP/REF/NEI labeling step (only that "sentences lacking self-contained context
   were filtered" during an earlier revision-selection stage, which does not resolve this question).
2. The public GitHub repository (TalSchuster/VitaminC) — contains only post-collection training/eval code;
   the original annotation interface (used by TransPerfect crowdworkers) is not published.
3. TransPerfect's internal annotation guidelines — not publicly available (commercial vendor, client-specific
   materials); no trace found via search.

**Status: unresolved limitation, not adjudicated.** We do not claim gold annotators lacked page context;
we also cannot confirm they had it. This is disclosed explicitly rather than assumed in either direction.

**Implication for the paper:** state this as an acknowledged information-asymmetry risk in Limitations —
"our blind relabeling protocol withheld page-title context that gold annotators may or may not have had
access to; this is undocumented and could not be resolved from public sources." Suggested future-work
mitigation: re-run a subset of blind relabeling WITH page-title visible, and compare human_type distributions
to see whether access to this single piece of context changes I/A/M judgments on subject-omitted cases
(e.g., id 4/5/51/144) — this would bound the size of the potential asymmetry empirically rather than leaving
it as pure speculation.

---

## v3 prototyping: fixed-schema entity extraction is insufficient, not impossible (2026-06-20)

Following the axis-level diagnosis (ignorance/ambiguity are frame-blind by design) and the
ensemble-is-bias-not-variance result, we prototyped a candidate frame-check signal: detect
whether claim and evidence share a referenced entity, using fixed-schema entity extraction.

**Method.** Two independent extraction approaches were tried: (1) spaCy NER restricted to an
8-label subset (PERSON, ORG, WORK_OF_ART, GPE, EVENT, FAC, NORP, PRODUCT); (2) GLiNER zero-shot
NER with an 8-label custom schema (video game/movie/TV/musical-work title, person, organization,
place, band). For each, `frame_uncertain` = no entity-string overlap (exact or substring) between
claim and evidence entities.

**Result — both approaches showed the same coverage-failure mechanism:**
- spaCy: AUROC (I vs M) = 0.545 overall, 0.573 excluding 5 confirmed data-corruption cases
  (Wikipedia lead-sentence subject omission, verified against the original `tals/vitaminc`
  dataset). Diagnosed false positive: id76 (video-game title with a colon was not recognized
  as any entity by either side, despite being lexically identical in claim and evidence).
- GLiNER (designed to fix exactly this gap via flexible zero-shot typing): AUROC = 0.488 overall,
  0.521 excluding data-corruption — *worse*, not better. Diagnosis: 13.7% of M cases had zero
  extracted entities on at least one side (vs. 0.8% for I), almost entirely because our 8-label
  schema omitted categories present in this subsample (e.g., treaty/legal-document titles,
  "Treaty on the Prohibition of Nuclear Weapons" — not covered by either label set).

**Mechanism (well-supported, retained):** schema projection (text → finite label set) creates
a coverage gap whenever the open-world referent space contains a category outside the schema.
When both claim and evidence entities are empty for a case, "no overlap" is returned by
construction — not because frame genuinely mismatches, but because the extractor found nothing
to compare. This inflates `frame_uncertain` for any case whose dominant entity type falls outside
the chosen schema, regardless of whether frame actually matches.

**On the direction of the resulting bias (important precision):** the coverage-gap mechanism
itself is non-directional — it inflates `frame_uncertain` wherever an out-of-schema category
appears, in I cases or M cases alike. The *appearance* of M-direction bias in our GLiNER run
(M's empty-entity rate 17x I's) reflects the composition of this specific 219-item sample (the
omitted category — legal/treaty documents — happened to cluster in M here, e.g., the Treaty on
the Prohibition of Nuclear Weapons cases), not a structural property of fixed-schema extraction
that would generalize to other samples or datasets. We do not claim schema projection is
inherently biased toward M; we claim it inflates noise non-directionally, and this sample's
particular composition happened to surface that noise as M-inflation.

**On the strength of the conclusion (calibrated):** we do NOT claim fixed-schema entity
extraction is fundamentally impossible for this task — that would require exhausting the
available schema space (e.g., the full ~18-category OntoNotes set including LAW, which our
8-label spaCy subset excluded, or a substantially expanded GLiNER label list), which we did not
do. What the evidence DOES support: **two independently-chosen partial label schemas both
failed via the same coverage mechanism**, which is suggestive but not dispositive evidence
against the fixed-schema approach in general, particularly under open-domain (Wikipedia-scale)
input distributions where the long tail of entity categories is large.

**Decision: stop here (diminishing returns), not because the question is resolved.** Two
schema attempts showing the same failure mode is enough signal to deprioritize further
schema-expansion attempts (chasing the long tail of categories is unlikely to converge) without
needing a third or fourth attempt to confirm the pattern. This is a scope/effort decision, not
a theoretical proof.

**v3 candidate directions (schema-free alternatives, ranked by what we learned):**
1. **LLM-as-judge** (direct entailment/coreference judgment, no fixed schema) — most likely to
   generalize across the open-domain entity space, but reintroduces exactly the validity problem
   this paper raises about LLM-based evaluation; would need its own small-sample human-agreement
   validation before use (cannot be trusted by default — same logic as the paper's critique of
   gold-conditioned UQ evaluation).
2. **Mention-span-based coreference** (e.g., fastcoref, AllenNLP coref) — considers all noun
   phrases as candidate mentions rather than only schema-tagged spans; would likely catch both
   pronoun cases (id66, "It") and out-of-schema titles (id141, treaty name) without needing a
   label list, but still depends on the underlying parser's span/boundary quality.
3. **Retrieval-augmented grounding** — ground claim/evidence entities against an external KB
   instead of a closed label schema; not attempted, flagged as a further future-work option.