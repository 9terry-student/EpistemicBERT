# Annotation Codebook — Confident-Error Epistemic Typing
### Frame-Based Sufficiency × Difficulty Scheme (gold-independent)

**Purpose:** Classify model confident-errors by the human-perceived epistemic
relationship between a *claim* and its *evidence*, independently of the dataset's
gold label. Annotators see ONLY claim + evidence (gold and model prediction hidden).

**Output:** two raw axes (sufficiency, difficulty) → derived type (I / A / M / O).

---

## 0. Core principle

Judge ONLY what a human reader can determine from the claim and evidence text.
- Do NOT consider how the model might process it.
- Do NOT consider the gold label (hidden during annotation).
- Do NOT supply unstated facts beyond common knowledge (see §4 background-knowledge rule).

The question is always: **"Does this evidence let a human decide whether the claim
is true or false — and if so, is it decisive or contestable?"**

---

## 1. Decision procedure (apply in order)

```
STEP 1 — Is there a FRAME?
   Shared semantic anchor present? (entity OR predicate-family OR event-domain)
     NO  → sufficiency = 0  → type I
     YES → go to Step 2

STEP 2 — Despite an anchor, is there a MISMATCH? (any one triggers I)
   · predicate-type mismatch  (execution≠distribution, place≠origin,
                               collaboration≠membership, write≠produce, etc.)
   · commitment mismatch      (planned/announced ≠ actualized/occurred,
                               role-assignment ≠ event-participation)
   · target / time / event mismatch (different year, different event, different referent)
     ANY mismatch → sufficiency = 0 → type I (frame broken)
     NONE         → frame holds → go to Step 3

STEP 3 — DECISIVENESS (only for frame-holding cases)
   Does the evidence DECISIVELY determine the claim (direct support OR direct contradiction)?
     YES → sufficiency = 1, difficulty = 1 → type M
     NO  (partial constraint, boundary value, requires inference, honest disagreement)
         → sufficiency = 1, difficulty = 0 → type A

O — broken input / suspected dataset-label error. Judge only AFTER gold is attached
    (not assignable during blind annotation).
```

### Axis encoding
| sufficiency | difficulty | derived type | meaning |
|-------------|-----------|--------------|---------|
| 0 | (blank) | **I** | Insufficient — frame mismatch or missing core predicate |
| 1 | 0 | **A** | Ambiguous — frame holds, evidence is partial/non-decisive |
| 1 | 1 | **M** | Misjudged — frame holds, evidence is decisive; model alone wrong |

---

## 2. Type definitions

**I — Insufficient evidence.** The evidence does not let a human decide the claim,
because (a) there is no shared anchor, or (b) there is an anchor but the evidence
addresses a *different kind of proposition* than the claim (mismatch in Step 2).
Key intuition: "not enough / not the right information," NOT "interpretations differ."

**A — Ambiguous / non-decisive.** The evidence is on-topic and relevant (frame holds),
but does not decisively settle the claim: boundary values, required inference,
or genuine reader disagreement. Key intuition: "information is here, but it doesn't
pin the claim down."

**M — Misjudged (clear but model-wrong).** Frame holds AND the evidence decisively
supports or contradicts the claim. A human reads it instantly; only the model errs.

**O — Other.** Broken/garbled input, or the gold label itself appears wrong
(evidence clearly supports/contradicts but gold disagrees). Assigned post-hoc.

---

## 3. FRAME definition (the heart of the scheme)

A frame requires a shared semantic anchor AND the absence of any disqualifying mismatch.

**Anchor** (need at least one): same entity, same predicate-family
(nationality, role, location, score, ranking, etc.), or same event-domain
(same film/game/match/article).

**Disqualifying mismatches** (any one breaks the frame → I), even if an anchor exists:

1. **Predicate-type mismatch** — the evidence speaks to a *different class* of
   proposition than the claim, even on the same entity:
   - execution vs distribution ("browser-based game" ≠ "has a PC version")
   - place vs origin, collaboration vs membership
   - writing vs producing (music industry roles are distinct)
   - production vs distribution (a producer ≠ a distributor)

2. **Commitment mismatch** — planned/announced vs actualized:
   - "announced to serve as commentator" ≠ "appears as commentator"
   - "set to replace X" (a plan) ≠ "replaced X" (an event)
   - Default to I. Collapsing prediction-space into fact-space is forbidden
     (a casting announcement does not establish participation).

3. **Target / time / event mismatch:**
   - different years (claim about 2001, evidence about 2011 census)
   - different events sharing only a surface token
     ("the game" = Super Bowl in evidence, but claim is about an NBA game)
   - "Same word present" does NOT make a frame — the anchor must be *semantic*.

---

## 4. Boundary rules established through case analysis

These resolve recurring traps. Apply uniformly.

**(a) Synonyms vs distinct roles (predicate granularity).**
Different words are NOT automatically different predicates, and same-looking relations
are NOT automatically the same. Test whether the field actually treats them as distinct:
- distinct (→ mismatch → I): write vs produce; production vs distribution;
  **published vs distributed** (a publisher ≠ a distributor); **"formed/founded" vs
  "under the management of"** (the entity that CREATED a group ≠ the entity that MANAGES
  it — evidence "formed under management of X" does not establish "X formed it").
- synonymous (→ frame holds): cinematography vs photography (film criticism);
  "slot" vs "draft pick" (sports draft); "approval rating %" vs "score out of 100".
Rule of thumb: if industry credits list the two roles separately (writer/producer,
publisher/distributor, founder/manager), treat as distinct → mismatch → I.

**(b) Background-knowledge / identifier inference → A.**
If the claim's key identifier (league, series, school, etc.) is NOT stated in the
evidence and must be inferred via background knowledge, the case is **A**
(not M, not I) — regardless of how strong the inference feels.
- "New England Patriots" implies NFL by background knowledge → A.
- "the Bulldogs" (many schools share it) → also A under this rule.
- Rationale: evidence does not directly establish the identifier; inference required.
(Note: this rule was adopted to treat all background-inference cases uniformly as A.
An alternative stricter scheme would split by identifier uniqueness; we chose the
uniform rule for consistency.)

**(c) Missing identifier in evidence (pronoun / category reference) → I.**
If the evidence refers to the claim's core entity only by pronoun or generic
category and never names it, the core predicate is unconfirmed → I.
- claim names "Off the Wall" track "Rock with You"; evidence starts "It was released…"
  with no track name → I.
- claim names series "You, Me and the Apocalypse"; evidence says only
  "the comedy/drama TV series" → I.
Distinguish from (b): (b) = identifier present elsewhere, inferable;
(c) = identifier absent from evidence text entirely.

**(d) Name suffix / honorific does not break entity identity.**
Unknown abbreviations or honorifics appended to a name do not create entity doubt
if other identifiers match. Ignore the unknown token; judge on name + birthdate +
occupation, etc.
- "Nelly Kim Furtado ComIH (born 2 Dec 1978), Portuguese-Canadian singer" — same
  person as claim's "Nelly Kim Furtado"; ComIH is an honorific → entity confirmed.
- Contrast: "Lenney" vs "Schick" — entirely different surnames, not a variant → not
  the same entity; evidence does not establish the claim's entity → I.

**(e) Compound claims (multiple entities/conjuncts).**
If a claim asserts multiple things (A and B) and the evidence confirms only one,
leaving the other unaddressed → A (frame holds, not decisive).
- "Demol AND Claesen played" / evidence confirms Demol, says nothing of Claesen → A.

**(f) Boundary values.**
- Evidence value clearly clears/misses the threshold → M (decisive).
  ("more than 44%", evidence "45%": 45>44 decisively → M)
- Evidence value sits exactly on or astride the boundary, or the qualifier's meaning
  is contestable → A.
  ("under 43%", evidence "43%": is 43 "under 43"? contestable → A)

**(g) Scope/qualifier mismatch on the same metric.**
If the claim and evidence use the same metric but different (possibly non-comparable)
qualifiers, and that affects decidability → A.
- claim "best-selling Latin album of 2001" (unscoped) / evidence "…in the United States"
  → US-vs-world scope unstated → A.
- claim "fourth single worldwide" / evidence "sixth single to [specific radio formats]"
  → different counting bases → A.

**(h) Different scenes / non-exclusive states (narrative claims).**
Evidence describing one action by a character does NOT contradict a claim about a
different action, because a film has many scenes — the two can both be true unless
the evidence marks finality. Default to A (frame holds, not decisive) rather than M,
unless evidence explicitly forecloses the claim.
- claim "Sarah returns to the photoshoot" / evidence "…walks off into the sunset"
  → could be different moments → A (not decisive contradiction).
(Contrast genuine M: "civilian" vs "hostile" classification of the same person in the
same situation — mutually exclusive → M.)

**(i) Implication strength.**
- Strong logical entailment → treat as decisive (M).
  (evidence "scored a goal" entails "played" → supports a single-person "played" claim)
- Weak/suggestive implication requiring an inferential leap → A.
  (evidence "heavily implies Deckard is a replicant" vs claim "forms a major plot point"
   → "implies" → "major plot point" needs a step → A)

**(j) Numeric/value mirror-contradiction → M, BUT only if the frame holds first.**
CRITICAL: Step 1/2 (frame) is checked BEFORE decisiveness. A value conflict yields M
ONLY if the evidence actually names/identifies the claim's subject. If the identifier
is absent (rule (c)), the case is I regardless of the value conflict.
- ranking "7th" vs "6th", population/score conflicts, WHEN the entity is named in
  evidence → M.
- box-office "4th" (claim) vs "5th" (evidence) where evidence reads "It…finishing 5th"
  with NO film name → **I** (rule (c) fires first; identifier absent). The number
  conflict is irrelevant because the frame never forms.
- name-swap in identical event: claim subject "Lenney" vs evidence subject "Schick"
  in same date/club/deal → I (evidence establishes a DIFFERENT named person's event
  and is silent on the claim's subject — omits, does not contradict).

---

**(k) Same predicate, different dimension/resolution → A (not I, not M).**
When evidence and claim address the SAME predicate-family but at different
dimensions/resolutions, and the two are mutually compatible (not exclusive), the
evidence neither confirms nor refutes the claim's specific dimension → **A** (frame
holds, non-decisive). This is NOT I: the predicate IS engaged, just at another
granularity. It is NOT M: compatibility means no contradiction.
- claim "born in a caravan" (dwelling type) / evidence "born in Beeston, Leeds"
  (administrative locality) → same predicate (birthplace), different dimension; a caravan
  can be IN Beeston → compatible, undecidable → A. (Pitfall: do not read "Beeston" as a
  rival VALUE to "caravan" — they are different axes, not competing values.)
- claim "saved with his right leg" (which leg) / evidence "saved with his legs" (general)
  → same predicate, evidence is less specific → A.
Contrast with M (competing values on the SAME dimension): "born in a caravan" vs
"born in a hospital" (both dwelling/venue type, mutually exclusive) → M.
Contrast with I (different predicate entirely): "born in a caravan" vs "is a comedian"
→ birthplace not addressed → I.

## 5. Worked examples (anchor → mismatch check → decisiveness → type)

| claim (abbrev) | evidence (abbrev) | reasoning | type |
|---|---|---|---|
| Onimusha Soul has a PC version | announced as browser-based game | execution≠distribution (predicate mismatch) | I |
| A,B served as commentators | announced they *would serve* | planned≠actualized (commitment) | I |
| Furtado is Portuguese-Canadian | "…ComIH… is a Portuguese-Canadian singer" | honorific ignored; direct support | M |
| Henry holds HS rushing record | Henry assumed control of Jaguars franchise | different predicate (record vs ownership) | I |
| Demol AND Claesen played | confirms Demol only | compound, one conjunct unaddressed | A |
| RT score more than 44%, >45 reviews | 45%, 49 reviews | both thresholds cleared decisively | M |
| RT score under 43% | 43% | "under 43"? boundary contestable | A |
| uncle was a local administrator | uncle "the black panther" [nickname] | predicate (role) not actually covered | I |
| Paulina is a platinum artist | sold ~15M records | platinum cert not stated; suggestive | A |
| Dirty Grandpa finished 4th | "It…finishing 5th" [no film name] | identifier absent from evidence | I |
| The House has abundant funny gags | consensus: "shortage of comic momentum" | decisive contradiction (synonymous axis) | M |

---

## 6. Reliability protocol (to be run with a time gap)

- **Test-retest:** re-annotate a ~50-item subset using ONLY this codebook, after a
  gap of several weeks (recall must have faded). Report Cohen's κ vs the original.
  Target κ ≥ 0.6 for the labels to count as stable.
- **Consistency audit:** confirm early- and late-annotated items obey the SAME final
  rules above (rules evolved during the first pass). Mirror-pair / duplicate check:
  identical or value-swapped claim-evidence pairs must receive consistent labels.
- **(Optional, strengthens claim) Inter-annotator:** a second annotator labels
  30–50 items from this codebook cold; report inter-annotator κ.

---

## 7. Notes on scope and honesty
- This codebook produces *human evidence-sufficiency judgments*, not ground truth.
  Refer to outputs as "manual annotation following a pre-specified protocol."
- The scheme is deliberately gold-independent: gold is hidden during annotation, and
  gold↔type association is measured AFTER labeling (Cramér's V) to verify independence.