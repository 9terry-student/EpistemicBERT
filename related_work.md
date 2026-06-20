# Related Work (draft)

> Scope note: this positions a NEGATIVE/EVALUATION finding — that model-internal
> uncertainty signals do not align with human evidence-sufficiency judgments, and that
> gold (NLI-style) labels and human evidence-adequacy judgments are distinct constructs.
> Claims are scoped to "this evaluation setting and this 219-item subset." Citations are
> placeholders to verify/complete before submission (years/venues checked Jun 2026).

## 1. Uncertainty quantification for hallucination detection

A large body of work detects hallucinations by estimating a model's uncertainty, on the
premise that low confidence indicates likely error. **Semantic entropy** (Kuhn et al.,
2023; Farquhar et al., 2024, Nature) clusters semantically equivalent generations and
computes entropy over meanings rather than tokens, detecting confabulations without
ground-truth labels. Cheaper or refined variants followed: **Semantic Entropy Probes**
(Kossen et al., 2024) approximate SE from a single generation's hidden states; **Kernel
Language Entropy** (Nikitin et al., 2024, NeurIPS) generalizes SE via von Neumann entropy
over semantic similarity kernels; **Semantic Energy** (2025) operates on penultimate-layer
logits to
capture inherent confidence SE misses. Consistency-based **SelfCheckGPT** (Manakul et
al., 2023) samples multiple generations and treats disagreement as a hallucination
signal. Classical token/sequence approaches (length-normalized entropy; Malinin & Gales,
2020/2021) and **evidential deep learning** (Sensoy et al., 2018), which outputs a
Dirichlet over class probabilities and reads epistemic uncertainty as K/S, and **MC
dropout** (Gal & Ghahramani, 2016) remain standard white-box baselines.

**Our relation.** All of these share one assumption: the *target* of uncertainty is
**output correctness** (is the generation factually right?). We evaluate a different
target — a human's judgment of whether the *available evidence is sufficient to decide
the claim* — and find that representative signals from this family (a 6-axis epistemic
decomposition of our own, evidential DL, and MC dropout) do not align with it in our
setting. This is not a claim that these methods fail at their intended task
(correctness); it is that correctness-oriented uncertainty and human evidence-sufficiency
are different things.

## 2. Fine-grained / evidence-grounded factuality evaluation

Beyond a single uncertainty scalar, **FActScore** (Min et al., 2023) decomposes long-form
text into atomic facts and scores each against a knowledge source, recognizing that
generations mix supported and unsupported content. Grounded/faithfulness evaluation
(McDonald, 2020; RAG-style pipelines) separates "true relative to world knowledge" from
"supported by the provided source." Fact-verification datasets such as **FEVER** and the
adversarial **VitaminC** (Schuster et al., 2021) supply (claim, evidence, label) triples
with labels SUPPORTS / REFUTES / NEI, where the label encodes an NLI-style entailment
relation between evidence and claim.

**Our relation.** We use VitaminC but interrogate the *meaning* of its gold label. The
SUPPORTS/REFUTES/NEI annotation answers "does the evidence entail/contradict the claim?"
— an NLI question. We ask annotators a different question — "is this evidence sufficient
for a human to decide the claim?" — and show the two diverge systematically. Atomic-fact
and entailment evaluations presuppose the gold relation is the construct of interest;
our finding is that human evidence-sufficiency is a *separate* construct that gold does
not capture.

## 3. Calibration, construct validity, and the model–human gap

Calibration work asks whether confidence matches accuracy; LLMs are repeatedly found
overconfident on low-accuracy inputs (e.g., verbalized-confidence studies, 2025), and
fact-checking evaluations report confidence–accuracy gaps resembling Dunning–Kruger
patterns across model scales (2025). Most relevant, recent **LLM-as-a-judge reliability**
analyses using Item Response Theory (e.g. Choi et al., 2026, diagnosing LLM-as-a-judge
reliability via IRT/GRM) explicitly distinguish **intrinsic consistency** (is the
measurement instrument stable?) from **human alignment** (does it correspond to human
judgments?) — a distinction that maps onto the classic **calibration mismatch** (same
construct, different scale) versus **validity gap** (a *different latent construct* than
intended, in the construct-validity sense of Cronbach & Meehl, 1955). Work on
**human-label uncertainty** (Baan et al., 2023; Nie et al., 2020; Elangovan et al., 2025)
similarly argues that human annotations are not monolithic and that evaluation must
stratify by label (dis)agreement.

**Our relation.** Our result is naturally framed as a **validity gap**, not a calibration
mismatch: gold (NLI entailment) and human evidence-sufficiency are different latent
constructs (Cramér's V ≈ 0.28 between gold and our typed labels; gold adds ~0.04 over
base-rate in predicting human type). This reframes a would-be "model failure" as a
**measurement/҂construct finding**: the reason no uncertainty signal recovers the human
judgment is, in part, that the field's standard supervision (gold) targets a different
construct. We contribute (i) a blind, gold-independent relabeling protocol that separates
the two constructs, and (ii) a re-evaluation showing existing uncertainty signals align
with neither human evidence-sufficiency (overall) nor, in the well-powered stratum, with
it under gold control.

## 4. Positioning summary (one paragraph for intro)

Prior UQ work detects hallucination by estimating uncertainty about *correctness* and
validates against correctness labels. We show that, on a fact-verification subset,
(a) the dataset's gold labels (NLI entailment) and human judgments of *evidence
sufficiency* are distinct constructs that standard evaluation conflates, producing
gold-leakage that can masquerade as diagnostic signal; and (b) once the constructs are
separated by blind relabeling, no representative model-internal uncertainty signal
(6-axis, evidential DL, MC dropout) aligns with human evidence-sufficiency in the
well-powered stratum. The contribution is an evaluation-methodology and negative-result
paper, not a new detector.

---

### Reference list — VERIFIED 2026-06-20 (✓ = confirmed against original)
- ✓ Kuhn, Gal, Farquhar. Semantic Uncertainty: Linguistic Invariances for Uncertainty
  Estimation in NLG. ICLR 2023 (Oral). arXiv:2302.09664
- ✓ Farquhar, Kossen, Kuhn, Gal. Detecting hallucinations in LLMs using semantic entropy.
  Nature 630(8017):625–630, 2024. doi:10.1038/s41586-024-07421-0
- ✓ Kossen, Han, Razzak, Schut, Malik, Gal. Semantic Entropy Probes. 2024. arXiv:2406.15927
- ✓ Manakul, Liusie, Gales. SelfCheckGPT: Zero-resource Black-box Hallucination Detection.
  EMNLP 2023, pp. 9004–9017. arXiv:2303.08896
- ✓ Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi. FActScore. EMNLP
  2023. arXiv:2305.14251  [arXiv number confirmed; double-check full author list at typeset]
- ✓ Sensoy, Kaplan, Kandemir. Evidential Deep Learning to Quantify Classification
  Uncertainty. NeurIPS 2018 (NIPS 31). arXiv:1806.01768
- ✓ Gal & Ghahramani. Dropout as a Bayesian Approximation. ICML 2016. arXiv:1506.02142
- ✓ Schuster, Fisch, Barzilay. Get Your Vitamin C! Robust Fact Verification with Contrastive
  Evidence. NAACL-HLT 2021, pp. 624–643. arXiv:2103.08541. doi:10.18653/v1/2021.naacl-main.52
- ✓ Malinin & Gales. Uncertainty Estimation in Autoregressive Structured Prediction. ICLR
  2021. arXiv:2002.07650. (NOT 2020 — that is the arXiv preprint date; ICLR 2021 is the
  venue to cite.)
- ✓ Nikitin, Kossen, Gal, Marttinen. Kernel Language Entropy: Fine-grained Uncertainty
  Quantification for LLMs from Semantic Similarities. NeurIPS 2024 (vol. 37).
  arXiv:2405.20003.
- Cronbach & Meehl. Construct validity. 1955 (for validity-gap framing)
- ✓ Baan, Daheim, Ilia, Ulmer, Li, Fernández, Plank, Sennrich, Zerva, Aziz. Uncertainty in
  Natural Language Generation: From Theory to Applications. arXiv:2307.15703, 2023.
  [survey/position paper — cite as "(Baan et al., 2023, arXiv)", not as a venue paper]
- ✓ Nie, Zhou, Bansal. "What Can We Learn from Collective Human Opinions on Natural
  Language Inference Data?" EMNLP 2020, pp. 9131–9143. doi:10.18653/v1/2020.emnlp-main.734.
  Dataset name: ChaosNLI. CONFIRMED via ACL Anthology official BibTeX + multiple
  independent citing papers (2026-06-20).
- ✓ Choi, Park, Cho, Park, Kim. Diagnosing the Reliability of LLM-as-a-Judge via Item
  Response Theory. arXiv:2602.00521, submitted 31 Jan 2026 (v2: 29 May 2026). CONFIRMED
  to exist (post-cutoff paper, verified via search). Scope note: this paper is about
  LLM-judge reliability via IRT/GRM (intrinsic consistency + human alignment for LLM
  judges), not directly about annotation-taxonomy construct validity. Use as a supporting
  citation for "psychometric approaches to evaluation reliability" (e.g., "recent work has
  examined reliability/validity in LLM-based evaluation using psychometric approaches such
  as IRT, Choi et al. 2026") rather than as the primary construct-validity citation for our
  §4.5 — Cronbach & Meehl (1955) remains the right anchor for that.