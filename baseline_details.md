# Baseline & Method Details (for §3 Setup + Reproducibility Appendix)

> Extracted verbatim from epistemic_bert.py and phase1_baselines.py. All three methods
> share the same DistilBERT-base-uncased backbone, the same VitaminC data, and the same
> training budget, so differences cannot be attributed to backbone or data scale.

## Shared experimental configuration
| item | value |
|------|-------|
| Backbone | DistilBERT-base-uncased (HuggingFace) |
| Dataset | VitaminC (tals/vitaminc); train 30,000 / val 5,000 |
| Input format | `[CLS] claim [SEP] evidence [SEP]`, max_length 192 |
| Labels | SUPPORTS=0, REFUTES=1, NEI=2 |
| Optimizer | AdamW, weight_decay 1e-2 |
| LR | backbone 2e-5, head 1e-3 |
| Batch size | 32 |
| Epochs | 3 |
| Mixed precision | AMP (CUDA) |
| Seeds | 42, 7, 123 (our model + MC dropout); evidential trained at seed 42 |
| Confident-error criterion | prediction ≠ gold AND max softmax prob > 0.95 |

Note on seeds: the evidential baseline is trained once (seed 42) and serves as a fixed
external scorer; it evaluates the confident-errors surfaced by each of our 3-seed models.
Evidential uncertainty depends only on the input text and the evidential model's weights,
so a single evidential model can score confident-errors from any seed. (If a reviewer asks,
3-seed evidential is a cheap add; the current single-seed is the standard baseline setup.)

## Baseline A — Evidential Deep Learning (Sensoy et al., 2018)
Same backbone, evidential head only (head differs from ours).
- Head: `Linear(768 → 3)`, then `evidence = softplus(logits)`, Dirichlet `α = evidence + 1`.
  (softplus, not exp — the numerically stable variant Sensoy recommends.)
- Epistemic uncertainty: **u = K / S**, where K = 3 (classes), S = Σα. Low evidence → u → 1.
- Expected class prob: `p = α / S`.
- Loss: Bayes-risk (MSE form, Sensoy Eq.5) + KL-to-uniform regularizer (Eq.9),
  KL annealed `min(1, epoch/(epochs//2))` × 0.1 (0→1 over the first half of training).
  KL term zeroes evidence on wrong classes via `α̃ = y + (1−y)·α`.
- Training: same optimizer/LR/epochs as above; saved to `evidential_seed42.pt`.
- Fairness check: report evidential val accuracy vs our model's (~0.63) — comparison is
  only valid if accuracies are comparable.

## Baseline B — MC Dropout (Gal & Ghahramani, 2016)
Applied to OUR trained model (no retraining); standard predictive-uncertainty baseline.
- At inference, set all `nn.Dropout` layers to train mode (rest eval), run N=20 stochastic
  forward passes.
- Predictive variance: `var = probs.var(dim=0).sum(-1)` (sum of per-class variances over
  the 20 samples) — the `mc_var` score.
- Mean entropy: entropy of the mean softmax over 20 samples — the `mc_ent` score.

## Our method — EpistemicBERT (the system under test, not a "win" claim here)
Six axes; the three uncertainty axes are sourced OFF the classification gradient (detached):
| axis | source | detached? |
|------|--------|-----------|
| novelty | token rarity: −log running-train frequency, special tokens masked | yes (token stats) |
| ambiguity | layer disagreement: directional variance of [CLS] over final 4 layers | yes (representation) |
| ignorance | cross-attention alignment: claim→evidence attention mass split at [SEP], 1−align | yes (attention) |
| truth / error / contradiction | dual-stream geometry: support/counter margin, energy, agreement | no (learned plane) |
- proj_dim 128, n_prototypes 32 (16 support + 16 counter).
- Loss (weights): 0.3·CE + 1.0·field_ranking + 0.2·margin + 0.1·diversity + 0.03·disentangle.
  field_ranking maps SUPPORTS→truth, REFUTES→error, NEI→ignorance.
- Uncertainty axes use running-mean/std (EMA) standardization → sigmoid.

## Per-id baseline score extraction (how the comparison set is built)
For each confident-error (per seed), record into `baseline_scores.csv`:
`id, seed, raw_idx, our_conf, evidential_u, mc_var, mc_ent`.
- evidential_u: that input passed through the (seed-42) evidential model.
- mc_var / mc_ent: 20-sample MC dropout on our model.
- The 6 axis scores (truth…ignorance) come from our model's field output for the same ids.
These scores are functions of input + model weights only — INDEPENDENT of human labels,
so re-labeling never requires re-extracting them (only re-running the analysis join).

## Evaluation against human labels
- Merge `baseline_scores.csv` + human I/A/M labels on `id`.
- Per signal, AUROC for separating human I vs M (and the sufficiency split I vs A+M).
- Stratify by gold (REFUTES/SUPPORTS/NEI) to control for gold confounding; bootstrap CIs
  (2000 iters) because some gold strata have small minority classes.

## Citations (VERIFIED against originals — 2026-06-20)
- Sensoy, M., Kaplan, L., Kandemir, M. "Evidential Deep Learning to Quantify Classification
  Uncertainty." Advances in Neural Information Processing Systems 31 (NeurIPS 2018).
  arXiv:1806.01768. [author order per official NeurIPS page: Sensoy, Kaplan, Kandemir]
- Gal, Y., Ghahramani, Z. "Dropout as a Bayesian Approximation: Representing Model
  Uncertainty in Deep Learning." ICML 2016. arXiv:1506.02142.
- Schuster, T., Fisch, A., Barzilay, R. "Get Your Vitamin C! Robust Fact Verification with
  Contrastive Evidence." NAACL-HLT 2021, pp. 624–643. arXiv:2103.08541.
  doi:10.18653/v1/2021.naacl-main.52.
- DistilBERT: Sanh, V., Debut, L., Chaumond, J., Wolf, T. "DistilBERT, a distilled version
  of BERT." 2019. arXiv:1910.01108. [Sanh author list — verify co-authors before final]