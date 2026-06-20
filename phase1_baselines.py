# ── 이전 셀에서 epistemic_bert.py 를 먼저 실행해야 함 ──────────────────────
# (CFG, EpistemicBERT, FEVERDataset, build_optimizer 등이 이미 정의됨)

"""
Phase 1 — Baseline 비교 (evidential DL + MC dropout).

목적: "confident error를 유형(I/A/M)으로 가르는 게 우리 6축 고유 능력인가,
       아니면 기존 uncertainty 방법도 하는가?"

공정성:
  - 같은 backbone (DistilBERT-base), 같은 데이터 (VitaminC), 같은 epoch/seed
  - evidential: 처음부터 재학습 (Sensoy 2018 표준: Dirichlet + Bayes risk + KL anneal)
  - MC dropout: 우리 모델에 dropout 켜고 추론 ×N (재학습 없음)

핵심 테스트:
  evidential의 uncertainty u 가 우리 사람-라벨 I형(증거부족)과 M형(판정실패)을
  가르나, 뭉개나? confident error라 둘 다 u 낮으면 → 못 가름 = 우리가 채우는 갭.

출력:
  baseline_scores.csv — confident error id별 evidential u, MC dropout variance.
  (cw_answer_key.csv 의 id 와 정렬 → 나중에 사람 라벨과 합쳐 분석)
"""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
from datasets import load_dataset


# ══════════════════════════════════════════════════════════════════════════════
# Part A — Evidential Deep Learning (Sensoy et al. 2018)
# ══════════════════════════════════════════════════════════════════════════════

class EvidentialBERT(nn.Module):
    """DistilBERT + evidential head. 우리 모델과 같은 backbone, head만 다름.

    logits 대신 evidence(≥0) 출력 → Dirichlet α = evidence + 1.
    uncertainty u = K / S,  S = Σα.
    """
    def __init__(self, n_classes=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.n_classes = n_classes
        self.head = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        # evidence ≥ 0 (softplus). exp 대신 softplus가 안정적 (Sensoy 권장)
        evidence = F.softplus(self.head(cls))     # (B, K)
        alpha = evidence + 1.0                     # Dirichlet param
        return evidence, alpha

    @staticmethod
    def uncertainty(alpha):
        """u = K / S. evidence 적을수록 1에 가까움 (= epistemic uncertainty)."""
        K = alpha.size(-1)
        S = alpha.sum(dim=-1)
        return K / S                               # (B,)

    @staticmethod
    def prob(alpha):
        """expected class probability = α / S."""
        return alpha / alpha.sum(dim=-1, keepdim=True)


def evidential_loss(alpha, target, epoch, total_epochs, n_classes=3):
    """Bayes risk (MSE form) + KL 정규화 (annealed).

    Sensoy 2018 Eq.5 (MSE) + Eq.9 (KL). KL coef는 epoch에 따라 0→1 anneal
    (초반에 KL 너무 세면 학습 안 됨).
    """
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S
    y = F.one_hot(target, n_classes).float()

    # Bayes risk (expected sum of squares): Σ (y - p)² + p(1-p)/(S+1)
    err = ((y - p) ** 2).sum(dim=-1)
    var = (p * (1 - p) / (S + 1)).sum(dim=-1)
    mse = err + var

    # KL(Dirichlet(α~) || Dirichlet(1)) — 틀린 클래스 evidence를 0으로
    alpha_tilde = y + (1 - y) * alpha              # 정답 클래스는 1로 고정
    kl = _kl_dirichlet_uniform(alpha_tilde, n_classes)

    anneal = min(1.0, epoch / max(1, total_epochs // 2))   # 절반까지 0→1
    return (mse + anneal * 0.1 * kl).mean()


def _kl_dirichlet_uniform(alpha, K):
    """KL( Dir(alpha) || Dir(1,...,1) ).  alpha=1(균등)이면 0."""
    S = alpha.sum(dim=-1, keepdim=True)
    # ln B(alpha) = Sum lnGamma(a_k) - lnGamma(S)
    ln_b_alpha = torch.lgamma(alpha).sum(dim=-1, keepdim=True) - torch.lgamma(S)
    # ln B(1..1) = Sum lnGamma(1) - lnGamma(K) = -lnGamma(K)
    ln_b_uniform = -torch.lgamma(torch.tensor(float(K), device=alpha.device))
    # Sum (a_k - 1)(psi(a_k) - psi(S))
    term = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1, keepdim=True)
    # KL = ln B(uniform) - ln B(alpha) + term
    kl = ln_b_uniform - ln_b_alpha + term
    return kl.squeeze(-1)


def train_evidential(seed=42, ckpt_out="/kaggle/working/evidential_seed42.pt"):
    """VitaminC로 evidential 모델 재학습 (우리 모델과 같은 조건)."""
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = CFG["device"]
    tcfg = CFG["train"]

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    vc = load_dataset("tals/vitaminc")
    train_split = vc["train"].select(range(min(tcfg["train_size"], len(vc["train"]))))
    val_split = vc["validation"].select(range(min(tcfg["val_size"], len(vc["validation"]))))

    train_ds = FEVERDataset(train_split, tokenizer, tcfg["max_length"])
    val_ds = FEVERDataset(val_split, tokenizer, tcfg["max_length"])
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = EvidentialBERT(n_classes=3).to(device)
    # 우리 모델과 동일한 lr 구조 (backbone 2e-5, head 1e-3)
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": tcfg["lr_bert"]},
        {"params": model.head.parameters(), "lr": tcfg["lr_head"]},
    ], weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(tcfg["epochs"]):
        model.train()
        tot_loss = tot_correct = tot = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                evidence, alpha = model(ids, mask)
                loss = evidential_loss(alpha, lbl, epoch, tcfg["epochs"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            pred = alpha.argmax(-1)
            tot_correct += (pred == lbl).sum().item()
            tot += lbl.size(0)
            tot_loss += loss.item()
        print(f"  evidential epoch {epoch+1}: loss={tot_loss/len(train_loader):.4f} "
              f"acc={tot_correct/tot:.4f}")

    # val acc (공정성 확인 — 우리 모델 acc와 비슷해야 비교 정당)
    model.eval()
    vc_correct = vt = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)
            with torch.cuda.amp.autocast():
                _, alpha = model(ids, mask)
            vc_correct += (alpha.argmax(-1) == lbl).sum().item()
            vt += lbl.size(0)
    print(f"  evidential val acc = {vc_correct/vt:.4f}  (우리 모델 ~0.63과 비교)")

    torch.save(model.state_dict(), ckpt_out)
    print(f"  saved → {ckpt_out}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Part B — MC Dropout (우리 모델에 dropout 켜고 추론 ×N)
# ══════════════════════════════════════════════════════════════════════════════

def enable_dropout(model):
    """eval 모드에서도 dropout layer만 train 모드로 (MC dropout 핵심)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_uncertainty(model, ids, mask, n_samples=20):
    """dropout 켜고 N번 추론 → predictive variance + mean entropy.

    반환: (variance, mean_entropy) 둘 다 (B,)
      variance: N개 softmax의 분산 합 (epistemic 근사)
      mean_entropy: 평균 softmax의 entropy (total uncertainty)
    """
    model.eval()
    enable_dropout(model)          # dropout만 다시 켬

    probs = []
    for _ in range(n_samples):
        with torch.cuda.amp.autocast():
            logits, _, _ = model(ids, mask)
        probs.append(F.softmax(logits.float(), dim=-1))
    probs = torch.stack(probs, dim=0)          # (N, B, K)

    mean_p = probs.mean(dim=0)                  # (B, K)
    # predictive variance: 클래스별 분산의 합
    var = probs.var(dim=0).sum(dim=-1)          # (B,)
    # mean entropy
    ent = -(mean_p * (mean_p + 1e-8).log()).sum(dim=-1)
    return var, ent


# ══════════════════════════════════════════════════════════════════════════════
# Part C — confident error id 별 baseline 점수 추출
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_baseline_scores(
        our_ckpt_fmt="/kaggle/input/datasets/terryterry9/epistemicbert-3seed/epistemic_bert_seed{}.pt",
        evidential_ckpt="/kaggle/working/evidential_seed42.pt",
        seeds=(42, 7, 123),
        conf_thresh=0.95,
        mc_samples=20,
        out_csv="/kaggle/working/baseline_scores.csv"):
    """cw_answer_key.csv 의 confident error 들에 대해 baseline 점수 계산.

    각 confident error(우리 모델 기준)에 대해:
      - evidential u (해당 입력을 evidential 모델에 통과)
      - MC dropout variance / entropy (우리 모델에 dropout 추론)
    → baseline_scores.csv (id, seed, evidential_u, mc_var, mc_ent)

    id 는 cw_for_labeling.csv / cw_answer_key.csv 와 동일 순서로 정렬됨.
    사람 라벨 채워지면 셋을 id 로 join 해서 분석.
    """
    device = CFG["device"]
    tcfg = CFG["train"]
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    vc = load_dataset("tals/vitaminc")
    val_split = vc["validation"].select(range(min(tcfg["val_size"], len(vc["validation"]))))
    val_ds = FEVERDataset(val_split, tokenizer, tcfg["max_length"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    # evidential 모델 로드
    ev_model = EvidentialBERT(n_classes=3).to(device)
    ev_model.load_state_dict(torch.load(evidential_ckpt, map_location=device))
    ev_model.eval()

    rows = []   # (id, seed, raw_idx, our_conf, ev_u, mc_var, mc_ent)
    global_id = 0
    for s in seeds:
        # 우리 모델 (MC dropout + confident error 판정용)
        our_model = EpistemicBERT(**CFG["model"]).to(device)
        our_model.token_nov.set_special_tokens(tokenizer.all_special_ids)
        our_model.attn_ign.sep_id = tokenizer.sep_token_id
        our_model.load_state_dict(torch.load(our_ckpt_fmt.format(s), map_location=device))
        our_model.eval()

        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)
            bidx = batch["idx"].to(device)

            # 우리 모델 confident error 판정
            with torch.cuda.amp.autocast():
                logits, _, _ = our_model(ids, mask)
            prob = F.softmax(logits.float(), dim=-1)
            pred = logits.argmax(-1)
            conf = prob.max(-1).values

            cw_mask = (pred != lbl) & (conf > conf_thresh)
            if not cw_mask.any():
                continue

            cw_ids = ids[cw_mask]
            cw_masks = mask[cw_mask]
            cw_bidx = bidx[cw_mask]
            cw_conf = conf[cw_mask]

            # evidential u (같은 입력)
            with torch.cuda.amp.autocast():
                _, alpha = ev_model(cw_ids, cw_masks)
            ev_u = EvidentialBERT.uncertainty(alpha.float())

            # MC dropout (우리 모델)
            mc_var, mc_ent = mc_dropout_uncertainty(our_model, cw_ids, cw_masks, mc_samples)
            our_model.eval()   # 복구

            for j in range(cw_ids.size(0)):
                rows.append((global_id, s, int(cw_bidx[j].item()),
                             cw_conf[j].item(), ev_u[j].item(),
                             mc_var[j].item(), mc_ent[j].item()))
                global_id += 1
        print(f"  seed {s}: baseline 점수 추출 완료 (누적 {global_id}개)")

    with open(out_csv, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id", "seed", "raw_idx", "our_conf",
                    "evidential_u", "mc_var", "mc_ent"])
        w.writerows(rows)
    print(f"\nbaseline 점수 → {out_csv}  ({len(rows)}개)")

    # 빠른 진단: evidential u 가 confident error 에서 어떤 분포인지
    ev_us = np.array([r[4] for r in rows])
    print(f"\n  evidential u (confident error): "
          f"mean={ev_us.mean():.4f}  std={ev_us.std():.4f}  "
          f"min={ev_us.min():.4f}  max={ev_us.max():.4f}")
    print(f"  → u 가 낮고 분산 작으면: evidential도 confident error를 '확신'으로 봄")
    print(f"     (= 유형 구분 불가 가능성. 사람 라벨과 합쳐서 최종 확인)")
    return out_csv


# ══════════════════════════════════════════════════════════════════════════════
# Part D — 분석 (사람 라벨 채워진 뒤 실행)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_vs_human(
        human_csv="/kaggle/working/cw_for_labeling.csv",   # human_type 채워진 것
        key_csv="/kaggle/input/datasets/terryterry9/epistemicbert-3seed/cw_answer_key.csv",
        baseline_csv="/kaggle/working/baseline_scores.csv"):
    """사람 라벨(I/A/M) ground truth 대비 세 방법의 복원율 비교.

    ⚠️ 사람 라벨링(human_type 채우기)이 끝난 뒤에 실행.
    각 방법:
      - ours: auto_type (cw_answer_key.csv)  vs human
      - evidential: u 로 I/A/M 분리되나
      - MC dropout: var 로 I/A/M 분리되나
    metric: Accuracy / Macro-F1 / Cohen's κ + confusion matrix.
    """
    import pandas as pd
    from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                                 confusion_matrix)

    human = pd.read_csv(human_csv)
    key = pd.read_csv(key_csv)
    base = pd.read_csv(baseline_csv)

    df = human.merge(key, on="id").merge(base, on="id", suffixes=("", "_b"))
    df = df[df["human_type"].isin(["I", "A", "M"])].copy()   # O 제외
    print(f"분석 대상: {len(df)}개 (O/빈칸 제외)")

    if len(df) == 0:
        print("  ⚠️ human_type 이 안 채워졌거나 I/A/M 없음. 라벨링 먼저.")
        return

    # 사람 라벨 → 코드북 매핑: I=ignorance, A=ambiguity, M=metacognitive
    type_map = {"I": "ignorance", "A": "ambiguity", "M": "metacognitive"}
    df["human_axis"] = df["human_type"].map(type_map)

    # ── 방법 1: ours (auto_type) ──
    print("\n══ ours (6축 auto_type) vs human ══")
    _report(df["human_axis"], df["auto_type"])

    # ── 방법 2: evidential u → 유형 분리 ──
    # u 는 스칼라라 직접 I/A/M 못 냄. "u 가 유형을 가르나"를 본다:
    #   I형(증거부족)이 M형(판정실패)보다 u 가 높아야 evidential이 가른다는 것
    print("\n══ evidential u — 유형별 분포 (가르나 뭉개나) ══")
    for t in ["ignorance", "ambiguity", "metacognitive"]:
        m = df["human_axis"] == t
        if m.any():
            print(f"    {t:>14}: u mean={df.loc[m,'evidential_u'].mean():.4f} "
                  f"± {df.loc[m,'evidential_u'].std():.4f}  (n={m.sum()})")
    print("    → I형과 M형의 u 가 비슷하면 evidential은 둘을 뭉갬 (우리가 채우는 갭)")

    # ── 방법 3: MC dropout var → 유형 분리 ──
    print("\n══ MC dropout variance — 유형별 분포 ══")
    for t in ["ignorance", "ambiguity", "metacognitive"]:
        m = df["human_axis"] == t
        if m.any():
            print(f"    {t:>14}: var mean={df.loc[m,'mc_var'].mean():.6f} "
                  f"± {df.loc[m,'mc_var'].std():.6f}  (n={m.sum()})")

    print("\n핵심: ours 가 human 을 높은 κ/F1 로 복원하고,")
    print("      evidential u / MC var 는 I형과 M형을 못 가르면 → 우리 기여 확립.")


def _report(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                                 confusion_matrix)
    labels = ["ignorance", "ambiguity", "metacognitive"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    print(f"    Accuracy={acc:.4f}  Macro-F1={f1:.4f}  Cohen κ={kappa:.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"    confusion (행=human, 열=pred): {labels}")
    for i, l in enumerate(labels):
        print(f"      {l:>14}: {cm[i].tolist()}")


# ── 실행 (새 셀에서 순서대로) ────────────────────────────────────────────────
# 1) evidential 재학습 (~9분):
#    train_evidential(seed=42)
#
# 2) baseline 점수 추출 (evidential u + MC dropout):
#    extract_baseline_scores()
#
# 3) (사람 라벨링 끝난 뒤) 분석:
#    analyze_vs_human()

print("Phase 1 baseline 모듈 로드됨.")
print("실행: train_evidential(seed=42) → extract_baseline_scores() → (라벨 후) analyze_vs_human()")