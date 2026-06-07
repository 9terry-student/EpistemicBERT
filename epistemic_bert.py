import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizerFast
from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression


# ─── 1. Dual-Stream Prototype Manifold ────────────────────────────────────────
class PrototypeManifold(nn.Module):
    def __init__(self, embed_dim: int, n_prototypes: int = 32):
        super().__init__()
        half = n_prototypes // 2
        self.proto_s = nn.Parameter(torch.randn(half, embed_dim))
        self.proto_c = nn.Parameter(torch.randn(half, embed_dim))

    def _stream_stats(self, z_norm, proto):
        p_norm = F.normalize(proto, dim=-1)
        sim = z_norm @ p_norm.T
        max_sim, _ = sim.max(dim=-1)
        return max_sim

    def forward(self, z: torch.Tensor) -> dict:
        z_norm = F.normalize(z, dim=-1)

        support = self._stream_stats(z_norm, self.proto_s)
        counter = self._stream_stats(z_norm, self.proto_c)

        # ── prototype collapse 방지 ──────────────────────────────────────
        s_norm = F.normalize(self.proto_s, dim=-1)
        c_norm = F.normalize(self.proto_c, dim=-1)

        cross_div = (s_norm @ c_norm.T).abs().mean()

        K_s = s_norm.size(0)
        K_c = c_norm.size(0)
        gram_s = s_norm @ s_norm.T
        gram_c = c_norm @ c_norm.T
        eye_s = torch.eye(K_s, device=z.device)
        eye_c = torch.eye(K_c, device=z.device)
        intra_s = (gram_s - eye_s).pow(2).mean()
        intra_c = (gram_c - eye_c).pow(2).mean()

        diversity_loss = cross_div + 0.5 * (intra_s + intra_c)

        return {
            "support": support,
            "counter": counter,
            "diversity_loss": diversity_loss,
        }


# ─── 2. Epistemic Field Classifier ────────────────────────────────────────────
class EpistemicFieldClassifier(nn.Module):

    AXES = ['truth', 'error', 'contradiction', 'novelty', 'ambiguity', 'ignorance']

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scales = nn.Parameter(torch.ones(6))
        self.truth_temp = nn.Parameter(torch.tensor(0.2))

        self.contra_bias = nn.Parameter(torch.tensor(1.0))
        self.contra_temp = nn.Parameter(torch.tensor(0.2))

    def forward(self, manifold_out: dict, ignorance: torch.Tensor) -> dict:
        support = manifold_out["support"]
        counter = manifold_out["counter"]
        novelty_score = manifold_out["novelty_score"]
        ambiguity_score = manifold_out["ambiguity_score"]

        temp = F.softplus(self.truth_temp).clamp(min=0.05)
        margin_s = (support - counter).clamp(min=0.0)
        margin_c = (counter - support).clamp(min=0.0)

        # ── evidence plane ───────────────────────────────────────────────
        truth = support * torch.sigmoid(margin_s / temp)
        error = counter * torch.sigmoid(margin_c / temp)

        energy_sc = support + counter
        agree = (support - counter).abs().clamp(0.0, 1.0)
        c_temp = F.softplus(self.contra_temp).clamp(min=0.05)
        contradiction = torch.sigmoid((energy_sc - F.softplus(self.contra_bias)) / c_temp) * (1.0 - agree)

        # ── independent uncertainty sources ──────────────────────────────
        novelty = novelty_score
        ambiguity = ambiguity_score

        raw = torch.stack(
            [truth, error, contradiction, novelty, ambiguity, ignorance], dim=-1
        )
        scales_norm = F.softmax(self.scales, dim=0) * 6.0
        field = raw * scales_norm

        if self.training:
            dominant_type = None
            active_states = None
        else:
            dominant_idx = field.argmax(dim=-1)
            dominant_type = [self.AXES[i] for i in dominant_idx.cpu().tolist()]
            active_states = []
            for sample in field:
                th = sample.mean()
                active = [axis for idx, axis in enumerate(self.AXES) if sample[idx] > th]
                active_states.append(active if active else ["ignorance"])

        return {
            "field": field,
            "dominant_type": dominant_type,
            "active_states": active_states,
            "truth": truth,
            "error": error,
            "contradiction": contradiction,
            "novelty": novelty,
            "ambiguity": ambiguity,
            "ignorance": ignorance,
            "support_raw": support,
            "counter_raw": counter,
        }


# ─── Token-Novelty Source ─────────────────────────────────────────────────────
class TokenNovelty(nn.Module):
    """학습 어휘 대비 입력의 신규성 — 표면 형태, task/z와 완전 독립."""

    def __init__(self, vocab_size: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.register_buffer("token_count", torch.zeros(vocab_size))
        self.register_buffer("total", torch.tensor(0.0))
        # 특수토큰 마스킹용 (PAD/CLS/SEP은 신규성 계산서 제외)
        self.register_buffer("special_mask", torch.zeros(vocab_size, dtype=torch.bool))
        self.detach_out = True

    def set_special_tokens(self, ids):
        self.special_mask[torch.tensor(ids)] = True

    @torch.no_grad()
    def _update(self, input_ids, attention_mask):
        valid = input_ids[attention_mask.bool()]
        self.token_count.index_add_(0, valid, torch.ones_like(valid, dtype=torch.float))
        self.total += valid.numel()

    def forward(self, input_ids, attention_mask):
        if self.training:
            self._update(input_ids, attention_mask)

        # 토큰별 학습빈도 → 희귀도 = -log(freq), 미등장이면 최대
        total = self.total.clamp(min=1.0)
        freq = self.token_count[input_ids] / total          # (B, T)
        rarity = -(freq + self.eps).log()                    # 희귀할수록 큼

        # 특수토큰·패딩 제외하고 문장 평균
        mask = attention_mask.bool() & ~self.special_mask[input_ids]
        rarity = rarity * mask.float()
        denom = mask.float().sum(dim=-1).clamp(min=1.0)
        sent_rarity = rarity.sum(dim=-1) / denom             # (B,)

        # 즉석 정규화 — log(eps) 기준 (미등장 토큰 = -log(eps) 부근)
        max_rarity = -torch.log(torch.tensor(self.eps, device=input_ids.device))
        novelty = (sent_rarity / max_rarity).clamp(0.0, 1.0)
        return novelty.detach() if self.detach_out else novelty


# ─── Attention-based Ignorance Source ─────────────────────────────────────────
class AttentionIgnorance(nn.Module):
    """claim→evidence cross-attention 정렬 부족 = ignorance.
    mode='dispersion'은 ablation용 (방향성 없는 attention entropy)."""

    def __init__(self, sep_id: int = 102, momentum: float = 0.01, eps: float = 1e-6):
        super().__init__()
        self.sep_id = sep_id
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("mis_mean", torch.tensor(0.5))
        self.register_buffer("mis_std", torch.tensor(0.15))
        self.register_buffer("initialized", torch.tensor(False))
        self.mode = "crossattn"      # "crossattn"(기본) | "dispersion"(ablation)
        self.detach_out = True

    @torch.no_grad()
    def _update(self, d):
        m = self.momentum
        if not self.initialized:
            self.mis_mean.copy_(d.mean())
            self.mis_std.copy_(d.std() + self.eps)
            self.initialized.fill_(True)
        else:
            self.mis_mean.mul_(1 - m).add_(d.mean(), alpha=m)
            self.mis_std.mul_(1 - m).add_(d.std() + self.eps, alpha=m)

    def forward(self, attentions, input_ids, attention_mask):
        A = attentions[-1].detach().mean(dim=1)               # (B, S, S) head 평균

        if self.mode == "dispersion":
            # ablation: cross-attention 대신 단순 attention entropy (방향성 없음)
            amask = attention_mask.bool()
            a = A * amask.unsqueeze(1).float()
            a = a / (a.sum(dim=-1, keepdim=True) + self.eps)
            ent = -(a * (a + self.eps).log()).sum(dim=-1)      # (B,S) 각 query entropy
            qf = amask.float()
            mis = (ent * qf).sum(-1) / qf.sum(-1).clamp(min=1.0)  # 길이평균 entropy
        else:
            # 기본: claim→evidence 정렬 (cross-attention)
            is_sep = (input_ids == self.sep_id)
            sep_cumsum = is_sep.cumsum(dim=1)
            amask = attention_mask.bool()
            claim_mask = (sep_cumsum == 0) & amask
            claim_mask[:, 0] = False
            evid_mask = (sep_cumsum == 1) & (~is_sep) & amask
            evid_f = evid_mask.unsqueeze(1).float()
            mass = (A * evid_f).sum(dim=-1)
            claim_f = claim_mask.float()
            align = (mass * claim_f).sum(dim=-1) / claim_f.sum(dim=-1).clamp(min=1.0)
            mis = 1.0 - align

        if self.training:
            self._update(mis.detach())
        mu = self.mis_mean.detach()
        std = self.mis_std.detach().clamp(min=self.eps)
        ignorance = torch.sigmoid((mis - mu) / std)
        return ignorance.detach() if self.detach_out else ignorance


# ─── Layer-Disagreement Ambiguity Source ──────────────────────────────────────
class LayerAmbiguity(nn.Module):
    """layer 간 [CLS] 방향 불일치 — 해석 비수렴. manifold(분류)와 독립 소스."""

    def __init__(self, momentum: float = 0.01, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("disp_mean", torch.tensor(0.3))
        self.register_buffer("disp_std", torch.tensor(0.15))
        self.register_buffer("initialized", torch.tensor(False))
        self.detach_out = True

    @torch.no_grad()
    def _update(self, d):
        m = self.momentum
        if not self.initialized:
            self.disp_mean.copy_(d.mean())
            self.disp_std.copy_(d.std() + self.eps)
            self.initialized.fill_(True)
        else:
            self.disp_mean.mul_(1 - m).add_(d.mean(), alpha=m)
            self.disp_std.mul_(1 - m).add_(d.std() + self.eps, alpha=m)

    def forward(self, hidden_states):
        # 마지막 4개 layer의 [CLS] 방향 분산 (초기 layer는 표면적이라 제외)
        cls_layers = torch.stack([h[:, 0] for h in hidden_states[-4:]], dim=1)  # (B, L, H)
        cls_dir = F.normalize(cls_layers, dim=-1)            # 방향만
        mean_dir = F.normalize(cls_dir.mean(dim=1), dim=-1)  # (B, H) 평균 방향

        cos = (cls_dir * mean_dir.unsqueeze(1)).sum(dim=-1).clamp(-1, 1)  # (B, L)
        disp = (1.0 - cos).mean(dim=-1)                      # (B,) 평균 불일치

        if self.training:
            self._update(disp.detach())

        mu = self.disp_mean.detach()
        std = self.disp_std.detach().clamp(min=self.eps)
        ambiguity = torch.sigmoid((disp - mu) / std)
        return ambiguity.detach() if self.detach_out else ambiguity


# ─── 3. EpistemicBERT ─────────────────────────────────────────────────────────
class EpistemicBERT(nn.Module):

    def __init__(
        self,
        n_classes: int = 3,
        n_prototypes: int = 32,
        proj_dim: int = 128,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert.config.output_attentions = True
        self.bert.config.output_hidden_states = True
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(768, proj_dim)
        self.manifold = PrototypeManifold(proj_dim, n_prototypes)
        self.token_nov = TokenNovelty(self.bert.config.vocab_size)
        self.attn_ign = AttentionIgnorance()
        self.layer_amb = LayerAmbiguity()
        self.epistemic = EpistemicFieldClassifier()

        self.field_proj = nn.Linear(6, n_classes)
        self.z_proj = nn.Linear(proj_dim, n_classes)

        self.margin_param = nn.Parameter(torch.tensor(0.2))
        self.energy_ceiling = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True,
        )
        cls = bert_out.last_hidden_state[:, 0]

        z = self.proj(cls)
        manifold_out = self.manifold(z)
        manifold_out["novelty_score"] = self.token_nov(input_ids, attention_mask)
        manifold_out["ambiguity_score"] = self.layer_amb(bert_out.hidden_states)

        ignorance = self.attn_ign(bert_out.attentions, input_ids, attention_mask)

        epistemic_out = self.epistemic(manifold_out, ignorance)

        field = epistemic_out["field"]
        logits = self.field_proj(field) + self.z_proj(z)

        return logits, epistemic_out, manifold_out["diversity_loss"]


# ─── Config ───────────────────────────────────────────────────────────────────
CFG = dict(
    model=dict(
        n_classes=3,
        n_prototypes=32,
        proj_dim=128,
        freeze_bert=False,
    ),
    train=dict(
        batch_size=32,
        epochs=3,
        lr_bert=2e-5,
        lr_head=1e-3,
        max_length=192,
        lambda_ce=0.3,
        lambda_field=1.0,
        lambda_margin=0.2,
        lambda_diversity=0.1,
        ranking_margin=0.1,
        train_size=30_000,
        val_size=5_000,
        lambda_disent=0.03,
    ),
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class FEVERDataset(Dataset):
    LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    def __init__(self, hf_split, tokenizer, max_length):
        self.data = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        lbl = item["label"]
        lbl = self.LABEL_MAP[lbl] if isinstance(lbl, str) else int(lbl)
        enc = self.tokenizer(
            item["claim"], item["evidence"],          # claim 먼저 → [CLS] claim [SEP] evidence [SEP]
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(lbl, dtype=torch.long),
        }


class OODDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length):
        self.data = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["sentence1"], item["sentence2"],
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ─── Loss ─────────────────────────────────────────────────────────────────────
def field_ranking_loss(field, labels, margin: float = 0.1, eps: float = 1e-8):
    loss = torch.tensor(0.0, device=field.device)
    n = torch.tensor(0, device=field.device)

    sup_mask = (labels == 0)   # SUPPORTS → truth(0)
    ref_mask = (labels == 1)   # REFUTES  → error(1)
    nei_mask = (labels == 2)   # NEI      → ignorance(5)

    if sup_mask.any():
        f = field[sup_mask]
        for idx in [1, 2, 3, 4, 5]:
            loss += F.relu(margin - (f[:, 0] - f[:, idx])).mean()
        n += sup_mask.sum()

    if ref_mask.any():
        f = field[ref_mask]
        for idx in [0, 2, 3, 4, 5]:
            loss += F.relu(margin - (f[:, 1] - f[:, idx])).mean()
        n += ref_mask.sum()

    if nei_mask.any():
        f = field[nei_mask]
        for idx in [0, 1, 2, 3, 4]:
            loss += F.relu(margin - (f[:, 5] - f[:, idx])).mean()
        n += nei_mask.sum()

    return loss / (n.float() + eps)


def margin_loss(support, counter, labels, margin_param, energy_ceiling):
    margin = F.softplus(margin_param).clamp(min=0.05)
    ceiling = F.softplus(energy_ceiling).clamp(min=0.2)

    loss = torch.tensor(0.0, device=support.device)
    sup_mask = (labels == 0)   # SUPPORTS: support > counter
    ref_mask = (labels == 1)   # REFUTES : counter > support
    nei_mask = (labels == 2)   # NEI     : 둘 다 낮게 (관련 증거 없음)

    if sup_mask.any():
        loss += F.relu(counter[sup_mask] - support[sup_mask] + margin).mean()

    if ref_mask.any():
        loss += F.relu(support[ref_mask] - counter[ref_mask] + margin).mean()

    if nei_mask.any():
        energy = support[nei_mask] ** 2 + counter[nei_mask] ** 2
        loss += F.relu(energy - ceiling).mean()

    return loss / 3.0


# ─── Optimizer ────────────────────────────────────────────────────────────────
def build_optimizer(model, cfg):
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.proj.parameters())
        + list(model.manifold.parameters())
        + list(model.epistemic.parameters())
        + list(model.field_proj.parameters())
        + list(model.z_proj.parameters())
        + [model.margin_param, model.energy_ceiling]
    )
    return torch.optim.AdamW([
        {"params": bert_params, "lr": cfg["lr_bert"]},
        {"params": head_params, "lr": cfg["lr_head"]},
    ], weight_decay=1e-2)


# ─── Train ────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, device, tcfg):

    def disentangle_loss(eout):
        def corr(a, b):
            a_c = a - a.mean()
            b_c = b - b.mean()
            return ((a_c * b_c).mean() / (a_c.std() * b_c.std() + 1e-8)).abs()

        nov = eout["novelty"]
        amb = eout["ambiguity"]
        ign = eout["ignorance"]
        return (corr(nov, amb) + corr(amb, ign)) / 2.0

    model.train()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = total_ce = total_field = total_correct = total = 0

    for batch in tqdm(loader, desc="train", mininterval=10.0, ncols=80):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits, eout, div_loss = model(input_ids, attention_mask)

            ce = ce_fn(logits, labels)
            f_loss = field_ranking_loss(eout["field"], labels, tcfg["ranking_margin"])
            m_loss = margin_loss(
                eout["support_raw"], eout["counter_raw"], labels,
                model.margin_param, model.energy_ceiling,
            )
            d_loss = disentangle_loss(eout)
            loss = (
                tcfg["lambda_ce"] * ce
                + tcfg["lambda_field"] * f_loss
                + tcfg["lambda_margin"] * m_loss
                + tcfg["lambda_diversity"] * div_loss
                + tcfg["lambda_disent"] * d_loss
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  nan/inf  ce={ce.item():.4f}  f={f_loss.item():.4f}  m={m_loss.item():.4f}")
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        total_ce += ce.item()
        total_field += f_loss.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "field": total_field / n,
        "acc": total_correct / total,
    }


# ─── Evaluate ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    AXES = EpistemicFieldClassifier.AXES

    axis_sums = defaultdict(lambda: defaultdict(float))
    axis_counts = defaultdict(int)
    s_sums = defaultdict(float)
    c_sums = defaultdict(float)
    total_loss = total_correct = total = 0
    field_all = []

    for batch in tqdm(loader, desc="eval", mininterval=10.0, ncols=80):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.cuda.amp.autocast():
            logits, eout, _ = model(input_ids, attention_mask)
            loss = ce_fn(logits, labels)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item()
        total += labels.size(0)
        field_all.append(eout["field"].cpu().float())

        for b in range(labels.size(0)):
            lbl = labels[b].item()
            axis_counts[lbl] += 1
            s_sums[lbl] += eout["support_raw"][b].item()
            c_sums[lbl] += eout["counter_raw"][b].item()
            for a, ax in enumerate(AXES):
                axis_sums[lbl][ax] += eout["field"][b, a].item()

    label_names = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI"}
    field_by_label = {
        label_names[lbl]: {ax: axis_sums[lbl][ax] / axis_counts[lbl] for ax in AXES}
        for lbl in label_names if axis_counts[lbl] > 0
    }
    field_cat = torch.cat(field_all, dim=0)
    monitor = {
        "field_mean": field_cat.mean(0).numpy().round(4).tolist(),
        "field_std": field_cat.std(0).numpy().round(4).tolist(),
    }
    return {
        "loss": total_loss / len(loader),
        "acc": total_correct / total,
        "field_by_label": field_by_label,
        "monitor": monitor,
        "support_by_label": {
            label_names[lbl]: {
                "support": s_sums[lbl] / axis_counts[lbl],
                "counter": c_sums[lbl] / axis_counts[lbl],
            }
            for lbl in label_names if axis_counts[lbl] > 0
        },
    }


@torch.no_grad()
def evaluate_ood(model, ood_loader, id_loader, device):
    model.eval()
    AXES = EpistemicFieldClassifier.AXES

    def collect(loader):
        fs = []
        for batch in tqdm(loader, desc="ood", mininterval=10.0, ncols=80):
            with torch.cuda.amp.autocast():
                _, eout, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
            fs.append(eout["field"].cpu())
        return torch.cat(fs, dim=0)

    id_f = collect(id_loader)
    ood_f = collect(ood_loader)
    return {
        ax: {"id_mean": id_f[:, i].mean().item(), "ood_mean": ood_f[:, i].mean().item()}
        for i, ax in enumerate(AXES)
    }


@torch.no_grad()
def identifiability_probe(model, loader, device):
    model.eval()
    AXES = EpistemicFieldClassifier.AXES

    fields = []
    for batch in tqdm(loader, desc="ident", mininterval=10.0, ncols=80):
        _, eout, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        fields.append(eout["field"].cpu().float())
    F_mat = torch.cat(fields, dim=0).numpy()

    # plane 축(0,1,2)은 redundant 정상, uncertainty 축(3,4,5)만 independent 기대
    print(f"\n{'axis':>16}  {'R² from others':>16}  {'group':>10}  {'verdict':>14}")
    print("─" * 62)
    groups = {0: "plane", 1: "plane", 2: "plane", 3: "uncert", 4: "uncert", 5: "uncert"}
    for i, ax in enumerate(AXES):
        y = F_mat[:, i]
        X = np.delete(F_mat, i, axis=1)
        r2 = LinearRegression().fit(X, y).score(X, y)
        if groups[i] == "plane":
            verdict = "plane-coord (OK)" if r2 > 0.6 else "unexpected-indep"
        else:
            verdict = "LEAK" if r2 > 0.6 else "independent (OK)"
        print(f"{ax:>16}  {r2:>16.4f}  {groups[i]:>10}  {verdict:>14}")

    corr = np.corrcoef(F_mat.T)
    print(f"\n  |correlation| matrix:")
    print(f"{'':>14}" + "".join(f"{a[:5]:>8}" for a in AXES))
    for i, ax in enumerate(AXES):
        row = "".join(f"{abs(corr[i, j]):>8.3f}" for j in range(6))
        print(f"{ax:>14}{row}")


@torch.no_grad()
def diagnose_ambiguity(model, loader, device):
    """ambiguity가 plane으로 새는 게 정규화 포화 때문인지, 소스 본질 결함인지 진단."""
    model.eval()
    la = model.layer_amb

    print(f"\n  LayerAmbiguity EMA 버퍼:")
    print(f"    disp_mean = {la.disp_mean.item():.4f}")
    print(f"    disp_std  = {la.disp_std.item():.4f}   (< 0.03이면 포화 의심)")
    print(f"    initialized = {la.initialized.item()}")

    raw_disps, amb_out, errs, contras = [], [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            out = model.bert(input_ids=ids, attention_mask=mask)
            _, eout, _ = model(ids, mask)
        hs = out.hidden_states
        cls_layers = torch.stack([h[:, 0] for h in hs[-4:]], dim=1)
        cls_dir = F.normalize(cls_layers.float(), dim=-1)
        mean_dir = F.normalize(cls_dir.mean(dim=1), dim=-1)
        cos = (cls_dir * mean_dir.unsqueeze(1)).sum(dim=-1).clamp(-1, 1)
        disp = (1.0 - cos).mean(dim=-1)
        raw_disps.append(disp.cpu().numpy())
        amb_out.append(eout["ambiguity"].float().cpu().numpy())
        errs.append(eout["error"].float().cpu().numpy())
        contras.append(eout["contradiction"].float().cpu().numpy())

    raw = np.concatenate(raw_disps)
    amb = np.concatenate(amb_out)
    err = np.concatenate(errs)
    con = np.concatenate(contras)

    print(f"\n  raw disp (정규화 전, task-free 신호):")
    print(f"    mean={raw.mean():.4f}  std={raw.std():.4f}  min={raw.min():.4f}  max={raw.max():.4f}")
    print(f"  ambiguity (정규화 후, field):")
    print(f"    mean={amb.mean():.4f}  std={amb.std():.4f}   (< 0.05면 거의 상수=포화)")

    def corr(a, b):
        return abs(np.corrcoef(a, b)[0, 1])
    print(f"\n  ── raw disp의 plane 상관 (소스 본질 결함 여부) ──")
    print(f"    raw_disp ↔ error  = {corr(raw, err):.4f}")
    print(f"    raw_disp ↔ contra = {corr(raw, con):.4f}")
    print(f"  ── 정규화 후 ambiguity의 plane 상관 ──")
    print(f"    ambiguity ↔ error  = {corr(amb, err):.4f}")
    print(f"    ambiguity ↔ contra = {corr(amb, con):.4f}")
    print(f"\n  판정: raw도 높으면(>0.5) 소스 본질 결함 / raw 낮고 정규화 후만 높으면 EMA 포화")


@torch.no_grad()
def evaluate_selective_prediction(model, loader, device):
    """by-design 확인용: uncertainty 축은 confidence-independent라 risk-coverage에서 baseline에 짐."""
    model.eval()

    correct_all = []
    acc = {
        "entropy (baseline)": [],
        "1-MSP (baseline)": [],
        "ambiguity (ours)": [],
        "error (ours)": [],
        "amb+error (ours)": [],
        "amb+ign+nov (ours)": [],
        "all-unc+error (ours)": [],
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.cuda.amp.autocast():
            logits, eout, _ = model(input_ids, attention_mask)

        prob = F.softmax(logits.float(), dim=-1)
        preds = logits.argmax(dim=-1)
        n_cls = prob.size(-1)

        amb = eout["ambiguity"].float()
        ign = eout["ignorance"].float()
        nov = eout["novelty"].float()
        err = eout["error"].float()

        ent = -(prob * (prob + 1e-8).log()).sum(dim=-1)
        ent = ent / torch.log(torch.tensor(float(n_cls), device=device))
        msp = 1.0 - prob.max(dim=-1).values

        correct_all.append((preds == labels).cpu())
        acc["entropy (baseline)"].append(ent.cpu())
        acc["1-MSP (baseline)"].append(msp.cpu())
        acc["ambiguity (ours)"].append(amb.cpu())
        acc["error (ours)"].append(err.cpu())
        acc["amb+error (ours)"].append((amb + err).cpu())
        acc["amb+ign+nov (ours)"].append((amb + ign + nov).cpu())
        acc["all-unc+error (ours)"].append((amb + ign + nov + err).cpu())

    correct = torch.cat(correct_all).numpy().astype(float)
    scores = {name: torch.cat(parts).numpy() for name, parts in acc.items()}
    return _risk_coverage_report(correct, scores)


def _risk_coverage_report(correct, scores):
    N = len(correct)
    base_risk = 1.0 - correct.mean()

    print(f"\n  base error rate (coverage=100%): {base_risk:.4f}")
    print(f"\n{'method':>24}  {'AURC':>8}  {'risk@90%':>9}  {'risk@80%':>9}  {'risk@70%':>9}")
    print("─" * 68)

    results = {}
    for name, unc in scores.items():
        order = np.argsort(unc)            # 확신 높은(불확실 낮은) 것부터
        c_sorted = correct[order]
        risks = 1.0 - np.cumsum(c_sorted) / np.arange(1, N + 1)
        aurc = risks.mean()

        def risk_at(cov):
            k = max(1, int(cov * N))
            return 1.0 - c_sorted[:k].mean()

        results[name] = {"aurc": aurc, "risk90": risk_at(0.9),
                         "risk80": risk_at(0.8), "risk70": risk_at(0.7)}
        print(f"{name:>24}  {aurc:>8.4f}  {results[name]['risk90']:>9.4f}"
              f"  {results[name]['risk80']:>9.4f}  {results[name]['risk70']:>9.4f}")

    base_keys = [k for k in scores if "baseline" in k]
    ours_keys = [k for k in scores if "ours" in k]
    best_base = min(results[k]["aurc"] for k in base_keys)
    best_ours_key = min(ours_keys, key=lambda k: results[k]["aurc"])
    best_ours = results[best_ours_key]["aurc"]

    print(f"\n  best baseline AURC = {best_base:.4f}")
    print(f"  best ours AURC     = {best_ours:.4f}  ({best_ours_key})")
    print(f"  → {'OURS WINS' if best_ours < best_base else 'baseline wins'} "
          f"(Δ={best_base - best_ours:+.4f})")
    return results


@torch.no_grad()
def failure_typing(model, loader, tokenizer, device, conf_thresh=0.95, k_examples=3):
    """메인 결과: confident wrong을 6축으로 유형 분리.
    silent-wrong + high-conf Δ(uncertainty 축이 안 움직임)도 함께 보고."""
    model.eval()
    AXES = EpistemicFieldClassifier.AXES
    unc_idx = {"novelty": 3, "ambiguity": 4, "ignorance": 5}

    rows = []   # (conf, ent, field[6], correct, ids, label, pred)
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl = batch["label"].to(device)
        with torch.cuda.amp.autocast():
            logits, eout, _ = model(ids, mask)
        prob = F.softmax(logits.float(), dim=-1)
        preds = logits.argmax(dim=-1)
        conf = prob.max(dim=-1).values
        ent = -(prob * (prob + 1e-8).log()).sum(-1) / torch.log(
            torch.tensor(float(prob.size(-1)), device=device))
        fld = eout["field"].float()
        for b in range(lbl.size(0)):
            rows.append((
                conf[b].item(), ent[b].item(), fld[b].cpu().numpy(),
                (preds[b] == lbl[b]).item(),
                ids[b].cpu(), lbl[b].item(), preds[b].item(),
            ))

    conf_a = np.array([r[0] for r in rows])
    ent_a = np.array([r[1] for r in rows])
    fld_a = np.stack([r[2] for r in rows])
    corr_a = np.array([r[3] for r in rows], dtype=bool)
    uncert = list(unc_idx.values())

    # ── silent wrong: 틀렸는데 uncertainty 축이 안 뜬 비율 ──
    thresh = np.median(fld_a[corr_a][:, uncert].max(axis=1))
    wrong_umax = fld_a[~corr_a][:, uncert].max(axis=1)
    silent = wrong_umax < thresh
    print(f"\n  silent wrong (틀렸으나 uncertainty 잠잠) = {silent.mean():.4f} "
          f"({silent.sum()}/{len(silent)})  [thresh={thresh:.4f}]")

    # ── high-conf wrong: uncertainty 축이 안 움직인다는 측정 사실 ──
    hc_c = corr_a & (conf_a > conf_thresh)
    hc_w = (~corr_a) & (conf_a > conf_thresh)
    if hc_c.sum() > 0 and hc_w.sum() > 0:
        dc = fld_a[hc_w].mean(0) - fld_a[hc_c].mean(0)
        print(f"  high-conf(>{conf_thresh}) wrong vs correct Δ:  "
              f"novelty={dc[3]:+.4f}  ambiguity={dc[4]:+.4f}  ignorance={dc[5]:+.4f}")

    # ── confident wrong 유형 분리 ──
    cw = hc_w
    n_cw = int(cw.sum())
    print(f"\n  confident wrong (conf>{conf_thresh}): {n_cw}개")
    if n_cw < 6:
        print("  표본 너무 적음 — conf_thresh 낮추거나 epoch 더.")
        return {"silent": float(silent.mean())}

    cw_fld, cw_conf, cw_ent = fld_a[cw], conf_a[cw], ent_a[cw]
    LOW = np.median(fld_a[corr_a][:, uncert], axis=0)
    types = []
    for s in cw_fld:
        u = {ax: s[i] for ax, i in unc_idx.items()}
        top_ax = max(u, key=u.get)
        types.append(top_ax if u[top_ax] > LOW[list(unc_idx).index(top_ax)] else "metacognitive")
    types = np.array(types)

    print(f"\n{'유형':>14}  {'n':>4}  {'conf':>7}  {'entropy':>8}  "
          f"{'novel':>7}  {'ambig':>7}  {'ignor':>7}  {'error':>7}")
    print("─" * 76)
    for t in ["ignorance", "ambiguity", "novelty", "metacognitive"]:
        m = types == t
        if m.sum() == 0:
            continue
        f = cw_fld[m]
        print(f"{t:>14}  {m.sum():>4}  {cw_conf[m].mean():>7.3f}  {cw_ent[m].mean():>8.3f}  "
              f"{f[:,3].mean():>7.3f}  {f[:,4].mean():>7.3f}  {f[:,5].mean():>7.3f}  {f[:,1].mean():>7.3f}")

    print(f"\n  ── conf/entropy로 유형 분리되나 (안 돼야 우리 주장 성립) ──")
    for t in ["ignorance", "ambiguity", "metacognitive"]:
        m = types == t
        if m.sum() > 0:
            print(f"    {t:>14}: conf={cw_conf[m].mean():.3f}±{cw_conf[m].std():.3f}  "
                  f"ent={cw_ent[m].mean():.3f}±{cw_ent[m].std():.3f}")

    LABELS = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI"}
    print(f"\n  ── 유형별 실제 사례 (claim ‖ evidence) ──")
    cw_rows = [r for r, m in zip(rows, cw) if m]
    for t in ["ignorance", "ambiguity", "metacognitive"]:
        idxs = np.where(types == t)[0]
        if len(idxs) == 0:
            continue
        order = (idxs[np.argsort(cw_conf[idxs])[::-1]] if t == "metacognitive"
                 else idxs[np.argsort(cw_fld[idxs][:, unc_idx[t]])[::-1]])
        print(f"\n  [{t}]")
        for j in order[:k_examples]:
            r = cw_rows[j]
            text = tokenizer.decode(r[4], skip_special_tokens=True)
            text = text[:160] + ("…" if len(text) > 160 else "")
            print(f"    gold={LABELS[r[5]]:>8} pred={LABELS[r[6]]:>8} conf={r[0]:.3f}  "
                  f"nov={r[2][3]:.2f} amb={r[2][4]:.2f} ign={r[2][5]:.2f}")
            print(f"      {text}")

    return {"types": types, "cw_fld": cw_fld, "cw_conf": cw_conf, "silent": float(silent.mean())}


@torch.no_grad()
def evaluate_fever_axes(model, loader, device):
    model.eval()
    AXES = EpistemicFieldClassifier.AXES
    LABELS = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI"}
    fields, labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            _, eout, _ = model(ids, mask)
        fields.append(eout["field"].float().cpu())
        labels.append(batch["label"])
    fld = torch.cat(fields).numpy()
    lab = torch.cat(labels).numpy()
    means = {l: fld[lab == l].mean(0) for l in [0, 1, 2] if (lab == l).any()}

    print(f"\n{'axis':>14}" + "".join(f"{LABELS[l]:>11}" for l in [0, 1, 2]))
    print("─" * 50)
    for i, ax in enumerate(AXES):
        row = "".join(f"{means[l][i]:>11.4f}" if l in means else f"{'-':>11}" for l in [0, 1, 2])
        print(f"{ax:>14}{row}")

    def hi(axis_i, target):
        vals = {l: means[l][axis_i] for l in means}
        top = max(vals, key=vals.get)
        return f"{'✓' if top == target else '✗'} (top={LABELS[top]} {vals[top]:.3f})"

    print(f"\n  매핑 점검:")
    print(f"    truth     → SUPPORTS  {hi(0, 0)}")
    print(f"    error     → REFUTES   {hi(1, 1)}")
    print(f"    ignorance → NEI       {hi(5, 2)}")

    print(f"\n  |corr| ignorance vs 나머지 (분리 확인):")
    for i, ax in enumerate(AXES):
        if ax == "ignorance":
            continue
        c = abs(np.corrcoef(fld[:, 5], fld[:, i])[0, 1])
        print(f"    ignorance ↔ {ax:>14}: {c:.4f}")
    return means


# ─── Pretty Print ─────────────────────────────────────────────────────────────
def print_field_by_label(d):
    AXES = EpistemicFieldClassifier.AXES
    print(f"{'':>15}" + "".join(f"{a:>14}" for a in AXES))
    print("─" * (15 + 14 * len(AXES)))
    for name, vals in d.items():
        print(f"{name:>15}" + "".join(f"{vals[a]:>14.4f}" for a in AXES))


def print_ood_comparison(r):
    AXES = EpistemicFieldClassifier.AXES
    print(f"\n{'axis':>16}  {'ID mean':>10}  {'OOD mean':>10}  {'OOD - ID':>10}")
    print("─" * 52)
    for ax in AXES:
        id_m, ood_m = r[ax]["id_mean"], r[ax]["ood_mean"]
        flag = "  ← ↑" if ax in ("novelty", "ignorance", "ambiguity") and (ood_m - id_m) > 0.05 else ""
        print(f"{ax:>16}  {id_m:>10.4f}  {ood_m:>10.4f}  {ood_m - id_m:>+10.4f}{flag}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(seed=None, ablation="none"):
    """ablation: "none" | "dispersion"(cross-attn→entropy) | "no_detach"(소스 detach 제거)."""
    if seed is None:
        seed = CFG["seed"]
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = CFG["device"]
    tcfg = CFG["train"]
    print(f"Device: {device}  |  seed={seed}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("Loading VitaminC...")
    vc = load_dataset("tals/vitaminc")
    train_split = vc["train"].select(range(min(tcfg["train_size"], len(vc["train"])))) if tcfg["train_size"] else vc["train"]
    val_split = vc["validation"].select(range(min(tcfg["val_size"], len(vc["validation"])))) if tcfg["val_size"] else vc["validation"]

    train_ds = FEVERDataset(train_split, tokenizer, tcfg["max_length"])
    val_ds = FEVERDataset(val_split, tokenizer, tcfg["max_length"])
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    print("Loading RTE (OOD)...")
    rte = load_dataset("glue", "rte")
    ood_ds = OODDataset(rte["validation"], tokenizer, tcfg["max_length"])
    ood_loader = DataLoader(ood_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    model = EpistemicBERT(**CFG["model"]).to(device)
    model.token_nov.set_special_tokens(tokenizer.all_special_ids)
    model.attn_ign.sep_id = tokenizer.sep_token_id

    # ── ablation 스위치 ──────────────────────────────────────────
    if ablation == "dispersion":
        model.attn_ign.mode = "dispersion"
    elif ablation == "no_detach":
        model.token_nov.detach_out = False
        model.layer_amb.detach_out = False
        model.attn_ign.detach_out = False
    print(f"  ablation = {ablation}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = build_optimizer(model, tcfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history = []
    for epoch in range(tcfg["epochs"]):
        print(f"\n═══ Epoch {epoch + 1}/{tcfg['epochs']} ═══")

        tr = train_epoch(model, train_loader, optimizer, scaler, device, tcfg)
        print(f"  train  loss={tr['loss']:.4f}  ce={tr['ce']:.4f}  field={tr['field']:.4f}  acc={tr['acc']:.4f}")
        print(f"  learnable params  margin={F.softplus(model.margin_param).item():.4f}  "
              f"energy_ceil={F.softplus(model.energy_ceiling).item():.4f}  "
              f"truth_temp={F.softplus(model.epistemic.truth_temp).item():.4f}")

        vl = evaluate(model, val_loader, device)
        m = vl["monitor"]
        print(f"  monitor  field_mean={m['field_mean']}")
        print(f"           field_std ={m['field_std']}")
        print(f"  val    loss={vl['loss']:.4f}  acc={vl['acc']:.4f}")

        print("\n  Epistemic field by label (val):")
        print_field_by_label(vl["field_by_label"])
        print("\n  Support / Counter by label:")
        for name, vals in vl["support_by_label"].items():
            print(f"  {name:>15}  support={vals['support']:.4f}  counter={vals['counter']:.4f}")

        history.append({"epoch": epoch + 1, "train": tr, "val": vl})

    print("\n\n═══ OOD Experiment ═══")
    print_ood_comparison(evaluate_ood(model, ood_loader, val_loader, device))

    print("\n\n═══ Identifiability Probe ═══")
    identifiability_probe(model, val_loader, device)

    print("\n\n═══ Ambiguity 진단 ═══")
    diagnose_ambiguity(model, val_loader, device)

    print("\n\n═══ Selective Prediction (by-design 확인) ═══")
    evaluate_selective_prediction(model, val_loader, device)

    print("\n\n═══ Failure Typing (메인 결과) ═══")
    ft = failure_typing(model, val_loader, tokenizer, device)

    print("\n\n═══ FEVER Axis ↔ Label Mapping ═══")
    fever_means = evaluate_fever_axes(model, val_loader, device)

    torch.save(model.state_dict(), "/kaggle/working/epistemic_bert.pt")
    with open("/kaggle/working/results.json", "w") as f:
        json.dump({"history": history}, f, indent=2)

    return model, val_loader, device, ft, fever_means


if __name__ == "__main__":
    # 본 실험: ablation="none". ablation 재현: ["none","dispersion","no_detach"] 루프.
    main(seed=42, ablation="none")
