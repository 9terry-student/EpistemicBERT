# Kaggle에서 실행 — 모델 checkpoint를 로드해서, 219개 confident-error claim/evidence에
# 직접 forward pass를 돌려 novelty/ambiguity/ignorance "원점수"를 뽑는다.
# (이 점수는 어디에도 저장된 적이 없어서, 새로 뽑아야 함)

import torch
import pandas as pd
from transformers import DistilBertTokenizerFast
# epistemic_bert.py가 같은 working dir / 경로에 있다고 가정
from epistemic_bert import EpistemicBERT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42  # 세 seed 중 하나로 먼저 확인; 필요하면 42/7/123 다 돌려서 평균

# 1) 모델 로드
model = EpistemicBERT(n_classes=3, n_prototypes=32, proj_dim=128)
ckpt_path = f"/kaggle/input/datasets/terryterry9/epistemicbert-3seed/epistemic_bert_seed{SEED}.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.token_nov.set_special_tokens(
    DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased").all_special_ids
)
model.to(DEVICE).eval()

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 2) 라벨 + claim/evidence 텍스트 불러오기
labels = pd.read_csv("/kaggle/working/relabel_final_for_analysis.csv")  # id, claim, evidence, human_type 포함

rows = []
with torch.no_grad():
    for _, r in labels.iterrows():
        enc = tokenizer(
            r["claim"], r["evidence"],
            truncation=True, max_length=192, padding="max_length",
            return_tensors="pt"
        ).to(DEVICE)
        out, epistemic_out, _ = model(enc["input_ids"], enc["attention_mask"])
        rows.append({
            "id": r["id"],
            "human_type": r["human_type"],
            "novelty": epistemic_out["novelty"].item(),
            "ambiguity": epistemic_out["ambiguity"].item(),
            "ignorance": epistemic_out["ignorance"].item(),
        })

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/axis_raw_scores.csv", index=False)
print(f"saved n={len(df)}")

# 3) human_type별 분산/평균 확인 (novelty)
for axis in ["novelty", "ambiguity", "ignorance"]:
    print(f"\n=== {axis} ===")
    print(df.groupby("human_type")[axis].describe()[["count","mean","std","min","max"]])
    overall_std = df[axis].std()
    means = df.groupby("human_type")[axis].mean()
    gap = abs(means.get("I", 0) - means.get("M", 0))
    print(f"overall std={overall_std:.4f}, |mean(I)-mean(M)|={gap:.4f}, ratio={gap/overall_std if overall_std>0 else float('nan'):.3f}")
    print("-> ratio가 1보다 훨씬 작으면: 분산 부족 가설(B) 지지")
    print("-> ratio가 어느 정도 크지만 AUROC가 낮으면: 분산은 있는데도 못 가름 -> A(설계) 문제 쪽")

# 4) AUROC도 같이
from sklearn.metrics import roc_auc_score
sub = df[df["human_type"].isin(["I","M"])].copy()
y = (sub["human_type"] == "I").astype(int)
for axis in ["novelty", "ambiguity", "ignorance"]:
    try:
        auc = roc_auc_score(y, sub[axis])
        print(f"{axis} AUROC (I vs M) = {auc:.3f}")
    except Exception as e:
        print(axis, "AUROC 실패:", e)