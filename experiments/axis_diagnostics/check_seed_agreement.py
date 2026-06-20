# Kaggle에서 실행 — 3개 seed 체크포인트 다 로드해서, 219개 confident-error에 대해
# 각 seed의 예측(predicted label)이 서로 얼마나 일치하는지, human_type(I/A/M)별로 확인.
#
# 목적: frame-blindness가 "bias"(세 seed가 다 같은 실수)인지 "variance"(seed마다 다름)인지 검증.
# - bias라면: ensemble(평균/투표)해도 안 사라짐 (셋이 합의해서 틀리니까)
# - variance라면: ensemble이 도움될 수 있음

import torch
import pandas as pd
from transformers import DistilBertTokenizerFast
from epistemic_bert import EpistemicBERT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7, 123]
CKPT_DIR = "/kaggle/input/datasets/terryterry9/epistemicbert-3seed"

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
labels = pd.read_csv("/kaggle/working/relabel_final_for_analysis.csv")  # id, claim, evidence, gold, human_type

LABEL_NAMES = ["SUPPORTS", "REFUTES", "NEI"]  # CFG 순서와 일치하는지 확인 필요

def load_model(seed):
    m = EpistemicBERT(n_classes=3, n_prototypes=32, proj_dim=128)
    ckpt = torch.load(f"{CKPT_DIR}/epistemic_bert_seed{seed}.pt", map_location=DEVICE)
    m.load_state_dict(ckpt)
    m.token_nov.set_special_tokens(tokenizer.all_special_ids)
    m.to(DEVICE).eval()
    return m

models = {s: load_model(s) for s in SEEDS}

rows = []
with torch.no_grad():
    for _, r in labels.iterrows():
        enc = tokenizer(
            r["claim"], r["evidence"],
            truncation=True, max_length=192, padding="max_length",
            return_tensors="pt"
        ).to(DEVICE)
        preds = {}
        for s in SEEDS:
            logits, _, _ = models[s](enc["input_ids"], enc["attention_mask"])
            preds[f"pred_seed{s}"] = LABEL_NAMES[logits.argmax(dim=-1).item()]
        rows.append({"id": r["id"], "human_type": r["human_type"], "gold": r["gold"], **preds})

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/per_seed_predictions.csv", index=False)
print(f"saved n={len(df)}")

# 3개 seed가 서로 얼마나 일치하는지 (전부 동일 prediction이면 agree=True)
pred_cols = [f"pred_seed{s}" for s in SEEDS]
df["all_agree"] = df[pred_cols].nunique(axis=1) == 1

print("\n=== human_type별 3-seed 일치율 ===")
agree_rate = df.groupby("human_type")["all_agree"].mean()
print(agree_rate)

print("\n=== 해석 ===")
for t in ["I", "A", "M"]:
    if t in agree_rate.index:
        rate = agree_rate[t]
        print(f"  {t}: 일치율={rate:.3f}  -> {'BIAS (seed 무관하게 다 같은 실수, ensemble 효과 적음)' if rate > 0.7 else 'VARIANCE 여지 있음 (seed마다 다름, ensemble 도움 가능성)'}")

# 추가: 일치할 때, 그 일치된 예측이 gold와 같은지 다른지 (= gold를 향한 공유된 bias인지)
agreed = df[df["all_agree"]]
agreed["pred_matches_gold"] = agreed["pred_seed42"] == agreed["gold"]
print("\n=== (일치 케이스만) human_type별 '셋 다 gold와 같게 틀림' 비율 ===")
print(agreed.groupby("human_type")["pred_matches_gold"].mean())
print("-> 이 비율이 높으면: 세 seed가 '다같이 gold 패턴을 따라간다'는 공유된 bias가 진짜 원인")