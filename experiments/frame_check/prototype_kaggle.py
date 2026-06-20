# Kaggle에서 실행 — spaCy NER 기반 frame-check 프로토타입.
# (이 동네 sandbox에선 spaCy 모델 다운로드가 막혀서, 로직은 정규식 버전으로 먼저
#  검증했고 — Resident Evil 케이스에서 "entity overlap 없음 -> frame uncertain"을
#  정확히 잡아냄을 확인함. 여기선 그 로직을 spaCy NER로 업그레이드.)
#
# !pip install -q spacy && python -m spacy download en_core_web_sm   # 처음 한 번만

import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

# frame-check에 의미 있는 entity 타입만 사용 (PERSON, ORG, WORK_OF_ART, GPE, EVENT, FAC 등)
RELEVANT_LABELS = {"PERSON", "ORG", "WORK_OF_ART", "GPE", "EVENT", "FAC", "NORP", "PRODUCT"}

def extract_entities(text):
    doc = nlp(text)
    return {(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in RELEVANT_LABELS}

def frame_check(claim, evidence):
    claim_ents = extract_entities(claim)
    evid_ents = extract_entities(evidence)
    claim_texts = {e[0] for e in claim_ents}
    evid_texts = {e[0] for e in evid_ents}

    exact_overlap = claim_texts & evid_texts
    fuzzy_overlap = {(c, e) for c in claim_texts for e in evid_texts
                      if len(c) > 2 and len(e) > 2 and (c in e or e in c)}

    has_overlap = bool(exact_overlap or fuzzy_overlap)
    return {
        "claim_entities": claim_ents,
        "evidence_entities": evid_ents,
        "exact_overlap": exact_overlap,
        "fuzzy_overlap": fuzzy_overlap,
        "frame_uncertain": not has_overlap,   # True = frame mismatch 의심 (사람의 I 판단과 맞춰볼 신호)
    }

# ── 219개 전체에 돌려서, frame_uncertain 신호가 human_type과 얼마나 맞는지 확인 ──
labels = pd.read_csv("/kaggle/working/relabel_final_for_analysis.csv")

rows = []
for _, r in labels.iterrows():
    res = frame_check(r["claim"], r["evidence"])
    rows.append({"id": r["id"], "human_type": r["human_type"], "frame_uncertain": res["frame_uncertain"]})

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/frame_check_results.csv", index=False)

print("=== human_type별 frame_uncertain 비율 ===")
print(df.groupby("human_type")["frame_uncertain"].mean())
print("\n(I 비율이 A/M보다 뚜렷이 높으면: 이 단순 entity-overlap 신호가 frame mismatch를 어느 정도 잡는다는 1차 증거)")

# AUROC로도 확인 (I vs M, frame_uncertain을 점수처럼 사용)
from sklearn.metrics import roc_auc_score
sub = df[df["human_type"].isin(["I", "M"])].copy()
y = (sub["human_type"] == "I").astype(int)
auc = roc_auc_score(y, sub["frame_uncertain"].astype(int))
print(f"\nframe_uncertain AUROC (I vs M) = {auc:.3f}  (참고: 현재 ignorance AUROC = 0.585)")