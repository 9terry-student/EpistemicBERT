import pandas as pd
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
LABELS = ["video game title", "movie title", "TV show title", "musical work title",
          "person", "organization", "place", "band or group name"]

labels = pd.read_csv("/kaggle/working/relabel_final_for_analysis.csv")

rows = []
for _, r in labels.iterrows():
    claim_ents = model.predict_entities(r["claim"], LABELS, threshold=0.3)
    evid_ents = model.predict_entities(r["evidence"], LABELS, threshold=0.3)
    rows.append({
        "id": r["id"],
        "human_type": r["human_type"],
        "claim_n_ents": len(claim_ents),
        "evidence_n_ents": len(evid_ents),
        "both_empty": (len(claim_ents) == 0 and len(evid_ents) == 0),
        "either_empty": (len(claim_ents) == 0 or len(evid_ents) == 0),
    })

df = pd.DataFrame(rows)

print("=== human_type별 'entity 아예 못 찾음(claim/evidence 둘 다 0개)' 비율 ===")
print(df.groupby("human_type")["both_empty"].mean())
print()
print("=== human_type별 'claim 또는 evidence 중 하나라도 0개' 비율 ===")
print(df.groupby("human_type")["either_empty"].mean())
print()
print("(M에서 이 비율이 I보다 뚜렷이 높으면: LABELS가 너무 좁아서 M 케이스들의")
print(" entity를 못 잡는 게 frame_uncertain 오탐의 주원인이라는 가설이 맞는 것)")

# 둘 다 0개인 M 케이스 샘플 직접 확인
print("\n=== 샘플: M인데 entity 아예 못 찾은 케이스 ===")
sample = df[(df["human_type"]=="M") & (df["both_empty"])].merge(
    labels[["id","claim","evidence"]], on="id")
for _, r in sample.head(5).iterrows():
    print(f"id={r['id']}")
    print(f"  claim: {r['claim']}")
    print(f"  evidence: {r['evidence']}")
    print()