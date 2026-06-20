from datasets import load_dataset
import pandas as pd

vitaminc = load_dataset("tals/vitaminc")

# train만 우선 pandas로 변환 (37만 행, 이 정도는 pandas 변환 자체는 빠름)
train_df = vitaminc["train"].to_pandas()
print(f"train rows: {len(train_df)}")

labels = pd.read_csv("/kaggle/working/relabel_final_for_analysis.csv")
suspect_ids = [51, 144]
suspects = labels[labels["id"].isin(suspect_ids)]

for _, r in suspects.iterrows():
    print(f"\n{'='*60}")
    print(f"id={r['id']}")
    our_claim = r['claim'].strip()
    our_evidence = r['evidence'].strip()
    print(f"우리 CSV claim:    {our_claim}")
    print(f"우리 CSV evidence: {our_evidence}")

    # 1) 정확히 같은 claim 텍스트로 매칭 (가장 신뢰도 높음)
    exact = train_df[train_df["claim"].str.strip() == our_claim]
    print(f"\n  [exact claim match] {len(exact)}건")
    if len(exact) > 0:
        for _, m in exact.iterrows():
            print(f"    page={m['page']}, revision={m['wiki_revision_id']}, label={m['label']}")
            print(f"    원본 evidence: {m['evidence']}")
            same = (m['evidence'].strip() == our_evidence)
            print(f"    우리 evidence와 동일한가: {same}")

    # 2) 못 찾았으면 claim 앞부분으로 느슨하게 검색
    if len(exact) == 0:
        loose = train_df[train_df["claim"].str.contains(our_claim[:25], regex=False, na=False)]
        print(f"  [loose claim match, 앞 25자] {len(loose)}건")
        for _, m in loose.head(3).iterrows():
            print(f"    claim={m['claim']}")
            print(f"    evidence={m['evidence']}")

# 같은 방식으로 validation/test도 필요하면 반복 (train에서 못 찾으면)