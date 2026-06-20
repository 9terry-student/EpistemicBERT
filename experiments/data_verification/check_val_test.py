from datasets import load_dataset
import pandas as pd

vitaminc = load_dataset("tals/vitaminc")

labels = pd.read_csv("/kaggle/working/relabel_final_for_analysis.csv")
suspect_ids = [51, 144]
suspects = labels[labels["id"].isin(suspect_ids)]

for split_name in ["validation", "test"]:
    df = vitaminc[split_name].to_pandas()
    print(f"\n{'#'*60}")
    print(f"### split = {split_name}  (rows={len(df)})")

    for _, r in suspects.iterrows():
        our_claim = r['claim'].strip()
        our_evidence = r['evidence'].strip()

        exact = df[df["claim"].str.strip() == our_claim]
        if len(exact) > 0:
            print(f"\n  id={r['id']}: [exact match in {split_name}] {len(exact)}건")
            for _, m in exact.iterrows():
                print(f"    page={m['page']}, revision={m['wiki_revision_id']}, label={m['label']}")
                print(f"    원본 evidence: {m['evidence']}")
                print(f"    동일한가: {m['evidence'].strip() == our_evidence}")
        else:
            # 느슨한 매칭: claim의 핵심 단어(따옴표/구두점 차이 무시하고) 일부로 시도
            key_phrase = our_claim.replace(" .", "").replace(" ,", ",").strip()[:20]
            loose = df[df["claim"].str.contains(key_phrase, regex=False, na=False)]
            print(f"\n  id={r['id']}: exact 0건, loose('{key_phrase}') {len(loose)}건")
            for _, m in loose.head(2).iterrows():
                print(f"    claim={m['claim']}")
                print(f"    evidence={m['evidence']}")