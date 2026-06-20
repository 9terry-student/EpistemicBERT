import re
import pandas as pd
from sklearn.metrics import roc_auc_score

AUX_START = re.compile(
    r'^(is|was|are|were|has|have|had|became|serves?|served|consists?|comprises?|'
    r'refers?|denotes?|includes?|features?|also known as)\b',
    re.IGNORECASE
)

def is_truncated_subject(evidence):
    text = re.sub(r'^[\s"\'`]+', '', str(evidence).strip())
    return bool(AUX_START.match(text))

# 1) 데이터 합치기
labels = pd.read_csv("/kaggle/working/relabel_final_for_analysis.csv")     # id, claim, evidence, human_type
frame = pd.read_csv("/kaggle/working/frame_check_results.csv")            # id, human_type, frame_uncertain

df = labels.merge(frame[["id", "frame_uncertain"]], on="id")
df["truncated_subject"] = df["evidence"].apply(is_truncated_subject)

print(f"전체 n={len(df)}")
print(f"truncated_subject=True (데이터 손상 의심) 비율: {df['truncated_subject'].mean():.3f}")
print()
print("=== human_type별 truncated_subject 비율 ===")
print(df.groupby("human_type")["truncated_subject"].mean())
print("(I에서 이 비율이 높다면, 기존 AUROC 일부가 데이터 손상 때문에 부풀려졌을 가능성)")
print()

# 2) "깨끗한" 부분집합 (주어 누락 케이스 제외) 에서 frame_uncertain AUROC 재계산
clean = df[~df["truncated_subject"]].copy()
print(f"\n깨끗한 부분집합 n={len(clean)} (전체 {len(df)}개 중 {len(df)-len(clean)}개 제외)")

print("\n=== 깨끗한 부분집합 - human_type별 frame_uncertain 비율 ===")
print(clean.groupby("human_type")["frame_uncertain"].mean())

sub = clean[clean["human_type"].isin(["I", "M"])].copy()
y = (sub["human_type"] == "I").astype(int)
auc_clean = roc_auc_score(y, sub["frame_uncertain"].astype(int))
print(f"\n[깨끗한 부분집합] frame_uncertain AUROC (I vs M) = {auc_clean:.3f}")

# 비교: 원래(오염된) AUROC도 같이 출력
sub_all = df[df["human_type"].isin(["I", "M"])].copy()
y_all = (sub_all["human_type"] == "I").astype(int)
auc_all = roc_auc_score(y_all, sub_all["frame_uncertain"].astype(int))
print(f"[전체, 오염 포함] frame_uncertain AUROC (I vs M) = {auc_all:.3f}")
print(f"[참고] 기존 ignorance AUROC = 0.585")

print("\n=== 해석 ===")
if auc_clean > auc_all and auc_clean > 0.585:
    print("-> 데이터 손상 제거 후 AUROC가 더 올라감: frame-check 신호가 진짜로 ignorance보다 낫다는 증거 강화")
elif auc_clean > auc_all:
    print("-> 데이터 손상 제거 후 AUROC는 올랐지만 ignorance(0.585)보다는 여전히 낮음: 약한 신호, 추가 개선 필요")
else:
    print("-> 데이터 손상 제거해도 AUROC 안 오름: entity-overlap 자체가 약한 신호일 가능성, NER 한계(예: 'the film' 같은 대명사성 표현을 못 잡음)가 진짜 원인일 수 있음")

# 3) 샘플 확인 - "깨끗한" 부분집합에서 frame_uncertain=True인 I케이스 몇 개 직접 보기
print("\n=== 샘플: 깨끗한 부분집합 중 frame_uncertain=True, human_type=I (진짜 frame mismatch 후보) ===")
real_fm = clean[(clean["human_type"]=="I") & (clean["frame_uncertain"]==True)]
for _, r in real_fm.head(5).iterrows():
    print(f"id={r['id']}")
    print(f"  claim:    {r['claim']}")
    print(f"  evidence: {r['evidence']}")
    print()