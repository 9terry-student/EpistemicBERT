# Kaggle에서 실행 — fastcoref로 known_cases 3개부터 검증.
# !! 이 샌드박스에서는 huggingface.co 접근이 막혀서 실제 추론 검증 못 했음.
#    spaCy도 의존성으로 깔림(기존 "spaCy 의존성 없음"이라는 전제는 틀렸음 - pip show로 확인).
#    Kaggle은 인터넷이 열려있어서 보통 될 거예요. 여기 3개 결과부터 보고 219개로 확장하세요.

# !pip install -q fastcoref

from fastcoref import FCoref

model = FCoref(device='cuda' if True else 'cpu')  # GPU 있으면 cuda, 없으면 'cpu'로 바꾸세요

known_cases = [
    ("id66 Dirty Grandpa (대명사 'It' - frame mismatch 의심, human=I)",
     "Dirty Grandpa finished 4th at the box office.",
     "It went on to gross $ 11.5 million in its opening weekend , finishing 5th at the box office."),
    ("id141 Treaty (GLiNER가 스키마 밖이라 놓친 케이스, human=M)",
     "The third draft of the Treaty on the Prohibition of Nuclear Weapons had signatures from September 19 , 2017 .",
     "A third draft as of 3 July 2017 provides signature from 19 September 2016 ."),
    ("id49 Furtado (frame 명확, human=M)",
     "Nelly Kim Furtado is Portuguese-Canadian .",
     "Nelly Kim Furtado ComIH ( ; born 2 December 1978 ) is a Portugese-Canadian singer and songwriter ."),
]

def frame_check_coref(claim, evidence, model):
    claim = claim.strip()
    evidence = evidence.strip()
    combined = claim + " " + evidence
    pred = model.predict(texts=[combined])[0]
    clusters = pred.get_clusters(as_strings=True)

    # claim이 끝나는 대략적 글자 위치 (이 위치 전/후로 mention이 나뉘면 claim쪽/evidence쪽)
    claim_char_len = len(claim) + 1  # +1 for the space we added

    linked = []
    for cluster in clusters:
        # cluster 안의 각 mention 문자열이 combined 텍스트에서 어디 있는지로 claim/evidence 쪽 판별
        # (간단한 근사: mention 문자열이 claim 부분에도 등장하는지 substring으로 체크)
        in_claim = any(m in claim for m in cluster)
        in_evidence = any(m in evidence for m in cluster)
        if in_claim and in_evidence:
            linked.append(cluster)

    return {"clusters": clusters, "claim_evidence_linked": linked,
            "frame_resolved": len(linked) > 0}

for name, claim, evidence in known_cases:
    r = frame_check_coref(claim, evidence, model)
    print(f"=== {name} ===")
    print(f"  claim: {claim}")
    print(f"  evidence: {evidence}")
    print(f"  전체 clusters: {r['clusters']}")
    print(f"  claim<->evidence 연결된 cluster: {r['claim_evidence_linked']}")
    print(f"  frame_resolved: {r['frame_resolved']}")
    print()

# 기대 결과:
# - id66: "Dirty Grandpa"와 "It"이 같은 cluster로 묶여서 frame_resolved=True가 나와야
#         이상적 (사람은 human=I로 봤지만, "It"이 실제로 Dirty Grandpa를 가리키는 건 맞음 -
#         frame 자체는 맞고, 다만 "5th" vs "4th" 숫자 불일치가 다른 이유로 I를 만든 케이스.
#         즉 여기서는 frame_resolved=True가 나와야 정상이고, 이 케이스의 I는 frame 문제가
#         아니라는 걸 보여주는 좋은 대조군이 됨)
# - id141: claim과 evidence에 명시적 entity가 거의 없어서(둘 다 "third draft"라는 표현은
#          공유하지만 고유명사가 부족) coref가 뭘 잡을지 불확실 - 직접 봐야 함
# - id49: "Nelly Kim Furtado"가 양쪽에 명시돼 있어서, mention detection이 NER 없이도
#         이걸 후보로 잡고 cluster로 묶을 가능성이 높음 -> frame_resolved=True 기대
#
# 주의: id66 케이스는 사실 frame_uncertain 가설을 재검토하게 만들 수 있는 케이스입니다.
# "It"이 Dirty Grandpa를 가리키는 게 명확하다면(frame_resolved=True), 이 케이스의 human=I
# 판정은 frame 문제가 아니라 다른 이유(4th vs 5th 숫자 불일치, decisive하지 않음)일 수 있고,
# 이러면 이 케이스는 애초에 "frame mismatch 후보" 목록에서 빼야 할 수도 있습니다 -
# 결과 보고 같이 재진단해야 합니다.