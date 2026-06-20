import re
import pandas as pd

def extract_entities(text):
    """아주 단순한 고유명사 추출: 연속된 대문자 시작 단어들을 entity로 본다.
    문장 맨 앞 단어(흔히 대문자로 시작하지만 고유명사가 아닐 수 있음)는 제외하지 않음 —
    프로토타입이라 단순함을 우선."""
    # 연속된 Capitalized 단어 시퀀스 찾기 (2단어 이상 또는 단독 고유명사 후보)
    candidates = re.findall(r"\b[A-Z][a-zA-Z']*(?:\s+[A-Z][a-zA-Z']*)*\b", text)
    # 너무 흔한 단일 대문자 단어(문장 시작 The, On, In 등) 제거
    stop = {"The", "On", "In", "A", "An", "Of", "At", "By", "For"}
    ents = set()
    for c in candidates:
        words = c.split()
        # stopword만 있는 경우 제외, 나머지는 entity 후보로
        if all(w in stop for w in words):
            continue
        ents.add(c.strip())
    return ents

def frame_check(claim, evidence):
    claim_ents = extract_entities(claim)
    evid_ents = extract_entities(evidence)
    overlap = claim_ents & evid_ents
    # 부분 문자열 매칭도 시도 (예: "Resident Evil" vs "Resident Evil : The Final Chapter")
    fuzzy_overlap = set()
    for ce in claim_ents:
        for ee in evid_ents:
            if ce in ee or ee in ce:
                fuzzy_overlap.add((ce, ee))
    return {
        "claim_entities": claim_ents,
        "evidence_entities": evid_ents,
        "exact_overlap": overlap,
        "fuzzy_overlap": fuzzy_overlap,
        "frame_flag": "OK (entity overlap found)" if (overlap or fuzzy_overlap) else "UNCERTAIN (no entity overlap -- frame may not match)"
    }

# 실제 케이스로 테스트
cases = [
    ("id4 (human=I, Resident Evil)",
     "Resident Evil : The Final Chapter scored under 43 % based on less than 47 reviews .",
     "On review aggregator website Rotten Tomatoes , the film has an approval rating of 43 % based on 48 reviews , and an average rating of 4.8/10 ."),
    ("id49 (human=M, Furtado)",
     "Nelly Kim Furtado is Portuguese-Canadian .",
     "Nelly Kim Furtado ComIH ( ; born 2 December 1978 ) is a Portugese-Canadian singer and songwriter ."),
    ("id110 (human=M, Blue Apple Theatre)",
     "Blue Apple Theatre is a company of actors .",
     "Winchester is the home of the award-winning Blue Apple Theatre , an inclusive company of actors with and without learning disability ."),
    ("id157 (human=M, Maze Runner)",
     "In Maze Runner : The Scorch Trials , Thomas and the Gladers battle the World Catastrophe Killzone Department .",
     "The plot of The Scorch Trials takes place immediately after the previous installment , with Thomas ( O'Brien ) and his fellow Gladers running away from the powerful World Catastrophe Killzone Department ( W.C.K.D. , or WICKED ) , while facing the perils of the Scorch , a desolate landscape filled with dangerous obstacles ."),
]

for name, claim, evidence in cases:
    r = frame_check(claim, evidence)
    print(f"=== {name} ===")
    print("claim entities:", r["claim_entities"])
    print("evidence entities:", r["evidence_entities"])
    print("exact overlap:", r["exact_overlap"])
    print("fuzzy overlap:", r["fuzzy_overlap"])
    print(">>>", r["frame_flag"])
    print()