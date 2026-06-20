# Kaggle에서 실행 — GLiNER (zero-shot NER, 임의의 타입 지정 가능, spaCy 의존성 없음)
# !! huggingface.co 접근이 필요해서 이 샌드박스에서는 검증 못 했음. Kaggle은 인터넷이
#    열려있어서 보통 바로 될 거예요. 먼저 known_cases로 확인하고 219개로 확장하세요.

# !pip install -q gliner

from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# 핵심 차이: 고정된 라벨셋(PERSON/ORG 등)이 아니라, 우리 데이터에 맞는 타입을 직접 지정
LABELS = ["video game title", "movie title", "TV show title", "musical work title",
          "person", "organization", "place", "band or group name"]

def extract_entities_gliner(text, threshold=0.3):
    ents = model.predict_entities(text, LABELS, threshold=threshold)
    return {(e["text"].strip(), e["label"]) for e in ents}

def frame_check_gliner(claim, evidence):
    claim_ents = extract_entities_gliner(claim)
    evid_ents = extract_entities_gliner(evidence)
    claim_texts = {e[0] for e in claim_ents}
    evid_texts = {e[0] for e in evid_ents}

    exact_overlap = claim_texts & evid_texts
    fuzzy_overlap = {(c, e) for c in claim_texts for e in evid_texts
                      if len(c) > 2 and len(e) > 2 and (c in e or e in c)}
    has_overlap = bool(exact_overlap or fuzzy_overlap)
    return {
        "claim_entities": claim_ents,
        "evidence_entities": evid_ents,
        "frame_uncertain": not has_overlap,
    }

# ── 알려진 케이스로 먼저 확인 ──
known_cases = [
    ("id76 Middle-earth (spaCy NER이 놓쳤던 케이스 - GLiNER가 잡는지 확인)",
     "Middle-earth : Shadow of War was released on 10 October 2017 .",
     "Middle-earth : Shadow of War is an upcoming action role-playing video game developed by Monolith Productions and published by Warner Bros. Interactive Entertainment ."),
    ("id46 Trump/Latham (다른 사람, frame mismatch 맞게 잡혀야 함)",
     "Donald Trump was once a Labour Party leader .",
     "Former Labor Party leader , Mark Latham , joined the party in November 2018 as leader for New South Wales ."),
    ("id49 Furtado (frame 명확, overlap 잡혀야 함)",
     "Nelly Kim Furtado is Portuguese-Canadian .",
     "Nelly Kim Furtado ComIH ( ; born 2 December 1978 ) is a Portugese-Canadian singer and songwriter ."),
]

for name, claim, evidence in known_cases:
    r = frame_check_gliner(claim, evidence)
    print(f"=== {name} ===")
    print(f"  claim entities:    {r['claim_entities']}")
    print(f"  evidence entities: {r['evidence_entities']}")
    print(f"  frame_uncertain: {r['frame_uncertain']}")
    print()

# 기대 결과:
# - id76: claim_entities와 evidence_entities 둘 다 'Middle-earth : Shadow of War'를
#         (video game title로) 잡아서 -> frame_uncertain=False 가 나와야 정상
#         (spaCy NER은 이걸 완전히 놓쳤던 케이스)
# - id46: overlap 없음 -> frame_uncertain=True (맞게 잡힘)
# - id49: 'Nelly Kim Furtado' 겹침 -> frame_uncertain=False
#
# id76에서 여전히 놓치면 -> threshold를 낮춰보거나 (0.3 -> 0.15), LABELS에
# "creative work" 같은 더 넓은 카테고리를 추가해서 재시도.