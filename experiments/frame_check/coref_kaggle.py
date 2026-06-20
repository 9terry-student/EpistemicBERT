# Kaggle에서 실행 — 대명사까지 잡는 coreference 기반 frame-check.
# !! 이 코드는 샌드박스 환경 제약(Python 3.12, thinc 빌드 실패)으로 로컬에서 직접
#    실행 검증을 못 했습니다. 먼저 known_cases 몇 개로 테스트하고, 결과가 말이 되는지
#    직접 확인한 뒤 219개 전체로 확장하세요.

# 설치 (en_core_web_trf 권장 - coreferee 정확도가 sm보다 훨씬 좋음, 근데 무겁고 느림.
#       처음엔 sm으로 시도하고, 결과가 너무 안 좋으면 trf로 바꾸세요)
# !pip install -q spacy coreferee
# !python -m spacy download en_core_web_trf
# !python -m coreferee install en

import spacy

nlp = spacy.load("en_core_web_trf")   # 안 되면 en_core_web_sm으로 바꿔서 시도
nlp.add_pipe("coreferee")

def frame_check_coref(claim, evidence):
    """claim+evidence를 하나의 텍스트로 합쳐서 coreference를 돌리고,
    evidence의 대명사/지칭표현이 claim의 명사(주로 주어, 첫 NP)와 같은 체인에
    묶이는지 확인."""
    claim = claim.strip()
    evidence = evidence.strip()
    combined = claim + " " + evidence
    doc = nlp(combined)

    claim_len_tokens = len(nlp(claim))  # claim이 차지하는 토큰 수 (대략적 경계)

    chains = doc._.coref_chains
    if chains is None or len(chains) == 0:
        return {"resolved": None, "note": "coref chain 없음 (대명사 자체가 없거나 resolver가 못 찾음)"}

    results = []
    for chain in chains:
        # 이 체인에 claim 쪽 토큰과 evidence 쪽 토큰이 둘 다 있는지 확인
        token_indices = [mention.root_index for mention in chain]
        has_claim_side = any(idx < claim_len_tokens for idx in token_indices)
        has_evidence_side = any(idx >= claim_len_tokens for idx in token_indices)
        if has_claim_side and has_evidence_side:
            mentions_text = [doc[idx].text for idx in token_indices]
            results.append(mentions_text)

    if results:
        return {"resolved": True, "linked_chains": results}
    else:
        return {"resolved": False, "note": "claim과 evidence 쪽을 잇는 coref chain 없음 -> frame uncertain 후보"}


# ── 알려진 케이스로 먼저 테스트 (219개 전체 돌리기 전에 꼭 확인) ──
known_cases = [
    ("id66 Dirty Grandpa (대명사 'It', frame mismatch 의심)",
     "Dirty Grandpa finished 4th at the box office .",
     "It went on to gross $ 11.5 million in its opening weekend , finishing 5th at the box office."),
    ("id46 Trump/Latham (다른 사람, 명확한 frame mismatch)",
     "Donald Trump was once a Labour Party leader .",
     "Former Labor Party leader , Mark Latham , joined the party in November 2018 as leader for New South Wales ."),
    ("id49 Furtado (frame 명확, M)",
     "Nelly Kim Furtado is Portuguese-Canadian .",
     "Nelly Kim Furtado ComIH ( ; born 2 December 1978 ) is a Portugese-Canadian singer and songwriter ."),
]

for name, claim, evidence in known_cases:
    r = frame_check_coref(claim, evidence)
    print(f"=== {name} ===")
    print(f"  claim: {claim}")
    print(f"  evidence: {evidence}")
    print(f"  결과: {r}")
    print()

# 기대하는 결과 (검증 전 가설):
# - id66: resolved=False 이거나, resolved=True인데 "It"이 다른 걸 가리킴 -> frame uncertain 맞게 잡혀야 함
# - id46: resolved=False (Latham과 Trump는 애초에 같은 체인에 안 묶임, 별개 entity) -> frame uncertain 맞게 잡혀야 함
# - id49: resolved=True ("Nelly Kim Furtado"가 양쪽에 다 명시적으로 나와서 entity 매칭만으로도 충분 -
#         이 경우는 coref보다 단순 NER overlap이 이미 잡았던 케이스)
#
# 만약 결과가 이 기대와 다르면 -> coreferee의 분석 단위(문장 경계 인식 등)가
# 우리가 합친 pseudo-document를 어색하게 처리하고 있을 수 있음. 그 경우 claim과
# evidence 사이에 명시적 구분자(예: 마침표+공백)를 더 분명히 하거나, 다른 coref
# 라이브러리(fastcoref 등)를 검토.