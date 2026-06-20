import spacy
nlp = spacy.load("en_core_web_sm")

claim = "Middle-earth : Shadow of War was released on 10 October 2017 ."
evidence = "Middle-earth : Shadow of War is an upcoming action role-playing video game developed by Monolith Productions and published by Warner Bros. Interactive Entertainment ."

print("=== claim 전체 entities (필터 없이 다 출력) ===")
for ent in nlp(claim).ents:
    print(f"  '{ent.text}'  ->  {ent.label_}")

print("\n=== evidence 전체 entities (필터 없이 다 출력) ===")
for ent in nlp(evidence).ents:
    print(f"  '{ent.text}'  ->  {ent.label_}")

print("\n(우리 도구가 쓴 RELEVANT_LABELS = PERSON, ORG, WORK_OF_ART, GPE, EVENT, FAC, NORP, PRODUCT)")
print("-> 'Middle-earth : Shadow of War'가 위 라벨 중 하나로 안 잡혔다면, 그게 오탐의 원인")