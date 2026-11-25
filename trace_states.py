"""trace_states.py - prints the states the system passes through for one article."""
from dataset import DatasetLoader
from detector import FakeNewsDetector
from validator import validate_input

det = FakeNewsDetector()
det.train(*DatasetLoader().load())

def check(text):
    print("[Idle] --submit(article)--> [Validating]")
    ok, msg = validate_input(text)
    if not ok:
        print("[Validating] --[invalid input]--> [Error Displayed]")
        print("   message:", msg)
        print("[Error Displayed] --close--> [Idle]\n")
        return
    print("[Validating] --[valid input]--> [Preprocessing]")
    clean = det.preprocessor.preprocess(text)
    n = len(clean.split())
    print(f"[Preprocessing] --cleaned text ({n} tokens)--> [Feature Extraction]")
    vec = det.vectorizer.transform([clean])
    print(f"[Feature Extraction] --TF-IDF ({vec.nnz} non-zero features)--> [Classifying]")
    p = float(det.model.predict_proba(vec)[0][1])
    res = det.predict(text)
    print(f"[Classifying] --P(fake)={p:.4f}, label={res.label}--> [Result Displayed]")
    print(f"[Result Displayed] confidence={res.confidence}% --clear--> [Idle]\n")

check("SHOCKING secret exposed! Miracle pill cures every disease overnight, "
      "doctors hate it, share before it gets deleted")
check("Fake news is bad")
