"""retrain.py - builds a new model version from user feedback and keeps it only if it is not worse.

The held-out test split of the base dataset (same seed as train.py) is used to compare the current
model with the candidate, so both are judged on identical, unseen articles.
"""
import os
import shutil
from sklearn.model_selection import train_test_split
from detector import ALGORITHMS, FakeNewsDetector
from preprocessor import TextPreprocessor


def algorithm_key(name):
    return next((k for k, (n, _) in ALGORITHMS.items() if n == name), "logreg")


def retrain_from_feedback(repo, model_path, seed=42, test_size=0.25):
    base = repo.latest_model()
    texts, labels = repo.dataset_rows(base["dataset_id"])
    fb_texts, fb_labels = repo.feedback_samples()
    known = set(texts)
    pairs = [(t, l) for t, l in zip(fb_texts, fb_labels) if t not in known]
    if not pairs:
        return {"accepted": False, "feedback_used": 0, "message": "There is no new feedback to learn from."}
    pre = TextPreprocessor()
    clean = [pre.preprocess(t) for t in texts]
    X_tr, X_te, y_tr, y_te = train_test_split(clean, labels, test_size=test_size,
                                              random_state=seed, stratify=labels)
    current = FakeNewsDetector()
    current.load(model_path)
    old = current.evaluate(X_te, y_te)
    candidate = FakeNewsDetector(algorithm_key(base["algorithm"]))
    candidate.fit_clean(X_tr + [pre.preprocess(t) for t, _ in pairs], y_tr + [l for _, l in pairs])
    new = candidate.evaluate(X_te, y_te)
    fixed = sum(candidate.predict_label_clean(pre.preprocess(t)) == l for t, l in pairs)
    accepted = new["accuracy"] >= old["accuracy"]
    new_id = None
    if accepted:
        if os.path.exists(model_path):
            shutil.copyfile(model_path, model_path + ".bak")
        candidate.save(model_path)
        new_id = repo.add_model(base["dataset_id"], new, model_path, candidate.algorithm_name)
    repo.add_retrain_run(base["model_id"], new_id, len(pairs), old["accuracy"], new["accuracy"], accepted)
    verdict = ("accepted: the new model is now in use" if accepted
               else "rejected: the current model was kept because accuracy would drop")
    return {"accepted": accepted, "feedback_used": len(pairs), "old_accuracy": old["accuracy"],
            "new_accuracy": new["accuracy"], "feedback_fixed": int(fixed), "model_id": new_id,
            "message": f"Learned from {len(pairs)} feedback article(s); test accuracy "
                       f"{old['accuracy']:.3f} -> {new['accuracy']:.3f}; {verdict}."}
