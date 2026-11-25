"""model_select.py - compares the four classifiers with stratified k-fold cross-validation."""
import time
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from detector import ALGORITHMS, FakeNewsDetector, add_passages
from preprocessor import TextPreprocessor


def compare_models(texts, labels, folds=5, seed=42, algorithms=None):
    """Return one row per algorithm, best F1 first:
    {algorithm, name, accuracy, accuracy_std, precision, recall, f1, fit_seconds}.
    Each fold is vectorised once and shared by all classifiers, so they see identical features."""
    pre = TextPreprocessor()
    clean = np.array([pre.preprocess(t) for t in texts], dtype=object)
    y = np.array(labels)
    folds = max(2, min(folds, int(min((y == 0).sum(), (y == 1).sum()))))
    keys = list(algorithms or ALGORITHMS)
    scores = {k: {"accuracy": [], "precision": [], "recall": [], "f1": [], "fit": []} for k in keys}
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(clean, y):
        vec = clone(FakeNewsDetector().vectorizer)
        tr_texts, y_tr = add_passages(list(clean[tr]), list(y[tr]))      # same training as the real model
        X_tr, X_te = vec.fit_transform(tr_texts), vec.transform(clean[te])
        for k in keys:
            clf = FakeNewsDetector(k).model
            start = time.perf_counter()
            clf.fit(X_tr, y_tr)
            scores[k]["fit"].append(time.perf_counter() - start)
            pred = clf.predict(X_te)
            scores[k]["accuracy"].append(accuracy_score(y[te], pred))
            scores[k]["precision"].append(precision_score(y[te], pred, zero_division=0))
            scores[k]["recall"].append(recall_score(y[te], pred, zero_division=0))
            scores[k]["f1"].append(f1_score(y[te], pred, zero_division=0))
    rows = [{"algorithm": k, "name": ALGORITHMS[k][0],
             "accuracy": float(np.mean(s["accuracy"])), "accuracy_std": float(np.std(s["accuracy"])),
             "precision": float(np.mean(s["precision"])), "recall": float(np.mean(s["recall"])),
             "f1": float(np.mean(s["f1"])), "fit_seconds": round(float(np.mean(s["fit"])), 3)}
            for k, s in scores.items()]
    rows.sort(key=lambda row: row["f1"], reverse=True)
    return rows


def print_table(rows):
    print(f"{'Algorithm':26s}{'Accuracy':>16s}{'Precision':>11s}{'Recall':>8s}{'F1':>7s}{'Fit s':>8s}")
    for r in rows:
        print(f"{r['name']:26s}{r['accuracy']:9.3f} ± {r['accuracy_std']:.3f}{r['precision']:11.3f}"
              f"{r['recall']:8.3f}{r['f1']:7.3f}{r['fit_seconds']:8.3f}")
