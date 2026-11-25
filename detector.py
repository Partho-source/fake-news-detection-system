"""detector.py - TF-IDF feature extraction and Fake/Real classification."""
import math
import pickle
import re
from dataclasses import dataclass, field
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessor import TextPreprocessor
from validator import validate_input


# The linear models are trained WITHOUT an intercept. With an intercept, a model trained on long
# articles learns a fixed lean towards one class that only many words can outweigh, so every short
# text got the same label. Without it, an article with no informative words scores exactly 50/50.
def _passive_aggressive():
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):           # PassiveAggressiveClassifier is deprecated from 1.8
        return SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
                             max_iter=1000, random_state=42, fit_intercept=False)
    from sklearn.linear_model import PassiveAggressiveClassifier
    return PassiveAggressiveClassifier(max_iter=1000, random_state=42, fit_intercept=False)


# key -> (display name, factory)
ALGORITHMS = {
    "logreg": ("Logistic Regression", lambda: LogisticRegression(max_iter=1000, fit_intercept=False)),
    "nb": ("Multinomial Naive Bayes", lambda: MultinomialNB()),
    "svm": ("Linear SVM", lambda: LinearSVC(fit_intercept=False)),
    "pa": ("Passive-Aggressive", _passive_aggressive),
}
PASSAGE_WORDS, MAX_PASSAGES = 40, 4


def add_passages(clean_texts, labels):
    """Training-only augmentation: long articles also contribute up to 4 passages of 40 words, so the
    model learns what a short excerpt of each class looks like (users often paste a few sentences)."""
    texts, ys = list(clean_texts), list(labels)
    for text, y in zip(clean_texts, labels):
        words = text.split()
        if len(words) < 2 * PASSAGE_WORDS:
            continue
        for i in range(0, min(len(words), PASSAGE_WORDS * MAX_PASSAGES), PASSAGE_WORDS):
            texts.append(" ".join(words[i:i + PASSAGE_WORDS]))
            ys.append(y)
    return texts, ys


@dataclass
class PredictionResult:
    label: str
    confidence: float
    top_terms: list = field(default_factory=list)


class FakeNewsDetector:
    def __init__(self, algorithm="logreg", max_features=30000):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.algorithm = algorithm
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
        self.model = ALGORITHMS[algorithm][1]()
        self.is_trained = False

    @property
    def algorithm_name(self):
        return ALGORITHMS[self.algorithm][0]

    def train(self, texts, labels, test_size=0.25, seed=42):
        clean = [self.preprocessor.preprocess(t) for t in texts]
        X_tr, X_te, y_tr, y_te = train_test_split(
            clean, labels, test_size=test_size, random_state=seed, stratify=labels)
        self.fit_clean(X_tr, y_tr)
        return self.evaluate(X_te, y_te)

    def fit_clean(self, clean_texts, labels):
        texts, ys = add_passages(clean_texts, labels)
        self.model.fit(self.vectorizer.fit_transform(texts), ys)
        self.is_trained = True

    def evaluate(self, clean_texts, y_true):
        y_pred = self.model.predict(self.vectorizer.transform(clean_texts))
        return {"accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}

    def predict_label_clean(self, clean_text):
        """1 = FAKE, 0 = REAL for already preprocessed text (used when retraining)."""
        return int(self.model.predict(self.vectorizer.transform([clean_text]))[0])

    def _prob_fake(self, vec):
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(vec)[0][1])
        margin = float(self.model.decision_function(vec)[0])      # SVM / PA: squash the margin
        return 1 / (1 + math.exp(-2 * margin))

    def predict(self, text):
        ok, msg = validate_input(text)
        if not ok:
            raise ValueError(msg)
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        vec = self.vectorizer.transform([self.preprocessor.preprocess(text)])
        prob_fake = self._prob_fake(vec)
        label = "FAKE" if prob_fake >= 0.5 else "REAL"
        confidence = prob_fake if label == "FAKE" else 1 - prob_fake
        return PredictionResult(label, round(confidence * 100, 2), self._top_terms(vec))

    def term_weights(self):
        """Weight of every vocabulary term; > 0 pushes towards FAKE, < 0 towards REAL."""
        if hasattr(self.model, "feature_log_prob_"):               # Naive Bayes
            return self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]
        return self.model.coef_[0]

    def _top_terms(self, vec, k=5):
        names = self.vectorizer.get_feature_names_out()
        w = self.term_weights()
        idx = vec.nonzero()[1]
        weights = [(names[i], w[i] * vec[0, i]) for i in idx]
        weights.sort(key=lambda t: abs(t[1]), reverse=True)
        return [t[0] for t in weights[:k]]

    def explain(self, text):
        """Split the original text into segments [(text, score)], score in -1..1 (FAKE > 0)."""
        vocab, w = self.vectorizer.vocabulary_, self.term_weights()
        vec = self.vectorizer.transform([self.preprocessor.preprocess(text)])
        contrib = {}
        for i in vec.nonzero()[1]:
            contrib[i] = w[i] * vec[0, i]
        top = max((abs(v) for v in contrib.values()), default=0) or 1
        segments = []
        for tok in re.findall(r"[A-Za-z]+|[^A-Za-z]+", text):
            i = vocab.get(tok.lower()) if tok[0].isalpha() else None
            score = contrib.get(i, 0.0) / top if i is not None else 0.0
            segments.append((tok, round(float(score), 3)))
        return segments

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model,
                         "algorithm": self.algorithm}, f)

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple):                                 # files from the first version
            data = {"vectorizer": data[0], "model": data[1], "algorithm": "logreg"}
        self.vectorizer, self.model, self.algorithm = data["vectorizer"], data["model"], data["algorithm"]
        self.is_trained = True
