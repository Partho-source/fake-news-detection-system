"""error_analysis.py - lists the test articles that the model classified wrongly."""
from sklearn.model_selection import train_test_split
from dataset import DatasetLoader
from detector import FakeNewsDetector

texts, labels = DatasetLoader().load()
det = FakeNewsDetector()
det.train(texts, labels)
clean = [det.preprocessor.preprocess(t) for t in texts]
_, X_te, _, y_te = train_test_split(range(len(texts)), labels, test_size=0.25, random_state=42, stratify=labels)
pred = det.model.predict(det.vectorizer.transform([clean[i] for i in X_te]))
print(f"Test articles: {len(y_te)}   wrong: {sum(p != y for p, y in zip(pred, y_te))}")
for i, p, y in zip(X_te, pred, y_te):
    if p != y:
        print(f"actual={'FAKE' if y else 'REAL'} predicted={'FAKE' if p else 'REAL'} : {texts[i][:60]}...")
