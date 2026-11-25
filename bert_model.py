"""bert_model.py - optional DistilBERT engine (needs: pip install -r requirements-bert.txt).

The model is fine-tuned with train_bert.py (free GPU on Google Colab recommended) and saved in
models/distilbert/. If the folder or the libraries are missing, the app simply hides this engine.
"""
import os
from detector import PredictionResult
from validator import validate_input

BERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "distilbert")


def available(path=BERT_DIR):
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    try:
        import torch, transformers  # noqa: F401
        return True
    except ImportError:
        return False


class TransformerDetector:
    algorithm_name = "DistilBERT"

    def __init__(self, path=BERT_DIR):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def prob_fake(self, text):
        enc = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        return float(self.torch.softmax(logits, dim=-1)[0][1])      # label 1 = FAKE

    def predict(self, text):
        ok, msg = validate_input(text)
        if not ok:
            raise ValueError(msg)
        p = self.prob_fake(text)
        label = "FAKE" if p >= 0.5 else "REAL"
        return PredictionResult(label, round((p if label == "FAKE" else 1 - p) * 100, 2), [])
