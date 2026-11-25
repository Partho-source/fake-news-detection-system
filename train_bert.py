"""train_bert.py - fine-tunes DistilBERT on the real dataset and compares it with the TF-IDF model.

Run on a GPU (Google Colab: Runtime > Change runtime type > T4 GPU):
    pip install -r requirements.txt -r requirements-bert.txt
    python get_dataset.py
    python train_bert.py --epochs 2
Then copy the models/distilbert folder into the project on your laptop and restart app.py.
On a CPU-only laptop use --max-rows 1500 --epochs 1 (slow, but works).
"""
import argparse
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bert_model import BERT_DIR
from dataset import DatasetLoader

DB_PATH = os.environ.get("FNDS_DB", "fnds.db")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-rows", type=int, default=0, help="use only this many articles (0 = all)")
    ap.add_argument("--base", default="distilbert-base-uncased")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorWithPadding)

    texts, labels = DatasetLoader().load_real()
    if args.max_rows:
        texts, _, labels, _ = train_test_split(texts, labels, train_size=args.max_rows,
                                               random_state=42, stratify=labels)
    # same 75/25 split and seed as the TF-IDF model, so the scores are comparable
    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)
    tok = AutoTokenizer.from_pretrained(args.base)
    enc = lambda b: tok(b["text"], truncation=True, max_length=args.max_len)
    train_ds = Dataset.from_dict({"text": X_tr, "label": y_tr}).map(enc, batched=True)
    test_ds = Dataset.from_dict({"text": X_te, "label": y_te}).map(enc, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label={0: "REAL", 1: "FAKE"}, label2id={"REAL": 0, "FAKE": 1})

    def metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        pr, rc, f1, _ = precision_recall_fscore_support(p.label_ids, pred, average="binary")
        return {"accuracy": accuracy_score(p.label_ids, pred), "precision": pr, "recall": rc, "f1": f1}

    targs = TrainingArguments(output_dir="bert_runs", num_train_epochs=args.epochs,
                              per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
                              learning_rate=3e-5, weight_decay=0.01, eval_strategy="epoch",
                              save_strategy="no", logging_steps=50, fp16=torch.cuda.is_available(),
                              report_to=[])
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=test_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    start = time.time()
    trainer.train()
    m = trainer.evaluate()
    print(f"DistilBERT  accuracy={m['eval_accuracy']:.3f}  precision={m['eval_precision']:.3f}  "
          f"recall={m['eval_recall']:.3f}  F1={m['eval_f1']:.3f}  ({(time.time() - start) / 60:.1f} min)")
    os.makedirs(BERT_DIR, exist_ok=True)
    trainer.save_model(BERT_DIR)
    tok.save_pretrained(BERT_DIR)
    print(f"Saved to {BERT_DIR}")

    from database import HistoryRepository
    repo = HistoryRepository(DB_PATH)
    ds = repo.add_dataset("fake_or_real_news.csv", "fake_or_real_news.csv (DistilBERT)", texts, labels)
    repo.add_model(ds, {"accuracy": m["eval_accuracy"], "precision": m["eval_precision"],
                        "recall": m["eval_recall"], "f1": m["eval_f1"]}, BERT_DIR, "DistilBERT")
    print(f"Registered in {DB_PATH}")


if __name__ == "__main__":
    main()
