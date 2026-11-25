"""train.py - trains the model, prints evaluation metrics, saves model.pkl and registers it in the database.

Usage:
    python train.py                         # real dataset if downloaded (python get_dataset.py), else the sample
    python train.py --dataset sample        # built-in 60-article sample
    python train.py --dataset real          # data/fake_or_real_news.csv
    python train.py --csv news.csv          # one CSV with text + label (1/0 or FAKE/REAL)
    python train.py --fake-csv Fake.csv --real-csv True.csv   # corpus split into two files (e.g. ISOT)
    python train.py --algorithm svm         # logreg | nb | svm | pa | auto (compare and pick the best F1)
"""
import argparse
import os
from database import HistoryRepository
from dataset import REAL_DATASET, DatasetLoader
from detector import ALGORITHMS, FakeNewsDetector
from model_select import compare_models, print_table

MODEL_PATH = os.environ.get("FNDS_MODEL", "model.pkl")
DB_PATH = os.environ.get("FNDS_DB", "fnds.db")


def load_data(args):
    loader = DatasetLoader()
    if args.csv:
        return (*loader.load_csv(args.csv, args.text_col, args.label_col), args.csv)
    if args.fake_csv and args.real_csv:
        ft, fl = loader.load_csv(args.fake_csv, args.text_col, label=1)
        rt, rl = loader.load_csv(args.real_csv, args.text_col, label=0)
        return ft + rt, fl + rl, f"{args.fake_csv} + {args.real_csv}"
    choice = args.dataset or os.environ.get("FNDS_DATASET", "auto")
    if choice == "auto" and not os.path.exists(REAL_DATASET):
        try:                                      # first run: fetch the public dataset if online
            import get_dataset
            get_dataset.main()
        except Exception as e:                    # offline -> fall back to the built-in sample
            print(f"Could not download the real dataset ({e.__class__.__name__}); using the sample.")
    if choice == "real" or (choice == "auto" and os.path.exists(REAL_DATASET)):
        return (*loader.load_real(), "fake_or_real_news.csv")
    return (*loader.load(), "built-in sample")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the Fake News Detection model.")
    ap.add_argument("--dataset", choices=["auto", "sample", "real"])
    ap.add_argument("--csv", help="CSV file with a text column and a label column")
    ap.add_argument("--fake-csv", help="CSV file containing only fake articles")
    ap.add_argument("--real-csv", help="CSV file containing only real articles")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--algorithm", default="auto", choices=["auto", *ALGORITHMS])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    texts, labels, source = load_data(args)
    if len(set(labels)) < 2:
        raise SystemExit("The dataset must contain both FAKE and REAL articles.")
    print(f"Dataset       : {source}")
    rows, algorithm = None, args.algorithm
    if algorithm == "auto":
        if len(texts) >= 200:
            print(f"Comparing {len(ALGORITHMS)} algorithms with {args.folds}-fold cross-validation ...")
            rows = compare_models(texts, labels, folds=args.folds)
            print_table(rows)
            algorithm = rows[0]["algorithm"]
        else:
            algorithm = "logreg"                       # too few articles for a fair comparison
    det = FakeNewsDetector(algorithm)
    m = det.train(texts, labels)
    print(f"Algorithm     : {det.algorithm_name}")
    print(f"Articles used : {len(texts)}  (FAKE={sum(labels)}, REAL={len(labels)-sum(labels)})")
    print(f"Accuracy      : {m['accuracy']:.2f}")
    print(f"Precision     : {m['precision']:.2f}")
    print(f"Recall        : {m['recall']:.2f}")
    print(f"F1-score      : {m['f1']:.2f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:", m['confusion'])
    det.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    repo = HistoryRepository(DB_PATH)
    dataset_id = repo.add_dataset(os.path.basename(source), source, texts, labels)
    model_id = repo.add_model(dataset_id, m, MODEL_PATH, det.algorithm_name)
    if rows:
        repo.add_comparison(model_id, rows)
    print(f"Model registered in {DB_PATH} (model_id={model_id}, {repo.latest_model()['version']})")
    repo.conn.close()
    return model_id


if __name__ == "__main__":
    main()
