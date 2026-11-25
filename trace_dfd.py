"""trace_dfd.py - prints the data flows of Level-1 DFD for one article."""
from database import HistoryRepository
from dataset import DatasetLoader
from detector import FakeNewsDetector
from validator import validate_input

det = FakeNewsDetector()
texts, labels = DatasetLoader().load()
metrics = det.train(texts, labels)                       # process 5.0 (offline)
d3 = HistoryRepository(":memory:")                       # D3 Prediction History (in memory)
model_id = d3.add_model(d3.add_dataset("sample", "built-in", texts, labels), metrics, "model.pkl")
article = ("Miracle herbal tea cures cancer in three days, big pharma furious "
           "and trying to ban this unbelievable discovery")
print("User --news article--> 1.0 Validate Input")
ok, msg = validate_input(article)
print(f"1.0 --valid article ({len(article.split())} words)--> 2.0 Preprocess & Extract")
clean = det.preprocessor.preprocess(article)
vec = det.vectorizer.transform([clean])
print(f"D2 --vocabulary, IDF--> 2.0 ; 2.0 --TF-IDF vector ({vec.nnz} non-zero)--> 3.0")
p = float(det.model.predict_proba(vec)[0][1])
print(f"D2 --classifier weights--> 3.0 ; 3.0 --P(fake)={p:.4f}--> 4.0 Generate Result")
r = det.predict(article)
pid = d3.save_prediction(article, r, model_id)
print(f"4.0 --prediction record #{pid}--> D3 ; 4.0 --{r.label}, {r.confidence}%--> User")
print(f"D3 now holds {len(d3.history())} prediction record(s)")
