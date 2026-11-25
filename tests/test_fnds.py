import sys, os, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validator import validate_input
from preprocessor import TextPreprocessor
from dataset import DatasetLoader
from detector import FakeNewsDetector, PredictionResult
from database import HistoryRepository

FAKE_TXT = "SHOCKING secret exposed! Miracle pill cures every disease overnight, doctors hate it, share before it gets deleted"
REAL_TXT = "The central bank kept its benchmark interest rate unchanged on Wednesday, citing steady inflation and moderate growth"

class TestValidator(unittest.TestCase):
    def test_TC01_empty(self):      self.assertFalse(validate_input("")[0])
    def test_TC02_none(self):       self.assertFalse(validate_input(None)[0])
    def test_TC03_spaces(self):     self.assertFalse(validate_input("     ")[0])
    def test_TC04_nine_words(self): self.assertFalse(validate_input("one two three four five six seven eight nine")[0])
    def test_TC05_ten_words(self):  self.assertTrue(validate_input("one two three four five six seven eight nine ten")[0])
    def test_TC06_max_len(self):    self.assertTrue(validate_input("word " * 2000)[0])
    def test_TC07_over_len(self):   self.assertFalse(validate_input("word " * 2001)[0])

class TestPreprocessor(unittest.TestCase):
    def setUp(self): self.p = TextPreprocessor()
    def test_TC08_lower_punct(self): self.assertEqual(self.p.clean("Hello, WORLD!!"), "hello world")
    def test_TC09_url(self):         self.assertNotIn("http", self.p.clean("see https://x.com/a now"))
    def test_TC10_html(self):        self.assertEqual(self.p.clean("<b>fake</b> news"), "fake news")
    def test_TC11_digits(self):      self.assertEqual(self.p.clean("rate 5 percent"), "rate percent")
    def test_TC12_stopwords(self):   self.assertEqual(self.p.preprocess("The cat is on the mat"), "cat mat")

class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = FakeNewsDetector()
        cls.det.train(*DatasetLoader().load())
    def test_TC13_fake(self):
        self.assertEqual(self.det.predict(FAKE_TXT).label, "FAKE")
    def test_TC14_real(self):
        self.assertEqual(self.det.predict(REAL_TXT).label, "REAL")
    def test_TC15_confidence_range(self):
        c = self.det.predict(FAKE_TXT).confidence
        self.assertTrue(50 <= c <= 100)
    def test_TC16_invalid_raises(self):
        with self.assertRaises(ValueError): self.det.predict("too short")
    def test_TC17_untrained(self):
        with self.assertRaises(RuntimeError): FakeNewsDetector().predict(FAKE_TXT)
    def test_TC18_save_load(self):
        self.det.save("/tmp/m.pkl"); d2 = FakeNewsDetector(); d2.load("/tmp/m.pkl")
        self.assertEqual(d2.predict(FAKE_TXT).label, "FAKE")

class TestHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = HistoryRepository(os.path.join(self.tmp.name, "t.db"))
        ds = self.repo.add_dataset("sample", "built-in sample", ["a b c", "d e f"], [1, 0])
        m = {"accuracy": 0.87, "precision": 0.78, "recall": 1.0, "f1": 0.88}
        self.mid = self.repo.add_model(ds, m, "model.pkl")
        self.res = PredictionResult("FAKE", 62.32, ["secret", "shocking"])
    def tearDown(self):
        self.repo.conn.close(); self.tmp.cleanup()
    def test_TC25_prediction_saved(self):
        pid = self.repo.save_prediction(FAKE_TXT, self.res, self.mid)
        row = self.repo.history()[0]
        self.assertEqual((row["prediction_id"], row["label"], row["confidence"]), (pid, "FAKE", 62.32))
    def test_TC26_feedback_saved(self):
        pid = self.repo.save_prediction(FAKE_TXT, self.res, self.mid)
        self.repo.add_feedback(pid, True); self.repo.add_feedback(pid, False)
        self.assertEqual(self.repo.feedback_summary(), {"total": 2, "correct": 1, "incorrect": 1})
    def test_TC27_feedback_unknown_prediction(self):
        with self.assertRaises(ValueError): self.repo.add_feedback(999, True)
    def test_TC28_data_persists(self):
        self.repo.save_prediction(FAKE_TXT, self.res, self.mid); self.repo.conn.close()
        self.repo = HistoryRepository(os.path.join(self.tmp.name, "t.db"))
        self.assertEqual((len(self.repo.history()), self.repo.admin_id,
                          self.repo.latest_model()["version"]), (1, 2, "v1"))

class TestCsvDataset(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); f.write(text); f.close()
        self.addCleanup(os.remove, f.name)
        return f.name
    def test_TC29_csv_labels(self):
        path = self.write("text,label\nfake one,FAKE\nreal one,0\n,1\nbad label,maybe\n")
        self.assertEqual(DatasetLoader().load_csv(path), (["fake one", "real one"], [1, 0]))
    def test_TC30_csv_fixed_label(self):
        path = self.write("title,text\nt1,first story\nt2,second story\n")
        self.assertEqual(DatasetLoader().load_csv(path, label=0)[1], [0, 0])

if __name__ == "__main__":
    unittest.main(verbosity=2)
