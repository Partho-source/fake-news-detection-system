"""preprocessor.py - text cleaning pipeline for the Fake News Detection System."""
import re

STOPWORDS = set("""a an the and or but if then of to in on at by for with from as is are was were be been
being this that these those it its he she they them his her their we you your i me my our not no so than
too very can will just do does did has have had into over under about after before also there here which
who whom what when where why how all any each more most other some such only own same up down out off""".split())

class TextPreprocessor:
    """Cleans raw news text and converts it into a normalised string."""
    def clean(self, text):
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # remove URLs
        text = re.sub(r"<[^>]+>", " ", text)                   # remove HTML tags
        text = re.sub(r"[^a-z\s]", " ", text)                  # keep letters only
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text):
        return text.split()

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    def preprocess(self, text):
        tokens = self.remove_stopwords(self.tokenize(self.clean(text)))
        return " ".join(tokens)
