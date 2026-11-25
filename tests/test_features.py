"""Unit tests for the add-on features (TC34-TC47). External services are replaced with fakes."""
import os
import pickle
import tempfile
import pytest
from conftest import FAKE_TXT, REAL_TXT

import factcheck
import llm
import signals
import sources
import urlfetch
from database import HistoryRepository
from dataset import DatasetLoader, strip_leaks
from detector import ALGORITHMS, FakeNewsDetector
from model_select import compare_models
from retrain import retrain_from_feedback


@pytest.fixture(scope="module")
def sample():
    return DatasetLoader().load()


@pytest.fixture
def repo(tmp_path):
    r = HistoryRepository(str(tmp_path / "t.db"), admin_password="admin123")
    yield r
    r.conn.close()


class FakeResponse:
    def __init__(self, status=200, data=None, text=""):
        self.status_code, self._data, self.text = status, data or {}, text

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def test_TC34_compare_models_ranks_all_algorithms(sample):
    rows = compare_models(*sample, folds=3)
    assert [r["algorithm"] for r in rows] and len(rows) == len(ALGORITHMS)
    assert all(0 <= r["accuracy"] <= 1 for r in rows)
    assert rows == sorted(rows, key=lambda r: r["f1"], reverse=True)


@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_TC35_every_algorithm_predicts(sample, algorithm):
    det = FakeNewsDetector(algorithm)
    det.train(*sample)
    r = det.predict(FAKE_TXT)
    assert r.label == "FAKE" and 50 <= r.confidence <= 100 and r.top_terms
    with pytest.raises(ValueError):
        FakeNewsDetector("unknown")


def test_TC36_label_leaks_are_removed():
    assert strip_leaks("WASHINGTON (Reuters) - The senate voted.") == "The senate voted."
    assert strip_leaks("Story text. Featured image via Getty Images") == "Story text."
    assert strip_leaks("Plain text stays.") == "Plain text stays."


def test_TC37_word_highlighting(sample):
    det = FakeNewsDetector()
    det.train(*sample)
    segs = det.explain("SHOCKING secret! The central bank said rates stay unchanged.")
    assert "".join(t for t, _ in segs) == "SHOCKING secret! The central bank said rates stay unchanged."
    scores = dict(segs)
    assert scores["SHOCKING"] > 0 and scores["bank"] < 0
    assert all(-1 <= s <= 1 for _, s in segs)


def test_TC38_writing_signals():
    bad = signals.analyse(FAKE_TXT + "!")
    assert any("clickbait" in w for w in bad["warnings"]) and any("exclamation" in w for w in bad["warnings"])
    good = signals.analyse("The minister said on Monday that growth reached 6.5 percent, according to official data.")
    assert not good["warnings"] and len(good["good"]) == 3


def test_TC39_source_credibility():
    assert sources.credibility("https://www.reuters.com/world/x")["level"] == "good"
    assert sources.credibility("https://news.theonion.com/a")["level"] == "warn"
    assert sources.credibility("http://infowars.com/b")["level"] == "bad"
    assert sources.credibility("https://myblog.example/c") == {"domain": "myblog.example", "level": "unknown",
                                                               "label": "Source not rated"}


@pytest.mark.parametrize("url", ["http://127.0.0.1:5000/", "http://localhost/", "http://10.0.0.5/x",
                                 "http://169.254.169.254/latest", "file:///etc/passwd", "ftp://example.com/a", "news"])
def test_TC40_url_fetch_refuses_private_addresses(url):
    with pytest.raises(urlfetch.FetchError):
        urlfetch._check_public(url)


ARTICLE_HTML = ("<html><head><title>Rates unchanged | Daily News</title></head><body><nav>Home Sport</nav>"
                "<article><h1>Central bank keeps rates unchanged</h1><p>" + REAL_TXT + ". The governor told "
                "reporters that inflation remained within the target band for the third month in a row.</p>"
                "</article><footer>Copyright</footer></body></html>")


def test_TC41_article_extraction(monkeypatch):
    title, text = urlfetch.extract(ARTICLE_HTML)
    assert "central bank" in text.lower() and "Copyright" not in text and title
    monkeypatch.setattr(urlfetch, "download", lambda url: ("https://news.example/rates", ARTICLE_HTML))
    final, title, text = urlfetch.fetch_article("https://news.example/rates")
    assert final == "https://news.example/rates" and len(text.split()) > 20
    monkeypatch.setattr(urlfetch, "download", lambda url: (url, "<html><body><p>Too short.</p></body></html>"))
    with pytest.raises(urlfetch.FetchError):
        urlfetch.fetch_article("https://news.example/empty")


def test_TC42_fact_check_lookup(monkeypatch):
    assert factcheck.search(FAKE_TXT) == []                      # no key -> switched off
    assert factcheck.build_query("Miracle pill cures all! Doctors hate it.") == "Miracle pill cures all"
    monkeypatch.setenv("FACTCHECK_API_KEY", "test-key")
    claim = {"text": "Pill cures all diseases", "claimant": "Social media",
             "claimReview": [{"publisher": {"name": "BOOM"}, "textualRating": "False",
                              "url": "https://boomlive.in/x", "reviewDate": "2026-03-01T00:00:00Z"}]}
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return FakeResponse(data={"claims": [claim]})
    monkeypatch.setattr(factcheck.requests, "get", fake_get)
    res = factcheck.search(FAKE_TXT)
    assert seen["key"] == "test-key" and res[0]["publisher"] == "BOOM" and res[0]["rating"] == "False"
    assert res[0]["date"] == "2026-03-01"


def test_TC43_grok_explanation(monkeypatch):
    with pytest.raises(llm.LLMUnavailable):
        llm.explain(FAKE_TXT, "FAKE", 90.0, ["shocking"])        # no key
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    monkeypatch.setenv("GROK_MODEL", "grok-retired")
    llm._resolved.clear()
    calls = []

    def fake_post(url, headers, timeout, json):
        calls.append(json["model"])
        assert headers["Authorization"] == "Bearer xai-test" and url.endswith("/chat/completions")
        if json["model"] == "grok-retired":
            return FakeResponse(404, text='{"error": "The model grok-retired does not exist"}')
        return FakeResponse(data={"choices": [{"message": {"content": "- Uses sensational words."}}]})

    def fake_get(url, headers, timeout):
        return FakeResponse(data={"data": [{"id": "grok-imagine-image"}, {"id": "grok-test-1"}]})
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.requests, "get", fake_get)
    text, model = llm.explain(FAKE_TXT, "FAKE", 90.0, ["shocking"])
    assert text == "- Uses sensational words." and model == "grok-test-1" and calls == ["grok-retired", "grok-test-1"]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(401, text="bad key"))
    llm._resolved.clear()
    with pytest.raises(llm.LLMUnavailable, match="rejected"):
        llm.explain(FAKE_TXT, "FAKE", 90.0, [])
    llm._resolved.clear()


def test_TC44_retrain_from_feedback(repo, tmp_path, sample):
    det = FakeNewsDetector()
    metrics = det.train(*sample)
    path = str(tmp_path / "model.pkl")
    det.save(path)
    mid = repo.add_model(repo.add_dataset("sample", "built-in", *sample), metrics, path)
    assert retrain_from_feedback(repo, path)["feedback_used"] == 0
    text = "Officials said the new bridge opened on Monday after three years of construction work in the city"
    pid = repo.save_prediction(text, det.predict(text), mid)
    repo.add_feedback(pid, False)
    out = retrain_from_feedback(repo, path)
    assert out["feedback_used"] == 1 and out["accepted"] == (out["new_accuracy"] >= out["old_accuracy"])
    assert repo.retrain_runs()[0]["feedback_used"] == 1
    if out["accepted"]:
        assert repo.latest_model()["version"] == "v2" and os.path.exists(path + ".bak")


def test_TC45_feedback_corrects_labels(repo, sample):
    det = FakeNewsDetector()
    m = det.train(*sample)
    mid = repo.add_model(repo.add_dataset("s", "s", ["a"], [1]), m, "x.pkl")
    p1 = repo.save_prediction(FAKE_TXT, det.predict(FAKE_TXT), mid)
    p2 = repo.save_prediction(REAL_TXT, det.predict(REAL_TXT), mid)
    repo.add_feedback(p1, True)                                  # FAKE confirmed -> 1
    repo.add_feedback(p2, True)
    repo.add_feedback(p2, False)                                 # latest answer wins: REAL was wrong -> 1
    texts, labels = repo.feedback_samples()
    assert sorted(zip(texts, labels)) == sorted([(FAKE_TXT, 1), (REAL_TXT, 1)])


def test_TC46_accounts(repo):
    uid = repo.create_user("Parth", "Parth@Example.com", "secret123")
    assert repo.check_login("parth@example.com", "secret123")["user_id"] == uid
    assert repo.check_login("parth@example.com", "wrong") is None
    assert repo.check_login("guest@fnds.local", "") is None          # guest can never log in
    with pytest.raises(ValueError):
        repo.create_user("Again", "parth@example.com", "secret123")
    assert repo.admin_uses_default_password()
    repo.set_password(repo.admin_id, "n3w-password")
    assert not repo.admin_uses_default_password()


def test_TC47_old_model_files_still_load(sample, tmp_path):
    det = FakeNewsDetector()
    det.train(*sample)
    path = tmp_path / "old.pkl"
    with open(path, "wb") as f:
        pickle.dump((det.vectorizer, det.model), f)               # format of the first version
    d2 = FakeNewsDetector()
    d2.load(str(path))
    assert d2.algorithm == "logreg" and d2.predict(FAKE_TXT).label == "FAKE"


class FakeRaw:
    def __init__(self, body):
        self.body = body

    def read(self, n, decode_content=True):
        return self.body[:n]


class FakePage:
    def __init__(self, status=200, location=None, body=b""):
        self.status_code, self.headers = status, {"Location": location} if location else {}
        self.is_redirect, self.raw, self.encoding = location is not None, FakeRaw(body), "utf-8"


def test_TC63_download_checks_every_redirect(monkeypatch):
    public = [(2, 1, 6, "", ("93.184.216.34", 0))]
    private = [(2, 1, 6, "", ("192.168.1.1", 0))]
    monkeypatch.setattr(urlfetch.socket, "getaddrinfo",
                        lambda host, port: private if host == "router.local" else public)
    pages = {"https://short.example/a": FakePage(301, "/story"),
             "https://short.example/story": FakePage(200, body=b"<html>ok</html>"),
             "https://evil.example/x": FakePage(302, "http://router.local/admin"),
             "https://big.example/": FakePage(200, body=b"x" * (urlfetch.MAX_BYTES + 1)),
             "https://gone.example/": FakePage(404)}
    monkeypatch.setattr(urlfetch.requests, "get", lambda url, **kw: pages[url])
    assert urlfetch.download("https://short.example/a") == ("https://short.example/story", "<html>ok</html>")
    for url, msg in [("https://evil.example/x", "Only public"), ("https://big.example/", "too large"),
                     ("https://gone.example/", "status 404")]:
        with pytest.raises(urlfetch.FetchError, match=msg):
            urlfetch.download(url)
    loop = {"https://loop.example/": FakePage(302, "https://loop.example/")}
    monkeypatch.setattr(urlfetch.requests, "get", lambda url, **kw: loop[url])
    with pytest.raises(urlfetch.FetchError, match="redirects"):
        urlfetch.download("https://loop.example/")


def test_TC64_env_file_settings(tmp_path, monkeypatch):
    import config
    env = tmp_path / ".env"
    env.write_text("# comment\nFNDS_TEST_A=from-file\nFNDS_TEST_B = 'quoted'\nFNDS_TEST_C=from-file\n")
    monkeypatch.setenv("FNDS_TEST_C", "from-environment")
    monkeypatch.delenv("FNDS_TEST_A", raising=False)
    monkeypatch.delenv("FNDS_TEST_B", raising=False)
    config.load_env(str(env))
    assert (config.get("FNDS_TEST_A"), config.get("FNDS_TEST_B"), config.get("FNDS_TEST_C")) == \
           ("from-file", "quoted", "from-environment")
    for k in ("FNDS_TEST_A", "FNDS_TEST_B"):
        monkeypatch.delenv(k)


def test_TC65_short_passages_added_for_training():
    from detector import add_passages
    long_text = " ".join(f"w{i}" for i in range(200))
    texts, labels = add_passages([long_text, "short text only"], [1, 0])
    assert texts[:2] == [long_text, "short text only"] and len(texts) == 2 + 4
    assert all(len(t.split()) == 40 for t in texts[2:]) and labels == [1, 0, 1, 1, 1, 1]
