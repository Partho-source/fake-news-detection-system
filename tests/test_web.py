"""System tests of the add-on web features and the REST API (TC48-TC58)."""
import io
import re
import pytest
from conftest import FAKE_TXT, REAL_TXT, get_app, token

mod = get_app()
app = mod.app


@pytest.fixture
def client():
    c = app.test_client()
    c.csrf = token(c)
    return c


def login(c, email="admin@fnds.local", password="admin123"):
    return c.post("/login", data={"email": email, "password": password, "csrf_token": c.csrf})


def test_TC48_forms_need_csrf_token(client):
    assert client.post("/", data={"article": FAKE_TXT}).status_code == 400
    assert client.post("/", data={"article": FAKE_TXT, "csrf_token": "forged"}).status_code == 400
    assert client.post("/", data={"article": FAKE_TXT, "csrf_token": client.csrf}).status_code == 200


def test_TC49_api_predict(client):
    r = client.post("/api/predict", json={"text": REAL_TXT})
    d = r.get_json()
    assert r.status_code == 200 and d["label"] == "REAL" and d["prediction_id"] > 0 and "signals" in d
    assert client.post("/api/predict", json={"text": "too short"}).status_code == 400
    assert client.post("/api/predict", data="text=" + REAL_TXT).status_code == 415   # JSON only
    h = client.get("/api/health").get_json()
    assert h["status"] == "ok" and h["grok"] is False and h["factcheck"] is False


def test_TC50_check_by_url(client, monkeypatch):
    article = REAL_TXT + ". " + "The governor said inflation stayed within the target band. " * 3
    monkeypatch.setattr(mod, "fetch_article", lambda url: ("https://www.reuters.com/x", "Rates unchanged", article))
    d = client.post("/api/predict", json={"url": "https://www.reuters.com/x"}).get_json()
    assert d["title"] == "Rates unchanged" and d["source"]["label"] == "Established news outlet"
    html = client.post("/", data={"mode": "url", "url": "https://www.reuters.com/x",
                                  "csrf_token": client.csrf}).get_data(as_text=True)
    assert "Result: REAL" in html and "Established news outlet" in html

    def fail(url):
        raise mod.FetchError("That website could not be found.")
    monkeypatch.setattr(mod, "fetch_article", fail)
    html = client.post("/", data={"mode": "url", "url": "https://nope.invalid", "csrf_token": client.csrf})
    assert "That website could not be found." in html.get_data(as_text=True)


def test_TC51_long_pages_are_cut_at_a_sentence(client):
    text, cut = mod.fit_text(("This sentence is part of a very long article. " * 400).strip())
    assert cut and len(text) <= 10000 and text.endswith(".")
    assert mod.fit_text(REAL_TXT) == (REAL_TXT, False)


def test_TC52_batch_csv(client):
    csv_data = f'title,text\nOne,"{FAKE_TXT}"\nTwo,"{REAL_TXT}"\nThree,too short\n'
    r = client.post("/batch", data={"file": (io.BytesIO(csv_data.encode()), "news.csv"), "csrf_token": client.csrf},
                    content_type="multipart/form-data")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "Download results" in html
    bid = re.search(r"/batch/(\w+)\.csv", html).group(1)
    out = client.get(f"/batch/{bid}.csv").get_data(as_text=True).splitlines()
    assert out[0].startswith("row,title,label") and ",FAKE," in out[1] and ",REAL," in out[2] and "too short" in out[3]
    bad = client.post("/batch", data={"file": (io.BytesIO(b"name\nx\n"), "x.csv"), "csrf_token": client.csrf},
                      content_type="multipart/form-data")
    assert bad.status_code == 400 and "column named text" in bad.get_data(as_text=True)
    assert client.get("/batch/unknown.csv").status_code == 404


def test_TC53_register_login_logout_and_own_history(client):
    r = client.post("/register", data={"name": "Asha", "email": "asha@example.com", "password": "password1",
                                       "csrf_token": client.csrf})
    assert r.status_code == 302
    client.post("/", data={"article": REAL_TXT + " in Mumbai today", "csrf_token": client.csrf})
    html = client.get("/history").get_data(as_text=True)
    assert "Your predictions" in html
    assert html.count("<tr>") == 2                                 # header + her one check
    client.post("/logout", data={"csrf_token": client.csrf})
    assert "Log in" in client.get("/").get_data(as_text=True)
    assert "Wrong email or password" in login(client, "asha@example.com", "nope").get_data(as_text=True)
    weak = client.post("/register", data={"name": "B", "email": "b@example.com", "password": "short",
                                          "csrf_token": client.csrf})
    assert "at least 8 characters" in weak.get_data(as_text=True)


def test_TC54_dashboard_is_admin_only(client):
    assert client.get("/admin").status_code == 302                  # not logged in -> login page
    client.post("/register", data={"name": "U", "email": "u@example.com", "password": "password1",
                                   "csrf_token": client.csrf})
    assert client.get("/admin").status_code == 403                  # ordinary user
    client.post("/logout", data={"csrf_token": client.csrf})
    login(client)
    html = client.get("/admin").get_data(as_text=True)
    assert "Retrain from feedback" in html and "Model versions" in html


def test_TC55_retrain_and_password_from_dashboard(client):
    login(client)
    pid = client.post("/api/predict", json={"text": REAL_TXT + " again"}).get_json()["prediction_id"]
    client.post("/feedback", data={"prediction_id": pid, "is_correct": "0", "csrf_token": client.csrf})
    html = client.post("/admin/retrain", data={"csrf_token": client.csrf}, follow_redirects=True).get_data(as_text=True)
    assert "test accuracy" in html and ("accepted" in html or "rejected" in html)
    msg = client.post("/admin/password", data={"password": "short", "csrf_token": client.csrf}, follow_redirects=True)
    assert "at least 8 characters" in msg.get_data(as_text=True)


def test_TC56_grok_explain_endpoint(client, monkeypatch):
    pid = client.post("/api/predict", json={"text": FAKE_TXT}).get_json()["prediction_id"]
    r = client.post(f"/api/explain/{pid}", json={})
    assert r.status_code == 503 and "XAI_API_KEY" in r.get_json()["error"]
    assert client.post(f"/api/explain/{pid}", data="x").status_code == 415
    assert client.post("/api/explain/99999", json={}).status_code == 404
    calls = []
    monkeypatch.setattr(mod.llm, "explain", lambda *a: calls.append(a) or ("- Sensational tone.", "grok-test"))
    for _ in range(2):
        d = client.post(f"/api/explain/{pid}", json={}).get_json()
        assert d == {"source": "grok", "model": "grok-test", "text": "- Sensational tone."}
    assert len(calls) == 1                                          # second answer comes from the cache


def test_TC57_factcheck_endpoint(client, monkeypatch):
    pid = client.post("/api/predict", json={"text": FAKE_TXT}).get_json()["prediction_id"]
    assert client.get(f"/api/factcheck/{pid}").status_code == 503
    monkeypatch.setattr(mod.factcheck, "enabled", lambda: True)
    monkeypatch.setattr(mod.factcheck, "search", lambda text, title=None: [{"publisher": "BOOM", "rating": "False"}])
    d = client.get(f"/api/factcheck/{pid}").get_json()
    assert d["results"][0]["publisher"] == "BOOM" and d["query"]
    assert client.get("/api/factcheck/99999").status_code == 404


def test_TC58_api_cors_only_for_extensions(client):
    ext = client.post("/api/predict", json={"text": REAL_TXT}, headers={"Origin": "chrome-extension://abcdef"})
    assert ext.headers["Access-Control-Allow-Origin"] == "chrome-extension://abcdef"
    site = client.post("/api/predict", json={"text": REAL_TXT}, headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in site.headers
    assert client.options("/api/predict", headers={"Origin": "chrome-extension://abcdef"}).status_code == 204
