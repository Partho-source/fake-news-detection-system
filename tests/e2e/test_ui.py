"""Browser tests with Playwright (TC59-TC62): a real Chromium drives the running app.
Install once with:  python -m playwright install chromium"""
import os
import threading
import pytest
from werkzeug.serving import make_server
from conftest import FAKE_TXT, REAL_TXT, get_app

pw = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e
EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "extension")


@pytest.fixture(scope="module")
def server():
    srv = make_server("127.0.0.1", 5055, get_app().app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:5055"
    srv.shutdown()


def _launch(p, **kw):
    for exe in (None, "/usr/bin/google-chrome", "/usr/bin/chromium"):
        try:
            return p.chromium.launch(executable_path=exe, **kw) if exe else p.chromium.launch(**kw)
        except Exception:
            continue
    pytest.skip("Chromium is not installed (python -m playwright install chromium)")


@pytest.fixture(scope="module")
def play():
    with pw.sync_playwright() as p:
        yield p


@pytest.fixture(scope="module")
def page(server, play):
    browser = _launch(play)
    pg = browser.new_page(viewport={"width": 1100, "height": 900})
    yield pg
    browser.close()


def test_TC59_check_highlight_and_clear(page, server):
    page.goto(server + "/")
    page.fill("#article", FAKE_TXT)
    assert "words" in page.inner_text("#counter")
    page.click("text=Check News")
    assert page.inner_text("#result h3") == "Result: FAKE"
    assert page.locator(".highlight span[title='towards FAKE']").count() >= 3
    assert "clickbait" in page.inner_text("ul.signals.bad")
    page.click("#clear-btn")
    assert page.input_value("#article") == "" and page.locator("#result").count() == 0


def test_TC60_feedback_then_history(page, server):
    page.goto(server + "/")
    page.fill("#article", REAL_TXT)
    page.click("text=Check News")
    page.click("button[name=is_correct][value='1']")
    assert "Thank you for your feedback." in page.inner_text("main")
    assert "Correct" in page.inner_text("table")


def test_TC61_admin_dashboard(page, server):
    page.goto(server + "/login")
    page.fill("#email", "admin@fnds.local")
    page.fill("#password", "admin123")
    page.click("button[type=submit]")
    page.click("text=Dashboard")
    assert "Learn from feedback" in page.inner_text("main")
    page.click("text=Retrain from feedback")
    page.wait_for_selector("p.ok")
    assert "test accuracy" in page.inner_text("p.ok")


def test_TC62_browser_extension_popup(server, play, tmp_path):
    """Loads the unpacked extension, points it at the test server and checks text from its popup."""
    p = play
    if True:
        args = [f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"]
        ctx = None
        for kw in ({"channel": "chromium"}, {"executable_path": "/usr/bin/google-chrome"}):
            try:                                     # the headless shell cannot run extensions
                ctx = p.chromium.launch_persistent_context(str(tmp_path), headless=True, args=args, **kw)
                break
            except Exception:
                continue
        if ctx is None:
            pytest.skip("This Chromium cannot load extensions")
        try:
            worker = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker", timeout=10000)
        except Exception:
            ctx.close()
            pytest.skip("Extension service worker did not start in this browser")
        ext_id = worker.url.split("/")[2]
        worker.evaluate(f"chrome.storage.sync.set({{server: '{server}'}})")
        popup = ctx.new_page()
        popup.goto(f"chrome-extension://{ext_id}/popup.html")
        popup.fill("#text", FAKE_TXT)
        popup.click("#check")
        pw.expect(popup.locator("#out")).to_contain_text("Result: FAKE", timeout=15000)
        ctx.close()
