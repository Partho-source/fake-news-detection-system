"""System tests (TC19-TC24, TC31-TC33): the web interface is tested with the Flask test client."""
import unittest
from conftest import FAKE_TXT, REAL_TXT, get_app, token

app = get_app().app


class TestWebInterface(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.csrf = token(self.client)

    def post(self, text):
        return self.client.post("/", data={"article": text, "csrf_token": self.csrf}).get_data(as_text=True)

    def test_TC19_home_page(self):
        r = self.client.get("/"); self.assertEqual(r.status_code, 200)
        self.assertIn("Fake News Detection System", r.get_data(as_text=True))
    def test_TC20_empty_article(self):    self.assertIn("Please enter a news article.", self.post(""))
    def test_TC21_short_article(self):    self.assertIn("too short", self.post("Fake news is bad"))
    def test_TC22_fake_article(self):     self.assertIn("Result: FAKE", self.post(FAKE_TXT))
    def test_TC23_real_article(self):     self.assertIn("Result: REAL", self.post(REAL_TXT))
    def test_TC24_script_is_escaped(self):
        html = self.post("<script>alert(1)</script> shocking secret miracle cure exposed share now before deleted please")
        self.assertNotIn("<script>alert", html)
    def test_TC31_history_saved(self):
        self.post(REAL_TXT)
        html = self.client.get("/history").get_data(as_text=True)
        self.assertIn("The central bank kept its benchmark", html)
    def test_TC32_feedback(self):
        html = self.post(FAKE_TXT)
        pid = html.split('name="prediction_id" value="')[1].split('"')[0]
        r = self.client.post("/feedback", data={"prediction_id": pid, "is_correct": "1", "csrf_token": self.csrf},
                             follow_redirects=True)
        self.assertIn("Thank you for your feedback.", r.get_data(as_text=True))
        bad = self.client.post("/feedback", data={"prediction_id": "999", "csrf_token": self.csrf})
        self.assertEqual(bad.status_code, 400)
    def test_TC33_check_another_clears(self):
        self.assertIn("Check another article", self.post(FAKE_TXT))
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('placeholder="Paste news article here..."></textarea>', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
