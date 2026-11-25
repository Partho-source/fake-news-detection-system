"""conftest.py - shared test setup: a temporary database and model, the sample dataset, and no API keys,
so the tests never touch your own fnds.db or call paid APIs even if a .env file exists."""
import contextlib
import io
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
TMP = tempfile.mkdtemp(prefix="fnds-test-")
os.environ.update({"FNDS_DB": os.path.join(TMP, "test.db"), "FNDS_MODEL": os.path.join(TMP, "model.pkl"),
                   "FNDS_DATASET": "sample", "FNDS_SECRET_KEY": "test-secret", "FNDS_ADMIN_PASSWORD": "admin123",
                   "XAI_API_KEY": "", "GROK_API_KEY": "", "FACTCHECK_API_KEY": ""})

FAKE_TXT = ("SHOCKING secret exposed! Miracle pill cures every disease overnight, doctors hate it, "
            "share before it gets deleted")
REAL_TXT = ("The central bank kept its benchmark interest rate unchanged on Wednesday, citing steady inflation "
            "and moderate growth")


def get_app():
    with contextlib.redirect_stdout(io.StringIO()):
        import app                                  # trains the sample model on first import
    return app


def token(client):
    html = client.get("/").get_data(as_text=True)
    return re.search(r'name="csrf-token" content="(\w+)"', html).group(1)
