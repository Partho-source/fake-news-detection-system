"""app.py - Flask web interface and REST API of the Fake News Detection System."""
import csv
import io
import os
import secrets
import threading
import uuid
from functools import wraps
from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user

import config
import bert_model
import factcheck
import llm
import signals
import sources
from database import HistoryRepository
from detector import FakeNewsDetector
from retrain import retrain_from_feedback
from urlfetch import FetchError, fetch_article
from validator import MAX_CHARS, validate_input

MODEL_PATH = config.get("FNDS_MODEL", "model.pkl")
DB_PATH = config.get("FNDS_DB", "fnds.db")
MAX_BATCH_ROWS = 500

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def _secret_key(path=os.path.join(config.HERE, ".secret_key")):
    """Kept in a file so logins survive a restart; never committed (see .gitignore)."""
    if config.get("FNDS_SECRET_KEY"):
        return config.get("FNDS_SECRET_KEY")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


app.secret_key = _secret_key()
repo = HistoryRepository(DB_PATH)
if not os.path.exists(MODEL_PATH) or repo.latest_model() is None:
    import train                                     # first run: train on the available dataset
    train.main([])


class Engines:
    """Holds the loaded models; the classic model can be swapped after retraining."""

    def __init__(self):
        self.lock = threading.Lock()
        self.bert = None
        self.reload()

    def reload(self):
        det = FakeNewsDetector()
        det.load(MODEL_PATH)
        with self.lock:
            self.classic, self.model = det, repo.latest_model()
        if bert_model.available() and self.bert is None and repo.latest_model(transformer=True):
            self.bert = bert_model.TransformerDetector()
            self.bert_row = repo.latest_model(transformer=True)

    def choose(self, name):
        if name == "bert" and self.bert is not None:
            return self.bert, self.bert_row
        return self.classic, self.model


engines = Engines()
batches = {}                                         # batch id -> CSV text (kept in memory)
explanations = {}                                    # prediction id -> Grok answer (cache)

# ---------------- login ----------------
login = LoginManager(app)
login.login_view = "login_page"
login.login_message = "Please log in to see that page."


class User(UserMixin):
    def __init__(self, row):
        self.id, self.name, self.email, self.role = row["user_id"], row["name"], row["email"], row["role"]

    @property
    def is_admin(self):
        return self.role == "ADMIN"


@login.user_loader
def load_user(user_id):
    row = repo.get_user(int(user_id))
    return User(row) if row and row["password_hash"] != "-" else None


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return login.unauthorized()
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def uid():
    return current_user.id if current_user.is_authenticated else None


# ---------------- CSRF protection for the HTML forms ----------------
def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def check_csrf():
    if request.method == "POST" and not request.path.startswith("/api/"):
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not sent or sent != session.get("csrf"):
            abort(400, "The form expired. Reload the page and try again.")


@app.after_request
def api_cors(resp):
    """Let browser extensions call the JSON API. Ordinary websites get no CORS headers, and the POST
    endpoints only accept JSON, so another site cannot use them from your browser."""
    origin = request.headers.get("Origin", "")
    if request.path.startswith("/api/") and origin.startswith(("chrome-extension://", "moz-extension://")):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.context_processor
def globals_for_templates():
    return {"model": engines.model, "bert_ready": engines.bert is not None,
            "llm_enabled": llm.enabled(), "factcheck_enabled": factcheck.enabled()}


# ---------------- core check ----------------
def run_check(text, engine="classic", title=None, source_url=None, save=True):
    det, row = engines.choose(engine)
    result = det.predict(text)                       # raises ValueError for invalid input
    pid = repo.save_prediction(text, result, row["model_id"], uid(), title, source_url) if save else None
    highlights = det.explain(text) if hasattr(det, "explain") else None
    return {"result": result, "prediction_id": pid, "engine": det.algorithm_name, "model_version": row["version"],
            "highlights": highlights, "signals": signals.analyse(text),
            "source": sources.credibility(source_url) if source_url else None}


def fit_text(text):
    """Long pages are cut at the last sentence that fits the 10,000-character limit."""
    if len(text) <= MAX_CHARS:
        return text, False
    cut = text[:MAX_CHARS]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[:end + 1] if end > MAX_CHARS // 2 else cut).strip(), True


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {"error": None, "text": "", "url": "", "mode": request.args.get("mode", "text"),
           "engine": "classic", "truncated": False}
    if request.method == "POST":
        ctx["mode"] = request.form.get("mode", "text")
        ctx["engine"] = request.form.get("engine", "classic")
        title = source_url = None
        if ctx["mode"] == "url":
            ctx["url"] = request.form.get("url", "").strip()
            try:
                source_url, title, text = fetch_article(ctx["url"])
                text, ctx["truncated"] = fit_text(text)
            except FetchError as e:
                ctx["error"] = str(e)
                return render_template("index.html", **ctx)
        else:
            text = request.form.get("article", "")
        ctx["text"], ctx["title"] = text, title
        ok, msg = validate_input(text)
        if ok:
            ctx.update(run_check(text, ctx["engine"], title, source_url))
        else:
            ctx["error"] = msg
    return render_template("index.html", **ctx)


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        repo.add_feedback(int(request.form.get("prediction_id", 0)),
                          request.form.get("is_correct") == "1", user_id=uid())
    except ValueError:
        return render_template("index.html", error="Feedback could not be saved.", text="", url="",
                               mode="text", engine="classic"), 400
    return redirect(url_for("history", thanks=1))


@app.route("/history")
def history():
    if current_user.is_authenticated and current_user.is_admin:
        rows = repo.history(50)
    else:
        rows = repo.history(50, user_id=uid() or repo.guest_id)
    return render_template("history.html", rows=rows, summary=repo.feedback_summary(),
                           thanks=request.args.get("thanks"))


# ---------------- batch CSV ----------------
@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html")
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return render_template("batch.html", error="Choose a .csv file."), 400
    try:
        reader = csv.DictReader(io.StringIO(f.read().decode("utf-8-sig", errors="replace")))
        rows = list(reader)
    except csv.Error:
        return render_template("batch.html", error="The file is not a valid CSV."), 400
    col = next((c for c in (reader.fieldnames or []) if c.strip().lower() in ("text", "article", "content")), None)
    if not col:
        return render_template("batch.html", error="The CSV needs a column named text, article or content."), 400
    if len(rows) > MAX_BATCH_ROWS:
        return render_template("batch.html", error=f"At most {MAX_BATCH_ROWS} rows per file."), 400
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["row", "title", "label", "confidence", "top_words", "status"])
    results = []
    for i, row in enumerate(rows, start=1):
        text, _ = fit_text((row.get(col) or "").strip())
        title = (row.get("title") or "").strip() or None
        ok, msg = validate_input(text)
        if ok:
            r = run_check(text, request.form.get("engine", "classic"), title)["result"]
            w.writerow([i, title or text[:60], r.label, r.confidence, " ".join(r.top_terms), "ok"])
            results.append((i, title or text[:60], r.label, r.confidence, "ok"))
        else:
            w.writerow([i, title or text[:60], "", "", "", msg])
            results.append((i, title or text[:60], "", "", msg))
    bid = uuid.uuid4().hex
    batches[bid] = out.getvalue()
    while len(batches) > 20:
        batches.pop(next(iter(batches)))
    checked = [r for r in results if r[2]]
    summary = {"rows": len(results), "checked": len(checked), "fake": sum(r[2] == "FAKE" for r in checked),
               "skipped": len(results) - len(checked)}
    return render_template("batch.html", results=results[:100], summary=summary, batch_id=bid)


@app.route("/batch/<bid>.csv")
def batch_download(bid):
    if bid not in batches:
        abort(404)
    return Response(batches[bid], mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=fnds_results.csv"})


# ---------------- accounts ----------------
@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        row = repo.check_login(request.form.get("email"), request.form.get("password"))
        if row:
            login_user(User(row))
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("home"))
        flash("Wrong email or password.")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name, email, pw = (request.form.get(k, "") for k in ("name", "email", "password"))
        if not name.strip() or "@" not in email or len(pw) < 8:
            flash("Enter your name, a valid email and a password of at least 8 characters.")
        else:
            try:
                login_user(User(repo.get_user(repo.create_user(name, email, pw))))
                return redirect(url_for("home"))
            except ValueError as e:
                flash(str(e))
    return render_template("register.html")


@app.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return redirect(url_for("home"))


# ---------------- admin ----------------
@app.route("/admin")
@admin_required
def dashboard():
    m = engines.model
    return render_template("dashboard.html", totals=repo.totals(), daily=[dict(r) for r in repo.daily_counts()],
                           models=repo.all_models(), comparison=repo.comparisons(m["model_id"]),
                           runs=repo.retrain_runs(), fb_models=[dict(r) for r in repo.feedback_by_model()],
                           summary=repo.feedback_summary(), default_pw=repo.admin_uses_default_password(),
                           message=request.args.get("msg"))


@app.route("/admin/retrain", methods=["POST"])
@admin_required
def admin_retrain():
    result = retrain_from_feedback(repo, MODEL_PATH)
    if result["accepted"]:
        engines.reload()
    return redirect(url_for("dashboard", msg=result["message"]))


@app.route("/admin/password", methods=["POST"])
@admin_required
def admin_password():
    pw = request.form.get("password", "")
    if len(pw) < 8:
        return redirect(url_for("dashboard", msg="The new password must have at least 8 characters."))
    repo.set_password(current_user.id, pw)
    return redirect(url_for("dashboard", msg="Password changed."))


# ---------------- JSON API ----------------
def api_error(msg, code=400):
    return jsonify({"error": msg}), code


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "model": engines.model["version"], "algorithm": engines.model["algorithm"],
                    "bert": engines.bert is not None, "grok": llm.enabled(), "factcheck": factcheck.enabled()})


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        return "", 204
    if not request.is_json:
        return api_error("Send JSON, e.g. {\"text\": \"...\"} or {\"url\": \"...\"}.", 415)
    data = request.get_json(silent=True) or {}
    title = source_url = None
    truncated = False
    if data.get("url"):
        try:
            source_url, title, text = fetch_article(str(data["url"]))
            text, truncated = fit_text(text)
        except FetchError as e:
            return api_error(str(e))
    else:
        text, truncated = fit_text(str(data.get("text", "")))
    ok, msg = validate_input(text)
    if not ok:
        return api_error(msg)
    out = run_check(text, data.get("engine", "classic"), title, source_url)
    r = out["result"]
    return jsonify({"prediction_id": out["prediction_id"], "label": r.label, "confidence": r.confidence,
                    "top_terms": r.top_terms, "engine": out["engine"], "model_version": out["model_version"],
                    "signals": out["signals"], "source": out["source"], "title": title,
                    "truncated": truncated, "history_url": url_for("history", _external=True)})


@app.route("/api/explain/<int:pid>", methods=["POST", "OPTIONS"])
def api_explain(pid):
    if request.method == "OPTIONS":
        return "", 204
    if not request.is_json:
        return api_error("Send the request as JSON.", 415)
    p = repo.get_prediction(pid)
    if not p:
        return api_error("Unknown prediction.", 404)
    if pid in explanations:
        return jsonify(explanations[pid])
    try:
        terms = [t for t in (p["top_terms"] or "").split(", ") if t]
        text, model = llm.explain(p["content"], p["label"], p["confidence"], terms,
                                  p["title"] if p["source_url"] else None)
    except llm.LLMUnavailable as e:
        return api_error(str(e), 503)
    explanations[pid] = {"source": "grok", "model": model, "text": text}
    return jsonify(explanations[pid])


@app.route("/api/factcheck/<int:pid>")
def api_factcheck(pid):
    p = repo.get_prediction(pid)
    if not p:
        return api_error("Unknown prediction.", 404)
    if not factcheck.enabled():
        return api_error("Add FACTCHECK_API_KEY to the .env file to enable fact-check lookup.", 503)
    return jsonify({"query": factcheck.build_query(p["content"], p["title"] if p["source_url"] else None),
                    "results": factcheck.search(p["content"], p["title"] if p["source_url"] else None)})


@app.errorhandler(413)
def too_large(_):
    return render_template("batch.html", error="The file is too large (limit 5 MB)."), 413


if __name__ == "__main__":
    app.run(host=config.get("HOST", "127.0.0.1"), port=int(config.get("PORT", "5000")),
            debug=config.get("FLASK_DEBUG", "0") == "1")
