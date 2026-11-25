"""database.py - SQLite storage of users, datasets, models, predictions and feedback (E-R design, Exp. 4)."""
import os
import sqlite3
from werkzeug.security import check_password_hash, generate_password_hash

SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
DEFAULT_ADMIN_PASSWORD = "admin123"


class HistoryRepository:
    """Saves and reads the records of the E-R model."""

    def __init__(self, path="fnds.db", admin_password=None):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with open(SCHEMA) as f:
            self.conn.executescript(f.read())
        self.guest_id = self._user("Guest", "guest@fnds.local", "USER", None)
        admin_pw = admin_password or os.environ.get("FNDS_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
        self.admin_id = self._user("Administrator", "admin@fnds.local", "ADMIN", admin_pw)
        if self.get_user(self.admin_id)["password_hash"] == "-":      # database from the first version
            self.set_password(self.admin_id, admin_pw)

    # ---------- users ----------
    def _user(self, name, email, role, password):
        row = self.conn.execute("SELECT user_id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            return row["user_id"]
        return self.create_user(name, email, password, role)

    def create_user(self, name, email, password, role="USER"):
        """password None -> the account cannot log in (used for the Guest user)."""
        email = email.strip().lower()
        if self.conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise ValueError("An account with this email already exists.")
        pw = generate_password_hash(password) if password else "-"
        cur = self.conn.execute("INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                                (name.strip(), email, pw, role))
        self.conn.commit()
        return cur.lastrowid

    def get_user(self, user_id):
        return self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()

    def check_login(self, email, password):
        row = self.conn.execute("SELECT * FROM users WHERE email = ?", ((email or "").strip().lower(),)).fetchone()
        if row and row["password_hash"] != "-" and check_password_hash(row["password_hash"], password or ""):
            return row
        return None

    def admin_uses_default_password(self):
        row = self.get_user(self.admin_id)
        return check_password_hash(row["password_hash"], DEFAULT_ADMIN_PASSWORD)

    def set_password(self, user_id, password):
        self.conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                          (generate_password_hash(password), user_id))
        self.conn.commit()

    # ---------- datasets and models ----------
    def add_dataset(self, name, source, texts, labels):
        cur = self.conn.execute("INSERT INTO datasets (name, source, num_articles) VALUES (?, ?, ?)",
                                (name, source, len(texts)))
        self.conn.executemany("INSERT INTO dataset_articles (dataset_id, text, label) VALUES (?, ?, ?)",
                              [(cur.lastrowid, t, l) for t, l in zip(texts, labels)])
        self.conn.commit()
        return cur.lastrowid

    def dataset_rows(self, dataset_id):
        rows = self.conn.execute("SELECT text, label FROM dataset_articles WHERE dataset_id = ?"
                                 " ORDER BY sample_id", (dataset_id,)).fetchall()
        return [r["text"] for r in rows], [r["label"] for r in rows]

    def add_model(self, dataset_id, metrics, file_path, algorithm="Logistic Regression"):
        count = self.conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        cur = self.conn.execute(
            "INSERT INTO models (dataset_id, trained_by, algorithm, version, accuracy,"
            " precision_val, recall_val, f1_score, file_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dataset_id, self.admin_id, algorithm, f"v{count + 1}", metrics["accuracy"],
             metrics["precision"], metrics["recall"], metrics["f1"], file_path))
        self.conn.commit()
        return cur.lastrowid

    def add_comparison(self, model_id, rows):
        self.conn.executemany(
            "INSERT INTO model_comparisons (model_id, algorithm, accuracy, accuracy_std, precision_val,"
            " recall_val, f1_score, fit_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(model_id, r["name"], r["accuracy"], r["accuracy_std"], r["precision"], r["recall"], r["f1"],
              r["fit_seconds"]) for r in rows])
        self.conn.commit()

    def comparisons(self, model_id):
        return self.conn.execute("SELECT * FROM model_comparisons WHERE model_id = ? ORDER BY f1_score DESC",
                                 (model_id,)).fetchall()

    def latest_model(self, transformer=False):
        op = "=" if transformer else "<>"
        return self.conn.execute(
            "SELECT m.*, d.name AS dataset_name, d.num_articles FROM models m"
            " JOIN datasets d ON d.dataset_id = m.dataset_id"
            f" WHERE m.algorithm {op} 'DistilBERT' ORDER BY m.model_id DESC LIMIT 1").fetchone()

    def all_models(self):
        return self.conn.execute(
            "SELECT m.*, d.name AS dataset_name, d.num_articles FROM models m"
            " JOIN datasets d ON d.dataset_id = m.dataset_id ORDER BY m.model_id DESC").fetchall()

    # ---------- predictions and feedback ----------
    def save_prediction(self, text, result, model_id, user_id=None, title=None, source_url=None):
        cur = self.conn.execute(
            "INSERT INTO articles (user_id, title, content, source_url) VALUES (?, ?, ?, ?)",
            (user_id or self.guest_id, (title or text)[:60], text, source_url))
        cur = self.conn.execute(
            "INSERT INTO predictions (article_id, model_id, label, confidence, top_terms) VALUES (?, ?, ?, ?, ?)",
            (cur.lastrowid, model_id, result.label, result.confidence, ", ".join(result.top_terms)))
        self.conn.commit()
        return cur.lastrowid

    def get_prediction(self, prediction_id):
        return self.conn.execute(
            "SELECT p.*, a.content, a.title, a.source_url, a.user_id FROM predictions p"
            " JOIN articles a ON a.article_id = p.article_id WHERE p.prediction_id = ?",
            (prediction_id,)).fetchone()

    def add_feedback(self, prediction_id, is_correct, comment="", user_id=None):
        if not self.get_prediction(prediction_id):
            raise ValueError("Unknown prediction.")
        cur = self.conn.execute(
            "INSERT INTO feedback (prediction_id, user_id, is_correct, comment) VALUES (?, ?, ?, ?)",
            (prediction_id, user_id or self.guest_id, int(bool(is_correct)), comment))
        self.conn.commit()
        return cur.lastrowid

    def history(self, limit=20, user_id=None):
        where, args = ("WHERE a.user_id = ?", (user_id, limit)) if user_id else ("", (limit,))
        return self.conn.execute(
            "SELECT p.prediction_id, a.title, a.source_url, p.label, p.confidence, p.top_terms,"
            " p.predicted_at, m.version, m.algorithm,"
            " (SELECT f.is_correct FROM feedback f WHERE f.prediction_id = p.prediction_id"
            "  ORDER BY f.feedback_id DESC LIMIT 1) AS is_correct"
            " FROM predictions p JOIN articles a ON a.article_id = p.article_id"
            " JOIN models m ON m.model_id = p.model_id "
            f"{where} ORDER BY p.prediction_id DESC LIMIT ?", args).fetchall()

    def feedback_summary(self):
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(is_correct), 0) AS correct FROM feedback").fetchone()
        return {"total": row["total"], "correct": row["correct"], "incorrect": row["total"] - row["correct"]}

    def feedback_samples(self):
        """Articles with feedback, labelled with the corrected answer (latest feedback wins)."""
        rows = self.conn.execute(
            "SELECT a.content, p.label, f.is_correct FROM feedback f"
            " JOIN predictions p ON p.prediction_id = f.prediction_id"
            " JOIN articles a ON a.article_id = p.article_id"
            " WHERE f.feedback_id IN (SELECT MAX(feedback_id) FROM feedback GROUP BY prediction_id)").fetchall()
        texts, labels = [], []
        for r in rows:
            predicted_fake = r["label"] == "FAKE"
            is_fake = predicted_fake if r["is_correct"] else not predicted_fake
            texts.append(r["content"])
            labels.append(int(is_fake))
        return texts, labels

    def add_retrain_run(self, base_model_id, new_model_id, used, old_acc, new_acc, accepted):
        self.conn.execute(
            "INSERT INTO retrain_runs (base_model_id, new_model_id, feedback_used, old_accuracy, new_accuracy,"
            " accepted) VALUES (?, ?, ?, ?, ?, ?)", (base_model_id, new_model_id, used, old_acc, new_acc, int(accepted)))
        self.conn.commit()

    def retrain_runs(self, limit=10):
        return self.conn.execute("SELECT * FROM retrain_runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()

    # ---------- dashboard statistics ----------
    def daily_counts(self, days=30):
        return self.conn.execute(
            "SELECT DATE(predicted_at) AS day, SUM(label = 'FAKE') AS fake, SUM(label = 'REAL') AS real"
            " FROM predictions WHERE predicted_at >= DATE('now', ?) GROUP BY day ORDER BY day",
            (f"-{days} days",)).fetchall()

    def totals(self):
        q = lambda sql: self.conn.execute(sql).fetchone()[0]
        return {"predictions": q("SELECT COUNT(*) FROM predictions"),
                "fake": q("SELECT COUNT(*) FROM predictions WHERE label = 'FAKE'"),
                "users": q("SELECT COUNT(*) FROM users WHERE password_hash <> '-'"),
                "models": q("SELECT COUNT(*) FROM models")}

    def feedback_by_model(self):
        return self.conn.execute(
            "SELECT m.version, m.algorithm, COUNT(f.feedback_id) AS total, COALESCE(SUM(f.is_correct), 0) AS correct"
            " FROM models m JOIN predictions p ON p.model_id = m.model_id"
            " JOIN feedback f ON f.prediction_id = p.prediction_id"
            " GROUP BY m.model_id ORDER BY m.model_id").fetchall()
