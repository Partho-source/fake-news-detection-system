-- schema.sql - database of the Fake News Detection System (E-R design of Experiment 4)
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('USER', 'ADMIN')),
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    source        TEXT,
    num_articles  INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS dataset_articles (
    sample_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL REFERENCES datasets(dataset_id),
    text          TEXT NOT NULL,
    label         INTEGER NOT NULL CHECK (label IN (0, 1))    -- 1 = FAKE, 0 = REAL
);
CREATE TABLE IF NOT EXISTS models (
    model_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL REFERENCES datasets(dataset_id),
    trained_by    INTEGER NOT NULL REFERENCES users(user_id),
    algorithm     TEXT NOT NULL,
    version       TEXT NOT NULL,
    accuracy      REAL, precision_val REAL, recall_val REAL, f1_score REAL,
    file_path     TEXT NOT NULL,
    trained_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS articles (
    article_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER REFERENCES users(user_id),
    title         TEXT,
    content       TEXT NOT NULL,
    source_url    TEXT,
    submitted_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id    INTEGER NOT NULL REFERENCES articles(article_id),
    model_id      INTEGER NOT NULL REFERENCES models(model_id),
    label         TEXT NOT NULL CHECK (label IN ('FAKE', 'REAL')),
    confidence    REAL NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    top_terms     TEXT,
    predicted_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id),
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    is_correct    INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    comment       TEXT,
    given_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS model_comparisons (
    comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      INTEGER NOT NULL REFERENCES models(model_id),
    algorithm     TEXT NOT NULL,
    accuracy      REAL, accuracy_std REAL, precision_val REAL, recall_val REAL, f1_score REAL,
    fit_seconds   REAL
);
CREATE TABLE IF NOT EXISTS retrain_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model_id INTEGER NOT NULL REFERENCES models(model_id),
    new_model_id  INTEGER REFERENCES models(model_id),         -- NULL when the candidate was rejected
    feedback_used INTEGER NOT NULL,
    old_accuracy  REAL, new_accuracy REAL,
    accepted      INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    run_at        TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(predicted_at);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON feedback(prediction_id);
