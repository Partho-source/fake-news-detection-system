# Fake News Detection System

**Live demo:** https://fake-news-detector-fmug.onrender.com (free plan: the first visit after it has been idle takes about a minute to wake up)

A Flask web app that classifies a news article as **FAKE** or **REAL**. It explains which words drove the decision, checks writing-style warning signs, can ask **Grok** for a plain-English explanation, and looks up published fact-checks. It also learns from user feedback.

## Features

| Area | What it does |
|---|---|
| Real dataset | 6,299 labelled articles (`fake_or_real_news.csv`), downloaded automatically on the first run |
| Model comparison | Logistic Regression, Naive Bayes, Linear SVM and Passive-Aggressive compared with 5-fold cross-validation; the best F1 is kept (Linear SVM, about 93%) |
| Word highlighting | Every word is coloured by how much it pushed the result towards FAKE (red) or REAL (green) |
| Writing signals | Rule-based warnings (capitals, clickbait, "share now", vague sources) and good signs (named sources, figures, dates). Works offline |
| Grok explanation | Optional plain-English explanation from xAI's Grok API (`XAI_API_KEY`) |
| Fact-checks | Optional search of published fact-checks via the Google Fact Check Tools API (`FACTCHECK_API_KEY`) |
| Check a link | Paste a URL; the article text is extracted and the site gets a credibility badge |
| Batch | Upload a CSV of up to 500 articles and download the results |
| Accounts | Sign up / log in; each user sees their own history. Admin dashboard with charts |
| Learn from feedback | The admin retrains with feedback-corrected articles; the new version is kept only if accuracy does not drop |
| REST API | `POST /api/predict` with `{"text": ...}` or `{"url": ...}` returns JSON |
| Chrome extension | Right-click selected text or a page and choose "Check with Fake News Detector" |
| DistilBERT (optional) | Fine-tune a transformer on Colab and switch between it and the classic model |
| Quality | 74 automated tests (unit, system, Playwright browser tests incl. the extension), about 93% coverage, GitHub Actions CI, Docker image, Render blueprint |

## Quick start (Windows PowerShell)

```powershell
cd path\to\fnds
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env              # optional: add API keys (see below)
python app.py
```

Open http://127.0.0.1:5000. The first start downloads the dataset (31 MB) and trains the models, which takes about a minute.

- **Admin login:** `admin@fnds.local` / `admin123`. Change it from the Dashboard. To start with a different password, set `FNDS_ADMIN_PASSWORD` in `.env` before the first run.
- If you change training data or retrain from the command line, restart `app.py`. Retraining from the Dashboard swaps the model automatically.

## API keys (optional)

Put the keys in `.env` (copied from `.env.example`). Never commit `.env`; it is in `.gitignore`.

| Key | Where to get it | Enables |
|---|---|---|
| `XAI_API_KEY` | console.x.ai, then API Keys (pay-as-you-go credits) | "Explain with Grok" button |
| `GROK_MODEL` | optional, default `grok-4` | If that model isn't available for your key, the app picks an available Grok model automatically |
| `FACTCHECK_API_KEY` | console.cloud.google.com: enable "Fact Check Tools API", then Credentials, then API key (free) | "Published fact-checks" panel |

Without keys, both panels say "Not set up" and everything else works.

## Training

```bash
python get_dataset.py                        # download data/fake_or_real_news.csv
python train.py                              # real dataset if present, compares 4 algorithms
python train.py --dataset sample             # the built-in 60-article sample
python train.py --algorithm logreg           # logreg | nb | svm | pa | auto
python train.py --csv news.csv               # your own CSV: text + label (1/0 or FAKE/REAL)
python train.py --fake-csv Fake.csv --real-csv True.csv   # e.g. the ISOT/Kaggle dataset
```

Two details that matter for the viva:

- **Label leaks are removed.** In ISOT every real article starts with "WASHINGTON (Reuters) -", so a model can score about 99% just by spotting the tag. `dataset.strip_leaks()` removes such fingerprints.
- **Short texts.** The linear models are trained without an intercept and also see 40-word passages of each training article. Before this change, a model trained on long articles called every short pasted text FAKE.

## Tests

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
pytest                                   # 74 tests
pytest --cov --cov-report=html           # coverage report in htmlcov/index.html
pytest -m "not e2e"                      # skip the browser tests
```

Tests use a temporary database and model and ignore your `.env`, so they never touch your data or call paid APIs. The old scripts for the practical file still work: `coverage_run.py`, `complexity.py`, `trace_states.py`, `trace_dfd.py`, `error_analysis.py`.

## Chrome extension

1. Start the app (`python app.py`).
2. In Chrome open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose the `extension` folder.
3. Select text on any news page, right-click and choose **Check with Fake News Detector**. A result box appears on the page. Alternatively, click the toolbar icon to paste text or check the current page.
4. If the app runs somewhere else (e.g. Render), set its address under **Server settings** in the popup.

## REST API

```http
POST /api/predict           {"text": "..."} or {"url": "https://..."}, optional "engine": "bert"
GET  /api/health
POST /api/explain/<id>      Grok explanation of a saved prediction (JSON body {})
GET  /api/factcheck/<id>    published fact-checks
```

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/predict -Method Post -ContentType "application/json" -Body '{"text":"Paste a news article of at least ten words here to test the API"}'
```

## Deploy (Docker / Render)

```bash
docker build -t fake-news-detector .      # downloads the data and trains during the build
docker run -p 8000:8000 -e XAI_API_KEY=... fake-news-detector
```

To deploy on Render: push the project to GitHub, then on render.com choose **New, then Blueprint**, and pick the repository; `render.yaml` sets everything up. Enter your keys and admin password when asked. API keys can be added later under the service's **Environment** settings. The live demo is deployed from the public repository URL, so after pushing changes use **Manual Deploy, then Deploy latest commit**. On the free plan the app sleeps when idle and the database resets on each deploy.

## DistilBERT (optional, needs a GPU)

On Google Colab (Runtime, then Change runtime type, then T4 GPU):

```bash
pip install -r requirements.txt -r requirements-bert.txt
python get_dataset.py
python train_bert.py --epochs 2          # about 10-15 minutes on a T4
```

Copy the resulting `models/distilbert` folder and `fnds.db` (or just re-run the registration) into the project on your laptop. Install `requirements-bert.txt` there too and restart. A "DistilBERT" option then appears next to **Check News**, and its scores show up in the model list. On a CPU-only laptop, `python train_bert.py --max-rows 1500 --epochs 1` works but is slow.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Flask routes, login, CSRF protection, dashboard, REST API |
| `detector.py` | TF-IDF + 4 classifiers, prediction, word highlighting, save/load |
| `model_select.py` | Cross-validated comparison of the classifiers |
| `retrain.py` | Retraining from feedback with an accuracy check |
| `dataset.py`, `get_dataset.py` | Sample data, CSV loading, leak removal, dataset download |
| `database.py`, `schema.sql` | SQLite (users, datasets, models, articles, predictions, feedback, comparisons, retrain runs) |
| `signals.py`, `sources.py` | Writing-style signals, website credibility lists |
| `urlfetch.py` | Safe page download (blocks private addresses) and article extraction |
| `llm.py`, `factcheck.py` | Grok and Google Fact Check clients |
| `bert_model.py`, `train_bert.py` | Optional DistilBERT engine |
| `preprocessor.py`, `validator.py`, `train.py`, `config.py` | Text cleaning, input rules, training CLI, `.env` loading |
| `templates/`, `static/` | Pages, stylesheet, bundled Chart.js |
| `extension/` | Chrome extension (Manifest V3) |
| `tests/` | pytest suite (`e2e/` = Playwright) |
| `Dockerfile`, `render.yaml`, `.github/workflows/ci.yml` | Deployment and CI |

Results are statistical predictions based on writing style, not proof that a story is true or false. The training data is mostly 2015–2017 US political news, so confidence is lower on other topics and regions.
