"""llm.py - plain-English explanation of a result using xAI's Grok API (optional).

Needs XAI_API_KEY (or GROK_API_KEY) in .env. GROK_MODEL picks the model; if that model is not
available to the key, the first Grok text model listed by the API is used instead.
"""
import requests
import config

BASE = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4"
_resolved = {}

SYSTEM = ("You are a careful media-literacy assistant inside a student fake-news detection app. "
          "You help readers judge an article; you never claim certainty about whether it is true.")


class LLMUnavailable(Exception):
    pass


def api_key():
    return config.get("XAI_API_KEY") or config.get("GROK_API_KEY")


def enabled():
    return bool(api_key())


def _headers():
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def _pick_model(timeout):
    r = requests.get(f"{BASE}/models", headers=_headers(), timeout=timeout)
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    text_models = [i for i in ids if i.startswith("grok") and not any(x in i for x in ("image", "vision", "imagine"))]
    if not text_models:
        raise LLMUnavailable("No Grok text model is available for this API key.")
    return text_models[0]


def _chat(model, messages, timeout):
    return requests.post(f"{BASE}/chat/completions", headers=_headers(), timeout=timeout,
                         json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 500})


def build_messages(text, label, confidence, top_terms, title=None):
    article = text if len(text) <= 6000 else text[:6000] + " [...]"
    prompt = (f"Our machine-learning model labelled this article {label} with {confidence}% confidence. "
              f"The words that influenced it most were: {', '.join(top_terms) or 'none'}.\n\n"
              f"Headline: {title or '(none)'}\nArticle:\n\"\"\"\n{article}\n\"\"\"\n\n"
              "In 4 to 6 short bullet points (start each with '- '), explain to an ordinary reader which features "
              "of the writing make it look credible or suspicious (sources, tone, evidence, specific claims), and "
              "name the 1 or 2 most important claims they should verify, and how. End with one sentence noting "
              "that this is guidance, not a verdict. Plain text only, no headings.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def explain(text, label, confidence, top_terms, title=None, timeout=45):
    if not enabled():
        raise LLMUnavailable("Add XAI_API_KEY to the .env file to enable Grok explanations.")
    messages = build_messages(text, label, confidence, top_terms, title)
    model = _resolved.get("model") or config.get("GROK_MODEL", DEFAULT_MODEL)
    try:
        r = _chat(model, messages, timeout)
        if r.status_code in (400, 404) and "model" in r.text.lower():
            model = _pick_model(timeout)
            r = _chat(model, messages, timeout)
        if r.status_code in (401, 403):
            raise LLMUnavailable("The Grok API key was rejected. Check XAI_API_KEY in .env.")
        if r.status_code == 429:
            raise LLMUnavailable("Grok rate limit or credit limit reached. Try again later.")
        r.raise_for_status()
        _resolved["model"] = model
        return r.json()["choices"][0]["message"]["content"].strip(), model
    except requests.RequestException as e:
        raise LLMUnavailable(f"Could not reach the Grok API ({e.__class__.__name__}).")
    except (KeyError, IndexError, ValueError):
        raise LLMUnavailable("Unexpected answer from the Grok API.")
