"""factcheck.py - looks up published fact-checks with the Google Fact Check Tools API (optional).

Needs FACTCHECK_API_KEY in .env; without it the feature is simply switched off.
"""
import re
import requests
import config

ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def enabled():
    return bool(config.get("FACTCHECK_API_KEY"))


def build_query(text, title=None, max_words=12):
    """Use the headline if known, otherwise the first sentence, cut to a short query."""
    src = title or re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    words = re.findall(r"[A-Za-z0-9']+", src)
    return " ".join(words[:max_words])


def search(text, title=None, limit=5, timeout=6):
    if not enabled():
        return []
    params = {"query": build_query(text, title), "key": config.get("FACTCHECK_API_KEY"),
              "languageCode": "en", "pageSize": limit}
    try:
        r = requests.get(ENDPOINT, params=params, timeout=timeout)
        r.raise_for_status()
        claims = r.json().get("claims", [])
    except (requests.RequestException, ValueError):
        return []
    results = []
    for c in claims[:limit]:
        review = (c.get("claimReview") or [{}])[0]
        results.append({"claim": c.get("text", ""), "claimant": c.get("claimant", ""),
                        "publisher": (review.get("publisher") or {}).get("name", ""),
                        "rating": review.get("textualRating", ""), "url": review.get("url", ""),
                        "date": (review.get("reviewDate") or "")[:10]})
    return results
