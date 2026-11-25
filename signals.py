"""signals.py - rule-based writing-style signals shown with every result (works offline, no API key)."""
import re

CLICKBAIT = ["you won't believe", "won't believe", "doctors hate", "they don't want you to know",
             "share before", "before it gets deleted", "before it's deleted", "wake up", "mainstream media",
             "the truth about", "miracle", "shocking", "exposed", "secret", "cover up", "cover-up",
             "this one trick", "100%", "guaranteed", "cure", "breaking"]
URGENCY = ["share now", "share with", "forward to", "forward this", "click this link", "act now",
           "before midnight", "rush to", "spread the word", "share this"]
VAGUE_SOURCES = ["anonymous source", "insiders say", "insider", "sources say", "experts say",
                 "it is said", "people are saying", "leaked"]
ATTRIBUTION = re.compile(r"\b(said|says|told|according to|reported|announced|stated|statement|"
                         r"spokesperson|spokesman|spokeswoman|minister|official|officials)\b", re.I)


def _found(text, phrases):
    low = text.lower()
    return [p for p in phrases if p in low]


def analyse(text):
    """Return {"warnings": [...], "good": [...]} with short human-readable findings."""
    warnings, good = [], []
    words = re.findall(r"[A-Za-z]+", text)
    caps = [w for w in words if len(w) >= 3 and w.isupper()]
    if len(caps) >= 2 or (words and len(caps) / len(words) > 0.05):
        warnings.append(f"Shouting in capitals ({', '.join(caps[:4])})")
    if text.count("!") >= 2:
        warnings.append(f"Many exclamation marks ({text.count('!')})")
    if hits := _found(text, CLICKBAIT):
        warnings.append("Sensational or clickbait wording: " + ", ".join(f'"{h}"' for h in hits[:4]))
    if hits := _found(text, URGENCY):
        warnings.append("Pushes you to share or act fast: " + ", ".join(f'"{h}"' for h in hits[:3]))
    if hits := _found(text, VAGUE_SOURCES):
        warnings.append("Vague or unnamed sources: " + ", ".join(f'"{h}"' for h in hits[:3]))
    attributions = ATTRIBUTION.findall(text)
    if attributions:
        good.append(f"Attributes statements to people or organisations ({len(attributions)}x)")
    elif len(words) > 40:
        warnings.append("No statement is attributed to a named person or organisation")
    if re.search(r"\b\d{1,4}([.,]\d+)?\s*(%|percent|per cent|million|billion|crore|lakh)\b", text, re.I):
        good.append("Gives specific figures")
    if re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|"
                 r"april|may|june|july|august|september|october|november|december)\b", text, re.I):
        good.append("Mentions specific dates or days")
    return {"warnings": warnings, "good": good}
