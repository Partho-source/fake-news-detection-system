"""sources.py - simple credibility badge for the website an article came from.

The lists are short, hand-picked examples for the prototype, not an authoritative rating.
"""
from urllib.parse import urlparse

RELIABLE = {"reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com",
            "washingtonpost.com", "npr.org", "aljazeera.com", "thehindu.com", "indianexpress.com",
            "hindustantimes.com", "ndtv.com", "livemint.com", "business-standard.com", "pib.gov.in",
            "economictimes.indiatimes.com", "timesofindia.indiatimes.com", "scroll.in", "theprint.in",
            "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", "nature.com", "who.int"}
FACT_CHECKERS = {"altnews.in", "boomlive.in", "factly.in", "snopes.com", "politifact.com",
                 "factcheck.org", "fullfact.org", "newschecker.in", "thequint.com"}
SATIRE = {"theonion.com", "babylonbee.com", "fakingnews.com", "thebeaverton.com", "waterfordwhispersnews.com"}
UNRELIABLE = {"infowars.com", "naturalnews.com", "beforeitsnews.com", "worldnewsdailyreport.com",
              "yournewswire.com", "newspunch.com", "thegatewaypundit.com", "zerohedge.com",
              "postcard.news", "empirenews.net", "abcnews.com.co", "nationalreport.net"}


def domain_of(url):
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _match(host, names):
    return any(host == n or host.endswith("." + n) for n in names)


def credibility(url):
    host = domain_of(url)
    if _match(host, FACT_CHECKERS):
        return {"domain": host, "level": "good", "label": "Fact-checking site"}
    if _match(host, RELIABLE):
        return {"domain": host, "level": "good", "label": "Established news outlet"}
    if _match(host, SATIRE):
        return {"domain": host, "level": "warn", "label": "Satire site (not meant as real news)"}
    if _match(host, UNRELIABLE):
        return {"domain": host, "level": "bad", "label": "Known for false or misleading stories"}
    return {"domain": host, "level": "unknown", "label": "Source not rated"}
