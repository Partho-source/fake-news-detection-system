"""urlfetch.py - downloads a news page and extracts the article text (requirement: check by URL).

Only public http(s) addresses are allowed: hosts that resolve to private, loopback or link-local
addresses are refused, so the server cannot be used to reach machines on the local network (SSRF).
"""
import ipaddress
import socket
from urllib.parse import urljoin, urlparse
import requests
import trafilatura

MAX_BYTES = 3_000_000
HEADERS = {"User-Agent": "Mozilla/5.0 (FakeNewsDetectionSystem student project)"}


class FetchError(Exception):
    pass


def _check_public(url):
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Enter a full web address starting with http:// or https://")
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except socket.gaierror:
        raise FetchError("That website could not be found.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise FetchError("Only public websites can be checked.")


def download(url, timeout=10):
    for _ in range(5):                                    # follow redirects ourselves, checking each hop
        _check_public(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True, allow_redirects=False)
        except requests.RequestException:
            raise FetchError("The page could not be downloaded.")
        if r.is_redirect:
            url = urljoin(url, r.headers.get("Location", ""))
            continue
        if r.status_code != 200:
            raise FetchError(f"The website answered with status {r.status_code}.")
        body = r.raw.read(MAX_BYTES + 1, decode_content=True)
        if len(body) > MAX_BYTES:
            raise FetchError("The page is too large.")
        return url, body.decode(r.encoding or "utf-8", errors="replace")
    raise FetchError("Too many redirects.")


def extract(html):
    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    meta = trafilatura.extract_metadata(html)
    title = meta.title if meta and meta.title else None
    return title, text.strip()


def fetch_article(url):
    """Return (final_url, title, text). Raises FetchError with a user-friendly message."""
    final_url, html = download(url.strip())
    title, text = extract(html)
    if len(text.split()) < 10:
        raise FetchError("No article text was found on that page. Paste the text instead.")
    return final_url, title, text
