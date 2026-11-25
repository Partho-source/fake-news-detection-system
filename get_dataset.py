"""get_dataset.py - downloads the public fake_or_real_news.csv dataset (about 6,300 labelled articles).

Source: GitHub mirror of the "fake_or_real_news" dataset (George McIntire), used in many fake-news
tutorials. Columns: id, title, text, label (FAKE / REAL).
"""
import os
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "data", "fake_or_real_news.csv")


def main():
    if os.path.exists(DEST) and "--force" not in sys.argv:
        print(f"Already downloaded: {DEST}")
        return DEST
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST + ".part")
    os.replace(DEST + ".part", DEST)
    print(f"Saved {os.path.getsize(DEST) / 1e6:.1f} MB to {DEST}")
    return DEST


if __name__ == "__main__":
    main()
