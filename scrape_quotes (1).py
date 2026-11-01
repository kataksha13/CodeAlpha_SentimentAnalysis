"""
CodeAlpha Data Analytics Internship - Task 4: Sentiment Analysis (step 1)
Scrapes quotes, authors and tags from https://quotes.toscrape.com (a practice site for scraping).

Usage:
    python scrape_quotes.py        # saves data/quotes.csv (about 100 quotes)
"""

import logging
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

START_URL = "https://quotes.toscrape.com/page/1/"
HEADERS = {"User-Agent": "Mozilla/5.0 (CodeAlpha internship learning project)"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def parse_page(soup):
    rows = []
    for box in soup.select("div.quote"):
        rows.append(
            {
                "quote": box.select_one("span.text").get_text(strip=True).strip("\u201c\u201d\""),
                "author": box.select_one("small.author").get_text(strip=True),
                "tags": ",".join(t.get_text(strip=True) for t in box.select("a.tag")),
            }
        )
    return rows


def scrape():
    session = requests.Session()
    url, data = START_URL, []
    while url:
        log.info("Scraping %s", url)
        try:
            response = session.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.RequestException as err:
            log.error("Failed on %s: %s", url, err)
            break
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        data.extend(parse_page(soup))
        nxt = soup.select_one("li.next a")
        url = urljoin(url, nxt["href"]) if nxt else None
        time.sleep(0.5)  # be polite
    return data


def main():
    rows = scrape()
    if not rows:
        log.error("No quotes collected. Check your internet connection.")
        return
    df = pd.DataFrame(rows).drop_duplicates(subset="quote").reset_index(drop=True)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/quotes.csv", index=False, encoding="utf-8-sig")
    log.info("Saved %d quotes to data/quotes.csv", len(df))
    print(df.head())


if __name__ == "__main__":
    main()
