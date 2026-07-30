from __future__ import annotations

import copy
import time
import urllib.parse
from datetime import datetime, timezone
import requests
import cloudscraper
from bs4 import BeautifulSoup

from reviews.base import BaseCollector
from reviews.models import RawReview
from reviews.utils import clean_text, stable_id


DEFAULT_QUERIES = [
    "Blinkit missing items",
    "Blinkit recommendations",
    "Blinkit search",
    "finding products on Blinkit",
    "Blinkit alternatives",
    "Blinkit checkout",
    "Blinkit grocery",
    "Blinkit categories",
    "Blinkit out of stock",
    "Blinkit catalog navigation",
]


class RedditCollector(BaseCollector):
    @property
    def source_name(self) -> str:
        return "reddit"

    def collect(self, limit: int, **kwargs) -> list[RawReview]:
        subreddits = kwargs.get("subreddits")
        queries = kwargs.get("queries")
        return collect_reddit_reviews(limit=limit, subreddits=subreddits, queries=queries)


def _extract_reddit_post(res: BeautifulSoup) -> str:
    """Extracts only the post title and self-text/snippet, discarding subreddit, author, flairs, flairs-class, etc."""
    element = copy.copy(res)

    unwanted_selectors = [
        ".search-author", ".search-subreddit", ".search-flair",
        ".search-comments", ".search-score", "time", ".search-time",
        ".search-info-header", ".search-result-meta", ".search-meta",
        ".flair", ".flair-rich", ".search-subreddit-link", ".author"
    ]
    for selector in unwanted_selectors:
        for tag in element.select(selector):
            tag.decompose()

    title_el = element.select_one("a.search-title")
    snippet_el = element.select_one(".search-result-text")

    title = ""
    snippet = ""

    if title_el:
        title = clean_text(title_el.get_text(" ", strip=True))
        title_el.decompose()

    if snippet_el:
        snippet = clean_text(snippet_el.get_text(" ", strip=True))
    else:
        snippet = clean_text(element.get_text(" ", strip=True))

    if title and snippet:
        if snippet.startswith(title):
            snippet = snippet[len(title):].strip()

    review_parts = []
    if title:
        review_parts.append(title)
    if snippet and snippet not in title:
        review_parts.append(snippet)

    return clean_text(" ".join(review_parts))


def collect_reddit_reviews(
    limit: int,
    subreddits: list[str] | None = None,
    queries: list[str] | None = None,
) -> list[RawReview]:
    """Collects Blinkit discovery and catalog-related reviews from public Reddit search results."""
    search_queries = queries or DEFAULT_QUERIES
    raw_reviews: list[RawReview] = []
    seen_texts: set[str] = set()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    with cloudscraper.create_scraper() as session:
        session.headers.update(headers)
        for query in search_queries:
            if len(raw_reviews) >= limit:
                break

            quoted_query = urllib.parse.quote(query)
            url = f"https://old.reddit.com/search?q={quoted_query}&sort=new"

            try:
                response = session.get(url, timeout=20)
                if response.status_code == 429:
                    time.sleep(2)
                    response = session.get(url, timeout=20)

                if response.status_code != 200:
                    print(f"Failed to fetch Reddit search results for query '{query}': {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                results = soup.select(".search-result")

                for res in results:
                    title_el = res.select_one("a.search-title")
                    if not title_el:
                        continue

                    url_path = title_el.get("href", "")
                    if url_path.startswith("/"):
                        post_url = f"https://www.reddit.com{url_path}"
                    else:
                        post_url = url_path

                    fullname = res.get("data-fullname") or ""
                    post_id = fullname.split("_")[-1] if "_" in fullname else ""
                    if not post_id and "comments/" in post_url:
                        try:
                            parts = post_url.split("comments/")
                            if len(parts) > 1:
                                post_id = parts[1].split("/")[0]
                        except Exception:
                            post_id = ""

                    review_text = _extract_reddit_post(res)
                    if not review_text:
                        continue

                    norm_text = " ".join(review_text.split()).strip().lower()
                    if norm_text in seen_texts:
                        continue
                    seen_texts.add(norm_text)

                    time_el = res.select_one("time")
                    date_str = ""
                    if time_el and time_el.has_attr("datetime"):
                        dt_val = time_el["datetime"]
                        if dt_val:
                            try:
                                if "T" in dt_val:
                                    date_str = dt_val.split("T")[0]
                                else:
                                    date_str = dt_val
                            except Exception:
                                date_str = ""

                    if not post_id:
                        post_id = stable_id("reddit", review_text, post_url)

                    raw_reviews.append(
                        RawReview(
                            id=post_id,
                            source="reddit",
                            review=review_text,
                            rating=None,
                            date=date_str,
                            url=post_url,
                        )
                    )

                    if len(raw_reviews) >= limit:
                        break

                time.sleep(1.0)

            except Exception as e:
                print(f"Error collecting Reddit reviews for query '{query}': {e}")

    return raw_reviews
