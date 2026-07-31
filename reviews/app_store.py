from __future__ import annotations

import requests
from reviews.base import BaseCollector
from reviews.models import RawReview
from reviews.utils import clean_text, iso_date, stable_id
from config import APP_STORE_COUNTRY, BLINKIT_APP_STORE_ID

BLINKIT_APP_STORE_URL = (
    f"https://apps.apple.com/{APP_STORE_COUNTRY}/app/blinkit-grocery-in-minutes/id{BLINKIT_APP_STORE_ID}"
)


class AppStoreCollector(BaseCollector):
    @property
    def source_name(self) -> str:
        return "app_store"

    def collect(self, limit: int = 100, **kwargs) -> list[RawReview]:
        country = kwargs.get("country", APP_STORE_COUNTRY)
        app_id = kwargs.get("app_id", BLINKIT_APP_STORE_ID)
        return collect_app_store_reviews(limit=limit, country=country, app_id=app_id)


def collect_app_store_reviews(
    limit: int = 100,
    country: str = APP_STORE_COUNTRY,
    app_id: str = BLINKIT_APP_STORE_ID,
) -> list[RawReview]:
    """Collects customer reviews for Blinkit iOS from Apple's public RSS endpoint, paginating up to limit."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    reviews: list[RawReview] = []
    seen_ids: set[str] = set()

    urls = [
        f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/json",
    ] + [
        f"https://itunes.apple.com/{country}/rss/customerreviews/page={p}/id={app_id}/json"
        for p in range(1, 11)
    ]

    for url in urls:
        if len(reviews) >= limit:
            break
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                continue

            payload = response.json()
            entries = payload.get("feed", {}).get("entry", [])
            if not isinstance(entries, list):
                entries = [entries] if isinstance(entries, dict) else []

            if not entries:
                continue

            for entry in entries:
                if len(reviews) >= limit:
                    break

                review_text = clean_text(
                    entry.get("content", {}).get("label") or entry.get("title", {}).get("label")
                )
                if not review_text:
                    continue

                entry_id = str(entry.get("id", {}).get("label") or stable_id("app_store", review_text))
                if entry_id in seen_ids:
                    continue
                seen_ids.add(entry_id)

                rating_val = None
                try:
                    rating_val = int(entry.get("im:rating", {}).get("label", ""))
                except (ValueError, TypeError):
                    pass

                reviews.append(
                    RawReview(
                        id=entry_id,
                        source="app_store",
                        review=review_text,
                        rating=rating_val,
                        date=iso_date(None),
                        url=BLINKIT_APP_STORE_URL,
                    )
                )

        except Exception as exc:
            print(f"Error collecting Apple App Store reviews on page {page}: {exc}")
            break

    return reviews
