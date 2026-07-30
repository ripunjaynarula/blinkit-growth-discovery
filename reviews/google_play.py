from __future__ import annotations

from typing import Iterable

from reviews.base import BaseCollector
from reviews.models import RawReview
from reviews.utils import clean_text, iso_date, stable_id
from config import GOOGLE_PLAY_COUNTRY, GOOGLE_PLAY_LANGUAGE


BLINKIT_ANDROID_APP_ID = "com.grofers.customerapp"
BLINKIT_PLAY_STORE_URL = (
    "https://play.google.com/store/apps/details?id=com.grofers.customerapp"
)


class GooglePlayCollector(BaseCollector):
    @property
    def source_name(self) -> str:
        return "google_play"

    def collect(self, limit: int, **kwargs) -> list[RawReview]:
        country = kwargs.get("country", GOOGLE_PLAY_COUNTRY)
        language = kwargs.get("language", GOOGLE_PLAY_LANGUAGE)
        return collect_google_play_reviews(limit=limit, country=country, language=language)


def collect_google_play_reviews(
    limit: int,
    country: str = GOOGLE_PLAY_COUNTRY,
    language: str = GOOGLE_PLAY_LANGUAGE,
) -> list[RawReview]:
    try:
        from google_play_scraper import Sort, reviews
    except ImportError as exc:
        raise RuntimeError(
            "google-play-scraper is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    collected, _ = reviews(
        BLINKIT_ANDROID_APP_ID,
        lang=language,
        country=country,
        sort=Sort.NEWEST,
        count=limit,
    )
    return list(_normalize_google_play_reviews(collected))


def _normalize_google_play_reviews(items: Iterable[dict]) -> Iterable[RawReview]:
    for item in items:
        review_text = clean_text(item.get("content"))
        review_id = str(item.get("reviewId") or stable_id("google_play", review_text))
        yield RawReview(
            id=review_id,
            source="google_play",
            review=review_text,
            rating=item.get("score"),
            date=iso_date(item.get("at")),
            url=BLINKIT_PLAY_STORE_URL,
        )
