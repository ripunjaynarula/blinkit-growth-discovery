from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from reviews.models import RAW_REVIEW_FIELDS, RawReview


def stable_id(source: str, *parts: object) -> str:
    payload = "|".join(str(part or "") for part in (source, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def iso_date(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.date().isoformat()
    return str(value)


def clean_text(text: object) -> str:
    return " ".join(str(text or "").split())


def dedupe_reviews(reviews: Iterable[RawReview]) -> list[RawReview]:
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    unique_reviews: list[RawReview] = []
    for review in reviews:
        cleaned = " ".join(review.review.split()).strip().lower()
        if not cleaned or review.id in seen_ids or cleaned in seen_texts:
            continue
        seen_ids.add(review.id)
        seen_texts.add(cleaned)
        unique_reviews.append(review)
    return unique_reviews


def write_raw_reviews_csv(reviews: Iterable[RawReview], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [review.to_dict() for review in dedupe_reviews(reviews)]
    dataframe = pd.DataFrame(rows, columns=RAW_REVIEW_FIELDS)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
