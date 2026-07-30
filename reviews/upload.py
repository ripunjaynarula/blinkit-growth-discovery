from __future__ import annotations

import io
import json
import pandas as pd

from reviews.base import BaseCollector
from reviews.models import RawReview
from reviews.utils import clean_text, iso_date, stable_id


class UploadCollector(BaseCollector):
    @property
    def source_name(self) -> str:
        return "upload"

    def collect(self, limit: int, **kwargs) -> list[RawReview]:
        raw_data = kwargs.get("data")
        file_type = kwargs.get("file_type", "csv")
        if not raw_data:
            return []

        if file_type == "csv":
            return parse_csv_reviews(raw_data, limit)
        elif file_type == "json":
            return parse_json_reviews(raw_data, limit)
        elif file_type == "manual":
            return parse_manual_reviews(str(raw_data), limit)
        return []


def parse_csv_reviews(content: str | bytes, limit: int = 500) -> list[RawReview]:
    """Parses uploaded CSV content into RawReview objects."""
    if isinstance(content, str):
        buf = io.StringIO(content)
    else:
        buf = io.BytesIO(content)

    df = pd.read_csv(buf).fillna("")
    review_col = next((c for c in df.columns if c.lower() in ("review", "content", "text", "comment", "feedback")), None)
    if not review_col:
        review_col = df.columns[0]

    reviews: list[RawReview] = []
    for idx, row in df.iterrows():
        if len(reviews) >= limit:
            break
        text = clean_text(row[review_col])
        if not text:
            continue
        row_id = str(row.get("id") or stable_id("upload_csv", text, idx))
        reviews.append(
            RawReview(
                id=row_id,
                source="upload_csv",
                review=text,
                rating=int(row["rating"]) if "rating" in row and str(row["rating"]).isdigit() else None,
                date=iso_date(row.get("date")),
                url=str(row.get("url", "")),
            )
        )
    return reviews


def parse_json_reviews(content: str | bytes, limit: int = 500) -> list[RawReview]:
    """Parses uploaded JSON content into RawReview objects."""
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    data = json.loads(content)
    items = data.get("reviews", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        items = [items]

    reviews: list[RawReview] = []
    for idx, item in enumerate(items):
        if len(reviews) >= limit:
            break
        if isinstance(item, str):
            text = clean_text(item)
            item_obj = {}
        elif isinstance(item, dict):
            text = clean_text(item.get("review") or item.get("content") or item.get("text"))
            item_obj = item
        else:
            continue

        if not text:
            continue

        row_id = str(item_obj.get("id") or stable_id("upload_json", text, idx))
        reviews.append(
            RawReview(
                id=row_id,
                source="upload_json",
                review=text,
                rating=item_obj.get("rating"),
                date=iso_date(item_obj.get("date")),
                url=str(item_obj.get("url", "")),
            )
        )
    return reviews


def parse_manual_reviews(text_block: str, limit: int = 500) -> list[RawReview]:
    """Parses line-separated or multi-line manual text entries into RawReview objects."""
    lines = [clean_text(line) for line in text_block.splitlines() if clean_text(line)]
    reviews: list[RawReview] = []
    for idx, line in enumerate(lines):
        if len(reviews) >= limit:
            break
        row_id = stable_id("manual_entry", line, idx)
        reviews.append(
            RawReview(
                id=row_id,
                source="manual_entry",
                review=line,
                rating=None,
                date=iso_date(None),
                url="",
            )
        )
    return reviews
