from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    RAW_REVIEWS_CSV,
    DEFAULT_LIMIT,
    GOOGLE_PLAY_COUNTRY,
    GOOGLE_PLAY_LANGUAGE,
)

from reviews.base import BaseCollector
from reviews.google_play import GooglePlayCollector, collect_google_play_reviews
from reviews.app_store import AppStoreCollector, collect_app_store_reviews
from reviews.reddit import RedditCollector, collect_reddit_reviews
from reviews.upload import UploadCollector, parse_csv_reviews, parse_json_reviews, parse_manual_reviews
from reviews.utils import write_raw_reviews_csv


DEFAULT_OUTPUT_PATH = RAW_REVIEWS_CSV
SUPPORTED_SOURCES = ("google_play", "app_store", "reddit", "upload")

COLLECTORS: dict[str, BaseCollector] = {
    "google_play": GooglePlayCollector(),
    "app_store": AppStoreCollector(),
    "reddit": RedditCollector(),
    "upload": UploadCollector(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Blinkit product & catalog discovery feedback into raw_reviews.csv."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SUPPORTED_SOURCES,
        default=["google_play"],
        help="Review sources to collect from.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max reviews per run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviews = []

    for source_key in args.sources:
        collector = COLLECTORS.get(source_key)
        if collector:
            reviews.extend(collector.collect(limit=args.limit))

    write_raw_reviews_csv(reviews, args.output)
    print(f"Wrote {len(reviews)} collected reviews to {args.output}")


if __name__ == "__main__":
    main()
