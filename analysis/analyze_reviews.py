from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

import config
from analysis.llm_client import (
    AnalysisError,
    ADAPTIVE_ENGINE,
    analyze_review_batch,
    generate_json_content_with_fallback,
    parse_json_response,
)
from analysis.schema import ANALYSIS_FIELDS, empty_analysis
from reviews.models import RAW_REVIEW_FIELDS

DEFAULT_INPUT_PATH = config.FILTERED_REVIEWS_CSV
DEFAULT_OUTPUT_PATH = config.ANALYZED_REVIEWS_CSV
DEFAULT_MODEL = config.DEFAULT_MODEL

ANALYSIS_OUTPUT_FIELDS = [
    *RAW_REVIEW_FIELDS,
    *ANALYSIS_FIELDS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Blinkit search/catalog discovery insights from filtered reviews via AI."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE_ANALYZE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=config.DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay-seconds", type=float, default=config.DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Write fallback rows for unanalyzed reviews instead of failing the entire process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    filtered_reviews = read_filtered_reviews(args.input)
    analysis_rows = analyze_reviews(filtered_reviews, args)
    write_analyzed_reviews(analysis_rows, args.output)
    print(f"Wrote {len(analysis_rows)} analyzed reviews to {args.output}")


def read_filtered_reviews(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Filtered reviews file not found: {input_path}")
    dataframe = pd.read_csv(input_path, dtype={"id": str}).fillna("")
    missing_fields = [field for field in RAW_REVIEW_FIELDS if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required filtered review fields: {missing_fields}")
    dataframe = dataframe[RAW_REVIEW_FIELDS].copy()
    dataframe["review"] = dataframe["review"].astype(str).str.strip()
    return dataframe[dataframe["review"] != ""].copy()


def analyze_reviews(
    filtered_reviews: pd.DataFrame,
    args: argparse.Namespace,
    *extra_args: Any,
    progress_callback: Callable[[int, int, str], None] | None = None,
    **kwargs: Any,
) -> list[dict[str, object]]:
    if progress_callback is None and "progress_callback" in kwargs:
        progress_callback = kwargs.pop("progress_callback")

    records = filtered_reviews.to_dict(orient="records")
    total_reviews = len(records)
    if total_reviews == 0:
        return []

    analysis_rows_by_id: dict[str, dict[str, object]] = {}
    api_key_override = getattr(args, "api_key", None)
    account_id_override = getattr(args, "account_id", None)

    start = 0
    progress = tqdm(
        total=total_reviews,
        desc="Extracting product insights via AI",
        unit="review",
    )
    while start < total_reviews:
        active_batch_size = min(ADAPTIVE_ENGINE.batch_size_analyze, args.batch_size)
        batch = records[start : start + active_batch_size]
        batch_results = _analyze_batch_with_fallback(
            reviews=_prompt_records(batch),
            model=args.model,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
            continue_on_error=args.continue_on_error,
            api_key_override=api_key_override,
            account_id_override=account_id_override,
        )
        for res in batch_results:
            analysis_rows_by_id[str(res["id"])] = res
        start += len(batch)
        progress.update(len(batch))
        if progress_callback:
            progress_callback(
                start,
                total_reviews,
                f"Extracting insights with {args.model} ({start}/{total_reviews} reviews processed)",
            )
    progress.close()

    raw_by_id = filtered_reviews.set_index(filtered_reviews["id"].astype(str))
    results: list[dict[str, object]] = []
    for record in records:
        review_id = str(record["id"])
        item = analysis_rows_by_id.get(review_id, _fallback_analysis_item(review_id))
        
        raw_row = raw_by_id.loc[review_id].to_dict() if review_id in raw_by_id.index else record
        if isinstance(raw_row, pd.DataFrame):
            raw_row = raw_row.iloc[0].to_dict()

        results.append(
            {
                "id": review_id,
                "source": str(raw_row.get("source", "unknown")),
                "review": str(raw_row.get("review", record.get("review", ""))),
                "rating": raw_row.get("rating", ""),
                "date": str(raw_row.get("date", "")),
                "url": str(raw_row.get("url", "")),
                **{field: item.get(field, empty_analysis()[field]) for field in ANALYSIS_FIELDS},
            }
        )
    return results


def _analyze_batch_with_fallback(
    reviews: list[dict[str, object]],
    model: str,
    max_retries: int,
    retry_delay_seconds: float,
    continue_on_error: bool,
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> list[dict[str, object]]:
    try:
        return analyze_review_batch(
            reviews=reviews,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            api_key_override=api_key_override,
            account_id_override=account_id_override,
        )
    except Exception as exc:
        if not continue_on_error:
            raise
        print(f"Batch analysis failed: {exc}. Writing fallback rows for batch.")
        return [_fallback_analysis_item(str(r["id"])) for r in reviews]


def write_analyzed_reviews(
    analysis_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(analysis_rows, columns=ANALYSIS_OUTPUT_FIELDS)
    dataframe.to_csv(
        output_path,
        columns=ANALYSIS_OUTPUT_FIELDS,
        index=False,
        encoding="utf-8-sig",
    )


def _prompt_records(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": str(row["id"]),
            "review": _truncate_review_text(row.get("review"), ADAPTIVE_ENGINE.max_review_chars),
        }
        for row in batch
    ]


def _truncate_review_text(review: object, max_characters: int) -> str:
    text = str(review or "").strip()
    if len(text) <= max_characters:
        return text
    return f"{text[: max_characters - 3].rstrip()}..."


def _fallback_analysis_item(review_id: str) -> dict[str, object]:
    return {"id": review_id, **empty_analysis()}


if __name__ == "__main__":
    main()
