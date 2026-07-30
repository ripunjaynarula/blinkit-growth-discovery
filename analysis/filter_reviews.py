from __future__ import annotations

import argparse
import json
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
    generate_parsed_json_with_fallback,
    parse_json_response,
)
from analysis.schema import clamp_confidence
from reviews.models import RAW_REVIEW_FIELDS

DEFAULT_INPUT_PATH = config.RAW_REVIEWS_CSV
DEFAULT_OUTPUT_PATH = config.FILTERED_REVIEWS_CSV
DEFAULT_REJECTED_OUTPUT_PATH = config.REJECTED_REVIEWS_CSV
DEFAULT_SUMMARY_OUTPUT_PATH = config.FILTER_SUMMARY_JSON
DEFAULT_MODEL = config.DEFAULT_MODEL

RELEVANCE_FIELDS = ["review_id", "is_relevant", "confidence", "reason", "discovered_signals"]

SYSTEM_PROMPT = """
You are a product research specialist filtering app reviews for a study on Blinkit's product discovery experience.

OBJECTIVE:
Identify reviews that contain evidence about how users search for, browse, navigate to, or discover products on Blinkit.
Optimize for PRECISION — only keep reviews where product discovery is the primary subject.

KEEP reviews that discuss:
- Searching for products (search bar, keywords, search results quality)
- Browsing categories, aisles, collections, or shelves
- Product recommendations (home page, cross-sell, "you might also like")
- Navigation and finding products within the app
- Product availability where it prevented the user from completing discovery
- Substitutions or alternatives when a product was unavailable
- Exploration behavior (discovering new products, categories, brands)
- Merchandising, catalog organization, product placement, variant selection
- Comparison shopping, brand discovery, or filter/sort usage
- Inventory gaps that forced the user to abandon or change their search

REJECT reviews primarily about:
- Delivery speed, delays, or late orders (reject unless the review ALSO discusses search/browse/discovery)
- Refunds, cancellations, missing items, or damaged goods
- Customer support or agent interactions
- Payment, checkout flow, or billing issues
- App crashes, loading errors, or technical bugs unrelated to search or browse
- Logistics, packaging, temperature, or cold chain
- Rider behavior or delivery personnel
- General praise or complaints with no product discovery context
- Unrelated Reddit discussions, news, or off-topic content
- Price complaints that do not involve discovering or comparing products

DECISION RULE:
Ask: "Does this review contain evidence about how the user searched, browsed, navigated, or discovered (or failed to discover) a product?"
If YES → is_relevant: true
If NO  → is_relevant: false
When in doubt, REJECT — prefer precision over recall.

EXAMPLES:
- "Searched for almond milk but only regular milk showed up" → KEEP (search relevance failure)
- "App recommended something I already bought yesterday" → KEEP (recommendation quality)
- "Can never find the ghee brand I want in the atta aisle" → KEEP (catalog organization)
- "Couldn't find a substitute when my product was out of stock" → KEEP (substitution/availability)
- "Delivery was 45 minutes late" → REJECT (delivery, no discovery content)
- "Customer support took 3 hours to respond" → REJECT (support, not discovery)
- "Payment failed twice" → REJECT (checkout, not discovery)
- "Great app, love it" → REJECT (no insight)

OUTPUT FORMAT:
Return a JSON object with one key "reviews" containing an array. Each object must include:
{
    "review_id": "...",
    "is_relevant": true or false,
    "confidence": 0.0 to 1.0,
    "reason": "one sentence explaining the discovery-relevance decision",
    "discovered_signals": []
}

Allowed discovered_signals values:
"search", "browse", "navigation", "recommendations", "availability", "substitution", "exploration", "merchandising", "competitor_comparison", "filter_sort"

Return JSON ONLY.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter raw Blinkit reviews for product/catalog discovery and search relevance."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE_FILTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=config.DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay-seconds", type=float, default=config.DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--min-review-length", type=int, default=config.DEFAULT_MIN_REVIEW_LENGTH)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only keep relevant reviews at or above this confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.min_review_length < 1:
        raise ValueError("--min-review-length must be at least 1")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")

    raw_reviews = read_raw_reviews(args.input)
    reviews_for_llm, removal_counts = prepare_reviews_for_llm(
        raw_reviews,
        min_review_length=args.min_review_length,
    )
    print_removal_counts(removal_counts)

    relevance_rows = classify_relevance(reviews_for_llm, args)
    filtered_reviews = filter_relevant_reviews(
        reviews_for_llm,
        relevance_rows,
        min_confidence=args.min_confidence,
    )
    rejected_reviews = build_rejected_reviews(reviews_for_llm, relevance_rows)
    write_filtered_reviews(filtered_reviews, args.output)
    write_rejected_reviews(rejected_reviews, DEFAULT_REJECTED_OUTPUT_PATH)
    write_filter_summary(
        summary=build_filter_summary(
            total_reviews=len(raw_reviews),
            reviews_sent_to_llm=len(reviews_for_llm),
            filtered_reviews=filtered_reviews,
            relevance_rows=relevance_rows,
            model=args.model,
            processing_time_seconds=time.perf_counter() - start_time,
        ),
        output_path=DEFAULT_SUMMARY_OUTPUT_PATH,
    )
    print(f"Wrote {len(filtered_reviews)} relevant reviews to {args.output}")
    print(f"Wrote {len(rejected_reviews)} rejected reviews to {DEFAULT_REJECTED_OUTPUT_PATH}")
    print(f"Wrote filter summary to {DEFAULT_SUMMARY_OUTPUT_PATH}")


def read_raw_reviews(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw reviews file not found: {input_path}")
    dataframe = pd.read_csv(input_path, dtype={"id": str}).fillna("")
    missing_fields = [field for field in RAW_REVIEW_FIELDS if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required raw review fields: {missing_fields}")
    dataframe = dataframe[RAW_REVIEW_FIELDS].copy()
    dataframe["review"] = dataframe["review"].astype(str).str.strip()
    return dataframe


def prepare_reviews_for_llm(
    raw_reviews: pd.DataFrame,
    min_review_length: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    reviews_series = raw_reviews["review"].astype(str).str.strip()
    
    empty_mask = reviews_series == ""
    short_mask = reviews_series.str.len() < min_review_length
    
    valid_mask = (~empty_mask) & (~short_mask)
    valid_reviews = raw_reviews[valid_mask]
    
    duplicate_mask = valid_reviews["review"].astype(str).str.strip().duplicated(keep="first")
    prepared_reviews = valid_reviews[~duplicate_mask].copy()

    removal_counts = {
        "empty_reviews": int(empty_mask.sum()),
        "short_reviews": int((~empty_mask & short_mask).sum()),
        "duplicate_review_texts": int(duplicate_mask.sum()),
    }
    return prepared_reviews, removal_counts


def print_removal_counts(removal_counts: dict[str, int]) -> None:
    print(f"Removed empty reviews: {removal_counts['empty_reviews']}")
    print(f"Removed short reviews: {removal_counts['short_reviews']}")
    print(f"Removed duplicate review texts: {removal_counts['duplicate_review_texts']}")


def deterministic_pre_filter(review_text: str) -> dict[str, object] | None:
    text_lower = review_text.lower().strip()

    # Reject obvious non-informative noise
    noise_phrases = {"good", "nice app", "ok", "okay", "good app", "very good", "nice", "best", "super", "love the app"}
    if text_lower in noise_phrases or len(text_lower) < 4:
        return {
            "is_relevant": False,
            "relevant": False,
            "reason": "Deterministic pre-filter: single-word greeting or uninformative rating.",
            "confidence": 1.0,
            "discovered_signals": [],
        }

    return None


def classify_relevance(
    raw_reviews: pd.DataFrame,
    args: argparse.Namespace,
    *extra_args: Any,
    progress_callback: Callable[[int, int, str], None] | None = None,
    **kwargs: Any,
) -> list[dict[str, object]]:
    if progress_callback is None and "progress_callback" in kwargs:
        progress_callback = kwargs.pop("progress_callback")

    records = raw_reviews.to_dict(orient="records")
    relevance_rows_by_id: dict[str, dict[str, object]] = {}
    
    llm_records: list[dict[str, object]] = []
    for record in records:
        review_id = str(record["id"])
        pre_filter_result = deterministic_pre_filter(record["review"])
        if pre_filter_result is not None:
            relevance_rows_by_id[review_id] = {
                "id": review_id,
                "review_id": review_id,
                **pre_filter_result
            }
        else:
            llm_records.append(record)
            
    total_llm_reviews = len(llm_records)
    if total_llm_reviews > 0:
        api_key_override = getattr(args, "api_key", None)
        account_id_override = getattr(args, "account_id", None)
        start = 0
        progress = tqdm(
            total=total_llm_reviews,
            desc="Filtering reviews via AI",
            unit="review",
        )
        while start < total_llm_reviews:
            active_batch_size = min(ADAPTIVE_ENGINE.batch_size_filter, args.batch_size)
            batch = llm_records[start : start + active_batch_size]
            batch_results = classify_relevance_batch(
                reviews=_prompt_records(batch),
                model=args.model,
                max_retries=args.max_retries,
                retry_delay_seconds=args.retry_delay_seconds,
                api_key_override=api_key_override,
                account_id_override=account_id_override,
            )
            for res in batch_results:
                relevance_rows_by_id[str(res["id"])] = res
            start += len(batch)
            progress.update(len(batch))
            if progress_callback:
                progress_callback(
                    start,
                    total_llm_reviews,
                    f"Filtering batch with {args.model} ({start}/{total_llm_reviews} reviews processed)",
                )
        progress.close()
            
    return [relevance_rows_by_id[str(record["id"])] for record in records]


def classify_relevance_batch(
    reviews: list[dict[str, object]],
    model: str,
    max_retries: int,
    retry_delay_seconds: float,
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> list[dict[str, object]]:
    remaining_reviews = reviews
    relevance_by_id: dict[str, dict[str, object]] = {}
    attempts = max(1, max_retries + 1)

    for attempt in range(attempts):
        batch_rows = _request_relevance_batch(
            reviews=remaining_reviews,
            model=model,
            max_retries=0,
            retry_delay_seconds=retry_delay_seconds,
            api_key_override=api_key_override,
            account_id_override=account_id_override,
        )
        for row in batch_rows:
            relevance_by_id[str(row["id"])] = row

        missing_reviews = _missing_reviews(remaining_reviews, relevance_by_id)
        if not missing_reviews:
            break

        missing_ids = [str(review["id"]) for review in missing_reviews]
        print(f"AI returned no result for review ids: {', '.join(missing_ids)}")
        if attempt == attempts - 1:
            for review in missing_reviews:
                fallback_row = _no_model_response(str(review["id"]))
                relevance_by_id[str(review["id"])] = fallback_row
            break

        delay_seconds = retry_delay_seconds * (2**attempt)
        print(f"Retrying missing review ids only in {delay_seconds} seconds...")
        time.sleep(delay_seconds)
        remaining_reviews = missing_reviews

    return [relevance_by_id[str(review["id"])] for review in reviews]


def _request_relevance_batch(
    reviews: list[dict[str, object]],
    model: str,
    max_retries: int,
    retry_delay_seconds: float,
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> list[dict[str, object]]:
    def request_and_parse() -> list[dict[str, object]]:
        parsed = generate_parsed_json_with_fallback(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_relevance_prompt(reviews),
            task_type="relevance",
            api_key_override=api_key_override,
            account_id_override=account_id_override,
        )
        return normalize_relevance_response(parsed, reviews)

    return _with_retries(request_and_parse, max_retries, retry_delay_seconds)


def build_relevance_prompt(reviews: list[dict[str, object]]) -> str:
    return (
        "Filter the following reviews. KEEP only those where product discovery (searching, browsing, "
        "navigation, recommendations, finding products, substitutions, exploration) is the primary subject. "
        "REJECT reviews primarily about delivery, payments, customer support, app crashes, or logistics. "
        "Optimize for precision — when in doubt, reject. "
        'Return a JSON object with one key "reviews" containing exactly one result per input review.\n\n'
        f"Input reviews:\n{reviews}"
    )


def normalize_relevance_response(
    parsed: dict[str, Any],
    input_reviews: list[dict[str, object]],
) -> list[dict[str, object]]:
    items = parsed.get("reviews", parsed.get("results", parsed.get("items")))
    if not isinstance(items, list):
        raise AnalysisError('AI response must contain a "reviews" array.')

    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict):
            r_id = str(item.get("review_id") or item.get("id") or "")
            if r_id:
                by_id[r_id] = item

    rows: list[dict[str, object]] = []
    for input_review in input_reviews:
        review_id = str(input_review["id"])
        item = by_id.get(review_id, {})
        rows.append({"id": review_id, **_normalize_relevance_item(review_id, item)})
    return rows


def filter_relevant_reviews(
    raw_reviews: pd.DataFrame,
    relevance_rows: list[dict[str, object]],
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    relevance_by_id = {str(row["id"]): row for row in relevance_rows}
    keep_ids = {
        review_id
        for review_id, row in relevance_by_id.items()
        if (row.get("is_relevant") is True or row.get("relevant") is True) and float(row["confidence"]) >= min_confidence
    }
    return raw_reviews[raw_reviews["id"].astype(str).isin(keep_ids)][RAW_REVIEW_FIELDS].copy()


def write_filtered_reviews(filtered_reviews: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_reviews.to_csv(
        output_path,
        columns=RAW_REVIEW_FIELDS,
        index=False,
        encoding="utf-8-sig",
    )


def build_rejected_reviews(
    raw_reviews: pd.DataFrame,
    relevance_rows: list[dict[str, object]],
) -> pd.DataFrame:
    raw_by_id = raw_reviews.set_index(raw_reviews["id"].astype(str))
    rows: list[dict[str, object]] = []
    for relevance_row in relevance_rows:
        is_rel = relevance_row.get("is_relevant") is True or relevance_row.get("relevant") is True
        if is_rel:
            continue
        review_id = str(relevance_row["id"])
        if review_id not in raw_by_id.index:
            continue
        rows.append(
            {
                "id": review_id,
                "review": raw_by_id.loc[review_id, "review"],
                "reason": relevance_row["reason"],
                "confidence": relevance_row["confidence"],
            }
        )
    return pd.DataFrame(rows, columns=["id", "review", "reason", "confidence"])


def write_rejected_reviews(rejected_reviews: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_reviews.to_csv(
        output_path,
        columns=["id", "review", "reason", "confidence"],
        index=False,
        encoding="utf-8-sig",
    )


def build_filter_summary(
    total_reviews: int,
    reviews_sent_to_llm: int,
    filtered_reviews: pd.DataFrame,
    relevance_rows: list[dict[str, object]],
    model: str,
    processing_time_seconds: float,
) -> dict[str, object]:
    confidence_values = [float(row["confidence"]) for row in relevance_rows]
    irrelevant_reviews = sum(
        1 for row in relevance_rows
        if not (row.get("is_relevant") is True or row.get("relevant") is True)
    )
    return {
        "total_reviews": total_reviews,
        "reviews_sent_to_llm": reviews_sent_to_llm,
        "relevant_reviews": len(filtered_reviews),
        "irrelevant_reviews": irrelevant_reviews,
        "average_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        ),
        "model": model,
        "processing_time_seconds": round(processing_time_seconds, 2),
    }


def write_filter_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
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


def _normalize_relevance_item(review_id: str, item: dict[str, Any]) -> dict[str, object]:
    is_rel = _as_bool(item.get("is_relevant", item.get("relevant")))
    signals = item.get("discovered_signals", [])
    if not isinstance(signals, list):
        signals = [str(signals)] if signals else []

    return {
        "review_id": review_id,
        "is_relevant": is_rel,
        "relevant": is_rel,
        "reason": str(item.get("reason") or "").strip(),
        "confidence": clamp_confidence(item.get("confidence")),
        "discovered_signals": [str(s).strip() for s in signals if str(s).strip()],
    }


def _missing_reviews(
    input_reviews: list[dict[str, object]],
    relevance_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return [
        review
        for review in input_reviews
        if str(review["id"]) not in relevance_by_id
    ]


def _no_model_response(review_id: str) -> dict[str, object]:
    return {
        "id": review_id,
        "review_id": review_id,
        "is_relevant": False,
        "relevant": False,
        "reason": "No model response",
        "confidence": 0.0,
        "discovered_signals": [],
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return False


def _with_retries(
    operation: Callable[[], list[dict[str, object]]],
    max_retries: int,
    retry_delay_seconds: float,
) -> list[dict[str, object]]:
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            delay_seconds = retry_delay_seconds * (2**attempt)
            print(f"Attempt {attempt + 1}/{attempts} failed.")
            print(f"Reason: {exc}")
            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise AnalysisError("Relevance filtering failed without a captured exception.")


if __name__ == "__main__":
    main()
