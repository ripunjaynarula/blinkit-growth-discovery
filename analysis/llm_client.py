from __future__ import annotations

import json
import re
import traceback
import time
from collections.abc import Callable
from json import JSONDecodeError
from typing import Any

import config
from analysis.llm_provider import get_llm_provider, LLMProviderError, LLMProvider
from analysis.schema import (
    ANALYSIS_FIELDS,
    SYSTEM_PROMPT,
    build_batch_prompt,
    empty_analysis,
    ROOT_CAUSES,
    DISCOVERY_SURFACES,
    USER_SEGMENTS,
    clamp_confidence,
)


class AnalysisError(RuntimeError):
    pass


class LLMRequestError(AnalysisError):
    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AdaptiveTokenEngine:
    """Dynamically adapts batch sizes, character limits, and token budgets based on provider API responses."""
    def __init__(self) -> None:
        self.batch_size_filter = config.DEFAULT_BATCH_SIZE_FILTER
        self.batch_size_analyze = config.DEFAULT_BATCH_SIZE_ANALYZE
        self.max_review_chars = config.DEFAULT_MAX_REVIEW_CHARACTERS

    def adapt_on_token_cap(self, stage: str, message: str) -> None:
        token_match = re.search(r"(\d+)\s*tokens?", message, re.IGNORECASE)
        cap_val = token_match.group(1) if token_match else "limit"
        
        if stage == "relevance":
            self.batch_size_filter = max(1, self.batch_size_filter - 1)
            self.max_review_chars = max(300, self.max_review_chars - 150)
            print(f"[Adaptive Token Engine] Detected token cap ({cap_val}). Filter batch size reduced to {self.batch_size_filter}, review text capped to {self.max_review_chars} chars.")
        else:
            self.batch_size_analyze = max(1, self.batch_size_analyze - 1)
            self.max_review_chars = max(300, self.max_review_chars - 150)
            print(f"[Adaptive Token Engine] Detected token cap ({cap_val}). Analyze batch size reduced to {self.batch_size_analyze}, review text capped to {self.max_review_chars} chars.")


ADAPTIVE_ENGINE = AdaptiveTokenEngine()


def estimate_tokens(text: str) -> int:
    """Rough token estimation (approx 4 chars per token for English text)."""
    return len(text) // 4


def get_model_max_tokens(model: str, task_type: str = "analysis") -> int:
    """Returns tailored per-call max_tokens based on active model and task type."""
    model_cfg = config.MODEL_TOKEN_LIMITS.get(
        model, config.MODEL_TOKEN_LIMITS.get("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {"relevance": 1500, "analysis": 2500, "summary": 4000})
    )
    return model_cfg.get(task_type, 1500)


def test_provider_connection(
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    account_id: str | None = None,
) -> tuple[bool, str]:
    """Validates connectivity for the selected provider."""
    provider = get_llm_provider(provider_name)
    return provider.test_connection(api_key=api_key, model=model, account_id=account_id)


def test_groq_connection(api_key: str | None = None, model: str | None = None) -> tuple[bool, str]:
    """Backward compatible connection test function."""
    return test_provider_connection("cloudflare", api_key=api_key, model=model)


def analyze_review_batch(
    reviews: list[dict[str, object]],
    model: str | None = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> list[dict[str, object]]:
    def request_and_parse() -> list[dict[str, object]]:
        parsed = generate_parsed_json_with_fallback(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_batch_prompt(reviews),
            task_type="analysis",
            api_key_override=api_key_override,
            account_id_override=account_id_override,
        )
        return normalize_batch_response(parsed, reviews)

    return _with_retries(request_and_parse, max_retries, retry_delay_seconds)


def generate_parsed_json_with_fallback(
    model: str | None,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
    task_type: str = "analysis",
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> dict[str, Any]:
    """Generates structured JSON responses with INSTANT model rotation on rate limits and multi-provider fallbacks."""
    provider = get_llm_provider()
    candidates: list[str] = []
    
    selected_model = model or provider.default_model
    if selected_model not in provider.fallback_chain:
        candidates.append(selected_model)
    candidates.extend(provider.fallback_chain)

    last_exc: Exception | None = None

    for candidate_model in candidates:
        computed_max_tokens = max_tokens or get_model_max_tokens(candidate_model, task_type)

        for attempt in range(2):
            try:
                raw_content, latency = provider.generate_raw_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=candidate_model,
                    max_tokens=computed_max_tokens,
                    api_key_override=api_key_override,
                    account_id_override=account_id_override,
                )
                print(f"[{provider.name.title()} AI] Model: '{candidate_model}' | Latency: {latency:.2f}s")
                return parse_json_response(raw_content)

            except LLMProviderError as exc:
                last_exc = exc
                msg = str(exc)

                if "context" in msg.lower() or "length" in msg.lower() or "too large" in msg.lower() or "token" in msg.lower():
                    ADAPTIVE_ENGINE.adapt_on_token_cap(task_type, msg)

                retry_sec = exc.retry_after_seconds

                # Short wait (<= 15s): quick sleep and retry
                if retry_sec is not None and 0 < retry_sec <= 15:
                    wait_time = retry_sec + 1.0
                    print(f"[{provider.name.title()} Quick Pause] Model '{candidate_model}' rate limit. Sleeping {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # Long wait (> 15s): Rotate immediately to next candidate model!
                if retry_sec is not None and retry_sec > 15:
                    mins = int(retry_sec // 60)
                    secs = int(retry_sec % 60)
                    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                    print(f"[{provider.name.title()} Model Rotation] Model '{candidate_model}' hit rate limit ({time_str} cooldown). Instantly switching to next model in chain...")
                    break

                print(f"[{provider.name.title()} Fallback] Model '{candidate_model}' error: {exc}. Trying next model...")
                break

            except (AnalysisError, Exception) as exc:
                last_exc = exc
                print(f"[{provider.name.title()} Fallback] Model '{candidate_model}' failed JSON parse/request: {exc}. Trying next model...")
                time.sleep(1.0)
                break

    if last_exc:
        raise LLMRequestError(str(last_exc))
    raise AnalysisError(f"All {provider.name.title()} model fallback candidates failed.")


def generate_json_content_with_fallback(
    model: str | None,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
    task_type: str = "analysis",
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> str:
    parsed = generate_parsed_json_with_fallback(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        task_type=task_type,
        api_key_override=api_key_override,
        account_id_override=account_id_override,
    )
    return json.dumps(parsed)


def parse_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    
    if "```" in cleaned:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return {"reviews": parsed}
        if isinstance(parsed, dict):
            return parsed
    except JSONDecodeError:
        pass

    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except JSONDecodeError:
            pass

    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group(0))
            if isinstance(parsed, list):
                return {"reviews": parsed}
        except JSONDecodeError:
            pass

    try:
        fixed_text = re.sub(r"'", '"', cleaned)
        fixed_text = re.sub(r",\s*([\}\]])", r"\1", fixed_text)
        
        obj_match = re.search(r"\{[\s\S]*\}", fixed_text)
        if obj_match:
            parsed = json.loads(obj_match.group(0))
            if isinstance(parsed, dict):
                return parsed

        arr_match = re.search(r"\[[\s\S]*\]", fixed_text)
        if arr_match:
            parsed = json.loads(arr_match.group(0))
            if isinstance(parsed, list):
                return {"reviews": parsed}
    except Exception:
        pass

    raise AnalysisError(f"Could not recover a valid JSON object or array from LLM response: '{cleaned[:120]}...'")


def normalize_batch_response(
    parsed: dict[str, Any],
    input_reviews: list[dict[str, object]],
) -> list[dict[str, object]]:
    items = parsed.get("reviews", parsed.get("results", parsed.get("items")))
    if not isinstance(items, list):
        raise AnalysisError('LLM response must contain a "reviews" array.')

    by_id = {
        str(item.get("id")): item
        for item in items
        if isinstance(item, dict) and item.get("id") is not None
    }

    normalized: list[dict[str, object]] = []
    for input_review in input_reviews:
        review_id = str(input_review["id"])
        item = by_id.get(review_id, {})
        normalized.append({"id": review_id, **_normalize_analysis_item(item)})
    return normalized


def _normalize_analysis_item(item: dict[str, Any]) -> dict[str, object]:
    root_cause_map = {rc.lower(): rc for rc in ROOT_CAUSES}
    discovery_surface_map = {ds.lower(): ds for ds in DISCOVERY_SURFACES}
    user_segment_map = {us.lower(): us for us in USER_SEGMENTS}

    normalized = empty_analysis()
    for field in ANALYSIS_FIELDS:
        value = item.get(field, normalized[field])
        if field == "confidence":
            normalized[field] = clamp_confidence(value)
            continue

        val_str = str(value or "unknown").strip()
        val_lower = val_str.lower()

        if field == "root_cause":
            if val_lower in root_cause_map:
                normalized[field] = root_cause_map[val_lower]
            else:
                normalized[field] = "unknown"
        elif field == "discovery_surface":
            if val_lower in discovery_surface_map:
                normalized[field] = discovery_surface_map[val_lower]
            else:
                normalized[field] = "unknown"
        elif field == "user_segment":
            if val_lower in user_segment_map:
                normalized[field] = user_segment_map[val_lower]
            else:
                normalized[field] = "unknown"
        else:
            normalized[field] = val_str if val_str else "unknown"
    return normalized


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
            print(
                f"[LLM attempt {attempt + 1}/{attempts} failed] "
                f"{type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )
            if attempt == attempts - 1:
                break
            retry_after_seconds = getattr(last_error, "retry_after_seconds", None)
            delay_seconds = (
                retry_after_seconds + 1.0
                if retry_after_seconds is not None and 0 < retry_after_seconds <= 15
                else retry_delay_seconds * (2**attempt)
            )
            print(f"Retrying in {delay_seconds:.1f}s…")
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise AnalysisError("LLM analysis failed after all attempts without a captured error.")
