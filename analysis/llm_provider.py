"""LLM Provider Abstraction Layer for Blinkit Growth Discovery Engine.

This module defines an extensible, object-oriented provider interface (LLMProvider)
with Cloudflare Workers AI as the primary default inference provider, alongside optional
Groq and OpenRouter secondary providers.
"""
from __future__ import annotations

import abc
import json
import re
import time
from typing import Any

import requests

import config


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
        is_quota_exceeded: bool = False,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.is_quota_exceeded = is_quota_exceeded


class LLMProvider(abc.ABC):
    """Abstract Base Class for LLM Inference Providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'cloudflare', 'groq', 'openrouter')."""
        pass

    @property
    @abc.abstractmethod
    def default_model(self) -> str:
        """Default model identifier for this provider."""
        pass

    @property
    @abc.abstractmethod
    def fallback_chain(self) -> list[str]:
        """Sequence of candidate models for this provider."""
        pass

    @abc.abstractmethod
    def generate_raw_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        api_key_override: str | None = None,
        account_id_override: str | None = None,
    ) -> tuple[str, float]:
        """Generates raw text response and returns (text_content, latency_seconds)."""
        pass

    @abc.abstractmethod
    def test_connection(
        self,
        api_key: str | None = None,
        model: str | None = None,
        account_id: str | None = None,
    ) -> tuple[bool, str]:
        """Validates API credentials and model availability."""
        pass


class CloudflareWorkersAIProvider(LLMProvider):
    """Cloudflare Workers AI Primary Inference Provider."""

    @property
    def name(self) -> str:
        return "cloudflare"

    @property
    def default_model(self) -> str:
        return config.CLOUDFLARE_MODEL or "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

    @property
    def fallback_chain(self) -> list[str]:
        return config.CLOUDFLARE_FALLBACK_CHAIN

    def _get_credentials(
        self,
        api_key_override: str | None = None,
        account_id_override: str | None = None,
    ) -> tuple[str, str]:
        token = (
            api_key_override
            or config.get_env_var("CLOUDFLARE_API_TOKEN")
            or config.CLOUDFLARE_API_TOKEN
        )
        account_id = (
            account_id_override
            or config.get_env_var("CLOUDFLARE_ACCOUNT_ID")
            or config.CLOUDFLARE_ACCOUNT_ID
        )

        if not token:
            raise LLMProviderError(
                "CLOUDFLARE_API_TOKEN is missing. Please configure it in .env, Streamlit Secrets, or Sidebar."
            )
        if not account_id:
            raise LLMProviderError(
                "CLOUDFLARE_ACCOUNT_ID is missing. Please configure it in .env, Streamlit Secrets, or Sidebar."
            )
        return token, account_id

    def generate_raw_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        api_key_override: str | None = None,
        account_id_override: str | None = None,
    ) -> tuple[str, float]:
        token, account_id = self._get_credentials(api_key_override, account_id_override)
        selected_model = model or self.default_model

        # Ensure model slug is formatted correctly for Cloudflare endpoint
        endpoint_model = selected_model.lstrip("/")
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{endpoint_model}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or 2000,
        }

        start_t = time.perf_counter()
        try:
            res = requests.post(url, headers=headers, json=body, timeout=60)
            latency = time.perf_counter() - start_t
            self._handle_response_status(res)
            
            payload = res.json()
            if not payload.get("success", False) and "result" not in payload:
                err_msg = json.dumps(payload.get("errors", "Cloudflare Workers AI returned an error."))
                raise LLMProviderError(f"Cloudflare Workers AI Error: {err_msg}")

            result = payload.get("result", {})
            if isinstance(result, dict):
                content = result.get("response") or result.get("text") or result.get("content", "")
            else:
                content = str(result or "")

            if not content:
                raise LLMProviderError("Cloudflare Workers AI returned an empty response.")
            return str(content), latency

        except requests.RequestException as exc:
            raise LLMProviderError(f"Cloudflare Workers AI connection error: {exc}") from exc

    def test_connection(
        self,
        api_key: str | None = None,
        model: str | None = None,
        account_id: str | None = None,
    ) -> tuple[bool, str]:
        try:
            token, acc_id = self._get_credentials(api_key, account_id)
            selected_model = (model or self.default_model).lstrip("/")
            url = f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run/{selected_model}"

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            body = {
                "messages": [{"role": "user", "content": "Ping"}],
                "max_tokens": 5,
            }

            res = requests.post(url, headers=headers, json=body, timeout=10)
            if res.status_code == 200:
                payload = res.json()
                if payload.get("success", True):
                    return True, f"Successfully connected to Cloudflare Workers AI model '{selected_model}'."
                return False, f"Cloudflare Workers AI error: {payload.get('errors')}"
            return False, f"Cloudflare API returned HTTP {res.status_code}: {res.text[:200]}"
        except Exception as exc:
            return False, f"Cloudflare Workers AI test failed: {exc}"

    def _handle_response_status(self, response: requests.Response) -> None:
        if response.status_code < 400:
            return

        body_text = response.text.strip()
        retry_after = _parse_retry_after(response.headers.get("Retry-After")) or _parse_retry_after_from_text(body_text)
        is_quota = (
            "10,000 neurons" in body_text.lower()
            or "used up your daily free allocation" in body_text.lower()
            or "4006" in body_text
            or "quota" in body_text.lower()
        )

        if response.status_code == 401:
            raise LLMProviderError(
                f"Cloudflare Authentication failed (HTTP 401). Invalid API Token. Response: {body_text}",
                retry_after_seconds=retry_after,
                is_quota_exceeded=is_quota,
            )
        if response.status_code == 403:
            raise LLMProviderError(
                f"Cloudflare Workers AI Permission Denied (HTTP 403). Check Account ID and Token scope. Response: {body_text}",
                retry_after_seconds=retry_after,
                is_quota_exceeded=is_quota,
            )
        if response.status_code == 404:
            raise LLMProviderError(
                f"Cloudflare Model or Account ID not found (HTTP 404). Response: {body_text}",
                retry_after_seconds=retry_after,
                is_quota_exceeded=is_quota,
            )
        if response.status_code == 429 or is_quota:
            raise LLMProviderError(
                f"Cloudflare Workers AI rate limit / daily free neuron quota exceeded: {body_text}",
                retry_after_seconds=retry_after,
                is_quota_exceeded=is_quota,
            )
        raise LLMProviderError(
            f"Cloudflare API request failed with HTTP {response.status_code}: {body_text}",
            retry_after_seconds=retry_after,
            is_quota_exceeded=is_quota,
        )


class GroqProvider(LLMProvider):
    """Groq Optional Backup Provider."""

    @property
    def name(self) -> str:
        return "groq"

    @property
    def default_model(self) -> str:
        return config.GROQ_MODEL or "llama-3.3-70b-versatile"

    @property
    def fallback_chain(self) -> list[str]:
        return config.GROQ_FALLBACK_CHAIN

    def generate_raw_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        api_key_override: str | None = None,
        account_id_override: str | None = None,
    ) -> tuple[str, float]:
        token = api_key_override or config.get_env_var("GROQ_API_KEY") or config.GROQ_API_KEY
        if not token:
            raise LLMProviderError("GROQ_API_KEY is not set.")

        selected_model = model or self.default_model
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        body = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or 2000,
            "response_format": {"type": "json_object"},
        }

        start_t = time.perf_counter()
        res = requests.post(url, headers=headers, json=body, timeout=60)
        latency = time.perf_counter() - start_t
        if res.status_code >= 400:
            retry_after = _parse_retry_after(res.headers.get("Retry-After")) or _parse_retry_after_from_text(res.text)
            raise LLMProviderError(f"Groq API returned HTTP {res.status_code}: {res.text}", retry_after_seconds=retry_after)

        payload = res.json()
        content = payload["choices"][0]["message"]["content"]
        return str(content), latency

    def test_connection(
        self,
        api_key: str | None = None,
        model: str | None = None,
        account_id: str | None = None,
    ) -> tuple[bool, str]:
        key = api_key or config.GROQ_API_KEY or config.get_env_var("GROQ_API_KEY")
        if not key:
            return False, "GROQ_API_KEY is missing."
        selected_model = model or self.default_model
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": selected_model, "messages": [{"role": "user", "content": "Ping"}], "max_tokens": 5}
        try:
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=10)
            if res.status_code == 200:
                return True, f"Successfully connected to Groq model '{selected_model}'."
            return False, f"Groq API returned HTTP {res.status_code}: {res.text[:200]}"
        except Exception as exc:
            return False, f"Groq connection failed: {exc}"


class OpenRouterProvider(LLMProvider):
    """OpenRouter Optional Backup Provider."""

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def default_model(self) -> str:
        return config.OPENROUTER_MODEL or "google/gemini-2.0-flash-lite-preview-02-05:free"

    @property
    def fallback_chain(self) -> list[str]:
        return config.OPENROUTER_FALLBACK_CHAIN

    def generate_raw_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        api_key_override: str | None = None,
        account_id_override: str | None = None,
    ) -> tuple[str, float]:
        token = api_key_override or config.get_env_var("OPENROUTER_API_KEY") or config.OPENROUTER_API_KEY
        if not token:
            raise LLMProviderError("OPENROUTER_API_KEY is not set.")

        selected_model = model or self.default_model
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://blinkit-growth-discovery.local",
            "X-Title": "Blinkit Growth Discovery Engine",
        }
        body = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or 2000,
            "response_format": {"type": "json_object"},
        }

        start_t = time.perf_counter()
        res = requests.post(url, headers=headers, json=body, timeout=60)
        latency = time.perf_counter() - start_t
        if res.status_code >= 400:
            retry_after = _parse_retry_after(res.headers.get("Retry-After")) or _parse_retry_after_from_text(res.text)
            raise LLMProviderError(f"OpenRouter API returned HTTP {res.status_code}: {res.text}", retry_after_seconds=retry_after)

        payload = res.json()
        content = payload["choices"][0]["message"]["content"]
        return str(content), latency

    def test_connection(
        self,
        api_key: str | None = None,
        model: str | None = None,
        account_id: str | None = None,
    ) -> tuple[bool, str]:
        key = api_key or config.OPENROUTER_API_KEY or config.get_env_var("OPENROUTER_API_KEY")
        if not key:
            return False, "OPENROUTER_API_KEY is missing."
        selected_model = model or self.default_model
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://blinkit-growth-discovery.local",
            "X-Title": "Blinkit Growth Discovery Engine",
        }
        body = {"model": selected_model, "messages": [{"role": "user", "content": "Ping"}], "max_tokens": 5}
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=10)
            if res.status_code == 200:
                return True, f"Successfully connected to OpenRouter model '{selected_model}'."
            return False, f"OpenRouter API returned HTTP {res.status_code}: {res.text[:200]}"
        except Exception as exc:
            return False, f"OpenRouter connection failed: {exc}"


_PROVIDERS: dict[str, LLMProvider] = {
    "cloudflare": CloudflareWorkersAIProvider(),
    "groq": GroqProvider(),
    "openrouter": OpenRouterProvider(),
}


def get_llm_provider(provider_name: str | None = None) -> LLMProvider:
    """Returns the requested LLMProvider instance (defaults to Cloudflare Workers AI)."""
    p_name = (provider_name or config.LLM_PROVIDER or "cloudflare").strip().lower()
    return _PROVIDERS.get(p_name, _PROVIDERS["cloudflare"])


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _parse_retry_after_from_text(text: str | None) -> float | None:
    if not text:
        return None
    h_m_s = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
    if h_m_s:
        hours = float(h_m_s.group(1) or 0)
        minutes = float(h_m_s.group(2) or 0)
        seconds = float(h_m_s.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    sec_match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s", text, re.IGNORECASE)
    if sec_match:
        try:
            return float(sec_match.group(1))
        except ValueError:
            pass
    return None
