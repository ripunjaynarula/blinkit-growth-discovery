"""Global configuration for the Blinkit Growth Discovery Engine."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_env_var(key: str, default: str = "") -> str:
    """Retrieve an environment variable safely, checking Streamlit secrets first if available."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else default

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Provider Configuration - Cloudflare Workers AI is default and preferred
LLM_PROVIDER = get_env_var("LLM_PROVIDER", "cloudflare").lower()

# Cloudflare Workers AI Credentials & Models
CLOUDFLARE_ACCOUNT_ID = get_env_var("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = get_env_var("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_MODEL = get_env_var("CLOUDFLARE_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

CLOUDFLARE_FALLBACK_CHAIN = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/mistral/mistral-7b-instruct-v0.2",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
]

# Optional Secondary Providers
GROQ_API_KEY = get_env_var("GROQ_API_KEY", "")
GROQ_MODEL = get_env_var("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

OPENROUTER_API_KEY = get_env_var("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = get_env_var("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite-preview-02-05:free")
OPENROUTER_FALLBACK_CHAIN = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "openrouter/free",
]

# Active Default Model Selection
DEFAULT_MODEL = CLOUDFLARE_MODEL if LLM_PROVIDER == "cloudflare" else GROQ_MODEL

# Pipeline Processing Defaults (Updated default limit to 250 reviews per source for 700 total)
DEFAULT_LIMIT = 250
DEFAULT_BATCH_SIZE_FILTER = 10
DEFAULT_BATCH_SIZE_ANALYZE = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_MIN_REVIEW_LENGTH = 10
DEFAULT_MAX_REVIEW_CHARACTERS = 800
DEFAULT_MIN_CONFIDENCE_SUMMARY = 0.60
DEFAULT_TOP_N_THEMES = 5

# Review Source Constants
GOOGLE_PLAY_APP_ID = get_env_var("GOOGLE_PLAY_APP_ID", "com.grofers.customerapp")
GOOGLE_PLAY_COUNTRY = get_env_var("GOOGLE_PLAY_COUNTRY", "in")
GOOGLE_PLAY_LANGUAGE = get_env_var("GOOGLE_PLAY_LANGUAGE", "en")

BLINKIT_APP_STORE_ID = get_env_var("BLINKIT_APP_STORE_ID", "960335206")
APP_STORE_APP_ID = BLINKIT_APP_STORE_ID
APP_STORE_COUNTRY = get_env_var("APP_STORE_COUNTRY", "in")

# File Paths
RAW_REVIEWS_CSV = RAW_DATA_DIR / "raw_reviews.csv"
FILTERED_REVIEWS_CSV = PROCESSED_DATA_DIR / "filtered_reviews.csv"
REJECTED_REVIEWS_CSV = PROCESSED_DATA_DIR / "rejected_reviews.csv"
ANALYZED_REVIEWS_CSV = PROCESSED_DATA_DIR / "analyzed_reviews.csv"
FILTER_SUMMARY_JSON = PROCESSED_DATA_DIR / "filter_summary.json"
THEME_SUMMARY_MD = OUTPUT_DIR / "theme_summary.md"
REPORT_MD = OUTPUT_DIR / "report.md"
EXPORT_BUNDLE_ZIP = OUTPUT_DIR / "blinkit_growth_discovery_bundle.zip"
REDDIT_CACHE_CSV = RAW_DATA_DIR / "reddit_cached_reviews.csv"

BEHAVIORS_JSON = OUTPUT_DIR / "behaviors.json"
BARRIERS_JSON = OUTPUT_DIR / "barriers.json"
REQUIREMENTS_JSON = OUTPUT_DIR / "requirements.json"
THEMES_JSON = OUTPUT_DIR / "themes.json"
ROOT_CAUSES_JSON = OUTPUT_DIR / "root_causes.json"
OPPORTUNITIES_JSON = OUTPUT_DIR / "opportunities.json"
HYPOTHESES_JSON = OUTPUT_DIR / "hypotheses.json"
INTERVIEW_PLANS_JSON = OUTPUT_DIR / "interview_plans.json"
METADATA_JSON = OUTPUT_DIR / "metadata.json"

IGNORE_VALUES = {"unknown", "n/a", "none", "", "null", "nan"}

MODEL_TOKEN_LIMITS = {
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": {"relevance": 1500, "analysis": 2500, "summary": 4000},
    "@cf/meta/llama-3.1-8b-instruct": {"relevance": 1000, "analysis": 1500, "summary": 2000},
    "@cf/mistral/mistral-7b-instruct-v0.2": {"relevance": 1200, "analysis": 1800, "summary": 2500},
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b": {"relevance": 1500, "analysis": 2500, "summary": 3500},
    "llama-3.3-70b-versatile": {"relevance": 1500, "analysis": 2500, "summary": 4000},
    "llama-3.1-8b-instant": {"relevance": 1000, "analysis": 1500, "summary": 2000},
}
