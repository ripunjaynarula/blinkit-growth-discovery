"""Blinkit Growth Discovery Engine – Streamlit application.

This module wires together the quick-commerce discovery analysis pipeline into a PM-facing
web application for Blinkit. It orchestrates review collection, relevance filtering,
LLM-based insight extraction via Cloudflare Workers AI, and theme report generation.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

# ── pipeline imports ──────────────────────────────────────────────────────────
from analysis.analyze_reviews import analyze_reviews, write_analyzed_reviews
from analysis.filter_reviews import (
    build_rejected_reviews,
    classify_relevance,
    filter_relevant_reviews,
    prepare_reviews_for_llm,
    read_raw_reviews,
    write_filtered_reviews,
    write_rejected_reviews,
)
from analysis.llm_client import test_provider_connection, test_groq_connection, estimate_tokens
from analysis.theme_summary import (
    build_theme_summary,
    read_analyzed_reviews,
    write_theme_summary,
)
import analysis.theme_summary as theme_summary_mod

def generate_all_artifacts(df, top_n=10):
    fn = getattr(theme_summary_mod, "generate_all_artifacts", None)
    if callable(fn):
        return fn(df, top_n)

from reviews.google_play import collect_google_play_reviews, GooglePlayCollector
from reviews.app_store import collect_app_store_reviews, AppStoreCollector
from reviews.reddit import collect_reddit_reviews, RedditCollector
from reviews.upload import parse_csv_reviews, parse_json_reviews, parse_manual_reviews
from reviews.utils import write_raw_reviews_csv
from utils.export import create_export_zip_bundle

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & BLINKIT THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blinkit Growth Discovery Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Blinkit Yellow #F7C600, Blinkit Green #0C831F, Outfit Font, Wrapped Logs)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, button, input {
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Blinkit Brand Yellow Primary Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #F7C600 !important;
        color: #121212 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #e5b600 !important;
        color: #000000 !important;
        box-shadow: 0 4px 12px rgba(247, 198, 0, 0.35) !important;
    }

    /* Secondary Buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #1c1c1c !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #0C831F !important;
        color: #0C831F !important;
    }

    /* Sidebar Theme */
    section[data-testid="stSidebar"] {
        background-color: #0d0d0d !important;
        border-right: 1px solid #222222 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #181818;
        border-radius: 6px;
        padding: 8px 16px;
        color: #b3b3b3;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F7C600 !important;
        color: #121212 !important;
        font-weight: 600 !important;
    }

    /* Execution Logs Word Wrap & Vertical Scroll Fix (No Horizontal Overflow) */
    .stCodeBlock code, div[data-testid="stCodeBlock"] pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-x: hidden !important;
        max-height: 450px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# TYPES & SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
SourceStatus = Literal["success", "cached", "skipped", "failed"]

if "pipeline_paused" not in st.session_state:
    st.session_state["pipeline_paused"] = False

if "last_logs" not in st.session_state:
    st.session_state["last_logs"] = ""


@dataclass
class CollectionResult:
    source: str
    status: SourceStatus
    reviews: list = field(default_factory=list)
    error: str = ""

    @property
    def count(self) -> int:
        return len(self.reviews)

    @property
    def mode(self) -> str:
        return "Cached" if self.status == "cached" else "Live"


# ─────────────────────────────────────────────────────────────────────────────
# LOG CAPTURE WITH IN-PLACE REPLACEMENT & CLEAN LINE WRAPPING
# ─────────────────────────────────────────────────────────────────────────────
class _LogCapture:
    def __init__(self, log_placeholder, notify_placeholder=None):
        self._log = log_placeholder
        self._notify = notify_placeholder
        self._buf = io.StringIO()

    def write(self, text: str) -> None:
        if not text:
            return
        
        if "\r" in text and "\n" not in text:
            parts = text.split("\r")
            text = parts[-1]

        self._buf.write(text)
        current_val = self._buf.getvalue()
        st.session_state["last_logs"] = current_val
        self._log.code(current_val, language="text")
        if self._notify and text.strip():
            self._surface_retry_notice(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def _surface_retry_notice(self, text: str) -> None:
        if "LLM returned no result for review ids" in text or "LLM returned no result" in text:
            try:
                ids_raw = text.split("review ids:")[-1].strip()
                count = len([x for x in ids_raw.split(",") if x.strip()])
                msg = (
                    f"Retrying {count} review(s) because AI returned "
                    "an incomplete response."
                )
            except Exception:
                msg = "Retrying reviews because AI returned an incomplete response."
            self._notify.warning(msg)
        elif "Auto-Pause" in text or "Rate limit" in text or "Model Rotation" in text:
            try:
                msg = text.strip()
                self._notify.info(f"⚡ AI Rate Limit Event: {msg}")
            except Exception:
                pass

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._buf.getvalue()


@contextlib.contextmanager
def capture_logs(log_placeholder, notify_placeholder=None):
    capture = _LogCapture(log_placeholder, notify_placeholder)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = capture
    sys.stderr = capture
    try:
        yield capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ─────────────────────────────────────────────────────────────────────────────
# REDDIT FAULT-TOLERANT COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
def collect_reddit_fault_tolerant(limit: int) -> CollectionResult:
    try:
        reviews = collect_reddit_reviews(limit=limit)
        if reviews:
            df = pd.DataFrame([r.__dict__ for r in reviews])
            df.to_csv(config.REDDIT_CACHE_CSV, index=False)
            return CollectionResult(source="reddit", status="success", reviews=reviews)
        raise RuntimeError("Live Reddit collection returned 0 reviews (likely HTTP 403 blocked).")
    except Exception as exc:
        error_str = str(exc)
        is_blocked = (
            "403" in error_str
            or "Forbidden" in error_str
            or "ConnectionError" in error_str
            or "timeout" in error_str.lower()
            or "0 reviews" in error_str
        )

        if config.REDDIT_CACHE_CSV.exists():
            try:
                from reviews.models import RawReview
                df = pd.read_csv(config.REDDIT_CACHE_CSV, dtype=str).fillna("")
                cached = [
                    RawReview(
                        id=row.get("id", ""), source="reddit", review=row.get("review", ""),
                        rating=None, date=row.get("date", ""), url=row.get("url", "")
                    )
                    for _, row in df.iterrows() if row.get("review", "").strip()
                ]
                return CollectionResult(source="reddit", status="cached", reviews=cached, error=error_str)
            except Exception as cache_exc:
                return CollectionResult(source="reddit", status="failed", error=f"Live failed ({error_str}); cache unreadable ({cache_exc})")

        return CollectionResult(source="reddit", status="skipped" if is_blocked else "failed", error=error_str)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def load_json(path: Path) -> dict | list | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def get_mode(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "unknown"
    series = df[~df[col].astype(str).str.lower().isin(config.IGNORE_VALUES)][col]
    return str(series.mode().iloc[0]) if not series.empty else "unknown"


def clean_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    return counts[~counts[col].astype(str).str.lower().isin(config.IGNORE_VALUES)]


def bar_chart(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    data = clean_counts(df, col).sort_values("count", ascending=True)
    fig = px.bar(data, x="count", y=col, orientation="h",
                 title=title, template="plotly_dark")
    fig.update_traces(marker_color="#F7C600")
    fig.update_layout(
        xaxis_title="Count", yaxis_title=None,
        height=350, margin=dict(l=180, r=20, t=44, b=40),
    )
    return fig


def donut_chart(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    data = clean_counts(df, col)
    fig = px.pie(data, values="count", names=col, hole=0.5,
                 title=title, template="plotly_dark",
                 color_discrete_sequence=["#F7C600", "#0C831F", "#E5B000", "#15A02B", "#1F7A33"])
    fig.update_layout(margin=dict(t=44, b=40, l=40, r=40), height=350)
    return fig


def insight_card(label: str, value: str, accent: str = "#F7C600") -> str:
    return f"""
    <div style="background:#181818;padding:18px 20px;border-radius:8px;
                border-left:4px solid {accent};height:100%;">
      <p style="color:#b3b3b3;margin:0;font-size:.78em;text-transform:uppercase;
                letter-spacing:.06em;">{label}</p>
      <p style="color:#fff;margin:6px 0 0 0;font-size:1.05em;
                font-weight:500;line-height:1.35;">{value}</p>
    </div>"""


def onboarding_card(icon: str, heading: str, body: str) -> str:
    return f"""
    <div style="background:#181818;padding:24px;border-radius:10px;
                border:1px solid #282828;text-align:center;">
      <div style="font-size:2em;margin-bottom:10px;">{icon}</div>
      <h4 style="color:#F7C600;margin:0 0 8px 0;font-weight:600;">{heading}</h4>
      <p style="color:#b3b3b3;margin:0;font-size:.9em;line-height:1.55;">{body}</p>
    </div>"""


def elapsed(start: float) -> str:
    secs = int(time.perf_counter() - start)
    return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"


def handle_runtime_error(
    exc: Exception,
    *,
    stage_status=None,
    progress=None,
    log_capture=None,
    message: str = "Operation failed",
) -> None:
    if log_capture is not None:
        log_capture.write("\n\n=== FULL EXCEPTION TRACEBACK ===\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=log_capture)
    if stage_status is not None:
        stage_status.error(f"{message}: {exc}")
    st.exception(exc)
    if progress is not None:
        progress.empty()


# ─────────────────────────────────────────────────────────────────────────────
# ARGS SHIMS
# ─────────────────────────────────────────────────────────────────────────────
class _FilterArgs:
    input = config.RAW_REVIEWS_CSV
    output = config.FILTERED_REVIEWS_CSV
    batch_size = config.DEFAULT_BATCH_SIZE_FILTER
    model = config.DEFAULT_MODEL
    max_retries = config.DEFAULT_MAX_RETRIES
    retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
    min_review_length = config.DEFAULT_MIN_REVIEW_LENGTH
    min_confidence = 0.0
    api_key = config.CLOUDFLARE_API_TOKEN
    account_id = config.CLOUDFLARE_ACCOUNT_ID


class _AnalyzeArgs:
    input = config.FILTERED_REVIEWS_CSV
    output = config.ANALYZED_REVIEWS_CSV
    batch_size = config.DEFAULT_BATCH_SIZE_ANALYZE
    model = config.DEFAULT_MODEL
    max_retries = config.DEFAULT_MAX_RETRIES
    retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
    continue_on_error = True
    api_key = config.CLOUDFLARE_API_TOKEN
    account_id = config.CLOUDFLARE_ACCOUNT_ID


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES WITH SAFE PROGRESS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def run_stage_collect(
    sources: list[str],
    limit: int = config.DEFAULT_LIMIT,
    uploaded_file=None,
    manual_text: str = "",
) -> tuple[list[CollectionResult], int]:
    results: list[CollectionResult] = []
    all_reviews = []

    if "google_play" in sources:
        try:
            reviews = collect_google_play_reviews(
                limit=limit,
                country=config.GOOGLE_PLAY_COUNTRY,
                language=config.GOOGLE_PLAY_LANGUAGE,
            )
            results.append(CollectionResult("google_play", "success", reviews))
            all_reviews.extend(reviews)
        except Exception as exc:
            results.append(CollectionResult("google_play", "failed", error=str(exc)))

    if "app_store" in sources:
        try:
            reviews = collect_app_store_reviews(limit=limit)
            results.append(CollectionResult("app_store", "success", reviews))
            all_reviews.extend(reviews)
        except Exception as exc:
            results.append(CollectionResult("app_store", "failed", error=str(exc)))

    if "reddit" in sources:
        res = collect_reddit_fault_tolerant(limit)
        results.append(res)
        all_reviews.extend(res.reviews)

    if "upload" in sources and uploaded_file is not None:
        try:
            content = uploaded_file.read()
            if uploaded_file.name.endswith(".json"):
                reviews = parse_json_reviews(content, limit)
            else:
                reviews = parse_csv_reviews(content, limit)
            results.append(CollectionResult("upload", "success", reviews))
            all_reviews.extend(reviews)
        except Exception as exc:
            results.append(CollectionResult("upload", "failed", error=str(exc)))

    if "manual" in sources and manual_text.strip():
        try:
            reviews = parse_manual_reviews(manual_text, limit)
            results.append(CollectionResult("manual", "success", reviews))
            all_reviews.extend(reviews)
        except Exception as exc:
            results.append(CollectionResult("manual", "failed", error=str(exc)))

    if not all_reviews:
        raise RuntimeError("All configured review sources failed to return data.")

    write_raw_reviews_csv(all_reviews, config.RAW_REVIEWS_CSV)
    return results, len(all_reviews)


def run_stage_filter(args=None, progress_callback=None) -> dict:
    args = args or _FilterArgs()
    
    raw = read_raw_reviews(args.input)
    reviews_for_llm, removal_counts = prepare_reviews_for_llm(
        raw, min_review_length=args.min_review_length
    )
    
    try:
        relevance_rows = classify_relevance(reviews_for_llm, args, progress_callback=progress_callback)
    except TypeError:
        relevance_rows = classify_relevance(reviews_for_llm, args)

    filtered = filter_relevant_reviews(reviews_for_llm, relevance_rows, args.min_confidence)
    rejected = build_rejected_reviews(reviews_for_llm, relevance_rows)
    write_filtered_reviews(filtered, args.output)
    write_rejected_reviews(rejected, config.REJECTED_REVIEWS_CSV)

    conf_vals = [float(r["confidence"]) for r in relevance_rows]
    irr = sum(1 for r in relevance_rows if not (r.get("is_relevant") is True or r.get("relevant") is True))
    summary = {
        "total_reviews": len(raw),
        "reviews_sent_to_llm": len(reviews_for_llm),
        "relevant_reviews": len(filtered),
        "irrelevant_reviews": irr,
        "average_confidence": sum(conf_vals) / len(conf_vals) if conf_vals else 0.0,
        "model": getattr(args, "model", config.DEFAULT_MODEL),
    }
    config.FILTER_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage_analyze(args=None, progress_callback=None) -> int:
    args = args or _AnalyzeArgs()

    df = pd.read_csv(args.input, dtype={"id": str})
    df = df[df["review"].astype(str).str.strip() != ""].copy()
    
    try:
        rows = analyze_reviews(df, args, progress_callback=progress_callback)
    except TypeError:
        rows = analyze_reviews(df, args)

    write_analyzed_reviews(rows, args.output)
    return len(rows)


def run_stage_summary(min_conf: float = None, top_n: int = None) -> None:
    min_conf = min_conf if min_conf is not None else config.DEFAULT_MIN_CONFIDENCE_SUMMARY
    top_n = top_n or config.DEFAULT_TOP_N_THEMES
    analyzed = read_analyzed_reviews(config.ANALYZED_REVIEWS_CSV, min_conf)
    md = build_theme_summary(analyzed, top_n)
    write_theme_summary(md, config.THEME_SUMMARY_MD)
    generate_all_artifacts(analyzed, top_n)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE STATUS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def render_source_status_table(results: list[CollectionResult]) -> None:
    rows = []
    for r in results:
        status_icon = {
            "success": "✅ Success",
            "cached": "⚠ Cached",
            "skipped": "⏭ Skipped",
            "failed": "❌ Failed",
        }.get(r.status, r.status)
        rows.append({
            "Source": r.source.replace("_", " ").title(),
            "Status": status_icon,
            "Reviews Collected": r.count,
            "Mode": r.mode,
            "Error": r.error or "—",
        })
    st.table(pd.DataFrame(rows))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (CLEAN PRODUCTION NAVIGATION)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:#F7C600;margin-bottom:0;'>Blinkit Growth</h2>", unsafe_allow_html=True)
st.sidebar.caption("AI Quick-Commerce Catalog Discovery Platform")
st.sidebar.markdown("---")

provider_key = config.LLM_PROVIDER
active_model = config.DEFAULT_MODEL
_FilterArgs.account_id = config.CLOUDFLARE_ACCOUNT_ID
_FilterArgs.api_key = config.CLOUDFLARE_API_TOKEN
_FilterArgs.model = config.CLOUDFLARE_MODEL
_AnalyzeArgs.account_id = config.CLOUDFLARE_ACCOUNT_ID
_AnalyzeArgs.api_key = config.CLOUDFLARE_API_TOKEN
_AnalyzeArgs.model = config.CLOUDFLARE_MODEL

navigation_page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Collect Reviews", "Filter Reviews",
     "Analyze Reviews", "Theme Summary", "Research Artifacts", "Outputs"],
)

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Powered by Cloudflare Workers AI")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────
st.info(
    "💡 **Best Results:** For the freshest review collection, run this workflow locally. "
    "Cloud deployments automatically fall back to cached snapshots for sources restricting cloud scraper requests."
)

df_raw = load_csv(config.RAW_REVIEWS_CSV)
df_filt = load_csv(config.FILTERED_REVIEWS_CSV)
df_rej = load_csv(config.REJECTED_REVIEWS_CSV)
df_ana = load_csv(config.ANALYZED_REVIEWS_CSV)

pipeline_started = df_raw is not None and len(df_raw) > 0


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if navigation_page == "Dashboard":
    st.title("Blinkit Growth Discovery Engine")
    st.markdown(
        "Executive product intelligence derived from Blinkit product search, catalog aisles, and recommendation feedback."
    )

    st.markdown("---")
    run_clicked = st.button(
        "Run Complete Pipeline", type="primary"
    )

    if run_clicked:
        t0 = time.perf_counter()
        progress = st.progress(0.0, text="Starting…")
        stage_status = st.empty()
        notify = st.empty()
        
        logs_exp = st.expander("Execution Logs & Real-Time Step Progress", expanded=True)
        with logs_exp:
            st.markdown("#### 📊 Live Processing Heartbeat & Stepwise Progress")
            sub_progress_ph = st.empty()
            status_metrics_ph = st.empty()
            st.markdown("#### 📜 Detailed Execution Log Stream")
            log_placeholder = st.empty()

        def make_progress_cb(stage_name: str):
            def cb(completed: int, total: int, text: str):
                pct = min(1.0, max(0.0, completed / total)) if total > 0 else 0.0
                sub_progress_ph.progress(pct, text=f"Stepwise Progress ({stage_name}): {completed}/{total} reviews ({int(pct*100)}%)")
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                status_metrics_ph.markdown(
                    f"""
                    <div style="background:#141414;padding:12px 16px;border-radius:8px;border:1px solid #282828;margin-bottom:12px;">
                        <span style="color:#F7C600;font-weight:600;">⚡ Active Stage:</span> {stage_name} &nbsp;|&nbsp;
                        <span style="color:#0C831F;font-weight:600;">📊 Processed:</span> {completed} / {total} ({int(pct*100)}%) &nbsp;|&nbsp;
                        <span style="color:#FFFFFF;">🤖 AI Engine:</span> <code>Cloudflare Workers AI</code> &nbsp;|&nbsp;
                        <span style="color:#888;">🕒 Last Heartbeat:</span> {now_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            return cb

        with capture_logs(log_placeholder, notify) as log_capture:
            try:
                # 1 / 4 – Collect
                stage_status.info(f"**Stage 1/4** — Collecting {config.DEFAULT_LIMIT} reviews per active source…")
                progress.progress(0.05, text="Collecting…")
                col_results, total_collected = run_stage_collect(
                    ["google_play", "app_store", "reddit"],
                    config.DEFAULT_LIMIT,
                )
                progress.progress(0.25, text=f"Collected {total_collected} reviews ({elapsed(t0)})")
                stage_status.success(f"**Stage 1/4 complete** — {total_collected} reviews collected.")

                # 2 / 4 – Filter
                stage_status.info(f"**Stage 2/4** — Filtering reviews for catalog discovery relevance via Cloudflare Workers AI…")
                progress.progress(0.3, text="Filtering…")
                filter_summary = run_stage_filter(progress_callback=make_progress_cb("Relevance Filtering"))
                n_rel = filter_summary["relevant_reviews"]
                progress.progress(0.55, text=f"{n_rel} relevant reviews ({elapsed(t0)})")
                stage_status.success(f"**Stage 2/4 complete** — {n_rel} relevant reviews kept.")

                # 3 / 4 – Analyze
                stage_status.info(f"**Stage 3/4** — Extracting product insights with Cloudflare Workers AI…")
                progress.progress(0.6, text="Analyzing…")
                n_analyzed = run_stage_analyze(progress_callback=make_progress_cb("Insight Extraction"))
                progress.progress(0.85, text=f"{n_analyzed} reviews analyzed ({elapsed(t0)})")
                stage_status.success(f"**Stage 3/4 complete** — {n_analyzed} reviews analyzed.")

                # 4 / 4 – Summary & Artifact Generation
                stage_status.info("**Stage 4/4** — Generating Growth Discovery report & 12 artifacts…")
                progress.progress(0.9, text="Summarizing…")
                run_stage_summary()
                progress.progress(1.0, text=f"Done in {elapsed(t0)}")
                stage_status.success(
                    f"**Pipeline complete** — finished in {elapsed(t0)}. "
                    "Refresh the page to see updated insights."
                )
                notify.empty()

            except Exception as exc:
                handle_runtime_error(
                    exc,
                    stage_status=stage_status,
                    progress=progress,
                    log_capture=log_capture,
                    message="Pipeline failed",
                )

        if st.session_state["last_logs"]:
            with logs_exp:
                st.markdown("#### 📋 Copyable Execution Log Text")
                st.text_area("Select all and copy (Ctrl+A, Ctrl+C):", st.session_state["last_logs"], height=200)

        st.markdown("---")

    elif st.session_state["last_logs"].strip():
        with st.expander("Previous Execution Logs", expanded=False):
            st.code(st.session_state["last_logs"], language="text")

    # ── Onboarding ────────────────────────────────────────────────────────────
    if not pipeline_started:
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            onboarding_card("📥", "Step 1 — Collect",
                            f"Click 'Run Complete Pipeline' above, or navigate to "
                            f"'Collect Reviews' to pull {config.DEFAULT_LIMIT} feedback entries each from Google Play, "
                            f"App Store, Reddit, or Upload custom CSV/JSON files."),
            unsafe_allow_html=True,
        )
        c2.markdown(
            onboarding_card("🔍", "Step 2 — Filter & Analyze",
                            "The Cloudflare Workers AI relevance filter removes off-topic posts. "
                            "The insight extractor then tags each review with pain "
                            "points, root causes, shopper segments and emotions."),
            unsafe_allow_html=True,
        )
        c3.markdown(
            onboarding_card("📊", "Step 3 — Explore & Export",
                            "Once the pipeline has run, explore all 12 structured JSON "
                            "artifacts, executive reports, and download the full ZIP bundle."),
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Executive insights ────────────────────────────────────────────────────
    if df_ana is not None and len(df_ana) > 0:
        st.subheader("Executive Customer Behavior Insights")

        top_pp = get_mode(df_ana, "pain_point")
        top_rc = get_mode(df_ana, "root_cause")
        top_seg = get_mode(df_ana, "user_segment")
        top_surf = get_mode(df_ana, "discovery_surface")

        cols = st.columns(4)
        for col, lbl, val in zip(
            cols,
            ["Top Pain Point", "Top Root Cause", "Critical Shopper Segment", "Primary Discovery Surface"],
            [top_pp, top_rc, top_seg, top_surf],
        ):
            col.markdown(insight_card(lbl, val, accent="#F7C600"), unsafe_allow_html=True)

        if "unknown" not in (top_pp, top_rc, top_seg, top_surf):
            opp = (
                f"Address **{top_rc}** on the **{top_surf}** surface "
                f"to resolve **{top_pp}** for **{top_seg}** users — "
                "How might we improve Blinkit's catalog discovery so users easily find new categories and increase their average order value?"
            )
        else:
            opp = "How might we improve Blinkit's catalog discovery so users easily find new categories and increase their average order value?"

        st.markdown(
            f"""<div style="background:#181818;padding:18px 22px;border-radius:8px;
                border-top:3px solid #F7C600;margin:18px 0 28px 0;">
              <p style="color:#b3b3b3;margin:0;font-size:.78em;text-transform:uppercase;
                        letter-spacing:.06em;">Largest Quick-Commerce Growth Opportunity</p>
              <p style="color:#fff;margin:8px 0 0 0;font-size:1.05em;line-height:1.5;">
                {opp}</p></div>""",
            unsafe_allow_html=True,
        )

    # ── Funnel & Diagram ──────────────────────────────────────────────────────
    raw_n = len(df_raw) if df_raw is not None else 0
    filt_n = len(df_filt) if df_filt is not None else 0
    ana_n = len(df_ana) if df_ana is not None else 0
    pre_n = raw_n

    if config.FILTER_SUMMARY_JSON.exists():
        try:
            pre_n = json.loads(
                config.FILTER_SUMMARY_JSON.read_text(encoding="utf-8")
            ).get("reviews_sent_to_llm", raw_n)
        except Exception:
            pass

    st.subheader("Review Processing Funnel")
    fig_funnel = go.Figure(go.Funnel(
        y=["Raw Collection", "Passed Pre-filter", "Relevance Filtered", "Analyzed"],
        x=[raw_n, pre_n, filt_n, ana_n],
        textinfo="value+percent initial",
        marker=dict(color=["#3d3d3d", "#2a2a2a", "#0C831F", "#F7C600"]),
    ))
    fig_funnel.update_layout(
        template="plotly_dark", height=300,
        margin=dict(t=20, b=20, l=160, r=160),
    )
    st.plotly_chart(fig_funnel, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – COLLECT REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Collect Reviews":
    st.title("Review Collection")
    st.markdown(
        "Pull customer feedback from Google Play, Apple App Store, Reddit, or Upload custom CSV/JSON files and manual entries."
    )

    sources = st.multiselect(
        "Sources to collect from",
        ["google_play", "app_store", "reddit", "upload", "manual"],
        default=["google_play", "app_store", "reddit"],
    )
    limit = st.slider("Max reviews per source", 10, 1000, config.DEFAULT_LIMIT)

    uploaded_file = None
    if "upload" in sources:
        uploaded_file = st.file_uploader("Upload CSV or JSON review file", type=["csv", "json"])

    manual_text = ""
    if "manual" in sources:
        manual_text = st.text_area("Enter manual review feedback (one per line)", placeholder="Great quick delivery but search for organic veggies was confusing...")

    if st.button("Run Collection", width="stretch", type="primary"):
        if not sources:
            st.error("Select at least one source.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0, text="Initialising…")
            stage_status = st.empty()
            logs_exp = st.expander("Execution Logs", expanded=True)
            with logs_exp:
                log_ph = st.empty()

            with capture_logs(log_ph) as log_capture:
                try:
                    stage_status.info("Connecting to review sources…")
                    progress.progress(0.1)
                    col_results, total = run_stage_collect(sources, limit, uploaded_file=uploaded_file, manual_text=manual_text)
                    progress.progress(1.0, text=f"Done — {total} reviews ({elapsed(t0)})")
                    stage_status.success(f"Collected **{total} reviews** in {elapsed(t0)}.")
                    render_source_status_table(col_results)
                    st.rerun()

                except RuntimeError as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Collection failed",
                    )
                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Unexpected error",
                    )

    if df_raw is not None:
        st.subheader("Raw Reviews Preview")
        st.dataframe(df_raw, width="stretch")

        if "source" in df_raw.columns:
            st.subheader("Source Distribution")
            df_src = df_raw["source"].value_counts().reset_index()
            df_src.columns = ["source", "count"]
            fig = px.pie(
                df_src, values="count", names="source", hole=0.5,
                template="plotly_dark",
                color_discrete_sequence=["#F7C600", "#0C831F", "#E5B000", "#15A02B", "#1F7A33"],
            )
            fig.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – FILTER REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Filter Reviews":
    st.title("Relevance Filtering")
    st.markdown(
        "Removes off-topic reviews to focus strictly on grocery/catalog discovery, search relevance, aisle navigation, and recommendations using Cloudflare Workers AI."
    )

    c1, c2 = st.columns(2)
    batch_size = c1.number_input("Batch size", 1, 200, config.DEFAULT_BATCH_SIZE_FILTER)
    min_len = c2.number_input("Min review length (chars)", 1, 500, config.DEFAULT_MIN_REVIEW_LENGTH)
    min_conf = st.slider("Min relevance confidence", 0.0, 1.0, 0.0, 0.05)

    if st.button("Run Relevance Filtering", width="stretch", type="primary"):
        if df_raw is None:
            st.error("Collect reviews first.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0, text="Loading…")
            stage_status = st.empty()
            notify = st.empty()
            logs_exp = st.expander("Execution Logs & Real-Time Step Progress", expanded=True)
            with logs_exp:
                st.markdown("#### 📊 Live Processing Heartbeat & Stepwise Progress")
                sub_progress_ph = st.empty()
                status_metrics_ph = st.empty()
                st.markdown("#### 📜 Detailed Execution Log Stream")
                log_ph = st.empty()

            def make_progress_cb(stage_name: str):
                def cb(completed: int, total: int, text: str):
                    pct = min(1.0, max(0.0, completed / total)) if total > 0 else 0.0
                    sub_progress_ph.progress(pct, text=f"Stepwise Progress: {completed}/{total} reviews ({int(pct*100)}%)")
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    status_metrics_ph.markdown(
                        f"""
                        <div style="background:#141414;padding:12px 16px;border-radius:8px;border:1px solid #282828;margin-bottom:12px;">
                            <span style="color:#F7C600;font-weight:600;">⚡ Active Task:</span> {stage_name} &nbsp;|&nbsp;
                            <span style="color:#0C831F;font-weight:600;">📊 Processed:</span> {completed} / {total} ({int(pct*100)}%) &nbsp;|&nbsp;
                            <span style="color:#FFFFFF;">🤖 Model:</span> <code>{active_model}</code> &nbsp;|&nbsp;
                            <span style="color:#888;">🕒 Last Heartbeat:</span> {now_str}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                return cb

            with capture_logs(log_ph, notify) as log_capture:
                try:
                    class _Args(_FilterArgs):
                        pass
                    _Args.batch_size = int(batch_size)
                    _Args.min_review_length = int(min_len)
                    _Args.min_confidence = float(min_conf)

                    stage_status.info("Stage 1/3 — Pre-filtering by catalog keywords…")
                    progress.progress(0.2)
                    stage_status.info(f"Stage 2/3 — Running Cloudflare Workers AI relevance assessment ({active_model})…")
                    progress.progress(0.4)
                    summary = run_stage_filter(_Args(), progress_callback=make_progress_cb("Relevance Filtering"))

                    stage_status.info("Stage 3/3 — Writing datasets…")
                    progress.progress(0.9)

                    n_rel = summary["relevant_reviews"]
                    progress.progress(1.0, text=f"Done ({elapsed(t0)})")
                    stage_status.success(
                        f"**Filtering complete** — {n_rel} relevant reviews retained "
                        f"out of {summary['total_reviews']} in {elapsed(t0)}."
                    )
                    notify.empty()
                    st.rerun()

                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Filtering failed",
                    )

    if df_filt is not None:
        st.subheader("Filtered Reviews Preview")
        st.dataframe(df_filt, width="stretch")

        rej_n = len(df_rej) if df_rej is not None else 0
        filt_n = len(df_filt)
        fig = px.pie(
            pd.DataFrame({"label": ["Relevant", "Irrelevant"],
                          "n": [filt_n, rej_n]}),
            values="n", names="label", hole=0.5,
            title="Relevance Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#F7C600", "#3d3d3d"],
        )
        fig.update_layout(height=300, margin=dict(t=44, b=40, l=40, r=40))
        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – ANALYZE REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Analyze Reviews":
    st.title("Insight Extraction")
    st.markdown(
        "Tags each relevant review with structured labels: pain point, root cause, "
        "discovery surface, shopper segment, emotion, and LLM confidence."
    )

    c1, c2 = st.columns(2)
    batch_size = c1.number_input("Batch size", 1, 50, config.DEFAULT_BATCH_SIZE_ANALYZE)
    cont_err = c2.checkbox("Continue on partial AI errors", value=True)

    if st.button("Run Insight Extraction", width="stretch", type="primary"):
        if df_filt is None:
            st.error("Run relevance filtering first.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0, text="Loading filtered data…")
            stage_status = st.empty()
            notify = st.empty()
            logs_exp = st.expander("Execution Logs & Real-Time Step Progress", expanded=True)
            with logs_exp:
                st.markdown("#### 📊 Live Processing Heartbeat & Stepwise Progress")
                sub_progress_ph = st.empty()
                status_metrics_ph = st.empty()
                st.markdown("#### 📜 Detailed Execution Log Stream")
                log_ph = st.empty()

            def make_progress_cb(stage_name: str):
                def cb(completed: int, total: int, text: str):
                    pct = min(1.0, max(0.0, completed / total)) if total > 0 else 0.0
                    sub_progress_ph.progress(pct, text=f"Stepwise Progress: {completed}/{total} reviews ({int(pct*100)}%)")
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    status_metrics_ph.markdown(
                        f"""
                        <div style="background:#141414;padding:12px 16px;border-radius:8px;border:1px solid #282828;margin-bottom:12px;">
                            <span style="color:#F7C600;font-weight:600;">⚡ Active Task:</span> {stage_name} &nbsp;|&nbsp;
                            <span style="color:#0C831F;font-weight:600;">📊 Processed:</span> {completed} / {total} ({int(pct*100)}%) &nbsp;|&nbsp;
                            <span style="color:#FFFFFF;">🤖 Model:</span> <code>{active_model}</code> &nbsp;|&nbsp;
                            <span style="color:#888;">🕒 Last Heartbeat:</span> {now_str}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                return cb

            with capture_logs(log_ph, notify) as log_capture:
                try:
                    class _Args(_AnalyzeArgs):
                        pass
                    _Args.batch_size = int(batch_size)
                    _Args.continue_on_error = cont_err

                    stage_status.info(f"Sending batches to Cloudflare Workers AI ({active_model}) for classification…")
                    progress.progress(0.2)
                    n = run_stage_analyze(_Args(), progress_callback=make_progress_cb("Insight Extraction"))
                    progress.progress(1.0, text=f"Done — {n} reviews ({elapsed(t0)})")
                    stage_status.success(
                        f"**Analysis complete** — {n} reviews classified in {elapsed(t0)}."
                    )
                    notify.empty()
                    st.rerun()

                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Analysis failed",
                    )

    if df_ana is not None:
        st.subheader("Analyzed Reviews Preview")
        st.dataframe(df_ana, width="stretch")

        st.markdown("---")
        st.subheader("Blinkit Growth Discovery Visualizations")

        c1, c2 = st.columns(2)
        if "root_cause" in df_ana.columns:
            c1.plotly_chart(
                bar_chart(df_ana, "root_cause", "Root Causes"),
                width="stretch",
            )
        if "pain_point" in df_ana.columns:
            c2.plotly_chart(
                bar_chart(df_ana, "pain_point", "Pain Points"),
                width="stretch",
            )

        c3, c4 = st.columns(2)
        if "discovery_surface" in df_ana.columns:
            c3.plotly_chart(
                bar_chart(df_ana, "discovery_surface", "Discovery Surfaces"),
                width="stretch",
            )
        if "user_segment" in df_ana.columns:
            seg_df = clean_counts(df_ana, "user_segment")
            if not seg_df.empty:
                fig_tree = px.treemap(
                    seg_df, path=["user_segment"], values="count",
                    title="Shopper Segments", template="plotly_dark",
                    color_discrete_sequence=["#F7C600", "#0C831F", "#E5B000", "#15A02B"],
                )
                fig_tree.update_layout(height=350, margin=dict(t=44, b=20, l=20, r=20))
                c4.plotly_chart(fig_tree, width="stretch")

        c5, c6 = st.columns(2)
        if "emotion" in df_ana.columns:
            c5.plotly_chart(
                donut_chart(df_ana, "emotion", "Customer Emotions Expressed"),
                width="stretch",
            )
        if "confidence" in df_ana.columns:
            fig_hist = px.histogram(
                df_ana, x="confidence", nbins=15,
                title="LLM Confidence Distribution", template="plotly_dark",
            )
            fig_hist.update_traces(marker_color="#F7C600")
            fig_hist.update_layout(
                xaxis_title="Confidence", yaxis_title="Count", height=350,
            )
            c6.plotly_chart(fig_hist, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – THEME SUMMARY & REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Theme Summary":
    st.title("Growth Discovery Executive Report")
    st.markdown(
        "Synthesizes structured labels into a Blinkit Growth Discovery executive report."
    )

    min_conf = st.slider(
        "Min confidence to include", 0.0, 1.0,
        config.DEFAULT_MIN_CONFIDENCE_SUMMARY, 0.05,
    )
    top_n = st.number_input("Top N themes per section", 1, 50, config.DEFAULT_TOP_N_THEMES)

    if st.button("Generate Report", width="stretch", type="primary"):
        if df_ana is None:
            st.error("Run insight extraction first.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0)
            stage_status = st.empty()
            logs_exp = st.expander("Execution Logs", expanded=True)
            with logs_exp:
                log_ph = st.empty()

            with capture_logs(log_ph) as log_capture:
                try:
                    stage_status.info("Clustering Blinkit catalog themes & generating all 12 artifacts…")
                    progress.progress(0.4)
                    run_stage_summary(float(min_conf), int(top_n))
                    progress.progress(1.0, text=f"Done ({elapsed(t0)})")
                    stage_status.success(f"Report & 12 research artifacts generated in {elapsed(t0)}.")
                    st.rerun()
                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Report generation failed",
                    )

    if config.THEME_SUMMARY_MD.exists():
        st.subheader("Report Preview")
        try:
            md_text = config.THEME_SUMMARY_MD.read_text(encoding="utf-8")
            st.markdown(md_text)
            st.download_button(
                "Download report.md",
                md_text.encode("utf-8"),
                "report.md",
                "text/markdown",
            )
        except Exception as exc:
            st.error(f"Could not read report: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 – RESEARCH ARTIFACTS VIEWER (12 Expected Outputs)
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Research Artifacts":
    st.title("Structured Research Artifacts")
    st.markdown("Explore the 12 expected production outputs generated by the pipeline.")

    tabs = st.tabs([
        "1. Summary", "2. Behaviors", "3. Barriers", "4. Requirements",
        "5. Themes", "6. Root Causes", "7. Opportunities", "8. Hypotheses",
        "9. Interview Plans", "10. Report", "11. Metadata"
    ])

    with tabs[0]:
        st.subheader("1. Research Summary")
        summary_data = load_json(config.FILTER_SUMMARY_JSON)
        if summary_data:
            st.json(summary_data)
        else:
            st.info("Run the pipeline to generate Research Summary.")

    with tabs[1]:
        st.subheader("2. User Behaviors")
        behaviors_data = load_json(config.BEHAVIORS_JSON)
        if behaviors_data:
            st.json(behaviors_data)
        else:
            st.info("Run the pipeline to generate Behaviors artifact.")

    with tabs[2]:
        st.subheader("3. User Barriers")
        barriers_data = load_json(config.BARRIERS_JSON)
        if barriers_data:
            st.json(barriers_data)
        else:
            st.info("Run the pipeline to generate Barriers artifact.")

    with tabs[3]:
        st.subheader("4. Product Requirements")
        req_data = load_json(config.REQUIREMENTS_JSON)
        if req_data:
            st.json(req_data)
        else:
            st.info("Run the pipeline to generate Product Requirements artifact.")

    with tabs[4]:
        st.subheader("5. Research Themes")
        themes_data = load_json(config.THEMES_JSON)
        if themes_data:
            st.json(themes_data)
        else:
            st.info("Run the pipeline to generate Themes artifact.")

    with tabs[5]:
        st.subheader("6. Root Causes")
        rc_data = load_json(config.ROOT_CAUSES_JSON)
        if rc_data:
            st.json(rc_data)
        else:
            st.info("Run the pipeline to generate Root Causes artifact.")

    with tabs[6]:
        st.subheader("7. Product Opportunities")
        opp_data = load_json(config.OPPORTUNITIES_JSON)
        if opp_data:
            st.json(opp_data)
        else:
            st.info("Run the pipeline to generate Opportunities artifact.")

    with tabs[7]:
        st.subheader("8. Research Hypotheses")
        hyp_data = load_json(config.HYPOTHESES_JSON)
        if hyp_data:
            st.json(hyp_data)
        else:
            st.info("Run the pipeline to generate Hypotheses artifact.")

    with tabs[8]:
        st.subheader("9. Interview Plans")
        ip_data = load_json(config.INTERVIEW_PLANS_JSON)
        if ip_data:
            st.json(ip_data)
        else:
            st.info("Run the pipeline to generate Interview Plans artifact.")

    with tabs[9]:
        st.subheader("10. Final Research Report (report.md)")
        if config.REPORT_MD.exists():
            st.markdown(config.REPORT_MD.read_text(encoding="utf-8"))
        else:
            st.info("Run the pipeline to generate Final Research Report.")

    with tabs[10]:
        st.subheader("11. Execution Metadata")
        meta_data = load_json(config.METADATA_JSON)
        if meta_data:
            st.json(meta_data)
        else:
            st.info("Run the pipeline to generate Execution Metadata artifact.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 – OUTPUTS & EXPORTS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Outputs":
    st.title("Outputs & Artifact Center")
    st.markdown("Download individual pipeline artifacts or export a complete ZIP bundle.")

    st.markdown("### 📦 Export Complete ZIP Bundle (12 Outputs)")
    zip_bytes = create_export_zip_bundle()
    st.download_button(
        "Download All Artifacts (blinkit_growth_discovery_bundle.zip)",
        data=zip_bytes,
        file_name="blinkit_growth_discovery_bundle.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )

    st.markdown("---")
    st.markdown("### 📄 Individual Machine-Readable & Report Artifacts")

    artifacts = [
        ("Raw Reviews (CSV)", config.RAW_REVIEWS_CSV),
        ("Filtered Reviews (CSV)", config.FILTERED_REVIEWS_CSV),
        ("Rejected Reviews (CSV)", config.REJECTED_REVIEWS_CSV),
        ("Analyzed Reviews (CSV)", config.ANALYZED_REVIEWS_CSV),
        ("Filter Summary (JSON)", config.FILTER_SUMMARY_JSON),
        ("Final Research Report (MD)", config.REPORT_MD),
        ("Behaviors (JSON)", config.BEHAVIORS_JSON),
        ("Barriers (JSON)", config.BARRIERS_JSON),
        ("Product Requirements (JSON)", config.REQUIREMENTS_JSON),
        ("Themes (JSON)", config.THEMES_JSON),
        ("Root Causes (JSON)", config.ROOT_CAUSES_JSON),
        ("Opportunities (JSON)", config.OPPORTUNITIES_JSON),
        ("Research Hypotheses (JSON)", config.HYPOTHESES_JSON),
        ("Interview Plans (JSON)", config.INTERVIEW_PLANS_JSON),
        ("Execution Metadata (JSON)", config.METADATA_JSON),
    ]

    left, right = st.columns(2)

    for idx, (label, path) in enumerate(artifacts):
        col = left if idx % 2 == 0 else right
        with col:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                ts = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%m-%d  %H:%M"
                )
                mime = (
                    "text/csv" if path.suffix == ".csv"
                    else "text/markdown" if path.suffix == ".md"
                    else "application/json"
                )
                st.markdown(
                    f"""<div style="background:#181818;padding:18px 20px;
                        border-radius:8px;border:1px solid #282828;
                        margin:12px 0 4px 0;">
                      <h4 style="color:#F7C600;margin:0 0 10px 0;font-weight:normal;">
                        {label}</h4>
                      <p style="margin:0;color:#b3b3b3;font-size:.82em;">
                        <code>{path.name}</code></p>
                      <p style="margin:3px 0;color:#b3b3b3;font-size:.82em;">
                        {size_kb:.1f} KB &nbsp;·&nbsp; {ts}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.download_button(
                    f"Download {path.name}",
                    path.read_bytes(),
                    path.name,
                    mime,
                    key=f"dl_{idx}",
                    width="stretch",
                )
            else:
                st.markdown(
                    f"""<div style="background:#181818;padding:18px 20px;
                        border-radius:8px;border:1px dashed #333;
                        margin:12px 0 18px 0;">
                      <h4 style="color:#555;margin:0 0 6px 0;font-weight:normal;">
                        {label}</h4>
                      <p style="margin:0;color:#555;font-size:.82em;">
                        Not generated yet.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
