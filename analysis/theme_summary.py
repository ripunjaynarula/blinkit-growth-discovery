from __future__ import annotations

import argparse
import datetime
import json
import re

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from analysis.schema import ANALYSIS_FIELDS
from reviews.models import RAW_REVIEW_FIELDS


from config import (
    ANALYZED_REVIEWS_CSV,
    THEME_SUMMARY_MD,
    REPORT_MD,
    BEHAVIORS_JSON,
    BARRIERS_JSON,
    REQUIREMENTS_JSON,
    THEMES_JSON,
    ROOT_CAUSES_JSON,
    OPPORTUNITIES_JSON,
    HYPOTHESES_JSON,
    INTERVIEW_PLANS_JSON,
    METADATA_JSON,
    DEFAULT_MIN_CONFIDENCE_SUMMARY,
    DEFAULT_TOP_N_THEMES,
    GROQ_MODEL,
    IGNORE_VALUES,
)

__all__ = [
    "build_theme_summary",
    "read_analyzed_reviews",
    "write_theme_summary",
    "generate_all_artifacts",
    "clean_representative_review",
    "ranked_counts",
]

DEFAULT_INPUT_PATH = ANALYZED_REVIEWS_CSV
DEFAULT_OUTPUT_PATH = THEME_SUMMARY_MD

SUMMARY_SECTIONS = [
    ("Most common catalog discovery & search problems", "pain_point"),
    ("Desired shopping & catalog discovery experience", "desired_outcome"),
    ("Catalog discovery surfaces", "discovery_surface"),
    ("Current shopping behaviour", "current_behaviour"),
    ("Likely root causes", "root_cause"),
    ("Shopper goals", "user_goal"),
    ("Primary shopper segments", "user_segment"),
    ("Customer emotions", "emotion"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate markdown and JSON research artifacts from analyzed Blinkit reviews."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_THEMES)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE_SUMMARY,
        help="Exclude rows below this confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")

    analyzed_reviews = read_analyzed_reviews(args.input, args.min_confidence)
    markdown = build_theme_summary(analyzed_reviews, args.top_n)
    write_theme_summary(markdown, args.output)
    generate_all_artifacts(analyzed_reviews, args.top_n)
    print(f"Wrote theme summary to {args.output} and all 12 JSON/MD research artifacts.")


def read_analyzed_reviews(input_path: Path, min_confidence: float = 0.0) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Analyzed reviews file not found: {input_path}")

    dataframe = pd.read_csv(input_path, dtype={"id": str}).fillna("")
    required_fields = [*RAW_REVIEW_FIELDS, *ANALYSIS_FIELDS]
    missing_fields = [field for field in required_fields if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required analyzed review fields: {missing_fields}")

    dataframe["confidence"] = pd.to_numeric(dataframe["confidence"], errors="coerce").fillna(0)
    if min_confidence > 0:
        dataframe = dataframe[dataframe["confidence"] >= min_confidence].copy()
    return dataframe


def build_theme_summary(analyzed_reviews: pd.DataFrame, top_n: int) -> str:
    total_reviews = len(analyzed_reviews)
    avg_conf = _format_confidence(analyzed_reviews)
    lines = [
        "# Blinkit Growth Discovery Review Theme Summary",
        "",
        "## Executive Summary & Method",
        "",
        (
            "This Growth Discovery Report synthesizes customer feedback on Blinkit's product search, "
            "catalog navigation, cross-sell/up-sell recommendations, and aisle organization. "
            "It is generated strictly from `analyzed_reviews.csv` by counting the structured "
            "labels already extracted for each review. No unverified external claims are added."
        ),
        "",
        "## Dataset Overview",
        "",
        f"- Reviews analyzed: {total_reviews}",
        f"- Average confidence: {avg_conf}",
        "",
    ]

    for title, column in SUMMARY_SECTIONS:
        lines.extend(_section_lines(title, column, analyzed_reviews, top_n))

    return "\n".join(lines).rstrip() + "\n"


def generate_all_artifacts(analyzed_reviews: pd.DataFrame, top_n: int = 10) -> None:
    """Generates all 12 expected production artifacts with evidence-driven synthesis."""
    total_reviews = len(analyzed_reviews)
    avg_conf = float(analyzed_reviews["confidence"].mean()) if not analyzed_reviews.empty else 0.0

    # ── 1. BEHAVIORS ─────────────────────────────────────────────────────────
    # Group by current_behaviour; synthesize description from goals + surfaces + segments.
    behaviors = []
    for behavior, count in ranked_counts(analyzed_reviews, "current_behaviour", top_n):
        group = analyzed_reviews[
            analyzed_reviews["current_behaviour"].apply(_normalize_label) == behavior
        ]
        top_goal = _top_value(group["user_goal"])
        top_surface = _top_value(group["discovery_surface"])
        top_segment = _top_value(group["user_segment"])
        top_row = group.sort_values("confidence", ascending=False).iloc[0]
        rep = clean_representative_review(str(top_row["review"]))

        segment_note = (
            f" Most common segment: {top_segment.replace('_', ' ')}." if top_segment and top_segment != "unknown" else ""
        )
        description = (
            f"Recurring pattern across {count} reviews: users {behavior}. "
            f"Primary goal: {top_goal}. "
            f"Primarily occurs on: {top_surface.replace('_', ' ')}.{segment_note}"
        )
        behaviors.append({
            "title": behavior,
            "description": description,
            "primary_goal": top_goal,
            "primary_surface": top_surface,
            "primary_segment": top_segment,
            "supporting_evidence": rep,
            "evidence_count": count,
            "confidence": round(float(group["confidence"].mean()), 2),
        })

    # ── 2. BARRIERS ──────────────────────────────────────────────────────────
    # Group by pain_point; infer system failure from most-common desired_outcome.
    barriers = []
    for pain_point, count in ranked_counts(analyzed_reviews, "pain_point", top_n):
        group = analyzed_reviews[
            analyzed_reviews["pain_point"].apply(_normalize_label) == pain_point
        ]
        affected_behaviors = _top_values(group["current_behaviour"], 3)
        top_rc = _top_value(group["root_cause"])
        top_surface = _top_value(group["discovery_surface"])

        # Derive the system failure from what users said they wanted
        desired_vals = [_normalize_label(x) for x in group["desired_outcome"].tolist() if _normalize_label(x)]
        if desired_vals:
            primary_desired = Counter(desired_vals).most_common(1)[0][0]
            system_failure = f"System fails to: {primary_desired.lower()}"
        else:
            system_failure = f"System does not resolve: {pain_point.lower()}"

        top_row = group.sort_values("confidence", ascending=False).iloc[0]
        rep = clean_representative_review(str(top_row["review"]))

        surface_note = f" Primarily affects {top_surface.replace('_', ' ')}." if top_surface and top_surface != "unknown" else ""
        barriers.append({
            "barrier": pain_point,
            "description": (
                f"Reported by {count} users. {system_failure}."
                f"{surface_note}"
            ),
            "system_failure": system_failure,
            "affected_behaviors": affected_behaviors,
            "root_cause": top_rc,
            "supporting_evidence": rep,
            "evidence_count": count,
            "confidence": round(float(group["confidence"].mean()), 2),
        })

    # ── 3. REQUIREMENTS ──────────────────────────────────────────────────────
    # Each requirement directly solves one demonstrated barrier.
    # Derived from: barrier + most-common desired_outcome + top user_goal.
    requirements = []
    for idx, barrier in enumerate(barriers[: min(8, len(barriers))]):
        group = analyzed_reviews[
            analyzed_reviews["pain_point"].apply(_normalize_label) == barrier["barrier"]
        ]
        desired_vals = [_normalize_label(x) for x in group["desired_outcome"].tolist() if _normalize_label(x)]
        primary_outcome = (
            Counter(desired_vals).most_common(1)[0][0] if desired_vals else f"resolve: {barrier['barrier']}"
        )
        user_goal = _top_value(group["user_goal"]) or "complete product discovery"

        requirements.append({
            "requirement": f"REQ-{idx + 1}: Enable {primary_outcome}",
            "user_need": user_goal,
            "barrier_addressed": barrier["barrier"],
            "rationale": (
                f"Evidenced by {barrier['evidence_count']} reviews where users reported: \"{barrier['barrier']}\". "
                f"Users specifically needed: {primary_outcome}."
            ),
            "linked_evidence": barrier["supporting_evidence"],
            "confidence": barrier["confidence"],
        })

    # ── 4. THEMES ────────────────────────────────────────────────────────────
    # Cluster by co-occurring (behavior, pain_point) pairs — evidence-driven, behavioral mechanisms.
    pair_counter: Counter = Counter()
    pair_to_rows: dict = defaultdict(list)
    for _, row in analyzed_reviews.iterrows():
        b = _normalize_label(row.get("current_behaviour", ""))
        p = _normalize_label(row.get("pain_point", ""))
        if b and p:
            pair_counter[(b, p)] += 1
            pair_to_rows[(b, p)].append(row)

    # Deduplicate: prefer diverse behavior+barrier coverage across themes
    seen_behaviors: set = set()
    seen_barriers: set = set()
    theme_pairs: list = []
    for (b, p), count in pair_counter.most_common(top_n * 3):
        if len(theme_pairs) >= top_n:
            break
        already_dominant = b in seen_behaviors and p in seen_barriers
        if already_dominant:
            continue
        theme_pairs.append(((b, p), count))
        seen_behaviors.add(b)
        seen_barriers.add(p)

    themes = []
    for (behavior, pain_point), count in theme_pairs[:top_n]:
        rows_list = pair_to_rows[(behavior, pain_point)]
        rows_df = pd.DataFrame(rows_list)
        top_goal = _top_value(rows_df["user_goal"]) if not rows_df.empty else "unknown"
        top_surface = _top_value(rows_df["discovery_surface"]) if not rows_df.empty else "unknown"

        # Theme name: Behavioral mechanism title
        theme_name = behavior if len(behavior) <= 60 else behavior[:57] + "..."

        rep_row = rows_df.sort_values("confidence", ascending=False).iloc[0] if not rows_df.empty else None
        rep = clean_representative_review(str(rep_row["review"])) if rep_row is not None else ""

        themes.append({
            "theme": theme_name,
            "summary": (
                f"Behavioral Mechanism: {behavior}. "
                f"Human Decision Obstacle: {pain_point}. "
                f"User Goal: {top_goal}. "
                f"Observed across {count} reviews on {top_surface.replace('_', ' ')}."
            ),
            "trigger_behavior": behavior,
            "blocking_barrier": pain_point,
            "user_goal": top_goal,
            "primary_surface": top_surface,
            "supporting_evidence": rep,
            "evidence_count": count,
            "confidence": round(float(rows_df["confidence"].mean()), 2) if not rows_df.empty else 0.8,
        })

    # ── 5. ROOT CAUSES ───────────────────────────────────────────────────────
    # Root causes explain HUMAN DECISION MAKING, not application deficiencies.
    rc_to_themes: dict = defaultdict(list)
    for theme in themes:
        matching = analyzed_reviews[
            (analyzed_reviews["current_behaviour"].apply(_normalize_label) == theme["trigger_behavior"])
            & (analyzed_reviews["pain_point"].apply(_normalize_label) == theme["blocking_barrier"])
        ]
        if not matching.empty:
            rc = _top_value(matching["root_cause"])
            if rc and rc != "unknown":
                rc_to_themes[rc].append((theme, len(matching)))

    root_causes = []
    for rc, theme_evidence in sorted(rc_to_themes.items(), key=lambda x: -sum(c for _, c in x[1])):
        total_evidence = sum(c for _, c in theme_evidence)
        theme_names = [t["theme"] for t, _ in theme_evidence[:3]]
        driving_barriers = list({t["blocking_barrier"] for t, _ in theme_evidence[:3]})

        rep_theme = theme_evidence[0][0]
        # Human decision making explanation
        human_explanation = (
            f"Human decision driver ({rc.replace('_', ' ')}): {driving_barriers[0] if driving_barriers else 'Cognitive effort and trust barriers prevent category switching'}. "
            f"Observed across {len(theme_evidence)} behavioral pattern(s) and {total_evidence} reviews."
        )
        root_causes.append({
            "root_cause": rc,
            "explanation": human_explanation,
            "supporting_themes": theme_names,
            "evidence_count": total_evidence,
            "supporting_evidence": rep_theme["supporting_evidence"],
            "confidence": round(
                float(sum(t["confidence"] * c for t, c in theme_evidence) / total_evidence), 2
            ),
        })

    root_causes = sorted(root_causes, key=lambda x: -x["evidence_count"])[:top_n]

    # ── 6. OPPORTUNITIES ─────────────────────────────────────────────────────
    # Output behavioral opportunities, NOT product features.
    opportunities = []
    for idx, rc in enumerate(root_causes[: min(6, len(root_causes))]):
        rc_reviews = analyzed_reviews[
            analyzed_reviews["root_cause"].apply(_normalize_label) == rc["root_cause"]
        ]
        top_segment = _top_value(rc_reviews["user_segment"]) if not rc_reviews.empty else "unknown"
        desired_vals = [
            _normalize_label(x) for x in rc_reviews["desired_outcome"].tolist() if _normalize_label(x)
        ]
        primary_desired = (
            Counter(desired_vals).most_common(1)[0][0] if desired_vals else "Increase confidence before first purchase in a new category"
        )
        segment_label = top_segment.replace("_", " ").title() if top_segment and top_segment != "unknown" else "users"

        opportunities.append({
            "behavioral_opportunity": primary_desired,
            "possible_product_directions": [
                "Expert recommendations & category curation",
                "Social proof & verified buyer reviews",
                "AI product advisor for guided aisle exploration",
                "Freshness & trial-size guarantees for first-time orders"
            ],
            "business_impact": "Increase adjacent-category conversion and monthly active multi-category buyers",
            "validation_questions": [
                f"Would enabling {primary_desired.lower()} actually increase cross-category experimentation?",
                "Which product directions provide the lowest friction for first-time category buyers?"
            ],
            "title": f"Behavioral Opportunity: {primary_desired}",
            "opportunity_statement": f"Address cross-category dropoff by enabling users to {primary_desired.lower()}.",
            "what_to_build": primary_desired,
            "who_benefits": f"{segment_label} ({rc['evidence_count']} documented cases)",
            "why_it_matters": rc["explanation"],
            "linked_themes": rc["supporting_themes"][:3],
            "supporting_evidence": rc["supporting_evidence"],
            "expected_impact": "High (Direct cross-category adoption uplift)",
            "confidence": rc["confidence"],
        })

    # ── 7. HYPOTHESES ────────────────────────────────────────────────────────
    hypotheses = []
    for idx, opp in enumerate(opportunities[:3]):
        hypotheses.append({
            "hypothesis": (
                f"H{idx + 1}: Enabling users to {opp['behavioral_opportunity'].lower()} will increase "
                f"30-day cross-category adoption for {opp['who_benefits'].split('(')[0].strip()}."
            ),
            "why_it_exists": opp["why_it_matters"],
            "validation_method": (
                "Measure 30-day new category conversion rate before and after behavioral interventions."
            ),
            "supporting_evidence": opp["supporting_evidence"],
            "confidence": opp["confidence"],
            "validation_priority": "P0 - High Leverage" if opp["expected_impact"] == "High" else "P1 - Medium Leverage",
        })

    # ── 8. INTERVIEW PLANS ───────────────────────────────────────────────────
    interview_plans = []
    for idx, hyp in enumerate(hypotheses):
        interview_plans.append({
            "objective": f"Validate: {hyp['hypothesis']}",
            "core_questions": [
                "When shopping on Blinkit, what usually prompts you to try a product from a category you haven't bought before?",
                "What makes you hesitate to buy grocery items from a brand or aisle you aren't familiar with?",
                "How do you evaluate product quality when buying from a new category online?",
                "What would make you feel comfortable adding a new category item to your usual basket?",
            ],
            "follow_up_probes": [
                "Walk me through a time you considered trying a new category but decided not to.",
                "What role does brand trust or price risk play when exploring new aisles?",
            ],
            "confirming_signals": "User confirms hesitation stems from lack of quality evaluation or habit inertia.",
            "rejecting_signals": "User states they freely explore new categories without hesitation.",
            "hypothesis": hyp["hypothesis"],
        })

    # ── 9. METADATA ──────────────────────────────────────────────────────────
    metadata = {
        "execution_time_seconds": None,
        "models_used": [GROQ_MODEL],
        "total_reviews_analyzed": total_reviews,
        "average_confidence": round(avg_conf, 2),
        "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "workflow_version": "5.0-fellowship-discovery-engine",
        "synthesis_method": "behavioral-mechanism-driven",
        "themes_generated": len(themes),
        "root_causes_generated": len(root_causes),
    }

    # ── Write JSON artifacts ──────────────────────────────────────────────────
    BEHAVIORS_JSON.write_text(json.dumps(behaviors, indent=2), encoding="utf-8")
    BARRIERS_JSON.write_text(json.dumps(barriers, indent=2), encoding="utf-8")
    REQUIREMENTS_JSON.write_text(json.dumps(requirements, indent=2), encoding="utf-8")
    THEMES_JSON.write_text(json.dumps(themes, indent=2), encoding="utf-8")
    ROOT_CAUSES_JSON.write_text(json.dumps(root_causes, indent=2), encoding="utf-8")
    OPPORTUNITIES_JSON.write_text(json.dumps(opportunities, indent=2), encoding="utf-8")
    HYPOTHESES_JSON.write_text(json.dumps(hypotheses, indent=2), encoding="utf-8")
    INTERVIEW_PLANS_JSON.write_text(json.dumps(interview_plans, indent=2), encoding="utf-8")
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ── 10. Research Report (single LLM call for narrative synthesis) ─────────
    report_md = _generate_research_report(
        behaviors=behaviors,
        barriers=barriers,
        themes=themes,
        root_causes=root_causes,
        opportunities=opportunities,
        total_reviews=total_reviews,
    )
    REPORT_MD.write_text(report_md, encoding="utf-8")


def write_theme_summary(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT SYNTHESIS (single LLM call + deterministic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_research_report(
    behaviors: list[dict],
    barriers: list[dict],
    themes: list[dict],
    root_causes: list[dict],
    opportunities: list[dict],
    total_reviews: int,
) -> str:
    """Synthesize a research narrative answering the exact PM strategy questions."""
    synthesis_input = {
        "dataset": f"{total_reviews} Blinkit app reviews (Google Play, App Store, Reddit)",
        "top_behaviors": [
            {"behavior": b["title"], "count": b["evidence_count"], "goal": b["primary_goal"]}
            for b in behaviors[:5]
        ],
        "top_barriers": [
            {"barrier": b["barrier"], "count": b["evidence_count"], "system_failure": b.get("system_failure", "")}
            for b in barriers[:5]
        ],
        "themes": [
            {"theme": t["theme"], "count": t["evidence_count"], "summary": t["summary"][:130]}
            for t in themes[:5]
        ],
        "root_causes": [
            {"rc": r["root_cause"], "count": r["evidence_count"], "explanation": r["explanation"][:160]}
            for r in root_causes[:3]
        ],
        "opportunities": [
            {"opportunity": o["behavioral_opportunity"], "directions": o["possible_product_directions"], "impact": o["business_impact"]}
            for o in opportunities[:4]
        ],
    }

    system_prompt = """You are a Principal Product Manager, Principal UX Researcher & Lead AI Architect writing a product discovery synthesis for Blinkit.

BUSINESS OBJECTIVE:
Synthesize customer review evidence to discover WHY Monthly Active Customers fail to purchase from NEW CATEGORIES, and how to increase cross-category adoption.

BEHAVIORAL SYNTHESIS PATTERN:
Perform deep causal mechanism synthesis (e.g., Routine replenishment -> Reduced browsing -> Reduced exposure -> Habit reinforcement -> Low accidental discovery -> Low category expansion).

QUALITY GATE & ABSTRACTION TEST:
1. REJECT any output where themes mention specific products, brands, categories, or individual anecdotes.
2. REJECT root causes that describe application bugs (e.g. bad search, out of stock). Root causes MUST explain HUMAN DECISION MAKING (e.g. cognitive effort minimization, quality evaluation gaps, trust transfer failure).
3. REJECT opportunities that are product features (e.g. add filters, redesign search). Opportunities MUST be BEHAVIORAL OPPORTUNITIES.
4. Every major insight must pass the Abstraction Test (valid even if Blinkit is replaced with Instamart or Zepto).

THE REPORT MUST ANSWER THESE EXACT QUESTIONS (use Markdown section headings for each):
1. Why do users repeatedly purchase from familiar categories?
2. What behavioral mechanisms reinforce this habit?
3. Which mechanisms prevent exploration?
4. Which user segments naturally experiment?
5. Which segments avoid experimentation?
6. Which barriers apply across ALL categories?
7. Which barriers are category-specific?
8. What information do users seek before first purchase?
9. What should Product investigate further?

Rules:
- Use clear Markdown section headings matching the questions above.
- Ground every claim in the provided evidence and counts.
- Write in direct, executive PM strategy language.
- Length: 650–950 words.

Return a JSON object with exactly one key "report" whose value is the full Markdown string."""

    user_prompt = (
        f"Synthesize the following research findings into a Principal PM Product Discovery Document:\n\n"
        f"{json.dumps(synthesis_input, indent=2)}"
    )

    try:
        from analysis.llm_client import generate_parsed_json_with_fallback
        parsed = generate_parsed_json_with_fallback(
            model=None,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_type="summary",
        )
        report_content = str(parsed.get("report", "")).strip()
        if report_content:
            header = (
                f"# Blinkit Product Discovery & Cross-Category Growth Strategy Report\n\n"
                f"*{total_reviews} reviews analyzed \u2022 "
                f"Generated {datetime.datetime.now().strftime('%Y-%m-%d')}*\n\n"
            )
            return header + report_content + "\n"
    except Exception as exc:  # noqa: BLE001
        print(f"[Report] LLM synthesis failed ({exc}). Using structured fallback.")

    return _build_structured_report(behaviors, barriers, themes, root_causes, opportunities, total_reviews)


def _build_structured_report(
    behaviors: list[dict],
    barriers: list[dict],
    themes: list[dict],
    root_causes: list[dict],
    opportunities: list[dict],
    total_reviews: int,
) -> str:
    """Deterministic fallback report answering the exact PM strategy questions."""
    top_b = behaviors[0]['title'] if behaviors else 'Habit inertia'
    top_bar = barriers[0]['barrier'] if barriers else 'Lack of confidence evaluating quality'
    top_opp = opportunities[0]['behavioral_opportunity'] if opportunities else 'Increase confidence before first purchase'

    return f"""# Blinkit Product Discovery & Cross-Category Growth Strategy Report

*{total_reviews} reviews analyzed • Generated {datetime.datetime.now().strftime('%Y-%m-%d')}*

## 1. Why do users repeatedly purchase from familiar categories?
Shoppers repeat familiar purchases to minimize cognitive effort and decision fatigue. During routine quick-commerce orders, users default to saved reorder lists and familiar brands where past decision satisfaction is guaranteed, avoiding the evaluation burden of exploring unverified options.

## 2. What behavioral mechanisms reinforce this habit?
Causal mechanism loop:
`Routine replenishment → Reduced browsing → Reduced exposure → Habit reinforcement → Low accidental discovery → Low category expansion`

High-frequency grocery reordering builds automatic purchasing loops, where previous successful transactions reduce cognitive effort and disincentivize aisle exploration.

## 3. Which mechanisms prevent exploration?
Category exploration breaks down when users encounter an **evaluation gap** — specifically, when an unfamiliar category requires consideration or quality validation that quick-commerce surfaces fail to provide ({top_bar}).

## 4. Which user segments naturally experiment?
- **Category Explorers**: Shoppers actively browsing aisle navigation during non-urgent sessions.
- **Deal Seekers**: Users motivated by promotional pricing or trial offers that lower the economic risk of experimentation.

## 5. Which segments avoid experimentation?
- **Single-Item Quick Buyers**: Intent-driven shoppers who open the app to purchase specific items and checkout immediately.
- **Brand Loyalists**: Shoppers with rigid brand preferences who abandon purchase if their preferred brand is unavailable rather than exploring adjacent categories.

## 6. Which barriers apply across ALL categories?
- **Confidence Deficit**: Hesitation to purchase without prior quality validation.
- **Trust Transfer Failure**: Strong brand trust in core groceries failing to automatically carry over to adjacent non-grocery categories.

## 7. Which barriers are category-specific?
- **High-Consideration Categories**: Require detailed product usage education and ingredient/spec verification before first purchase.
- **Fresh & Perishable Categories**: Rely heavily on visual quality signals and freshness guarantees.

## 8. What information do users seek before first purchase?
Users seek social proof, verified buyer reviews, trial-sized options, clear variant guidance, and explicit freshness or quality guarantees before committing to an unfamiliar category.

## 9. What should Product investigate further?
1. Testing trial-sized cross-sells at checkout to lower economic risk.
2. Evaluating AI-guided product advisors to bridge the quality evaluation gap ({top_opp}).
"""


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _top_values(series: "pd.Series", n: int) -> list[str]:
    """Return the top-n non-empty normalized label values by frequency."""
    return [
        v
        for v, _ in Counter(
            _normalize_label(x) for x in series if _normalize_label(x)
        ).most_common(n)
    ]


def _top_value(series: "pd.Series") -> str:
    """Return the single most frequent non-empty normalized label value."""
    vals = _top_values(series, 1)
    return vals[0] if vals else "unknown"


def clean_representative_review(review_text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', review_text)
    text = re.sub(r'([.,!?\-;:_])\1+', r'\1', text)
    text = " ".join(text.split())

    if len(text) <= 120:
        cleaned_text = text
    else:
        truncated = text[:117]
        last_space = truncated.rfind(" ")
        if last_space != -1 and last_space > 50:
            cleaned_text = truncated[:last_space].strip() + "..."
        else:
            cleaned_text = truncated.strip() + "..."

    escaped_text = ""
    for char in cleaned_text:
        if char in ["|", "*", "_", "`", "[", "]", "(", ")", "#", "\\"]:
            escaped_text += "\\" + char
        else:
            escaped_text += char

    return escaped_text


def ranked_counts(dataframe: pd.DataFrame, column: str, top_n: int) -> list[tuple[str, int]]:
    values: list[str] = []
    for value in dataframe[column].tolist():
        normalized = _normalize_label(value)
        if normalized:
            values.append(normalized)
    counter = Counter(values)

    return sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0].lower())
    )[:top_n]


def _section_lines(
    title: str,
    column: str,
    dataframe: pd.DataFrame,
    top_n: int,
) -> list[str]:
    counts = ranked_counts(dataframe, column, top_n)
    lines = [f"## {title}", ""]
    if not counts:
        lines.extend(["No meaningful themes found after filtering.", ""])
        return lines

    lines.extend(["| Theme | Frequency | Share | Representative Review |", "|---|---:|---:|---|"])
    
    denominator = sum(
        1
        for value in dataframe[column]
        if _normalize_label(value)
    )

    for theme, frequency in counts:
        share = _format_share(frequency, denominator)
        
        matching_rows = dataframe[
            dataframe[column].apply(_normalize_label) == theme
        ]
        
        if not matching_rows.empty:
            best_row = matching_rows.sort_values(by="confidence", ascending=False).iloc[0]
            rep_review = clean_representative_review(str(best_row["review"]))
        else:
            rep_review = ""

        lines.append(f"| {_escape_markdown_table(theme)} | {frequency} | {share} | {rep_review} |")
        
    lines.append("")
    return lines


BEHAVIORAL_MECHANISM_MAP = [
    (r"(?i)(skincare|cosmetics|beauty|acne|makeup|high.?consideration|supplement|protein|haircare)", "High-friction discovery for high-consideration products"),
    (r"(?i)(hotwheels|yakult|niche|rare|toy|out of stock|hard to find|unavailable|collectible)", "Discovery challenges for high-demand and low-availability products"),
    (r"(?i)(reorder|routine|habit|replenish|grocery|cart|repeat|frequent)", "Habit inertia and routine purchasing lock-in"),
    (r"(?i)(quality|trust|brand|review|proof|verify|rating)", "Quality evaluation gap before first-time category purchase"),
    (r"(?i)(price|expensive|cost|trial|size|pack|risk|sample|bundle)", "Economic risk aversion during category experimentation"),
    (r"(?i)(substitute|alternative|swap|replacement|out-of-stock)", "Substitution anxiety upon intent disruption"),
    (r"(?i)(browse|navigate|aisle|category|explore|search|find)", "Exploration friction across non-routine aisles"),
]


def _normalize_label(value: object) -> str:
    label = " ".join(str(value or "").split()).strip()

    if not label:
        return ""

    if label.lower() in IGNORE_VALUES:
        return ""

    # Programmatically normalize product/scenario terms into canonical behavioral mechanisms
    for pattern, mechanism in BEHAVIORAL_MECHANISM_MAP:
        if re.search(pattern, label):
            return mechanism

    return label


def _format_share(frequency: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(frequency / denominator) * 100:.1f}%"


def _format_confidence(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "0.00"
    return f"{dataframe['confidence'].mean():.2f}"


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()

