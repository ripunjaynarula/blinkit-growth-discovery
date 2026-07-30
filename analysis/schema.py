from __future__ import annotations

ROOT_CAUSES = [
    "search_irrelevance",
    "category_misplacement",
    "poor_aisle_organization",
    "missing_high_demand_item",
    "ineffective_recommendations",
    "poor_cross_category_discovery",
    "unclear_product_variants",
    "broken_reorder_flow",
    "unknown",
]

DISCOVERY_SURFACES = [
    "search_bar",
    "category_browse_aisle",
    "home_page_recommendations",
    "checkout_upsell",
    "reorder_buy_again",
    "filter_sort_facet",
    "product_detail_page",
    "unknown",
]

USER_SEGMENTS = [
    "single_item_quick_buyer",
    "category_explorer",
    "brand_loyalist",
    "deal_seeker",
    "bulk_grocery_planner",
    "unknown",
]

DISCOVERY_SIGNALS = [
    "delivery_delay",
    "pricing",
    "inventory",
    "checkout",
    "search",
    "customer_support",
    "shopping_behavior",
    "trust",
    "retention",
    "feature_request",
    "competitor_comparison",
    "product_delight",
    "workflow_friction",
]

ANALYSIS_FIELDS = [
    "pain_point",
    "root_cause",
    "discovery_surface",
    "user_goal",
    "current_behaviour",
    "desired_outcome",
    "user_segment",
    "emotion",
    "confidence",
]


def clamp_confidence(val: object) -> float:
    try:
        f = float(val)  # type: ignore[arg-type]
        return max(0.0, min(1.0, f))
    except (ValueError, TypeError):
        return 0.0


def empty_analysis() -> dict[str, object]:
    return {
        "pain_point": "unknown",
        "root_cause": "unknown",
        "discovery_surface": "unknown",
        "user_goal": "unknown",
        "current_behaviour": "unknown",
        "desired_outcome": "unknown",
        "user_segment": "unknown",
        "emotion": "neutral",
        "confidence": 0.0,
    }


SYSTEM_PROMPT = """
You are a Principal Product Manager & Lead Behavioral Architect analyzing customer feedback for Blinkit (Quick-Commerce Grocery & Instant Delivery app in India).

FELLOWSHIP BUSINESS OBJECTIVE:
Increase the percentage of Monthly Active Customers who purchase from at least one NEW CATEGORY every month.
Your objective is to discover WHY users fail to cross category boundaries and what behavioral mechanisms prevent cross-category adoption.

THE ABSTRACTION TEST:
Every extracted insight MUST pass the abstraction test: replace the specific product category with any other category (e.g., skincare -> baby products -> supplements). If the insight is still true, it is valid.

CRITICAL EXTRACTION RULES:

1. CURRENT_BEHAVIOUR — Must be a RECURRING BEHAVIORAL MECHANISM of human decision making.
   Examples of valid behavioral mechanisms:
   - "Habit inertia" / "Routine reinforcement" (reordering familiar items to minimize cognitive effort)
   - "Confidence gap before first purchase" (hesitating to buy from an unfamiliar category due to lack of quality proof)
   - "Exploration friction" (struggling to discover or evaluate non-routine categories during quick shopping)
   - "Economic risk aversion during experimentation" (reluctance to spend on full-sized items in new categories)
   - "Trust transfer deficit across categories" (brand trust in one category failing to carry over to adjacent categories)
   - "Decision simplification" (defaulting to known brands to avoid decision paralysis)
   - "Substitution uncertainty" (abandoning purchase when familiar item is out of stock instead of trying a new category)

   NEVER extract product scenarios like "Searches for acne skincare" or "Wants Hotwheels".

2. PAIN_POINT — Must explain HUMAN DECISION MAKING, NOT application deficiencies.
   GOOD: "Users avoid unfamiliar categories because they lack confidence evaluating product quality remotely"
   GOOD: "Users repeat familiar purchases because previous successful decisions reduce cognitive effort"
   GOOD: "Users hesitate to try new categories when brand trust cannot transfer across aisle boundaries"
   BAD: "Poor search", "Bad recommendations", "Out of stock items", "App crash" (these are app bugs, not human decision obstacles)

3. USER_GOAL — Behavioral goal for category expansion.
   GOOD: "Evaluate product quality before committing to a purchase in an unfamiliar category"
   GOOD: "Reduce cognitive effort during routine grocery restock without missing relevant new items"
   GOOD: "Minimize financial and decision risk when experimenting with a new product category"

4. DESIRED_OUTCOME — Behavioral opportunity (NOT a product feature).
   GOOD: "Increase confidence before first purchase in a new category"
   GOOD: "Reduce effort evaluating unfamiliar products"
   GOOD: "Encourage low-risk experimentation across categories"
   GOOD: "Help users transfer trust across categories"
   BAD: "Add filters", "Improve search bar", "Redesign recommendations" (these are product features, not behavioral opportunities)

5. ROOT_CAUSE — Must be exactly one of:
   search_irrelevance, category_misplacement, poor_aisle_organization, missing_high_demand_item,
   ineffective_recommendations, poor_cross_category_discovery, unclear_product_variants,
   broken_reorder_flow, unknown

6. DISCOVERY_SURFACE — Must be exactly one of:
   search_bar, category_browse_aisle, home_page_recommendations, checkout_upsell,
   reorder_buy_again, filter_sort_facet, product_detail_page, unknown

7. USER_SEGMENT — Must be exactly one of:
   single_item_quick_buyer, category_explorer, brand_loyalist, deal_seeker, bulk_grocery_planner, unknown

8. EMOTION — Must be exactly one of: frustrated, delighted, disappointed, confused, neutral

9. CONFIDENCE — float between 0.0 and 1.0.

Return a JSON object with one key "reviews" containing an array of objects.
Return JSON ONLY.
""".strip()


def build_batch_prompt(reviews: list[dict[str, object]]) -> str:
    return (
        "Extract behavioral mechanisms and human decision obstacles from the following Blinkit reviews. "
        "Optimize for understanding WHY users fail to cross category boundaries. "
        "Every extraction must pass the Abstraction Test (reusable across any category). "
        "Extract current_behaviour as a BEHAVIORAL MECHANISM (e.g. Habit inertia, Confidence before first purchase, Trust transfer deficit). "
        "Extract pain_point as a HUMAN DECISION MAKING OBSTACLE (e.g. Lacks confidence evaluating quality). "
        "Extract desired_outcome as a BEHAVIORAL OPPORTUNITY (e.g. Increase confidence before first purchase). "
        "Return a JSON object with key 'reviews' containing an array of objects.\n\n"
        f"Input reviews:\n{reviews}"
    )


