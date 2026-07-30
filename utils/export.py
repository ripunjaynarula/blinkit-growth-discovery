from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import config


def create_export_zip_bundle() -> bytes:
    """Creates an in-memory ZIP archive containing all 12 production research outputs & raw data artifacts."""
    buf = io.BytesIO()
    artifacts = [
        ("raw_reviews.csv", config.RAW_REVIEWS_CSV),
        ("filtered_reviews.csv", config.FILTERED_REVIEWS_CSV),
        ("rejected_reviews.csv", config.REJECTED_REVIEWS_CSV),
        ("analyzed_reviews.csv", config.ANALYZED_REVIEWS_CSV),
        ("filter_summary.json", config.FILTER_SUMMARY_JSON),
        ("theme_summary.md", config.THEME_SUMMARY_MD),
        ("report.md", config.REPORT_MD),
        ("behaviors.json", config.BEHAVIORS_JSON),
        ("barriers.json", config.BARRIERS_JSON),
        ("requirements.json", config.REQUIREMENTS_JSON),
        ("themes.json", config.THEMES_JSON),
        ("root_causes.json", config.ROOT_CAUSES_JSON),
        ("opportunities.json", config.OPPORTUNITIES_JSON),
        ("hypotheses.json", config.HYPOTHESES_JSON),
        ("interview_plans.json", config.INTERVIEW_PLANS_JSON),
        ("metadata.json", config.METADATA_JSON),
    ]

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in artifacts:
            if path.exists():
                zf.write(path, arcname=arcname)
            else:
                zf.writestr(arcname, f"# {arcname}\nNot generated yet in this run.\n")

    buf.seek(0)
    return buf.getvalue()
