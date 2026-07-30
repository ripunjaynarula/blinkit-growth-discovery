from __future__ import annotations

from dataclasses import asdict, dataclass


RAW_REVIEW_FIELDS = ["id", "source", "review", "rating", "date", "url"]


@dataclass(frozen=True)
class RawReview:
    id: str
    source: str
    review: str
    rating: int | None
    date: str
    url: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
