from __future__ import annotations

from abc import ABC, abstractmethod
from reviews.models import RawReview


class BaseCollector(ABC):
    """Abstract Base Class for review collectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Returns the identifier name of the review source."""
        pass

    @abstractmethod
    def collect(self, limit: int, **kwargs) -> list[RawReview]:
        """Collects reviews from the target source up to `limit`."""
        pass
