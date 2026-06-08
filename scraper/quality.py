"""Quality gates for generated roasted coffee data."""

from __future__ import annotations

import math
from collections.abc import Sequence

from scraper.models import RoastedCoffeeProduct

MIN_PRODUCT_COUNT = 100
MIN_ORIGIN_COUNTRY_COUNT = 100
MIN_ORIGIN_COUNTRY_RATIO = 0.30
MIN_WATCHLIST_MATCH_COUNT = 10
MIN_IMAGE_RATIO = 0.95
SUSPICIOUS_PROCESS_VALUES = {
    "nirvana",
    "moon shadow natural #3 (msn #3)",
}


class OutputQualityError(RuntimeError):
    """Raised when generated data is too degraded to publish."""


def validate_output_quality(products: Sequence[RoastedCoffeeProduct]) -> None:
    """Raise if generated data looks like a failed scrape/extraction run."""
    total = len(products)
    origin_count = sum(1 for p in products if p.origin_country)
    watchlist_count = sum(1 for p in products if p.watchlist_tier)
    image_count = sum(1 for p in products if p.image_url)
    required_origins = max(
        MIN_ORIGIN_COUNTRY_COUNT,
        math.ceil(total * MIN_ORIGIN_COUNTRY_RATIO),
    )
    required_images = math.ceil(total * MIN_IMAGE_RATIO)
    suspicious_processes = [
        p.title
        for p in products
        if p.process and p.process.strip().lower() in SUSPICIOUS_PROCESS_VALUES
    ]

    problems: list[str] = []
    if total < MIN_PRODUCT_COUNT:
        problems.append(f"only {total} products generated")
    if origin_count < required_origins:
        problems.append(
            f"only {origin_count}/{total} products have origin_country "
            f"(minimum {required_origins})"
        )
    if watchlist_count < MIN_WATCHLIST_MATCH_COUNT:
        problems.append(
            f"only {watchlist_count} products have watchlist_tier "
            f"(minimum {MIN_WATCHLIST_MATCH_COUNT})"
        )
    if image_count < required_images:
        problems.append(
            f"only {image_count}/{total} products have image_url "
            f"(minimum {required_images})"
        )
    if suspicious_processes:
        sample = ", ".join(suspicious_processes[:3])
        problems.append(f"suspicious process values on: {sample}")

    if problems:
        raise OutputQualityError("; ".join(problems))
