"""Deterministic product publishing filters for scraped coffee products."""

from __future__ import annotations

import re
from typing import Any


_REJECT_TAGS = {
    "b2b",
    "exclude",
    "hide-from-retail",
    "internal",
    "white-label",
    "white label",
    "wholesale_list",
    "wholesale list",
}

_REJECT_TYPE_RE = re.compile(
    r"\b("
    r"bundle|gift\s*card|gift\s*box|gear|equipment|apparel|"
    r"merch(?:andise)?|mug|tumbler|capsule|pod|instant|ready\s*to\s*drink"
    r")\b",
    re.IGNORECASE,
)

_REJECT_WHOLESALE_TYPE_RE = re.compile(r"(^|[/,\s])wholesale\b", re.IGNORECASE)
_REJECT_NUMBERED_WHOLESALE_TAG_RE = re.compile(r"^wholesale_\d", re.IGNORECASE)

_REJECT_TITLE_RE = re.compile(
    r"\b("
    r"wholesale|bulk|case|internal|white\s*label|gift\s*card|gift\s*box|"
    r"capsules?|pods?|instant|ready\s*to\s*drink|canned"
    r")\b",
    re.IGNORECASE,
)


def _field(product: Any, name: str, default: Any = "") -> Any:
    if isinstance(product, dict):
        return product.get(name, default)
    return getattr(product, name, default)


def _tags(product: Any) -> list[str]:
    raw = _field(product, "tags", []) or []
    if isinstance(raw, str):
        raw = [t.strip() for t in raw.split(",")]
    return [str(t).strip().lower() for t in raw if str(t).strip()]


def should_publish_product(product: Any, extracted: Any | None = None) -> bool:
    """Return True when a product should appear in the public explorer.

    LLM extraction decides whether something is a bag of coffee. These checks add
    a conservative deterministic gate for private-label, wholesale, bundle, and
    non-bean artifacts that commonly leak through Shopify product feeds.
    """
    if extracted is not None and not _field(extracted, "is_coffee_product", True):
        return False

    title = str(_field(product, "title") or "")
    vendor = str(_field(product, "vendor") or "")
    product_type = str(_field(product, "product_type") or "")
    tags = _tags(product)
    product_type_norm = product_type.lower()

    if "wholesale" in vendor.lower():
        return False
    if (
        _REJECT_WHOLESALE_TYPE_RE.search(product_type)
        and "wholebeancoffee" not in product_type_norm
    ):
        return False
    if _REJECT_TYPE_RE.search(product_type):
        return False
    if _REJECT_TITLE_RE.search(title):
        return False

    for tag in tags:
        if tag in _REJECT_TAGS:
            return False
        if tag in {"wholesale", "wholesale quick order"}:
            return False
        if _REJECT_NUMBERED_WHOLESALE_TAG_RE.search(tag):
            return False

    return True
