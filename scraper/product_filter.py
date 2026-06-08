"""Deterministic product publishing filters for scraped coffee products.

Structured Shopify metadata is the primary signal. Title checks are reserved for
feeds that leave product_type/tags sparse or misleading.
"""

from __future__ import annotations

import re
from typing import Any


_REJECT_TAGS = {
    "aeropress",
    "apparel",
    "apparel & gifts",
    "apparel & gifts-1",
    "b2b",
    "box-set",
    "brew gear",
    "brewing equipment",
    "brewing gear",
    "cafe-supplies",
    "coffee filters",
    "dropship",
    "equipment",
    "exclude",
    "excludedgorilla",
    "espresso equipment",
    "gift subscription",
    "hide-from-retail",
    "homebrew",
    "internal",
    "merch",
    "merchgorilla",
    "notvisible",
    "shopify collective",
    "single-use",
    "swag",
    "tea",
    "t-shirt",
    "type: tea",
    "warehouse",
    "white-label",
    "white label",
    "wholesale_list",
    "wholesale-hide",
    "wholesale list",
}

_REJECT_PRODUCT_TYPES = {
    "cold brew",
    "essentials",
    "mostra classics",
    "traditional",
}

_SPARSE_COFFEE_ROASTERS = {
    "helm",
    "qima",
}

_COUNTRY_TAGS = {
    "bolivia",
    "brazil",
    "burundi",
    "china",
    "colombia",
    "costa rica",
    "ecuador",
    "el salvador",
    "ethiopia",
    "guatemala",
    "honduras",
    "indonesia",
    "kenya",
    "mexico",
    "nicaragua",
    "panama",
    "peru",
    "rwanda",
    "sumatra",
    "thailand",
    "uganda",
    "yemen",
}

_PROCESS_TAGS = {
    "anaerobic natural",
    "anaerobic washed",
    "co-ferment",
    "honey",
    "natural",
    "washed",
}

_COFFEE_PRODUCT_TYPE_RE = re.compile(
    r"\b(bag of coffee|coffee|espresso|retail so|retail yr|roasted coffee|whole\s*bean|wholebeancoffee)\b",
    re.IGNORECASE,
)

_REJECT_TYPE_RE = re.compile(
    r"\b("
    r"box\s*set|brewing|bundle|cafe\s*products|carbon\s*offset|"
    r"chocolate|drinkware|dropship|espresso\s*machine|gift\s*card|"
    r"gift\s*box|gift\s*subscription|gear|equipment|apparel|hat|hoodie|membership|"
    r"merch(?:andise)?|mug|onesie|pastry|pin|plate|preset\s*box|"
    r"public\s*session|shirt|socks?|stickers?|subscription|tea|tumbler|"
    r"capsule|pod|instant|ready\s*to\s*drink"
    r")\b",
    re.IGNORECASE,
)

_REJECT_WHOLESALE_TYPE_RE = re.compile(r"(^|[/,\s])wholesale\b", re.IGNORECASE)
_REJECT_NUMBERED_WHOLESALE_TAG_RE = re.compile(r"^wholesale_\d", re.IGNORECASE)

_REJECT_TITLE_RE = re.compile(
    r"\b("
    r"wholesale|bulk|case|internal|white\s*label|gift\s*card|gift\s*box|box(?:es)?|"
    r"capsules?|pods?|instant|ready\s*to\s*drink|canned|subscription|membership|"
    r"aeropress|brew\s*guides?|brewing\s*sets?|cascara|chemex|coffee\s*makers?|"
    r"coffee\s*blossom\s*honey|coffeemakers?|coffee\s*servers?|coffee\s*systems?|coffee\s*tools?|"
    r"cold\s*brew\s*bottles?|cupping\s*spoons?|drippers?|filter\s*holders?|"
    r"grinders?|kettles?|moccamaster|portafilters?|posters?|pour-?over|"
    r"loose\s*leaf|sibarist|snapbacks?|stagg|tickets?|t-?shirts?|hoodies?|mugs?|"
    r"tumblers?|glassware|"
    r"aeropress\s*filters?|coffee\s*filters?|filter\s*paper|paper\s*filters?|seasoning\s*beans|chocolate\s*bar|"
    r"cocoa\s*powder|syrup|tea\s+(box|catalog|collection|set)|water\s*bottles?|"
    r"sample\s*(pack|set)|coffee\s*sample|matcha|chai"
    r")\b",
    re.IGNORECASE,
)


def _has_extracted_metadata(product: Any) -> bool:
    return any(
        _field(product, name)
        for name in (
            "producer_or_farm",
            "origin_country",
            "origin_region",
            "process",
            "elevation",
            "watchlist_match",
            "watchlist_tier",
        )
    ) or bool(_field(product, "variety", []) or _field(product, "tasting_notes", []))


def _has_coffee_metadata(product_type: str, tags: list[str], product: Any) -> bool:
    roaster_slug = str(_field(product, "roaster_slug") or "")
    if _COFFEE_PRODUCT_TYPE_RE.search(product_type):
        return True
    if any(
        tag
        in {
            "coffee",
            "whole bean coffees",
            "whole beans",
            "type:single origin",
            "type:blend",
        }
        or tag in _COUNTRY_TAGS
        or tag in _PROCESS_TAGS
        or tag.startswith(("origin:", "process:", "profile:"))
        or "wholebeancoffee" in tag
        or tag == "single origin"
        for tag in tags
    ):
        return True
    if roaster_slug in _SPARSE_COFFEE_ROASTERS:
        return True
    return _has_extracted_metadata(product)


def _has_rejected_metadata(vendor: str, product_type: str, tags: list[str]) -> bool:
    product_type_norm = product_type.lower().strip()
    if "wholesale" in vendor.lower():
        return True
    if product_type_norm in _REJECT_PRODUCT_TYPES:
        return True
    if (
        _REJECT_WHOLESALE_TYPE_RE.search(product_type)
        and "wholebeancoffee" not in product_type_norm
    ):
        return True
    if _REJECT_TYPE_RE.search(product_type):
        return True

    for tag in tags:
        if tag in _REJECT_TAGS:
            return True
        if tag in {"wholesale", "wholesale quick order"}:
            return True
        if _REJECT_NUMBERED_WHOLESALE_TAG_RE.search(tag):
            return True

    return False


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
    if extracted is not None and not _field(extracted, "is_coffee_product", False):
        return False

    title = str(_field(product, "title") or "")
    vendor = str(_field(product, "vendor") or "")
    product_type = str(_field(product, "product_type") or "")
    tags = _tags(product)

    if _has_rejected_metadata(vendor, product_type, tags):
        return False
    if _REJECT_TITLE_RE.search(title):
        return False

    return _has_coffee_metadata(product_type, tags, product) or bool(
        extracted is not None and _field(extracted, "is_coffee_product", False)
    )
