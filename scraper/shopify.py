"""Fetch products from Shopify /products.json API with pagination."""

import logging
import re
import time

import requests

from scraper.models import RoasterConfig, ShopifyProduct

log = logging.getLogger(__name__)

# Shopify returns max 250 products per page
PAGE_LIMIT = 250
REQUEST_TIMEOUT = 30
POLITE_DELAY = 1.0  # seconds between paginated requests

# Headers to look like a normal browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# ---------- Deterministic non-coffee filter ----------
# Products matching these patterns are definitively not roasted coffee beans.
# Applied before the LLM filter to catch obvious non-coffee items cheaply.
NON_COFFEE_PATTERNS = [
    r"nespresso", r"capsule", r"pod\b", r"cold brew bottle",
    r"chocolate.?covered", r"gift\s*card", r"gift\s*box", r"cometeer",
    r"energy drink", r"glassware", r"ceramics", r"\bmug\b", r"tumbler",
    r"t-shirt", r"\btote\b", r"\bhat\b", r"beanie", r"candle",
    r"instant coffee", r"drip bag",
]
_NON_COFFEE_RE = re.compile("|".join(NON_COFFEE_PATTERNS), re.IGNORECASE)

NON_COFFEE_PRODUCT_TYPES = {
    "gift cards", "glassware", "apparel", "merchandise", "accessories",
    "gear", "equipment",
}


def _is_non_coffee_deterministic(title: str, product_type: str, tags: list[str]) -> bool:
    """Return True if the product is definitively not roasted coffee beans."""
    if _NON_COFFEE_RE.search(title):
        return True
    if product_type.lower().strip() in NON_COFFEE_PRODUCT_TYPES:
        return True
    tags_str = " ".join(tags).lower()
    if _NON_COFFEE_RE.search(tags_str):
        return True
    return False


# ---------- Variant selection ----------

def _parse_variant_grams(variant: dict) -> int:
    """Extract weight in grams from a variant, trying grams field then title."""
    grams = variant.get("grams", 0) or 0
    if grams > 0:
        return grams

    # Try to parse from variant title (e.g. "250g", "10oz", "1lb")
    title = variant.get("title", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(oz|g|gr|lb|kg)\b", title, re.IGNORECASE)
    if not m:
        return 0

    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("g", "gr"):
        return int(val)
    if unit == "oz":
        return int(val * 28.3495)
    if unit == "lb":
        return int(val * 453.592)
    if unit == "kg":
        return int(val * 1000)
    return 0


def _variant_weight_label(variant: dict, grams: int) -> str:
    """Build a human-readable weight label for a variant."""
    raw_label = variant.get("title", "")
    if raw_label and re.search(r"\d+\s*(oz|g|gr|lb|kg)\b", raw_label, re.IGNORECASE):
        return raw_label
    if grams > 0:
        oz = grams / 28.3495
        return f"{oz:.0f}oz" if oz >= 1 else f"{grams}g"
    return ""


def _pick_variants(variants: list[dict]) -> tuple[dict, dict | None]:
    """Return (standard_variant, sample_variant_or_None).

    Standard: available variant closest to 300g (8-12oz retail bag).
    Sample: smallest available variant under 100g, if one exists.
    Falls back to unavailable variants if no available ones exist.
    """
    if not variants:
        return {}, None

    # Parse grams for all variants
    parsed = [(v, _parse_variant_grams(v)) for v in variants]

    # Separate available from unavailable
    available = [(v, g) for v, g in parsed if v.get("available", False)]
    pool = available if available else parsed  # fall back to all if none available

    # Standard variant: closest to 300g target, but prefer variants >= 100g
    TARGET_GRAMS = 300
    standard_candidates = [(v, g) for v, g in pool if g >= 100]
    if not standard_candidates:
        # No variant >= 100g — just pick the one closest to target from whole pool
        standard_candidates = pool

    if any(g > 0 for _, g in standard_candidates):
        # Among those with known weight, pick closest to target
        with_weight = [(v, g) for v, g in standard_candidates if g > 0]
        standard_v, standard_g = min(with_weight, key=lambda x: abs(x[1] - TARGET_GRAMS))
    else:
        # No weights known — fall back to first variant
        standard_v, standard_g = standard_candidates[0]

    # Sample variant: smallest variant under 100g (different from standard)
    sample_candidates = [(v, g) for v, g in pool if 0 < g < 100 and v is not standard_v]
    if sample_candidates:
        sample_v, sample_g = min(sample_candidates, key=lambda x: x[1])
    else:
        sample_v = None

    return standard_v, sample_v


def fetch_products(roaster: RoasterConfig) -> list[ShopifyProduct]:
    """Fetch all products from a Shopify store's /products.json endpoint.

    Paginates automatically if needed (>250 products).
    Returns parsed ShopifyProduct objects.
    """
    all_products: list[ShopifyProduct] = []
    page = 1

    while True:
        url = f"{roaster.base_url}/products.json"
        params = {"limit": PAGE_LIMIT, "page": page}

        log.info(
            "[%s] Fetching page %d from %s ...",
            roaster.slug, page, url,
        )

        try:
            resp = requests.get(
                url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error("[%s] Failed to fetch products: %s", roaster.slug, e)
            break

        data = resp.json()
        raw_products = data.get("products", [])

        if not raw_products:
            break

        for p in raw_products:
            variants = p.get("variants", [])

            # Tags come as comma-separated string or list depending on endpoint
            tags_raw = p.get("tags", [])
            if isinstance(tags_raw, str):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            else:
                tags = tags_raw

            title = p.get("title", "")
            product_type = p.get("product_type", "")

            # Availability: product is in-stock if ANY variant is available
            is_available = any(v.get("available", False) for v in variants)

            # Pick best standard + sample variants
            standard_v, sample_v = _pick_variants(variants)

            # Standard variant fields
            price = standard_v.get("price", "")
            weight_grams = _parse_variant_grams(standard_v)
            weight_label = _variant_weight_label(standard_v, weight_grams)

            # Sample variant fields
            sample_price = ""
            sample_grams = 0
            sample_label = ""
            if sample_v is not None:
                sample_price = sample_v.get("price", "")
                sample_grams = _parse_variant_grams(sample_v)
                sample_label = _variant_weight_label(sample_v, sample_grams)

            # Extract first image URL
            images = p.get("images", [])
            image_url = images[0].get("src", "") if images else ""

            product = ShopifyProduct(
                id=p["id"],
                title=title,
                handle=p.get("handle", ""),
                vendor=p.get("vendor", ""),
                product_type=product_type,
                tags=tags,
                body_html=p.get("body_html", "") or "",
                created_at=p.get("created_at", ""),
                is_available=is_available,
                price=price,
                weight_grams=weight_grams,
                weight_label=weight_label,
                sample_price=sample_price,
                sample_grams=sample_grams,
                sample_label=sample_label,
                image_url=image_url,
            )

            # Deterministic non-coffee filter — mark before LLM sees it
            if _is_non_coffee_deterministic(title, product_type, tags):
                log.debug(
                    "[%s] Deterministic filter: skipping non-coffee '%s'",
                    roaster.slug, title,
                )
                # Still include in results so scrape.py can skip it,
                # but we set a flag via the product_type convention.
                # Actually, we just won't append it — the LLM won't see it either.
                continue

            all_products.append(product)

        log.info(
            "[%s] Got %d products on page %d (total so far: %d)",
            roaster.slug, len(raw_products), page, len(all_products),
        )

        # If we got fewer than the limit, there are no more pages
        if len(raw_products) < PAGE_LIMIT:
            break

        page += 1
        time.sleep(POLITE_DELAY)

    log.info("[%s] Total products fetched: %d", roaster.slug, len(all_products))
    return all_products
