"""Fetch products from Shopify /products.json API with pagination."""

import logging
import re
from fractions import Fraction

from bs4 import BeautifulSoup
from scraper.catalog import CatalogError, fetch_shopify_catalog, get, tags as product_tags

from scraper.models import ExtractedCoffee, RoasterConfig, ShopifyProduct
from scraper.product_filter import is_excluded_product

log = logging.getLogger(__name__)


def parse_granja_details(html: str) -> ExtractedCoffee:
    """Granja stores coffee facts in page accordions, outside Shopify body_html."""
    soup = BeautifulSoup(html, "html.parser")
    specs = {}
    for detail in soup.select("details"):
        title = detail.select_one("summary")
        value = detail.select_one(".accordion__content")
        if title and value:
            specs[title.get_text(" ", strip=True).upper()] = value.get_text(" ", strip=True)
    if not all(specs.get(key) for key in ("FARM", "VARIETY", "PROCESO")):
        raise CatalogError("Granja product is missing its coffee specifications")
    return ExtractedCoffee(
        producer_or_farm=specs["FARM"], variety=[specs["VARIETY"]],
        process=specs["PROCESO"].title(), elevation=specs.get("ALTITUDE") or None,
        tasting_notes=[v.strip() for v in re.split(r"[,;]", specs.get("CUP PROFLE", specs.get("CUP PROFILE", ""))) if v.strip()],
    )

# ---------- Variant selection ----------

_WEIGHT_RE = re.compile(
    r"(?<![\d.,])(?:(?P<count>\d+)\s*[x×]\s*)?"
    r"(?P<weight>(?:\d+\s+)?\d+\s*/\s*\d+|\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<unit>oz|grams?|gr|g|lbs?|kg)s?\b"
    r"(?:\s*[x×]\s*(?P<count_after>\d+))?",
    re.IGNORECASE,
)


def _parse_variant_grams(variant: dict) -> int:
    """Prefer labeled net bag weight; Shopify grams may include packaging."""
    title = variant.get("title", "")
    m = _WEIGHT_RE.search(title)
    if not m:
        return variant.get("grams", 0) or 0

    weight = re.sub(r"\s*/\s*", "/", m.group("weight").replace(",", ""))
    try:
        # Fraction accepts decimals and simple fractions; summing also handles
        # mixed sizes such as "1 1/2lb".
        val = sum(Fraction(part) for part in weight.split())
    except (ValueError, ZeroDivisionError):
        return variant.get("grams", 0) or 0
    val = float(val) * int(m.group("count") or 1) * int(m.group("count_after") or 1)
    unit = m.group("unit").lower().rstrip("s")  # normalize plurals
    if unit in ("g", "gr", "gram"):
        return int(val)
    if unit == "oz":
        return int(val * 28.3495)
    if unit == "lb":
        return int(val * 453.592)
    if unit == "kg":
        return int(val * 1000)
    return 0


def _variant_weight_label(variant: dict, grams: int) -> str:
    """Extract a concise weight label like '12oz' or '250g' from a variant."""
    raw_label = variant.get("title", "")
    m = _WEIGHT_RE.search(raw_label)
    if m:
        return m.group(0)  # just "100gr", "12oz", "5LB" — not the surrounding text
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

    # Parse grams for all variants; exclude zero-priced placeholder variants
    parsed = [
        (v, _parse_variant_grams(v))
        for v in variants
        if float(v.get("price", "0") or "0") > 0
    ]
    if not parsed:
        # All variants are zero-priced — fall back to raw list
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
    raw_products = fetch_shopify_catalog(roaster.base_url, roaster.collection)
    for p in raw_products:
        variants = p.get("variants", [])

        tags = product_tags(p)

        title = p.get("title", "")
        product_type = p.get("product_type", "")

        # Skip products tagged as archived by the roaster
        if any(t.lower() == "archived" for t in tags):
            continue

        # Availability: product is in-stock if ANY variant is available
        is_available = any(v.get("available", False) for v in variants)

        # Pick best standard + sample variants
        standard_v, sample_v = _pick_variants(variants)

        # Standard variant fields
        price_raw = float(standard_v.get("price", "0") or "0")
        price = f"{price_raw / roaster.price_divisor:.2f}" if price_raw else ""
        weight_grams = _parse_variant_grams(standard_v)
        weight_label = _variant_weight_label(standard_v, weight_grams)

        # Sample variant fields
        sample_price = ""
        sample_grams = 0
        sample_label = ""
        if sample_v is not None:
            sample_raw = float(sample_v.get("price", "0") or "0")
            sample_price = f"{sample_raw / roaster.price_divisor:.2f}" if sample_raw else ""
            sample_grams = _parse_variant_grams(sample_v)
            sample_label = _variant_weight_label(sample_v, sample_grams)

        # Extract first image URL
        images = p.get("images", [])
        image_url = images[0].get("src", "") if images else ""
        if not image_url and p.get("image"):
            image_url = p["image"].get("src", "")

        product = ShopifyProduct(
            roaster_slug=roaster.slug,
            source_collection=roaster.collection,
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

        if roaster.slug == "granja-la-esperanza" and not is_excluded_product(product):
            response = get(f"{roaster.base_url}/products/{product.handle}")
            try:
                product.structured_coffee = parse_granja_details(response.text)
            finally:
                response.close()

        all_products.append(product)

    log.info("[%s] Total products fetched: %d", roaster.slug, len(all_products))
    return all_products
