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
            # Extract first variant price and weight
            variants = p.get("variants", [])
            first_variant = variants[0] if variants else {}
            price = first_variant.get("price", "")
            weight_grams = first_variant.get("grams", 0) or 0

            # Only use variant title as weight_label if it looks like a weight
            # (e.g. "10oz", "250g", "1lb", "2 oz") — not "Whole Bean", "Ground", etc.
            raw_label = first_variant.get("title", "")
            if raw_label and re.search(r"\d+\s*(oz|g|gr|lb|kg)\b", raw_label, re.IGNORECASE):
                weight_label = raw_label
            elif weight_grams > 0:
                # Derive label from grams
                oz = weight_grams / 28.3495
                weight_label = f"{oz:.0f}oz" if oz >= 1 else f"{weight_grams}g"
            else:
                weight_label = ""

            # Extract first image URL
            images = p.get("images", [])
            image_url = images[0].get("src", "") if images else ""

            # Tags come as comma-separated string or list depending on endpoint
            tags_raw = p.get("tags", [])
            if isinstance(tags_raw, str):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            else:
                tags = tags_raw

            product = ShopifyProduct(
                id=p["id"],
                title=p.get("title", ""),
                handle=p.get("handle", ""),
                vendor=p.get("vendor", ""),
                product_type=p.get("product_type", ""),
                tags=tags,
                body_html=p.get("body_html", "") or "",
                created_at=p.get("created_at", ""),
                price=price,
                weight_grams=weight_grams,
                weight_label=weight_label,
                image_url=image_url,
            )
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
