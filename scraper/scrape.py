"""Main orchestrator for the coffee-explorer scraper pipeline.

Steps:
  1. Fetch products from all Shopify roasters (sync, fast)
  2. Extract structured data via Gemini LLM (async, parallel)
  3. Match against producer watchlist
  4. Write docs/roasted-data.json for the frontend
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import requests as req_sync

from scraper.extract import extract_products
from scraper.match import load_watchlist, match_products
from scraper.models import RoastedCoffeeProduct
from scraper.roasters import ROASTERS
from scraper.shopify import fetch_products

LIVENESS_CONCURRENCY = 20
LIVENESS_TIMEOUT = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT_FILE = DOCS_DIR / "roasted-data.json"


async def _filter_locked(
    products: list[RoastedCoffeeProduct],
) -> list[RoastedCoffeeProduct]:
    """Remove products whose storefront pages are locked or 404.

    Uses requests (sync) in a thread pool, grouped by host with polite
    delays. Hosts checked in parallel via asyncio threads.
    """
    import time
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import urlparse

    # Only check sold-out products — in-stock products are always live
    candidates = [p for p in products if not p.is_available]
    if not candidates:
        log.info("Liveness check: no sold-out products to check")
        return products

    by_host: dict[str, list[RoastedCoffeeProduct]] = defaultdict(list)
    for p in candidates:
        by_host[urlparse(p.product_url).netloc].append(p)

    locked_urls: set[str] = set()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def check_host(host: str, host_products: list[RoastedCoffeeProduct]) -> list[str]:
        dead: list[str] = []
        session = req_sync.Session()
        session.headers.update(headers)
        # Warm session with homepage to get cookies (avoids 429s)
        try:
            session.get(f"https://{host}/", timeout=LIVENESS_TIMEOUT)
            time.sleep(0.5)
        except Exception:
            pass
        for p in host_products:
            try:
                resp = session.get(p.product_url, timeout=LIVENESS_TIMEOUT)
                if resp.status_code == 404:
                    dead.append(p.product_url)
                elif resp.status_code == 200 and "<h1>Locked</h1>" in resp.text[:50_000]:
                    dead.append(p.product_url)
                elif resp.status_code == 429:
                    # Rate limited — stop checking this host, keep remaining products
                    log.warning("  429 rate-limited on %s, skipping remaining", host)
                    break
            except Exception:
                pass
            time.sleep(1.5)
        return dead

    log.info("Liveness check: scanning %d sold-out products across %d hosts...",
             len(candidates), len(by_host))

    with ThreadPoolExecutor(max_workers=len(by_host)) as pool:
        futures = {pool.submit(check_host, host, prods): host for host, prods in by_host.items()}
        for fut in futures:
            dead = fut.result()
            locked_urls.update(dead)

    if locked_urls:
        log.info("Liveness check: %d locked/dead pages removed", len(locked_urls))
        for url in sorted(locked_urls):
            log.info("  Removed: %s", url)
    else:
        log.info("Liveness check: all pages accessible")

    return [p for p in products if p.product_url not in locked_urls]


async def run() -> None:
    watchlist = load_watchlist()
    all_products: list[RoastedCoffeeProduct] = []

    for roaster in ROASTERS:
        log.info("=" * 60)
        log.info("Processing roaster: %s (%s)", roaster.name, roaster.slug)
        log.info("=" * 60)

        # Step 1: Fetch from Shopify (sync — fast, one HTTP call per page)
        raw_products = fetch_products(roaster)
        if not raw_products:
            log.warning("[%s] No products fetched, skipping", roaster.slug)
            continue

        # Step 2: LLM extraction (async, parallel with semaphore)
        extractions = await extract_products(raw_products, roaster.slug)

        # Step 3: Assemble RoastedCoffeeProduct records
        for product in raw_products:
            extracted = extractions.get(product.id)
            if extracted and not extracted.is_coffee_product:
                continue

            # Skip archived products (sold out + $0 price = dead listing)
            if not product.is_available and not float(product.price or "0"):
                continue

            record = RoastedCoffeeProduct(
                roaster_slug=roaster.slug,
                roaster_name=roaster.name,
                product_url=f"{roaster.base_url}/products/{product.handle}",
                image_url=product.image_url,
                title=product.title,
                handle=product.handle,
                vendor=product.vendor,
                product_type=product.product_type,
                tags=product.tags,
                is_available=product.is_available,
                price=product.price,
                weight_grams=product.weight_grams,
                weight_label=product.weight_label,
                sample_price=product.sample_price,
                sample_grams=product.sample_grams,
                sample_label=product.sample_label,
                created_at=product.created_at,
            )

            if extracted:
                record.producer_or_farm = extracted.producer_or_farm
                record.origin_country = extracted.origin_country
                record.origin_region = extracted.origin_region
                record.variety = extracted.variety
                record.process = extracted.process
                record.elevation = extracted.elevation
                record.tasting_notes = extracted.tasting_notes
                record.is_coffee_product = extracted.is_coffee_product

            all_products.append(record)

    log.info("=" * 60)
    log.info("Total coffee products: %d", len(all_products))

    # Step 4: Liveness check — remove locked/dead product pages
    all_products = await _filter_locked(all_products)

    # Step 5: Watchlist matching (async for Tier 2 LLM)
    all_products = await match_products(all_products, watchlist)

    # Step 6: Write output
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output_data = [p.model_dump() for p in all_products]
    OUTPUT_FILE.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    matched_count = sum(1 for p in all_products if p.watchlist_match)
    log.info("Wrote %d products to %s", len(all_products), OUTPUT_FILE)
    log.info("Watchlist matches: %d", matched_count)
    log.info("Done.")


if __name__ == "__main__":
    asyncio.run(run())
