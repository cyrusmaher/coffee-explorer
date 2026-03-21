"""Main orchestrator for the coffee-explorer scraper pipeline.

Steps:
  1. Fetch products from all Shopify roasters (sync, fast)
  2. Extract structured data via Gemini LLM (async, parallel)
  3. Match against producer watchlist
  4. Write docs/roasted-data.json for the frontend
  5. (Separate step) Run `python -m scraper.liveness` to remove locked pages
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from scraper.extract import extract_products
from scraper.match import load_watchlist, match_products
from scraper.models import RoastedCoffeeProduct
from scraper.roasters import ROASTERS
from scraper.shopify import fetch_products

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT_FILE = DOCS_DIR / "roasted-data.json"


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

    # Step 4: Watchlist matching (async for Tier 2 LLM)
    all_products = await match_products(all_products, watchlist)

    # Step 5: Write output
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output_data = [p.model_dump() for p in all_products]
    OUTPUT_FILE.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    matched_count = sum(1 for p in all_products if p.watchlist_match)
    log.info("Wrote %d products to %s", len(all_products), OUTPUT_FILE)
    log.info("Watchlist matches: %d", matched_count)
    log.info("Done. Run `python -m scraper.liveness` to check for locked pages.")


if __name__ == "__main__":
    asyncio.run(run())
