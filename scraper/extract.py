"""LLM-based extraction of structured coffee data from Shopify body_html.

Uses NVIDIA-hosted inference through the shared async LLM client. Results are
cached by a hash of the product title and stripped description so unchanged
products skip the LLM call.

Parallelism follows the pattern from ds_utils.py: asyncio.Semaphore +
asyncio.create_task + asyncio.as_completed with tqdm progress.
"""

import asyncio
import hashlib
import json
import logging
import re
import threading
from pathlib import Path

from bs4 import BeautifulSoup
from scraper.llm import NvidiaClient
from pydantic import ValidationError

from scraper.models import ExtractedCoffee, ShopifyProduct

log = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "llm_cache.json"

# The shared client limits concurrency and request rate across all LLM stages.
CONCURRENCY = 4

# Save cache every N new LLM calls
CACHE_SAVE_INTERVAL = 20


class ExtractionParseError(ValueError):
    """Raised when an LLM response cannot be parsed as extraction JSON."""


def _too_many_failures(failed_count: int, work_count: int) -> bool:
    """Return True when extraction failures are too risky to publish."""
    if failed_count <= 0 or work_count <= 0:
        return False
    failure_limit = max(1, (work_count + 9) // 10)
    return failed_count > failure_limit or failed_count == work_count


EXTRACTION_PROMPT = """\
You are a specialty coffee data extractor. Given a coffee product title and description \
from a roaster's website, extract structured information.

Product title: {title}

Product description:
{description}

Extract the following fields. If a field is not mentioned, return null for strings \
or an empty list for arrays. Be precise — only extract what is explicitly stated.

Return a JSON object with these fields:
- producer_or_farm (string | null): The name of the farm, estate, producer, or washing station. \
  Look for phrases like "from Finca ...", "produced by ...", "farm: ...", etc.
- origin_country (string | null): Country of origin
- origin_region (string | null): Specific region, department, or area within the country
- variety (list[string]): Coffee variety/cultivar names (e.g. "Geisha", "Bourbon", "SL-28", "Caturra"). \
  Normalize: "gesha" → "Geisha", "sl28"/"sl-28" → "SL-28"
- process (string | null): Processing method (e.g. "Washed", "Natural", "Honey", "Anaerobic Natural")
- elevation (string | null): Growing elevation/altitude (e.g. "1800 masl", "1600-1900m")
- tasting_notes (list[string]): Flavor/tasting notes (e.g. ["jasmine", "stone fruit", "dark chocolate"])
- is_coffee_product (bool): true ONLY if this is a bag of coffee beans (roasted or green/unroasted) \
  that a consumer would brew or roast at home. Return false for ALL of the following: \
  Nespresso/capsules/pods, ready-to-drink beverages (cold brew bottles, canned drinks), \
  chocolate-covered beans, gift cards/gift boxes, apparel (t-shirts, hoodies, hats, turtlenecks), \
  mugs/tumblers/glassware, brewing equipment/accessories, subscriptions/memberships, \
  stickers/posters/candles, instant coffee, drip bags, \
  or anything else that is not a bag of coffee beans.

Return ONLY the JSON object, no markdown fences or explanation."""


def _content_hash(text: str) -> str:
    """Return first 16 chars of sha256 hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load LLM cache (%s), starting fresh", e)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _parse_llm_response(text: str) -> ExtractedCoffee:
    """Parse LLM JSON response into ExtractedCoffee, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            # Models may use null for an absent array despite the prompt. Keep
            # unknown values empty without accepting strings or other shapes.
            for field in ("variety", "tasting_notes"):
                if data.get(field) is None:
                    data[field] = []
        return ExtractedCoffee.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("Failed to parse LLM response: %s\nResponse: %s", e, text[:200])
        raise ExtractionParseError(str(e)) from e


async def extract_products(
    products: list[ShopifyProduct],
    roaster_slug: str,
) -> dict[int, ExtractedCoffee]:
    """Extract structured data from products using NVIDIA with async concurrency.

    Uses asyncio.Semaphore to limit concurrent LLM calls (ds_utils pattern).
    Returns a dict mapping product ID -> ExtractedCoffee.
    """
    cache = _load_cache()
    client = NvidiaClient()
    results: dict[int, ExtractedCoffee] = {}
    cache_hits = 0
    llm_calls = 0
    failed_jobs: dict[str, str] = {}
    cache_lock = threading.Lock()

    # Pre-process: resolve cache hits and build work items
    work_items: list[tuple[ShopifyProduct, str, str]] = []  # (product, cache_key, prompt)

    for product in products:
        description_text = _strip_html(product.body_html)
        cache_key = _content_hash(f"{product.title}|{description_text}")

        if cache_key in cache:
            try:
                results[product.id] = ExtractedCoffee(**cache[cache_key])
                cache_hits += 1
                continue
            except ValidationError:
                pass

        if not description_text:
            results[product.id] = ExtractedCoffee(is_coffee_product=False)
            cache[cache_key] = results[product.id].model_dump()
            continue

        if len(description_text) > 2000:
            description_text = description_text[:2000] + "..."

        prompt = EXTRACTION_PROMPT.format(
            title=product.title,
            description=description_text,
        )
        work_items.append((product, cache_key, prompt))

    log.info(
        "[%s] %d cache hits, %d need LLM extraction (concurrency=%d)",
        roaster_slug, cache_hits, len(work_items), CONCURRENCY,
    )

    if not work_items:
        _save_cache(cache)
        return results

    # Async extraction with semaphore-bounded concurrency
    sem = asyncio.Semaphore(CONCURRENCY)
    completed = 0

    async def extract_one(product: ShopifyProduct, cache_key: str, prompt: str):
        nonlocal completed
        async with sem:
            try:
                response = await client.generate(prompt)
                extracted = _parse_llm_response(response)

                with cache_lock:
                    cache[cache_key] = extracted.model_dump()

                completed += 1
                if completed % CACHE_SAVE_INTERVAL == 0:
                    with cache_lock:
                        _save_cache(cache)
                    log.info(
                        "[%s] Progress: %d/%d LLM calls complete",
                        roaster_slug, completed, len(work_items),
                    )

                return product.id, extracted, None
            except ExtractionParseError as e:
                completed += 1
                with cache_lock:
                    cache.pop(cache_key, None)
                fallback = ExtractedCoffee(is_coffee_product=False)
                return product.id, fallback, str(e)
            except Exception as e:
                completed += 1
                with cache_lock:
                    cache.pop(cache_key, None)
                fallback = ExtractedCoffee(is_coffee_product=False)
                return product.id, fallback, str(e)

    # Create all tasks and run concurrently
    tasks = [
        asyncio.create_task(extract_one(product, cache_key, prompt))
        for product, cache_key, prompt in work_items
    ]

    for fut in asyncio.as_completed(tasks):
        product_id, extracted, err = await fut
        if err is None:
            results[product_id] = extracted
            llm_calls += 1
        else:
            failed_jobs[str(product_id)] = err
            results[product_id] = extracted or ExtractedCoffee(is_coffee_product=False)
            llm_calls += 1

    # Final cache save
    _save_cache(cache)

    if failed_jobs:
        log.warning(
            "[%s] %d extraction failures: %s",
            roaster_slug, len(failed_jobs),
            list(failed_jobs.values())[:3],
        )
        if _too_many_failures(len(failed_jobs), len(work_items)):
            raise RuntimeError(
                f"[{roaster_slug}] refusing to continue after "
                f"{len(failed_jobs)}/{len(work_items)} extraction failures"
            )

    log.info(
        "[%s] Extraction complete: %d products, %d LLM calls, %d cache hits, %d failures",
        roaster_slug, len(products), llm_calls, cache_hits, len(failed_jobs),
    )
    return results
