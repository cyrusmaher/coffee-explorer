"""LLM-based extraction of structured coffee data from Shopify body_html.

Uses Gemini 3 Flash via the google-genai SDK (Vertex AI backend with global
endpoint). Caches results keyed by sha256(body_html)[:16] so unchanged
descriptions skip the LLM call.

Parallelism follows the pattern from ds_utils.py: asyncio.Semaphore +
asyncio.create_task + asyncio.as_completed with tqdm progress.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path

from bs4 import BeautifulSoup
from google import genai
from pydantic import ValidationError

from scraper.models import ExtractedCoffee, ShopifyProduct

log = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "llm_cache.json"

# Gemini 3 Flash — fast, cheap, structured output
MODEL = "gemini-3-flash-preview"

# Concurrency: Gemini Flash can handle high parallelism
CONCURRENCY = 30

# Save cache every N new LLM calls
CACHE_SAVE_INTERVAL = 20

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


def _init_client() -> genai.Client:
    """Initialize Gemini client via Vertex AI (global endpoint) or API key fallback."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_AI_LOCATION", "global")

    if project:
        log.info("Using Vertex AI: project=%s, location=%s", project, location)
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        log.info("Using Gemini API key (direct)")
        return genai.Client(api_key=api_key)

    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT + VERTEX_AI_LOCATION=global for Vertex AI, "
        "or GEMINI_API_KEY for direct API access."
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
        return ExtractedCoffee(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("Failed to parse LLM response: %s\nResponse: %s", e, text[:200])
        return ExtractedCoffee(is_coffee_product=True)


async def extract_products(
    products: list[ShopifyProduct],
    roaster_slug: str,
) -> dict[int, ExtractedCoffee]:
    """Extract structured data from products using Gemini Flash with async concurrency.

    Uses asyncio.Semaphore to limit concurrent LLM calls (ds_utils pattern).
    Returns a dict mapping product ID -> ExtractedCoffee.
    """
    cache = _load_cache()
    client = _init_client()
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
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                )
                extracted = _parse_llm_response(response.text)

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
            except Exception as e:
                completed += 1
                return product.id, None, str(e)

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
            results[product_id] = ExtractedCoffee(is_coffee_product=True)
            llm_calls += 1

    # Final cache save
    _save_cache(cache)

    if failed_jobs:
        log.warning(
            "[%s] %d extraction failures: %s",
            roaster_slug, len(failed_jobs),
            list(failed_jobs.values())[:3],
        )

    log.info(
        "[%s] Extraction complete: %d products, %d LLM calls, %d cache hits, %d failures",
        roaster_slug, len(products), llm_calls, cache_hits, len(failed_jobs),
    )
    return results
