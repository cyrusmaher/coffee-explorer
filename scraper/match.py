"""Producer watchlist matching — deterministic (Tier 1) + LLM fallback (Tier 2).

Loads data/producer_watchlist.csv and matches extracted product data against it.
Tier 2 uses async concurrency (same pattern as extract.py).
"""

import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import unicodedata
from pathlib import Path

from google import genai
from pydantic import ValidationError

from scraper.models import ExtractedCoffee, RoastedCoffeeProduct

log = logging.getLogger(__name__)

WATCHLIST_FILE = Path(__file__).resolve().parent.parent / "data" / "producer_watchlist.csv"
MATCH_CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "match_cache.json"

# Gemini 3 Flash for Tier 2 matching
MODEL = "gemini-3-flash-preview"


def _normalize(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if not s:
        return ""
    # Decompose unicode, strip combining marks (accents)
    nfkd = unicodedata.normalize("NFKD", s)
    without_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", without_accents.lower().strip())


def load_watchlist() -> list[dict]:
    """Load producer watchlist CSV. Fails loudly if missing."""
    assert WATCHLIST_FILE.exists(), f"Watchlist not found: {WATCHLIST_FILE}"
    with open(WATCHLIST_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    log.info("Loaded %d producers from watchlist", len(rows))
    return rows



# Words too common in coffee product text to use as standalone match terms
_STOPWORDS = frozenset({
    "coffee", "coffees", "cafe", "farm", "farms", "family", "estate", "estates",
    "cooperative", "coop", "union", "project", "special", "reserve", "natural",
    "washed", "honey", "roast", "roasted", "blend", "single", "origin",
    "bourbon", "geisha", "gesha", "java", "nova", "halo", "goro",
})


def _build_match_terms(producer: dict) -> list[str]:
    """Build a list of normalized search terms for a watchlist producer.

    We want to match against product titles, producer_or_farm fields, etc.
    Generates terms from both producer_name and farm_or_station.
    Only uses full names — no single last-name extraction (too many false positives).
    """
    terms = []

    # Producer full name only (no last-name splitting — "family", "savage", etc. are too generic)
    name = producer.get("producer_name", "").strip()
    if name:
        terms.append(_normalize(name))

    # Farm/station name — split on "/" for multi-farm entries
    farm = producer.get("farm_or_station", "").strip()
    if farm:
        for part in farm.split("/"):
            part = part.strip()
            if part:
                normalized = _normalize(part)
                terms.append(normalized)
                # Also try without "Finca" / "Hacienda" prefix
                if normalized.startswith("finca "):
                    terms.append(normalized[6:])
                elif normalized.startswith("hacienda "):
                    terms.append(normalized[9:])

    # Filter: must be 5+ chars, not a stopword
    terms = [t for t in terms if len(t) >= 5 and t not in _STOPWORDS]
    return terms


def _tier1_match(
    product: RoastedCoffeeProduct,
    watchlist: list[dict],
    match_terms: dict[int, list[str]],
) -> dict | None:
    """Deterministic string matching. Returns matched watchlist row or None."""
    # Build search corpus from product
    search_text = _normalize(" ".join([
        product.title or "",
        product.producer_or_farm or "",
        product.vendor or "",
        " ".join(product.tags),
    ]))

    for i, producer in enumerate(watchlist):
        for term in match_terms[i]:
            if term in search_text:
                # Avoid false positives from very common words
                return producer

    return None


def _load_match_cache() -> dict:
    if MATCH_CACHE_FILE.exists():
        try:
            return json.loads(MATCH_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_match_cache(cache: dict) -> None:
    MATCH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MATCH_CACHE_FILE.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def _tier2_batch_match(
    unmatched: list[RoastedCoffeeProduct],
    watchlist: list[dict],
) -> dict[str, dict | None]:
    """LLM-based matching for products that didn't match in Tier 1.

    Sends batches to Gemini with the full watchlist for fuzzy matching.
    Uses async concurrency (semaphore pattern from ds_utils.py).
    Returns dict mapping product_url -> matched watchlist row or None.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_AI_LOCATION", "global")
    api_key = os.environ.get("GEMINI_API_KEY")

    if project:
        client = genai.Client(vertexai=True, project=project, location=location)
    elif api_key:
        client = genai.Client(api_key=api_key)
    else:
        log.warning("No GOOGLE_CLOUD_PROJECT or GEMINI_API_KEY — skipping Tier 2 matching")
        return {}

    cache = _load_match_cache()
    results: dict[str, dict | None] = {}

    # Build concise watchlist reference
    watchlist_ref = "\n".join(
        f"- {p.get('producer_name', '?')} | {p.get('farm_or_station', '?')} | "
        f"{p.get('country', '?')} | {p.get('tier', '?')}"
        for p in watchlist
    )

    # Build batches, resolving cache hits first
    batch_size = 10
    batches: list[list[RoastedCoffeeProduct]] = []  # uncached batches to send to LLM

    for batch_start in range(0, len(unmatched), batch_size):
        batch = unmatched[batch_start:batch_start + batch_size]
        uncached = []
        for product in batch:
            cache_key = _content_hash(
                f"{product.title}|{product.producer_or_farm or ''}"
            )
            if cache_key in cache:
                cached_val = cache[cache_key]
                if cached_val is None:
                    results[product.product_url] = None
                else:
                    for wp in watchlist:
                        if wp.get("producer_name") == cached_val.get("producer_name"):
                            results[product.product_url] = wp
                            break
                    else:
                        results[product.product_url] = None
            else:
                uncached.append(product)
        if uncached:
            batches.append(uncached)

    log.info(
        "Tier 2: %d unmatched products, %d cached, %d batches to send (concurrency=10)",
        len(unmatched), len(unmatched) - sum(len(b) for b in batches), len(batches),
    )

    if not batches:
        _save_match_cache(cache)
        return results

    sem = asyncio.Semaphore(10)
    completed = 0

    async def match_batch(uncached: list[RoastedCoffeeProduct]):
        nonlocal completed
        async with sem:
            products_text = "\n".join(
                f"{i+1}. Title: {p.title} | Producer: {p.producer_or_farm or 'unknown'} | "
                f"Country: {p.origin_country or 'unknown'} | Roaster: {p.roaster_name}"
                for i, p in enumerate(uncached)
            )

            prompt = f"""\
You are matching roasted coffee products to a watchlist of elite coffee producers.

WATCHLIST (producer | farm | country | tier):
{watchlist_ref}

PRODUCTS TO MATCH:
{products_text}

For each numbered product, determine if it comes from any producer on the watchlist.
Consider that product titles and descriptions may use different naming conventions
(e.g. "Esmeralda" for "Hacienda La Esmeralda", "Paraiso 92" for "Granja Paraiso 92").

Return a JSON array with one object per product:
[
  {{"product_number": 1, "matched_producer": "producer_name or null", "confidence": "high/medium/low"}},
  ...
]

Only include matches with high or medium confidence. Return null for low confidence or no match.
Return ONLY the JSON array."""

            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                )

                text = response.text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```\w*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)
                text = text.strip()

                matches = json.loads(text)
                batch_results = {}

                for match_result in matches:
                    idx = match_result.get("product_number", 0) - 1
                    if 0 <= idx < len(uncached):
                        product = uncached[idx]
                        matched_name = match_result.get("matched_producer")
                        confidence = match_result.get("confidence", "low")
                        cache_key = _content_hash(
                            f"{product.title}|{product.producer_or_farm or ''}"
                        )

                        if matched_name and confidence in ("high", "medium"):
                            found = False
                            for wp in watchlist:
                                if (_normalize(wp.get("producer_name", "")) == _normalize(matched_name or "") or
                                    _normalize(wp.get("farm_or_station", "")) == _normalize(matched_name or "")):
                                    batch_results[product.product_url] = wp
                                    cache[cache_key] = {"producer_name": wp.get("producer_name")}
                                    found = True
                                    break
                            if not found:
                                batch_results[product.product_url] = None
                                cache[cache_key] = None
                        else:
                            batch_results[product.product_url] = None
                            cache[cache_key] = None

                completed += 1
                if completed % 10 == 0:
                    _save_match_cache(cache)
                    log.info("Tier 2 progress: %d/%d batches complete", completed, len(batches))

                return batch_results

            except Exception as e:
                log.error("Tier 2 batch match failed: %s", e)
                completed += 1
                return {p.product_url: None for p in uncached}

    tasks = [asyncio.create_task(match_batch(batch)) for batch in batches]

    for fut in asyncio.as_completed(tasks):
        batch_results = await fut
        results.update(batch_results)

    _save_match_cache(cache)

    match_count = sum(1 for v in results.values() if v is not None)
    log.info(
        "Tier 2 matching: %d products, %d batches, %d matches",
        len(unmatched), len(batches), match_count,
    )
    return results


async def match_products(
    products: list[RoastedCoffeeProduct],
    watchlist: list[dict],
) -> list[RoastedCoffeeProduct]:
    """Apply Tier 1 (deterministic) + Tier 2 (LLM) matching to all products.

    Mutates and returns the products list with watchlist_match/tier fields set.
    """
    match_terms = {i: _build_match_terms(p) for i, p in enumerate(watchlist)}

    tier1_matched = 0
    unmatched_products: list[RoastedCoffeeProduct] = []

    def _apply_match(product: RoastedCoffeeProduct, matched: dict) -> None:
        product.watchlist_match = matched.get("producer_name") or matched.get("farm_or_station", "")
        product.watchlist_farm = matched.get("farm_or_station", "")
        product.watchlist_tier = matched.get("tier", "")
        product.watchlist_credential_type = matched.get("credential_type", "")
        product.watchlist_credential_detail = matched.get("credential_detail", "")
        product.watchlist_notes = matched.get("notes", "")

    for product in products:
        if not product.is_coffee_product:
            continue

        matched = _tier1_match(product, watchlist, match_terms)
        if matched:
            _apply_match(product, matched)
            tier1_matched += 1
        else:
            unmatched_products.append(product)

    log.info("Tier 1 matching: %d/%d products matched", tier1_matched, len(products))

    if unmatched_products:
        tier2_results = await _tier2_batch_match(unmatched_products, watchlist)
        for product in unmatched_products:
            matched = tier2_results.get(product.product_url)
            if matched:
                _apply_match(product, matched)

    total_matched = sum(1 for p in products if p.watchlist_match)
    log.info("Total watchlist matches: %d/%d products", total_matched, len(products))

    return products
