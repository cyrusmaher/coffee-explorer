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

# Gemini 3 Flash for Tier 2 proposal, Pro for review
MODEL_PROPOSE = "gemini-3-flash-preview"
MODEL_REVIEW = "gemini-3.1-pro-preview"


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


def _parse_json_response(text: str) -> list | dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned.strip())


def _find_watchlist_row(name: str, watchlist: list[dict]) -> dict | None:
    """Find a watchlist row by producer name or farm name."""
    norm = _normalize(name)
    for wp in watchlist:
        if _normalize(wp.get("producer_name", "")) == norm:
            return wp
        if _normalize(wp.get("farm_or_station", "")) == norm:
            return wp
    return None


async def _tier2_batch_match(
    unmatched: list[RoastedCoffeeProduct],
    watchlist: list[dict],
) -> dict[str, dict | None]:
    """Two-step LLM matching: Flash proposes, Pro reviews.

    Step 1 (Gemini 3 Flash): Propose candidate matches with strong bias toward
    "no match" as the default.
    Step 2 (Gemini 3.1 Pro): Review each proposed match — only keep matches
    where the product text contains grounded evidence of the association.

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

    watchlist_ref = "\n".join(
        f"- {p.get('producer_name', '?')} | {p.get('farm_or_station', '?')} | "
        f"{p.get('country', '?')} | {p.get('tier', '?')}"
        for p in watchlist
    )

    # Resolve cache hits and build uncached work items
    uncached_products: list[RoastedCoffeeProduct] = []
    for product in unmatched:
        cache_key = _content_hash(f"{product.title}|{product.producer_or_farm or ''}")
        if cache_key in cache:
            cached_val = cache[cache_key]
            if cached_val is None:
                results[product.product_url] = None
            else:
                wp = _find_watchlist_row(cached_val.get("producer_name", ""), watchlist)
                results[product.product_url] = wp
        else:
            uncached_products.append(product)

    log.info(
        "Tier 2: %d unmatched, %d cached, %d to process",
        len(unmatched), len(unmatched) - len(uncached_products), len(uncached_products),
    )

    if not uncached_products:
        _save_match_cache(cache)
        return results

    # --- Step 1: Flash proposes candidate matches ---
    batch_size = 10
    batches = [
        uncached_products[i:i + batch_size]
        for i in range(0, len(uncached_products), batch_size)
    ]
    sem = asyncio.Semaphore(10)
    proposals: list[tuple[RoastedCoffeeProduct, str]] = []  # (product, matched_name)
    completed = 0

    async def propose_batch(batch: list[RoastedCoffeeProduct]):
        nonlocal completed
        async with sem:
            products_text = "\n".join(
                f"{i+1}. Title: {p.title} | Producer: {p.producer_or_farm or 'unknown'} | "
                f"Country: {p.origin_country or 'unknown'} | Roaster: {p.roaster_name}"
                for i, p in enumerate(batch)
            )

            prompt = f"""\
You are matching roasted coffee products to a watchlist of elite producers.

IMPORTANT: Most products will NOT match any watchlist producer. "No match" is the \
expected answer for the majority of products. Only return a match if the product \
title or producer field explicitly names the producer, farm, or estate on the watchlist.

DO NOT match based on:
- Shared country or region alone (e.g. "Ethiopia Bensa" does NOT mean "Daye Bensa")
- Shared processing method or variety
- Vague associations or educated guesses

WATCHLIST (producer | farm | country | tier):
{watchlist_ref}

PRODUCTS:
{products_text}

Return a JSON array with one entry per product:
[{{"product_number": 1, "matched_producer": null}}, ...]

Set matched_producer to the exact watchlist producer name ONLY if you are certain \
the product is from that producer. Otherwise null.
Return ONLY the JSON array."""

            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_PROPOSE, contents=prompt,
                )
                matches = _parse_json_response(response.text)
                batch_proposals = []
                for match_result in matches:
                    idx = match_result.get("product_number", 0) - 1
                    if 0 <= idx < len(batch):
                        matched_name = match_result.get("matched_producer")
                        if matched_name:
                            batch_proposals.append((batch[idx], matched_name))
                        else:
                            # No match proposed — cache as None
                            product = batch[idx]
                            ck = _content_hash(f"{product.title}|{product.producer_or_farm or ''}")
                            cache[ck] = None
                            results[product.product_url] = None

                completed += 1
                if completed % 10 == 0:
                    log.info("Tier 2 propose: %d/%d batches", completed, len(batches))
                return batch_proposals

            except Exception as e:
                log.error("Tier 2 propose batch failed: %s", e)
                completed += 1
                for p in batch:
                    results[p.product_url] = None
                return []

    tasks = [asyncio.create_task(propose_batch(b)) for b in batches]
    for fut in asyncio.as_completed(tasks):
        batch_proposals = await fut
        proposals.extend(batch_proposals)

    _save_match_cache(cache)  # Save nulls from proposal step

    log.info("Tier 2 propose: %d candidates from %d products", len(proposals), len(uncached_products))

    if not proposals:
        return results

    # --- Step 2: Pro reviews each proposed match ---
    review_sem = asyncio.Semaphore(5)  # Pro is heavier, lower concurrency
    reviewed = 0

    async def review_one(product: RoastedCoffeeProduct, proposed_name: str):
        nonlocal reviewed
        async with review_sem:
            wp = _find_watchlist_row(proposed_name, watchlist)
            if not wp:
                return product, None

            prompt = f"""\
Review whether this coffee product is actually from the proposed producer.

PRODUCT:
- Title: {product.title}
- Extracted producer/farm: {product.producer_or_farm or 'none'}
- Country: {product.origin_country or 'unknown'}
- Roaster: {product.roaster_name}

PROPOSED MATCH:
- Producer: {wp.get('producer_name', '?')}
- Farm: {wp.get('farm_or_station', '?')}
- Country: {wp.get('country', '?')}

RULES:
- ACCEPT only if the product title or extracted producer explicitly references \
the producer's name, farm name, or a well-known abbreviation of it.
- REJECT if the match is based only on shared country, region, or variety.
- REJECT if there is no textual evidence in the product fields linking it to this producer.

Return a JSON object:
{{"verdict": "accept" or "reject", "evidence": "quote the specific text that justifies the match, or explain why rejected"}}
Return ONLY the JSON object."""

            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_REVIEW, contents=prompt,
                )
                result = _parse_json_response(response.text)
                reviewed += 1
                if reviewed % 10 == 0:
                    log.info("Tier 2 review: %d/%d reviewed", reviewed, len(proposals))

                if result.get("verdict") == "accept":
                    return product, wp
                else:
                    log.info(
                        "Tier 2 REJECTED: '%s' != %s (%s)",
                        product.title[:40], proposed_name, result.get("evidence", "")[:60],
                    )
                    return product, None

            except Exception as e:
                log.error("Tier 2 review failed for '%s': %s", product.title[:40], e)
                reviewed += 1
                return product, None

    review_tasks = [
        asyncio.create_task(review_one(product, proposed_name))
        for product, proposed_name in proposals
    ]

    for fut in asyncio.as_completed(review_tasks):
        product, wp = await fut
        cache_key = _content_hash(f"{product.title}|{product.producer_or_farm or ''}")
        if wp:
            results[product.product_url] = wp
            cache[cache_key] = {"producer_name": wp.get("producer_name")}
        else:
            results[product.product_url] = None
            cache[cache_key] = None

    _save_match_cache(cache)

    match_count = sum(1 for v in results.values() if v is not None)
    log.info(
        "Tier 2 final: %d proposed -> %d accepted out of %d unmatched",
        len(proposals), match_count, len(unmatched),
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
