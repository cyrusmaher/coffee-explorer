"""Shopify adapters for the green explorer's existing output schemas.

The green repository supplies its record builders and output paths. Transport,
current storefront selectors, cache freshness and publication checks live here
alongside the roasted parsers so both repositories exercise the same fetcher.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import re
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from scraper.catalog import CatalogError, fetch_shopify_catalog, get, tags
from scraper.shopify import _parse_variant_grams

log = logging.getLogger(__name__)
SOURCES = {
    "sweetmarias": ("https://www.sweetmarias.com", "green-coffee"),
    "cbc": ("https://coffeebeancorral.com", "all-green-coffee-beans-collection"),
    "hacea": ("https://haceacoffee.com", "green-coffee"),
    "royal": ("https://royalcoffee.com", "all-coffees"),
}
CACHE_VERSION = 1
CACHE_TTL = 24 * 60 * 60


def text(node) -> str:
    if node is None:
        return ""
    # BeautifulSoup excludes template strings from get_text by default.
    fragment = BeautifulSoup(node.decode_contents(), "html.parser")
    for template in fragment.select("template"):
        template.replace_with(BeautifulSoup(template.decode_contents(), "html.parser"))
    for element in fragment.select("script,style"):
        element.decompose()
    return fragment.get_text(" ", strip=True)


def table_specs(soup) -> dict:
    specs = {}
    for row in soup.select("tr"):
        cells = row.select("th,td")
        if len(cells) == 2:
            specs[text(cells[0]).strip(": ").lower()] = text(cells[1])
    return specs


def fetch_infographic(uuid: str) -> dict:
    if not re.fullmatch(r"[0-9a-f-]{36}", uuid):
        raise CatalogError("Invalid infographic identifier")
    response = get(f"https://e.infogram.com/{uuid}")
    match = re.search(r"window\.infographicData\s*=\s*", response.text)
    if not match:
        raise CatalogError(f"Missing infographic data: {uuid}")
    try:
        data, _ = json.JSONDecoder().raw_decode(response.text, match.end())
        return data
    except ValueError as exc:
        raise CatalogError(f"Malformed infographic: {uuid}") from exc
    finally:
        response.close()


def source_infographic(data: dict) -> dict:
    """Read ordered label/value pairs from Royal's Source graphic."""
    try:
        content = data["elements"]["content"]["content"]
        entities = content["entities"]
        labels = []
        for block in content["blocks"].values():
            for entity_id in block.get("entities", []):
                entity = entities.get(str(entity_id), {})
                if entity.get("type") == "TEXT":
                    value = " ".join(b.get("text", "") for b in entity.get("props", {}).get("content", {}).get("blocks", [])).strip()
                    if value:
                        labels.append(value)
    except (KeyError, TypeError, AttributeError) as exc:
        raise CatalogError("Malformed Royal source infographic") from exc
    names = {"grower": "grower", "producer": "grower", "region": "region", "elevation": "altitude",
             "altitude": "altitude", "cultivar": "variety", "variety": "variety", "process": "process", "harvest": "harvest"}
    result = {}
    for label, value in zip(labels, labels[1:]):
        key = names.get(label.lower().strip(": "))
        if key and value.lower().strip(": ") not in names:
            result[key] = value
    if not result:
        raise CatalogError("Royal source infographic contains no coffee specifications")
    return result


def parse_detail(source: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    if not soup.select_one("h1"):
        raise CatalogError(f"{source}: missing product title")
    if source == "sweetmarias":
        chart = soup.select_one('[data-chart-id="cupping-chart"]')
        if not chart or not soup.select_one(".additional-attributes-table"):
            raise CatalogError("sweetmarias: missing coffee charts/specifications")
        specs = table_specs(soup)
        flavor = soup.select_one('[data-chart-id="flavor-chart"]')
        # Keep the legacy record builder's field names/capitalization.
        names = ["Region", "Processing", "Drying Method", "Arrival date", "Lot size",
                 "Bag size", "Packaging", "Cultivar Detail", "Grade", "Appearance",
                 "Roast Recommendations", "Type", "Recommended for Espresso"]
        detail = {
            "specs": {name: specs.get(name.lower(), "") for name in names},
            "overview_attrs": {name: specs.get(name.lower(), "") for name in
                               ["Process Method", "Cultivar", "Farm Gate"]},
            "total_score": chart.get("data-chart-score", ""),
            "cupping_values": chart.get("data-chart-value", ""),
            "cupper_correction": chart.get("data-cupper-correction", ""),
            "flavor_values": flavor.get("data-chart-value", "") if flavor else "",
            "full_cupping_notes": text(soup.select_one(".overview-notes__body")),
            "farm_notes": text(soup.select_one(".origin-notes__notes")),
        }
        detail["specs"]["Recommended for Espresso"] = specs.get("recommended espresso", "")
        return detail
    if source == "cbc":
        rows = soup.select(".pdp-spec-row")
        if not rows and not soup.select_one("product-page[data-id]"):
            raise CatalogError("cbc: missing coffee product content")
        specs = {text(row.select_one("dt")).lower(): text(row.select_one("dd")) for row in rows}
        mapping = {"Local Region": "local region", "Category": "bean category", "Process": "process",
                   "Variety": "variety", "Altitude": "altitude", "Harvest": "harvest",
                   "Organic Certification": "organic", "Fair Trade": "fair trade",
                   "Rainforest Alliance Certified": "rainforest alliance", "Decaffeinated": "decaf"}
        detail = {"country": specs.get("country", ""),
                  "specs": {key: specs.get(value, "") for key, value in mapping.items()},
                  "attributes": {}, "flavors": {}}
        for row in soup.select(".pdp-fp-row"):
            key = text(row.select_one(".pdp-fp-label"))
            value = text(row.select_one(".pdp-fp-val"))
            if re.fullmatch(r"\d+(?:\.\d+)?/[47]", value):
                group = "attributes" if value.endswith("/7") else "flavors"
                detail[group][key] = value.split("/")[0]
        for node in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(node.string or "")
            except ValueError:
                continue
            if isinstance(data, dict) and data.get("@type") == "Product":
                rating = data.get("aggregateRating") or {}
                detail["rating"] = str(rating.get("ratingValue", ""))
                detail["review_count"] = str(rating.get("reviewCount", ""))
        return detail
    if source == "hacea":
        sections = {text(d.select_one("summary")).lower(): d.select_one(".accordion-content")
                    for d in soup.select("details") if d.select_one("summary")}
        details = sections.get("details") or soup.select_one("#tab1")
        if not details:
            raise CatalogError("hacea: missing coffee specifications")
        specs = {}
        for strong in details.select("strong"):
            parts = []
            for sibling in strong.next_siblings:
                if getattr(sibling, "name", None) in {"br", "strong", "p"}:
                    break
                parts.append(text(sibling) if getattr(sibling, "name", None) else str(sibling))
            specs[text(strong).strip(": ")] = " ".join(parts).strip()
        lookup = {key.lower(): value for key, value in specs.items()}
        for old, new in [("Farmer/Grower", "Grower"), ("Altitude", "Elevation"), ("Varieties", "Variety"),
                         ("Harvest Date", "Harvest date"), ("Arrival Date", "Arrival date"),
                         ("SCA Cupping Score", "SCI cupping score")]:
            specs[old] = lookup.get(old.lower()) or lookup.get(new.lower(), "")
        return {"specs": specs,
                "tasting_notes": text(sections.get("tasting notes") or soup.select_one("#tab2")),
                "roasting_notes": text(sections.get("roasting notes") or soup.select_one("#tab3")),
                "story": " ".join(text(e) for e in soup.select(".image-with-text__text, .image-with-text__content .rte"))}
    if source != "royal":
        raise ValueError(f"Unknown green source: {source}")
    if not soup.select_one(".rc-product__details"):
        raise CatalogError("royal: missing coffee product details")
    detail = {"title": text(soup.select_one("h1")), "infogram_urls": {}, "pane_texts": []}
    for row in soup.select(".rc-product-attribute__copy"):
        key = text(row.select_one(".rc-product-attribute__label")).lower()
        detail["certifications" if key == "certification" else key] = text(row.select_one(".rc-product-attribute__value"))
    for row in soup.select(".rc-product-summary__fact"):
        key = text(row.select_one(".rc-product-summary__fact-label")).lower()
        value = text(row.select_one(".rc-product-summary__fact-value"))
        detail["inventory" if key in {"bags", "boxes"} else key] = value
    detail["position"] = text(soup.select_one(".rc-product__position-line"))
    for section in soup.select("details.collapsible-tab"):
        if text(section.select_one("summary")).lower() == "cup notes":
            detail["flavor_text"] = text(section.select_one(".collapsible-tab__text"))
    for button in soup.select(".rc-product-tabs__trigger"):
        panel = soup.find(id=button.get("aria-controls"))
        if not panel:
            continue
        title = text(button)
        detail["pane_texts"].append(("Grower\n" if title == "About This Coffee" else title + "\n") + text(panel))
        embed = panel.select_one('script[id^="infogram_0_"]')
        if embed:
            detail["infogram_urls"][title] = embed["id"].removeprefix("infogram_0_")
    pricing = soup.select_one(".product-pricing")
    price = soup.select_one(".product-pricing .price__current .money")
    if not price:
        price = soup.select_one(".product-pricing .money")
    unit = pricing.get("data-price-unit-suffix", "") if pricing else ""
    detail["price_per_lb_text" if unit == "per pound" else "price_per_box_text"] = text(price)
    if pricing and unit == "per pound":
        detail["jsonld_weight"] = pricing.get("data-weight-per-unit", "")
    # A hidden quote dialog also exists on buy-online pages; inspect the actual control.
    quote = soup.select_one(".product-form__quote-button")
    detail["is_quote_only"] = bool(quote and "display: none" not in quote.get("style", ""))
    return detail


def royal_online_handles() -> set[str]:
    url = SOURCES["royal"][0] + "/collections/all-coffees?filter.p.m.custom.sale_type=Buy+Online"
    handles = set()
    visited = set()
    while url:
        if url in visited or len(visited) >= 100:
            raise CatalogError("royal: repeated or excessive collection pagination")
        visited.add(url)
        response = get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        response.close()
        links = soup.select('[data-product-item] a[href*="/products/"]')
        page_handles = {urlparse(a["href"]).path.split("/products/")[-1] for a in links}
        if not page_handles or handles.intersection(page_handles):
            raise CatalogError("royal: empty or repeated buy-online page")
        handles.update(page_handles)
        nxt = soup.select_one('a[aria-label="Go to next page"]')
        url = urljoin(url, nxt["href"]) if nxt else None
        time.sleep(0.2)
    return handles


def choose_variant(product: dict) -> dict | None:
    variants = []
    for variant in product["variants"]:
        try:
            price = float(variant.get("price") or 0)
        except (ValueError, TypeError) as exc:
            raise CatalogError(f"Invalid price on {product['handle']}") from exc
        if not math.isfinite(price) or price < 0:
            raise CatalogError(f"Invalid price on {product['handle']}")
        if variant.get("available") and price > 0:
            variants.append(variant)
    retail = [v for v in variants if "sample" not in v.get("title", "").lower()]
    if not retail:
        return None
    return min(retail, key=lambda v: (_parse_variant_grams(v) or math.inf, float(v["price"])))


def fetch_listings(source: str) -> list[tuple[dict, dict, dict]]:
    base, collection = SOURCES[source]
    products = fetch_shopify_catalog(base, collection)
    online = royal_online_handles() if source == "royal" else None
    if online is not None and online - {p["handle"] for p in products}:
        raise CatalogError("royal: buy-online collection and API disagree")
    listings = []
    for product in products:
        kind = product.get("product_type", "").lower()
        title = product["title"].lower()
        if "archived" in kind or "roasted" in kind or "seasoning" in title:
            continue
        if kind == "sets" or re.search(r"\b(subscription|sampler|sample set|flavor set|coffee set|bundle)\b|^set:", title):
            continue
        if source == "sweetmarias" and kind != "green coffee":
            continue
        if online is not None and product["handle"] not in online:
            continue
        variant = choose_variant(product)
        if variant is None:
            continue
        listing = {"name": product["title"], "url": f"{base}/products/{product['handle']}",
                   "price": f"${float(variant['price']):.2f}", "tags": ", ".join(tags(product)),
                   "origin_category": "", "image_url": (product.get("images") or [{}])[0].get("src", "")}
        if source == "royal":
            listing["bag_type"] = {"Crown Jewels": "Crown Jewel", "Royal Gems": "Royal Gem"}.get(product["product_type"], "Full Bag")
        listings.append((listing, product, variant))
    if not listings:
        raise CatalogError(f"{source}: no available coffees")
    return listings


def validate_publication(records: list[dict], output_file, price_field="price", *, shopify_migration=False) -> None:
    if not records:
        raise CatalogError("Refusing to publish an empty coffee dataset")
    urls = set()
    for record in records:
        url = record.get("url", "")
        if not record.get("name") or not url.startswith("https://") or url in urls:
            raise CatalogError("Missing identity or duplicate product URL")
        urls.add(url)
        value = str(record.get(price_field, "")).replace("$", "").replace(",", "").removesuffix("/lb")
        try:
            price = float(value)
        except ValueError as exc:
            raise CatalogError(f"Invalid price for {url}") from exc
        if not math.isfinite(price) or price <= 0:
            raise CatalogError(f"Invalid price for {url}")
    if output_file.exists():
        previous = json.loads(output_file.read_text())
        if (shopify_migration and previous
                and all(r.get("url") and "/products/" not in urlparse(r["url"]).path for r in previous)):
            log.info("Replacing legacy storefront URLs; establishing a fresh Shopify count baseline")
            return
        if len(previous) >= 10 and len(records) < len(previous) * 0.5:
            raise CatalogError(f"Refusing catalog collapse: {len(previous)} -> {len(records)} coffees")


def run_green(source: str, legacy: dict) -> list[dict]:
    """Refresh fresh listings/details, then call the existing schema builders."""
    listings = fetch_listings(source)
    cache = legacy["load_cache"]()

    def load_one(row):
        listing, product, variant = row
        url = listing["url"]
        entry = cache.get(url, {})
        if (isinstance(entry, dict) and entry.get("version") == CACHE_VERSION
                and entry.get("updated_at") == product.get("updated_at")
                and time.time() - entry.get("fetched_at", 0) < CACHE_TTL
                and entry.get("detail")):
            return entry["detail"]
        response = get(url)
        try:
            detail = parse_detail(source, response.text)
        except CatalogError as exc:
            raise CatalogError(f"{url}: {exc}") from exc
        finally:
            response.close()
        description = text(BeautifulSoup(product.get("body_html") or "", "html.parser"))
        if source == "sweetmarias":
            detail["short_description"] = description
        if source == "cbc":
            detail["sku"] = variant.get("sku", "")
            detail["description"] = description
            match = re.search(r"Cupping Notes\s*:\s*(.*)", description, re.I)
            detail["cupping_notes"] = match[1] if match else ""
        if source == "royal":
            detail["jsonld_price"] = variant["price"]
            # Crown Jewels and Royal Gems have labeled net package weights.
            pounds = {"Crown Jewel": 22, "Royal Gem": 50}.get(listing["bag_type"])
            detail["jsonld_weight"] = str(pounds or detail.get("jsonld_weight") or variant.get("grams", 0) / 453.59237)
            detail["jsonld_weight_unit"] = "LBR"
        cache[url] = {"version": CACHE_VERSION, "updated_at": product.get("updated_at"),
                      "fetched_at": time.time(), "detail": detail}
        time.sleep(0.3)
        return detail

    with ThreadPoolExecutor(max_workers=4) as pool:
        try:
            details = list(pool.map(load_one, listings))
        finally:
            # Do not save while workers are still mutating the cache.
            pool.shutdown(wait=True, cancel_futures=True)
            legacy["save_cache"](cache)
    records = []
    info_cache = legacy["load_infogram_cache"]() if source == "royal" else {}
    for (listing, product, variant), detail in zip(listings, details):
        url = listing["url"]
        if source == "royal":
            taste = None
            for tab, uuid in detail.get("infogram_urls", {}).items():
                tab = tab.lower()
                if not tab.startswith(("taste", "source")):
                    continue
                entry = info_cache.get(uuid, {})
                if not entry or time.time() - entry.get("fetched_at", 0) >= CACHE_TTL:
                    data = fetch_infographic(uuid)
                    info = source_infographic(data) if tab.startswith("source") else legacy["_parse_infogram"](data)
                    if not info:
                        raise CatalogError(f"royal: missing {tab} chart for {url}")
                    entry = {"fetched_at": time.time(), "data": info}
                    info_cache[uuid] = entry
                if tab.startswith("source"):
                    detail.update(entry["data"])
                else:
                    taste = entry["data"]
            staff = {url} if any(t.lower() in {"staff-picks", "staff picks"} for t in tags(product)) else set()
            record = legacy["build_record"](listing, detail, staff, taste)
            if not record["price_per_lb"] or "Shanghai" in record.get("warehouse", ""):
                continue
        else:
            record = legacy["build_record"](listing, detail)
        if source == "cbc":
            record["review_count"] = detail.get("review_count", "")
        records.append(record)
    if source == "royal":
        legacy["save_infogram_cache"](info_cache)
    price_field = "price_per_lb" if source == "royal" else "price"
    validate_publication(records, legacy["JSON_FILE"], price_field,
                         shopify_migration=source in {"sweetmarias", "cbc", "royal"})
    enriched = sum(bool(r.get("country") or r.get("total_score") or r.get("process") or r.get("processing")) for r in records)
    if enriched < len(records) * 0.5:
        raise CatalogError(f"{source}: too few products have coffee specifications")
    legacy["match_products"](records, corpus_fields=[
        "name", "country", "region", "grower", "farmer_grower", "description",
        "farm_notes", "story", "overview_text", "source_text",
    ])
    legacy["write_csv"](records)
    legacy["write_json"](records)
    return records
