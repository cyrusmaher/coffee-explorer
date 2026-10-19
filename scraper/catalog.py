"""Public catalog transport shared by roasted and green coffee scrapers."""

import logging
import threading
import time
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "CoffeeExplorer/1.0 (+https://github.com/cyrusmaher/coffee-explorer)"}
RETRYABLE = {408, 429, 500, 502, 503, 504}
_pace_lock = threading.Lock()
_next_request: dict[str, float] = {}


def _pace(url: str) -> None:
    host = urlparse(url).netloc
    with _pace_lock:
        now = time.monotonic()
        scheduled = max(now, _next_request.get(host, now))
        _next_request[host] = scheduled + 0.75
    time.sleep(max(0, scheduled - now))


class CatalogError(RuntimeError):
    """A source could not be read completely; do not publish partial data."""


def get(url: str, *, params=None) -> requests.Response:
    for attempt in range(3):
        try:
            _pace(url)
            response = requests.get(url, params=params, headers=HEADERS, timeout=(10, 45))
            if response.status_code in RETRYABLE and attempt < 2:
                try:
                    delay = float(response.headers.get("Retry-After", 2 ** attempt))
                except ValueError:
                    delay = 2 ** attempt
                if delay > 60:
                    raise CatalogError(f"{url}: source requests a later retry")
                if response.status_code == 429:
                    delay = max(30, delay)
                    with _pace_lock:
                        host = urlparse(url).netloc
                        _next_request[host] = max(_next_request.get(host, 0), time.monotonic() + delay)
                response.close()
                time.sleep(max(0, delay))
                continue
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 2:
                raise CatalogError(f"{url}: network failure after 3 attempts") from exc
            time.sleep(2 ** attempt)
        except requests.RequestException as exc:
            raise CatalogError(f"{url}: {exc}") from exc
    raise CatalogError(f"{url}: request failed")


def fetch_shopify_catalog(base_url: str, collection: str | None = None) -> list[dict]:
    """Fetch every page, rejecting malformed or repeated pages and empty feeds."""
    path = f"/collections/{collection}" if collection else ""
    url = f"{base_url.rstrip('/')}{path}/products.json"
    products = []
    seen = set()
    for page in range(1, 101):
        response = get(url, params={"limit": 250, "page": page})
        try:
            data = response.json()
        except ValueError as exc:
            raise CatalogError(f"{url}: invalid JSON on page {page}") from exc
        finally:
            response.close()
        raw = data.get("products") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            raise CatalogError(f"{url}: missing products array on page {page}")
        for product in raw:
            if (not isinstance(product, dict) or not product.get("id")
                    or not product.get("handle") or not product.get("title")
                    or not isinstance(product.get("variants"), list)):
                raise CatalogError(f"{url}: malformed product on page {page}")
            if product["id"] in seen:
                raise CatalogError(f"{url}: repeated product/page at page {page}")
            seen.add(product["id"])
            products.append(product)
        log.info("%s page %d: %d products", base_url, page, len(raw))
        if len(raw) < 250:
            if not products:
                raise CatalogError(f"{url}: empty catalog")
            return products
        time.sleep(0.5)
    raise CatalogError(f"{url}: exceeded pagination limit")


def tags(product: dict) -> list[str]:
    raw = product.get("tags") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    return [str(tag).strip() for tag in raw if str(tag).strip()]
