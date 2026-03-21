"""Check product URLs for locked/dead pages and remove them from the data.

Run separately from the scraper to avoid compounding rate limits:
    python -m scraper.liveness

Caches known-good URLs so only new or sold-out products are re-checked.
"""

import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "docs" / "roasted-data.json"
CACHE_FILE = REPO_ROOT / "data" / "liveness_cache.json"

DELAY_BETWEEN_REQUESTS = 1.5  # seconds, per host
TIMEOUT = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _load_cache() -> set[str]:
    """Load set of known-good URLs from cache."""
    if CACHE_FILE.exists():
        try:
            return set(json.loads(CACHE_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def _save_cache(good_urls: set[str]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(sorted(good_urls), indent=2), encoding="utf-8",
    )


def _check_host(host: str, urls: list[str]) -> tuple[list[str], list[str]]:
    """Check URLs for a single host. Returns (dead_urls, good_urls)."""
    dead: list[str] = []
    good: list[str] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    # Warm session with homepage to get cookies
    try:
        session.get(f"https://{host}/", timeout=TIMEOUT)
        time.sleep(1.0)
    except Exception:
        pass

    for url in urls:
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                dead.append(url)
                log.info("  DEAD (404): %s", url)
            elif resp.status_code == 200 and "<h1>Locked</h1>" in resp.text[:50_000]:
                dead.append(url)
                log.info("  LOCKED: %s", url)
            elif resp.status_code == 429:
                log.warning("  429 rate-limited on %s, stopping host", host)
                break
            else:
                good.append(url)
        except Exception as e:
            log.warning("  Error checking %s: %s", url, e)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return dead, good


def run() -> None:
    if not DATA_FILE.exists():
        log.error("Data file not found: %s", DATA_FILE)
        sys.exit(1)

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    log.info("Loaded %d products from %s", len(data), DATA_FILE)

    cache = _load_cache()
    log.info("Liveness cache: %d known-good URLs", len(cache))

    # Check all products not in cache
    to_check: list[dict] = []
    for p in data:
        url = p["product_url"]
        if url in cache:
            continue
        to_check.append(p)

    if not to_check:
        log.info("No sold-out products need checking. Done.")
        return

    # Group by host
    by_host: dict[str, list[str]] = defaultdict(list)
    for p in to_check:
        host = urlparse(p["product_url"]).netloc
        by_host[host].append(p["product_url"])

    log.info("Checking %d sold-out products across %d hosts...",
             len(to_check), len(by_host))

    all_dead: list[str] = []
    all_good: list[str] = []

    with ThreadPoolExecutor(max_workers=len(by_host)) as pool:
        futures = {
            pool.submit(_check_host, host, urls): host
            for host, urls in by_host.items()
        }
        for fut in futures:
            dead, good = fut.result()
            all_dead.extend(dead)
            all_good.extend(good)

    # Update cache with newly confirmed good URLs
    cache.update(all_good)
    # Also add all in-stock product URLs to cache
    for p in data:
        if p.get("is_available", True):
            cache.add(p["product_url"])
    _save_cache(cache)

    if not all_dead:
        log.info("All checked pages are accessible. Done.")
        return

    # Remove dead products and rewrite data file
    dead_set = set(all_dead)
    filtered = [p for p in data if p["product_url"] not in dead_set]
    log.info("Removing %d dead products (%d -> %d)",
             len(all_dead), len(data), len(filtered))

    DATA_FILE.write_text(
        json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    log.info("Wrote %s. Done.", DATA_FILE)


if __name__ == "__main__":
    run()
