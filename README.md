# Roasted Coffee Explorer

The site is published from `docs/` on `main`. A weekly GitHub Actions workflow
fetches roaster listings, extracts structured coffee data, proposes producer
watchlist matches, and reviews those proposals before publishing.

## NVIDIA setup

The pipeline uses NVIDIA's hosted Nemotron model through the chat-completions API for extraction and
watchlist matching. It does not require Gemini or Google Cloud credentials.

- Set the repository Actions secret `NVIDIA_API_KEY` to your NVIDIA API key.
- `NVIDIA_MODEL` defaults to `nvidia/nemotron-3.5-lightning-30b-a3b`, also configured
  explicitly in `.github/workflows/scrape.yml`. Check NVIDIA's authenticated
  `/v1/models` listing before changing it.
- For local runs, set `NVIDIA_API_KEY` in your shell or an ignored `.env` file,
  install `scraper/requirements.txt`, and run `python -m scraper.scrape`.
- Trigger **Scrape Roasted Coffee** with **Run workflow** for a manual refresh.

Requests are paced at 30 per minute with at most four in flight, with bounded
retries for throttling and transient failures. HTTP 429 pauses the entire request
queue for at least a minute; persistent throttling stops further calls until a
later run. Listings excluded by the publishing rules skip inference entirely.
Actual access and quotas depend on
the NVIDIA account; the scraper does not purchase credits or use a paid fallback.
Existing extraction and matching caches are reused. New responses are validated;
malformed or truncated model answers get up to three attempts through the same
paced request queue. Exhausted API failures do not receive extra validation retries.
Failed reviews are not cached as rejected matches. A failed extraction/matching
stage or output quality check leaves the published JSON unchanged.

Unique, explicit producer names are matched directly before model review or
cached rejections. Farm-name matches additionally require a matching country;
ambiguous and partial names still go through proposal and review. Variant sizing
uses the labeled net bag weight before Shopify's shipping weight, keeping the
selected price and price-per-ounce calculation consistent.

Actions checks NVIDIA availability before scraping. Successful inference is
saved to an Actions cache even when the scraper fails, and failed runs also
upload their cache files as a seven-day artifact. Re-running the workflow on the
same branch resumes this progress. A change to the committed cache files starts
a new cache lineage so restored files cannot overwrite committed corrections.

Run checks with `python -m pytest tests` (install `pytest` separately).

## Shared roasted and green parsers

`scraper/catalog.py` provides paginated Shopify fetching, per-host pacing and
bounded retries. A missing source, malformed response or failed later page stops
publication. Roasted output is checked per roaster as well as across the whole
dataset. George Howell uses its public `all-coffee` collection; Granja's coffee
facts come from its product-page accordions when the API description is empty.

The neighboring `green-coffee-explorer` repository uses `scraper/green.py` for
Sweet Maria's, Coffee Bean Corral, Hacea and Royal Coffee, retaining its existing
output schemas. Royal's Source and Taste infographics provide structured facts
and charts without model calls. Green producer matches use explicit identities
and compatible countries. Genuine Origin keeps its SuiteCommerce adapter in the
green repository and uses the shared transport/publication checks.

Keep the two repositories beside each other or set `COFFEE_EXPLORER_ROOT` in the
green checkout. Deploy this repository's shared changes before updating the green
workflow; that workflow checks out shared `main` and tests both repositories.
Green refreshes do not require an LLM account or a Chromium installation.

Regression fixtures cover the current storefront layouts, charts, package prices,
empty/error pages, failed pagination, stale negative extractions, and publication
guards. Each repository also runs its tests on pushes and pull requests. The live
scrapers remain scheduled weekly; run the green entry points in a temporary copy
when auditing without replacing that checkout's generated data.
