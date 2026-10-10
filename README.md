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
failed reviews are not cached as rejected matches. A failed extraction/matching
stage or output quality check leaves the published JSON unchanged.

Actions checks NVIDIA availability before scraping. Successful inference is
saved to an Actions cache even when the scraper fails, and failed runs also
upload their cache files as a seven-day artifact. Re-running the workflow on the
same branch resumes this progress. A change to the committed cache files starts
a new cache lineage so restored files cannot overwrite committed corrections.

Run checks with `python -m pytest tests` (install `pytest` separately).
