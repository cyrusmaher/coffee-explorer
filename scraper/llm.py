"""NVIDIA chat-completions transport shared by extraction and watchlist matching."""

import asyncio
import logging
import os
import time
from email.utils import parsedate_to_datetime
from typing import Callable, TypeVar

import requests

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
log = logging.getLogger(__name__)
T = TypeVar("T")


class NvidiaRateLimitError(RuntimeError):
    """The provider is still throttling; preserve progress for a later run."""


class NvidiaCompletionError(RuntimeError):
    """A successful HTTP response did not contain a complete model answer."""


async def generate_validated(client: "NvidiaClient", prompt: str, validate: Callable[[str], T]) -> T:
    """Retry malformed model answers, while preserving transport retry limits."""
    for attempt in range(3):
        try:
            response = await client.generate(prompt)
        except NvidiaCompletionError:
            if attempt == 2:
                raise
        else:
            try:
                return validate(response)
            except ValueError:
                if attempt == 2:
                    raise
        log.warning("Invalid model answer; retrying validation (%d/3)", attempt + 2)
        if attempt == 0:
            prompt += (
                "\nYour previous answer was incomplete or did not follow the requested JSON schema. "
                "Generate a complete answer again, with valid JSON and exactly the requested field types. "
                "Do not include comments or explanations outside the JSON."
            )
    raise AssertionError("Unreachable validation retry state")


class NvidiaClient:
    """Bounded requests with pacing, transient retries, and redacted failures."""

    def __init__(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Set NVIDIA_API_KEY to run the coffee scraper")
        self.model = os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL)
        self._slots = asyncio.Semaphore(4)
        self._pace = asyncio.Lock()
        self._next_request = 0.0
        self._cooldown_until = 0.0
        self._rate_limit_exhausted = False

    async def _wait_for_slot(self):
        async with self._pace:
            while True:
                delay = max(self._next_request, self._cooldown_until) - time.monotonic()
                if delay <= 0:
                    break
                await asyncio.sleep(delay)
            # Conservative default for the hosted API: at most 30 requests/minute.
            self._next_request = time.monotonic() + 2.0

    def _request(self, prompt):
        try:
            response = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 4096,
                    "stream": False,
                    "chat_template_kwargs": {
                        "thinking" if self.model.startswith("deepseek-ai/") else "enable_thinking": False
                    },
                },
                timeout=(15, 180),
                allow_redirects=False,
            )
        except requests.RequestException:
            return None, None, 0
        with response:
            retry_after = response.headers.get("Retry-After", "0")
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    delay = 0
            delay = max(0, delay)
            if response.status_code != 200:
                return response.status_code, None, delay
            try:
                choice = response.json()["choices"][0]
                content = choice["message"]["content"]
                if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
                    raise ValueError("Missing or incomplete completion")
            except (ValueError, KeyError, IndexError, TypeError):
                raise NvidiaCompletionError("NVIDIA returned a missing, invalid, or truncated completion") from None
            return 200, content, 0

    async def generate(self, prompt: str) -> str:
        async with self._slots:
            for attempt in range(4):
                if self._rate_limit_exhausted:
                    raise NvidiaRateLimitError("NVIDIA remains rate limited; retry in a later run")
                await self._wait_for_slot()
                if self._rate_limit_exhausted:
                    raise NvidiaRateLimitError("NVIDIA remains rate limited; retry in a later run")
                status, content, retry_after = await asyncio.to_thread(self._request, prompt)
                if status == 200:
                    return content
                if status is not None and status not in RETRYABLE:
                    raise RuntimeError(f"NVIDIA request failed (HTTP {status}); check credentials and model")
                if status == 429:
                    if attempt == 3 or retry_after > 300:
                        self._rate_limit_exhausted = True
                        raise NvidiaRateLimitError("NVIDIA remains rate limited (HTTP 429); retry in a later run")
                    # A rate limit applies to the whole queue, not just this job.
                    retry_after = max(60, retry_after)
                    self._cooldown_until = max(self._cooldown_until, time.monotonic() + retry_after)
                    log.warning("NVIDIA HTTP 429; pausing requests for %.0f seconds", retry_after)
                if attempt < 3:
                    if retry_after > 300:
                        raise RuntimeError("NVIDIA requested a long retry delay; retry in a later run")
                    await asyncio.sleep(max(retry_after, 2 ** (attempt + 1)))
            raise RuntimeError(f"NVIDIA request failed after 4 attempts (HTTP {status or 'network error'})")
