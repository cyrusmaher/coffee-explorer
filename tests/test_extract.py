import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from scraper import extract
from scraper.extract import ExtractionParseError, _parse_llm_response, _too_many_failures
from scraper.models import ShopifyProduct


def test_failure_guard_rejects_total_failure_for_tiny_batch():
    assert _too_many_failures(failed_count=1, work_count=1)


def test_failure_guard_allows_one_failure_in_small_batch():
    assert not _too_many_failures(failed_count=1, work_count=5)


def test_failure_guard_rejects_more_than_ten_percent():
    assert _too_many_failures(failed_count=3, work_count=20)


@pytest.mark.parametrize("field", ["variety", "tasting_notes"])
def test_absent_array_fields_are_normalized(field):
    payload = {
        "producer_or_farm": "San Antonio",
        "origin_country": "Colombia",
        "variety": ["Geisha"],
        "tasting_notes": ["red apple"],
        "is_coffee_product": True,
    }
    payload[field] = None

    result = _parse_llm_response(json.dumps(payload))

    assert getattr(result, field) == []
    other_field = "tasting_notes" if field == "variety" else "variety"
    assert getattr(result, other_field) == payload[other_field]
    assert result.producer_or_farm == "San Antonio"
    assert result.origin_country == "Colombia"
    assert result.is_coffee_product is True


@pytest.mark.parametrize("payload", [
    {"variety": "Geisha"},
    {"tasting_notes": "red apple"},
    {"variety": [None]},
    {"tasting_notes": {}},
    {"tasting_notes": False},
    [],
    None,
])
def test_invalid_extraction_shapes_remain_rejected(payload):
    with pytest.raises(ExtractionParseError):
        _parse_llm_response(json.dumps(payload))


def test_null_arrays_are_cached_as_lists_without_extraction_failure(monkeypatch, tmp_path):
    cache_file = tmp_path / "llm_cache.json"
    monkeypatch.setattr(extract, "CACHE_FILE", cache_file)
    client = Mock()
    client.generate = AsyncMock(return_value=json.dumps({
        "variety": None,
        "tasting_notes": None,
        "is_coffee_product": False,
    }))
    monkeypatch.setattr(extract, "NvidiaClient", lambda: client)
    product = ShopifyProduct(id=1, title="Mug", handle="mug", body_html="A ceramic mug")

    result = asyncio.run(extract.extract_products([product], "test"))
    cached = next(iter(json.loads(cache_file.read_text()).values()))

    assert result[1].is_coffee_product is False
    assert cached["variety"] == []
    assert cached["tasting_notes"] == []
    assert cached["is_coffee_product"] is False
