from unittest.mock import Mock

import pytest
import requests

from scraper import catalog
from scraper.models import RoasterConfig
from scraper.shopify import fetch_products


def product(index=1):
    return {"id": index, "title": f"Coffee {index}", "handle": f"coffee-{index}",
            "variants": [{"title": "250g", "price": "12", "available": True}]}


def response(payload):
    result = Mock()
    result.json.return_value = payload
    return result


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(catalog.time, "sleep", lambda _: None)
    monkeypatch.setattr(catalog, "_pace", lambda _: None)


def test_fetches_every_page(monkeypatch):
    get = Mock(side_effect=[response({"products": [product(i) for i in range(1, 251)]}),
                            response({"products": [product(251)]})])
    monkeypatch.setattr(catalog, "get", get)
    assert len(catalog.fetch_shopify_catalog("https://example.com", "green-coffee")) == 251
    assert get.call_args.args[0] == "https://example.com/collections/green-coffee/products.json"
    assert get.call_args.kwargs["params"]["page"] == 2


@pytest.mark.parametrize("payload", [{}, [], {"products": None}, {"products": []},
                                      {"products": [{"id": 1}]}])
def test_rejects_invalid_or_empty_feed(monkeypatch, payload):
    monkeypatch.setattr(catalog, "get", lambda *a, **k: response(payload))
    with pytest.raises(catalog.CatalogError):
        catalog.fetch_shopify_catalog("https://example.com")


def test_repeated_page_cannot_loop_or_duplicate_products(monkeypatch):
    monkeypatch.setattr(catalog, "get", lambda *a, **k: response({"products": [product(i) for i in range(1, 251)]}))
    with pytest.raises(catalog.CatalogError, match="repeated"):
        catalog.fetch_shopify_catalog("https://example.com")


def test_failed_later_page_does_not_return_partial_roaster(monkeypatch):
    monkeypatch.setattr(catalog, "get", Mock(side_effect=[
        response({"products": [product(i) for i in range(1, 251)]}), catalog.CatalogError("503")]))
    with pytest.raises(catalog.CatalogError, match="503"):
        fetch_products(RoasterConfig(slug="test", name="Test", base_url="https://example.com"))


def test_bounded_network_retries(monkeypatch):
    request = Mock(side_effect=requests.Timeout())
    monkeypatch.setattr(catalog.requests, "get", request)
    with pytest.raises(catalog.CatalogError, match="3 attempts"):
        catalog.get("https://example.com")
    assert request.call_count == 3


def test_retry_server_failure(monkeypatch):
    busy = Mock(status_code=503, headers={})
    ok = Mock(status_code=200)
    request = Mock(side_effect=[busy, ok])
    monkeypatch.setattr(catalog.requests, "get", request)
    assert catalog.get("https://example.com") is ok
    assert request.call_count == 2


def test_shopify_string_tags_are_not_split_into_characters(monkeypatch):
    data = product()
    data["tags"] = "Colombia, Washed"
    monkeypatch.setattr(catalog, "get", lambda *a, **k: response({"products": [data]}))
    result = fetch_products(RoasterConfig(slug="test", name="Test", base_url="https://example.com"))
    assert result[0].tags == ["Colombia", "Washed"]
    assert result[0].weight_grams == 250
