import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scraper import green
from scraper.catalog import CatalogError

FIXTURES = Path(__file__).parent / "fixtures" / "green"


def parse(source, fixture=None):
    return green.parse_detail(source, (FIXTURES / f"{fixture or source}.html").read_text())


def test_sweetmarias_shopify_preserves_charts_and_specs():
    detail = parse("sweetmarias")
    assert detail["total_score"] == "87.8"
    assert "Dry Fragrance:8.2" in detail["cupping_values"]
    assert "Floral:0" in detail["flavor_values"]
    assert detail["specs"]["Recommended for Espresso"] == "Yes"
    assert detail["overview_attrs"]["Process Method"] == "Swiss Water Decaf"


def test_cbc_shopify_scales_and_country():
    detail = parse("cbc")
    assert detail["country"] == "Ethiopia"
    assert detail["specs"]["Local Region"] == "Yirgacheffe"
    assert detail["attributes"]["Brightness"] == "5"
    assert detail["flavors"]["Chocolaty"] == "4"


def test_cbc_blend_can_have_no_single_origin_specs():
    detail = green.parse_detail("cbc", '<h1>Red Sea Blend</h1><product-page data-id="123"></product-page>')
    assert detail["country"] == ""


def test_hacea_accordion_aliases():
    detail = parse("hacea")
    assert detail["specs"]["Country"] == "Brazil"
    assert detail["specs"]["Altitude"] == "950-1120 m.a.s.l."
    assert detail["specs"]["SCA Cupping Score"] == "83.75"
    assert detail["specs"]["Harvest Date"] == "September 2025"
    assert detail["tasting_notes"]


def test_royal_full_bag_prices_are_already_per_pound():
    detail = parse("royal")
    assert detail["price_per_lb_text"] == "$5.20"
    assert float(detail["jsonld_weight"]) == pytest.approx(151.85)
    assert detail["warehouse"] == "Oakland"
    assert not detail["is_quote_only"]


def test_royal_crown_template_text_and_infogram_identifiers():
    detail = parse("royal", "royal-crown")
    assert detail["price_per_box_text"] == "$314.34"
    assert detail["infogram_urls"]["Source"] == "510159c6-195c-45b3-8470-8b61ac1ec090"
    assert any(p.startswith("Source\n") and len(p) > 100 for p in detail["pane_texts"])


def test_royal_specs_use_infographic_order_without_an_llm():
    data = json.loads((FIXTURES / "royal-source.json").read_text())
    detail = green.source_infographic(data)
    assert detail["variety"] == "Typica"
    assert detail["altitude"] == "1442 masl"
    assert "Asbi" in detail["grower"]
    assert "Wet Hulled" in detail["process"]


@pytest.mark.parametrize("source", green.SOURCES)
def test_error_page_is_not_accepted_as_detail(source):
    with pytest.raises(CatalogError):
        green.parse_detail(source, "<h1>Please try again later</h1>")


def test_green_retail_price_excludes_samples_and_unavailable_sizes():
    sample = {"title": "100 gram sample", "price": "5", "available": True}
    retail = {"title": "500 gram ($6.39 / lb)", "price": "7.03", "available": True, "grams": 525}
    sold = {"title": "250g", "price": "4", "available": False}
    assert green.choose_variant({"variants": [sample, sold, retail]}) is retail


@pytest.mark.parametrize("price", ["NaN", "Infinity", "-5", "free"])
def test_green_rejects_invalid_prices(price):
    with pytest.raises(CatalogError):
        green.choose_variant({"handle": "coffee", "variants": [{"price": price, "available": True}]})


def test_source_failure_keeps_published_files_and_does_not_cache_empty_details(monkeypatch, tmp_path):
    output = tmp_path / "data.json"
    output.write_text('[{"old": true}]')
    cache = {"https://example.com/products/test": {}}
    product = {"title": "Coffee", "updated_at": "today"}
    listing = {"name": "Coffee", "url": "https://example.com/products/test"}
    monkeypatch.setattr(green, "fetch_listings", lambda _: [(listing, product, {})])
    monkeypatch.setattr(green, "get", lambda _: Mock(text="<h1>Error</h1>"))
    save = Mock()
    writer = Mock()
    legacy = {"load_cache": lambda: cache, "save_cache": save, "JSON_FILE": output,
              "write_csv": writer, "write_json": writer}
    with pytest.raises(CatalogError):
        green.run_green("hacea", legacy)
    writer.assert_not_called()
    assert output.read_text() == '[{"old": true}]'
    assert cache[listing["url"]] == {}


def test_empty_or_collapsed_output_is_not_published(tmp_path):
    output = tmp_path / "data.json"
    output.write_text(json.dumps([{}] * 40))
    with pytest.raises(CatalogError, match="empty"):
        green.validate_publication([], output)
    with pytest.raises(CatalogError, match="collapse"):
        green.validate_publication([{"name": "Coffee", "url": "https://example.com/product", "price": "$4"}], output)
    assert len(json.loads(output.read_text())) == 40


def test_royal_distinct_lots_with_matching_names_and_prices_are_preserved(monkeypatch, tmp_path):
    listings = []
    cache = {}
    for lot, position in [("older", "Spot"), ("newer", "Arriving")]:
        url = f"https://example.com/products/{lot}"
        listings.append(({"name": "Colombia Lot", "url": url}, {"updated_at": "today"}, {}))
        cache[url] = {"version": green.CACHE_VERSION, "updated_at": "today",
                      "fetched_at": green.time.time(), "detail": {
                          "country": "Colombia", "warehouse": "Oakland",
                          "price_per_lb": "$6.00", "position": position,
                      }}
    monkeypatch.setattr(green, "fetch_listings", lambda _: listings)
    legacy = {
        "load_cache": lambda: cache, "save_cache": Mock(),
        "load_infogram_cache": dict, "save_infogram_cache": Mock(),
        "build_record": lambda listing, detail, *args: {**listing, **detail},
        "JSON_FILE": tmp_path / "royal-data.json", "match_products": Mock(),
        "write_csv": Mock(), "write_json": Mock(),
    }
    records = green.run_green("royal", legacy)
    assert len(records) == 2
    assert {r["position"] for r in records} == {"Spot", "Arriving"}
    assert len({r["url"] for r in records}) == 2
    legacy["write_json"].assert_called_once_with(records)
