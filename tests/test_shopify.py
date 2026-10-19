import pytest
from pathlib import Path
from unittest.mock import Mock

from scraper import shopify
from scraper.models import RoasterConfig
from scraper.shopify import _parse_variant_grams, _pick_variants, _variant_weight_label
from scraper.shopify import parse_granja_details


@pytest.mark.parametrize(
    "title,expected",
    [
        ("50g", 50),
        ("250g / Whole Bean", 250),
        ("1,000g", 1000),
        ("250 G", 250),
        ("250gr", 250),
        ("250 grams", 250),
        ("0.25kg", 250),
        ("12oz", 340),
        ("1lb", 453),
        ("Producer Set - 3 x 100g bags", 300),
        ("2×250g", 500),
        ("3 x 4oz", 340),
        ("1/2lb", 226),
        ("1 / 2 lb", 226),
        ("1 1/2lb", 680),
        ("250g x 2", 500),
    ],
)
def test_parse_weight_from_variant_title(title, expected):
    assert _parse_variant_grams({"title": title, "grams": 0}) == expected


def test_gram_labels_select_retail_bag_and_sample():
    sample = {"title": "50g", "price": "8", "available": True}
    retail = {"title": "250g", "price": "30", "available": True}
    bulk = {"title": "1,000g", "price": "95", "available": True}

    standard, selected_sample = _pick_variants([sample, retail, bulk])

    assert standard is retail
    assert selected_sample is sample
    assert _variant_weight_label(standard, _parse_variant_grams(standard)) == "250g"
    assert _variant_weight_label(bulk, _parse_variant_grams(bulk)) == "1,000g"


def test_net_bag_weight_takes_precedence_over_shipping_weight():
    small = {'title':'100g', 'grams':354, 'price':'14.00', 'available':True}
    retail = {'title':'12oz', 'grams':354, 'price':'30.00', 'available':True}
    assert _parse_variant_grams(small) == 100
    assert _parse_variant_grams(retail) == 340
    standard, _ = _pick_variants([small, retail])
    assert standard is retail
    assert _variant_weight_label(standard, _parse_variant_grams(standard)) == '12oz'


def test_shipping_weight_is_fallback_when_title_has_no_size():
    assert _parse_variant_grams({'title':'Whole Bean', 'grams':250}) == 250


def test_multipack_label_and_price_use_total_net_weight():
    pack = {'title':'Producer Set - 3 x 100g bags', 'grams':354, 'price':'45.00', 'available':True}
    grams = _parse_variant_grams(pack)
    assert grams == 300
    assert _variant_weight_label(pack, grams) == '3 x 100g'


@pytest.mark.parametrize('title,grams,expected', [('1/2lb',227,226),('250g x 2',500,500)])
def test_fraction_and_suffix_multipack_do_not_override_with_partial_size(title, grams, expected):
    assert _parse_variant_grams({'title':title,'grams':grams}) == expected


def test_granja_coffee_facts_come_from_product_accordions():
    html = (Path(__file__).parent / "fixtures/granja.html").read_text()
    coffee = parse_granja_details(html)
    assert coffee.producer_or_farm == "Potosi Caicedonia - Valle"
    assert coffee.variety == ["MANDELA"]
    assert coffee.process == "Natural"
    assert coffee.elevation == "1420-1660"
    assert "blueberries" in coffee.tasting_notes
    assert coffee.is_coffee_product


def test_granja_excluded_listings_do_not_require_coffee_accordions(monkeypatch):
    rows = [
        {"id": 1, "title": "Mandela", "handle": "mandela", "product_type": "Coffee"},
        {"id": 2, "title": "Logo Mug", "handle": "mug", "product_type": "Merch"},
        {"id": 3, "title": "Private Lot", "handle": "private", "tags": ["hide-from-retail"]},
    ]
    for row in rows:
        row["variants"] = [{"title": "250g", "price": "20", "available": True}]
    monkeypatch.setattr(shopify, "fetch_shopify_catalog", lambda *args: rows)
    response = Mock(text=(Path(__file__).parent / "fixtures/granja.html").read_text())
    get = Mock(return_value=response)
    monkeypatch.setattr(shopify, "get", get)

    products = shopify.fetch_products(RoasterConfig(
        slug="granja-la-esperanza", name="Granja", base_url="https://example.com",
    ))

    get.assert_called_once_with("https://example.com/products/mandela")
    assert products[0].structured_coffee.producer_or_farm == "Potosi Caicedonia - Valle"
    assert products[1].structured_coffee is products[2].structured_coffee is None
