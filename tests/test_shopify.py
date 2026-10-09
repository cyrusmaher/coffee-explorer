import pytest

from scraper.shopify import _parse_variant_grams, _pick_variants, _variant_weight_label


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
