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
