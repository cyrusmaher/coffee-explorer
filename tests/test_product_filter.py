from types import SimpleNamespace

import pytest

from scraper.product_filter import should_publish_product


def product(**overrides):
    base = {
        "title": "Colombia Producer Lot",
        "vendor": "Example Roaster",
        "product_type": "Coffee",
        "tags": ["coffee", "wholesale-coffee"],
    }
    base.update(overrides)
    return base


def test_rejects_extracted_non_coffee_product():
    extracted = SimpleNamespace(is_coffee_product=False)

    assert not should_publish_product(product(), extracted)


@pytest.mark.parametrize(
    "tag",
    [
        "b2b",
        "hide-from-retail",
        "internal",
        "white-label",
        "wholesale",
        "wholesale quick order",
        "wholesale_1_private",
        "wholesale_list",
    ],
)
def test_rejects_non_public_tags(tag):
    assert not should_publish_product(product(tags=[tag]))


@pytest.mark.parametrize(
    ("title", "product_type"),
    [
        ("Holiday Gift Card", "Coffee"),
        ("Espresso Pod Box", "Coffee"),
        ("Instant Coffee Pack", "Coffee"),
        ("Smiley Blend 5lb Case - Wholesale", "Coffee"),
        ("Colombia Producer Lot", "Gift Box"),
        ("Colombia Producer Lot", "Merchandise"),
        ("Colombia Producer Lot", "/wholesale"),
    ],
)
def test_rejects_non_coffee_titles_and_types(title, product_type):
    assert not should_publish_product(product(title=title, product_type=product_type))


def test_rejects_wholesale_vendor():
    assert not should_publish_product(product(vendor="Example Roaster (WHOLESALE)"))


def test_allows_public_wholesale_coffee_tag():
    assert should_publish_product(product(tags=["coffee", "wholesale-coffee"]))


def test_allows_moonwake_wholebeancoffee_type_with_wholesale_segment():
    assert should_publish_product(
        product(
            product_type="/wholesale, /cocoa-forward, /wholebeancoffee",
            tags=["Cocoa-forward", "Light"],
        )
    )
