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
        ("Sibarist Espresso High Cellulose Filter", ""),
        ("V60 Paper Filters", ""),
        ("Coffee Filters", ""),
        ("Blue Haven Tea - Loose Leaf", ""),
        ("Cascara Tea | Coffee Cherry", "Roasted Coffee Beans"),
        ("Coffee Blossom Honey | Francisco Cardona | La Colmenita", ""),
        ("Miir Water Bottle", "25% OFF"),
        ("The Standout Coffee Sample Pack", "Roasted Coffee"),
        ("Four Coffee Saber Sample Set", "Coffee"),
        ("Whole Bean Membership", "Roasted Coffee Beans"),
    ],
)
def test_rejects_non_coffee_titles_and_types(title, product_type):
    assert not should_publish_product(product(title=title, product_type=product_type))


def test_rejects_wholesale_vendor():
    assert not should_publish_product(product(vendor="Example Roaster (WHOLESALE)"))


def test_allows_public_wholesale_coffee_tag():
    assert should_publish_product(product(tags=["coffee", "wholesale-coffee"]))


def test_allows_subscription_tag_on_public_bag_of_coffee():
    assert should_publish_product(
        product(
            title="Standout Signature Espresso",
            product_type="Roasted Coffee",
            tags=["Coffee", "Subscription"],
        )
    )


def test_allows_coffee_name_with_tea_tasting_note():
    assert should_publish_product(
        product(title="Rodrigo Sanchez - Green Tea", product_type="Retail SO")
    )


def test_allows_brewers_cup_coffee():
    assert should_publish_product(
        product(
            title="Isaiah Sheese 2026 Brewers Cup Competition Blend",
            product_type="Coffee",
            tags=[],
        )
    )


def test_allows_brewers_cup_product_type():
    assert should_publish_product(
        product(
            title="Colombia Producer Lot",
            product_type="Brewer's Cup Coffee",
            tags=[],
        )
    )


def test_allows_filter_roast_coffee():
    assert should_publish_product(
        product(
            title="Colombia Gesha Filter Roast",
            product_type="Coffee",
            tags=[],
        )
    )


def test_allows_moonwake_wholebeancoffee_type_with_wholesale_segment():
    assert should_publish_product(
        product(
            product_type="/wholesale, /cocoa-forward, /wholebeancoffee",
            tags=["Cocoa-forward", "Light"],
        )
    )


def test_allows_whole_beans_tag_metadata():
    assert should_publish_product(product(product_type="Reserve", tags=["Whole Beans"]))


def test_allows_country_and_process_tag_metadata():
    assert should_publish_product(product(product_type="", tags=["Panama", "Natural", "retail"]))


def test_allows_known_sparse_coffee_roaster_after_rejects():
    assert should_publish_product(
        product(
            roaster_slug="qima",
            title="Ecuador . Guillermo Ortiz",
            product_type="",
            tags=[],
        )
    )


def test_rejects_known_sparse_roaster_subscription():
    assert not should_publish_product(
        product(
            roaster_slug="helm",
            title="Decaf Subscription",
            product_type="",
            tags=[],
        )
    )


def test_rejects_metadata_weak_unknown_product_without_extraction():
    assert not should_publish_product(product(product_type="", tags=[]))


def test_rejects_metadata_weak_product_with_negative_extraction():
    extracted = SimpleNamespace(is_coffee_product=False)

    assert not should_publish_product(product(product_type="", tags=[]), extracted)


def test_allows_metadata_weak_product_with_positive_extraction():
    extracted = SimpleNamespace(is_coffee_product=True)

    assert should_publish_product(product(product_type="", tags=[]), extracted)
