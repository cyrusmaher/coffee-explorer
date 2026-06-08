import pytest

from scraper.models import RoastedCoffeeProduct
from scraper.quality import OutputQualityError, validate_output_quality


def coffee(
    index: int,
    *,
    country: str | None = "Colombia",
    tier: str | None = None,
    image_url: str = "https://example.com/coffee.jpg",
    process: str | None = "Washed",
):
    return RoastedCoffeeProduct(
        roaster_slug="test",
        roaster_name="Test Roaster",
        product_url=f"https://example.com/products/{index}",
        image_url=image_url,
        title=f"Coffee {index}",
        handle=f"coffee-{index}",
        origin_country=country,
        process=process,
        watchlist_tier=tier,
    )


def test_quality_gate_allows_enriched_dataset():
    products = [
        coffee(i, tier="WBC Elite" if i < 10 else None)
        for i in range(120)
    ]

    validate_output_quality(products)


def test_quality_gate_rejects_missing_countries():
    products = [
        coffee(i, country=None, tier="WBC Elite" if i < 10 else None)
        for i in range(120)
    ]

    with pytest.raises(OutputQualityError, match="origin_country"):
        validate_output_quality(products)


def test_quality_gate_rejects_missing_watchlist_tiers():
    products = [coffee(i) for i in range(120)]

    with pytest.raises(OutputQualityError, match="watchlist_tier"):
        validate_output_quality(products)


def test_quality_gate_rejects_tiny_output():
    products = [coffee(i, tier="WBC Elite") for i in range(10)]

    with pytest.raises(OutputQualityError, match="only 10 products"):
        validate_output_quality(products)


def test_quality_gate_rejects_missing_images():
    products = [
        coffee(
            i,
            image_url="" if i < 20 else "https://example.com/coffee.jpg",
            tier="WBC Elite" if i < 10 else None,
        )
        for i in range(120)
    ]

    with pytest.raises(OutputQualityError, match="image_url"):
        validate_output_quality(products)


def test_quality_gate_rejects_lot_name_process_values():
    products = [
        coffee(
            i,
            process="Nirvana" if i == 0 else "Washed",
            tier="WBC Elite" if i < 10 else None,
        )
        for i in range(120)
    ]

    with pytest.raises(OutputQualityError, match="suspicious process"):
        validate_output_quality(products)
