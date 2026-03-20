"""Pydantic models for the coffee-explorer scraper pipeline."""

from pydantic import BaseModel, Field


class RoasterConfig(BaseModel):
    """Configuration for a single Shopify-based roaster."""
    slug: str
    name: str
    base_url: str  # e.g. "https://onyxcoffeelab.com"


class ExtractedCoffee(BaseModel):
    """LLM-extracted structured data from a product's body_html."""
    producer_or_farm: str | None = None
    origin_country: str | None = None
    origin_region: str | None = None
    variety: list[str] = Field(default_factory=list)
    process: str | None = None
    elevation: str | None = None
    tasting_notes: list[str] = Field(default_factory=list)
    is_coffee_product: bool = True  # False for merch, subscriptions, gear


class ShopifyProduct(BaseModel):
    """Raw Shopify product fields we care about."""
    id: int
    title: str
    handle: str
    vendor: str = ""
    product_type: str = ""
    tags: list[str] = Field(default_factory=list)
    body_html: str = ""
    created_at: str = ""
    # First variant price
    price: str = ""
    # First image URL
    image_url: str = ""


class RoastedCoffeeProduct(BaseModel):
    """Final output record combining Shopify data + LLM extraction + watchlist match."""
    # Source info
    roaster_slug: str
    roaster_name: str
    product_url: str
    image_url: str = ""

    # Shopify fields
    title: str
    handle: str
    vendor: str = ""
    product_type: str = ""
    tags: list[str] = Field(default_factory=list)
    price: str = ""
    created_at: str = ""

    # LLM-extracted fields
    producer_or_farm: str | None = None
    origin_country: str | None = None
    origin_region: str | None = None
    variety: list[str] = Field(default_factory=list)
    process: str | None = None
    elevation: str | None = None
    tasting_notes: list[str] = Field(default_factory=list)
    is_coffee_product: bool = True

    # Watchlist match fields
    watchlist_match: str | None = None  # producer_name from watchlist
    watchlist_farm: str | None = None   # farm_or_station from watchlist
    watchlist_tier: str | None = None   # e.g. "Legend", "WBC Elite", etc.
    watchlist_credential_type: str | None = None   # e.g. "ACE Legend of Excellence"
    watchlist_credential_detail: str | None = None  # e.g. "5x CoE winner; 1st place 2022 & 2024"
    watchlist_notes: str | None = None  # additional context from watchlist
