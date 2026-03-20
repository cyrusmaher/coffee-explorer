"""Registry of Shopify-based roasters to scrape."""

from scraper.models import RoasterConfig

ROASTERS: list[RoasterConfig] = [
    RoasterConfig(
        slug="onyx",
        name="Onyx Coffee Lab",
        base_url="https://onyxcoffeelab.com",
    ),
    RoasterConfig(
        slug="george-howell",
        name="George Howell Coffee",
        base_url="https://georgehowellcoffee.com",
    ),
    RoasterConfig(
        slug="black-white",
        name="Black & White Coffee Roasters",
        base_url="https://www.blackwhiteroasters.com",
    ),
    RoasterConfig(
        slug="moonwake",
        name="Moonwake Coffee",
        base_url="https://moonwakecoffeeroasters.com",
    ),
    RoasterConfig(
        slug="coffee-bros",
        name="Coffee Bros",
        base_url="https://coffeebros.com",
    ),
    RoasterConfig(
        slug="standout",
        name="Standout Coffee",
        base_url="https://standoutcoffee.com",
    ),
    RoasterConfig(
        slug="dune",
        name="Dune Coffee",
        base_url="https://dunecoffee.com",
    ),
    RoasterConfig(
        slug="corvus",
        name="Corvus Coffee",
        base_url="https://corvuscoffee.com",
    ),
    RoasterConfig(
        slug="elixr",
        name="Elixr Coffee",
        base_url="https://elixrcoffee.com",
    ),
    RoasterConfig(
        slug="haven",
        name="Haven Coffee Roaster",
        base_url="https://havencoffeeroaster.com",
    ),
    RoasterConfig(
        slug="granja-la-esperanza",
        name="Cafe Granja La Esperanza",
        base_url="https://cafegranjalaesperanza.com",
    ),
    RoasterConfig(
        slug="savage",
        name="Savage Coffees",
        base_url="https://savagecoffees.co",
    ),
    RoasterConfig(
        slug="qima",
        name="Qima Cafe",
        base_url="https://qimacafe.com",
    ),
    RoasterConfig(
        slug="formative",
        name="Formative Coffee",
        base_url="https://formative.coffee",
    ),
]
