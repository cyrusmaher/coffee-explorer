import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from scraper import match
from scraper.models import RoastedCoffeeProduct


@pytest.fixture
def setup(monkeypatch, tmp_path):
    cache_path = tmp_path / 'matches.json'
    monkeypatch.setattr(match, 'MATCH_CACHE_FILE', cache_path)
    client = Mock()
    client.generate = AsyncMock()
    monkeypatch.setattr(match, 'NvidiaClient', lambda: client)
    product = RoastedCoffeeProduct(roaster_slug='test', roaster_name='Test',
        product_url='https://example.com/coffee', title='Diego Bermudez', handle='coffee', producer_or_farm='Diego Bermudez')
    watchlist = [{'producer_name': 'Diego Bermudez', 'country': 'Colombia'}]
    return client, product, watchlist, cache_path


def test_matching_uses_nvidia_for_proposal_and_review(setup):
    client, product, watchlist, cache_path = setup
    client.generate.side_effect = [
        '[{"product_number":1,"matched_producer":"Diego Bermudez"}]',
        '{"verdict":"accept","evidence":"Diego Bermudez"}',
    ]
    result = asyncio.run(match._tier2_batch_match([product], watchlist))
    assert result[product.product_url] == watchlist[0]
    assert client.generate.await_count == 2
    assert next(iter(json.loads(cache_path.read_text()).values())) == {'producer_name': 'Diego Bermudez'}


@pytest.mark.parametrize('review', [RuntimeError('API unavailable'), '{"verdict":"maybe"}'])
def test_review_failures_are_not_cached_as_rejections(setup, review):
    client, product, watchlist, cache_path = setup
    client.generate.side_effect = ['[{"product_number":1,"matched_producer":"Diego Bermudez"}]', review]
    with pytest.raises(RuntimeError, match='refusing to publish'):
        asyncio.run(match._tier2_batch_match([product], watchlist))
    assert json.loads(cache_path.read_text()) == {}


@pytest.mark.parametrize('proposal', ['[]', '{}', '[{"product_number":1}]',
    '[{"product_number":true,"matched_producer":null}]', RuntimeError('API unavailable')])
def test_incomplete_proposals_stop_publication(setup, proposal):
    client, product, watchlist, cache_path = setup
    client.generate.side_effect = [proposal]
    with pytest.raises(RuntimeError, match='refusing to publish'):
        asyncio.run(match._tier2_batch_match([product], watchlist))
    assert json.loads(cache_path.read_text()) == {}


def test_cache_reused_without_api_calls(setup):
    client, product, watchlist, cache_path = setup
    key = match._content_hash(f'{product.title}|{product.producer_or_farm}')
    cache_path.write_text(json.dumps({key: {'producer_name': 'Diego Bermudez'}}))
    assert asyncio.run(match._tier2_batch_match([product], watchlist))[product.product_url] == watchlist[0]
    client.generate.assert_not_awaited()


def test_malformed_proposal_and_review_retry_before_caching(setup):
    client, product, watchlist, cache_path = setup
    client.generate.side_effect = [
        '[{"product_number":1 "matched_producer":null}]',
        '[{"product_number":1,"matched_producer":"Diego Bermudez"}]',
        '{"verdict":["accept"]}',
        '{"verdict":"accept","evidence":"Diego Bermudez"}',
    ]
    result = asyncio.run(match._tier2_batch_match([product], watchlist))
    assert result[product.product_url] == watchlist[0]
    assert client.generate.await_count == 4
    assert next(iter(json.loads(cache_path.read_text()).values())) == {'producer_name': 'Diego Bermudez'}


def test_persistently_malformed_proposal_is_bounded_and_not_cached(setup):
    client, product, watchlist, cache_path = setup
    client.generate.return_value = '[{"product_number":1}]'
    with pytest.raises(RuntimeError, match='refusing to publish'):
        asyncio.run(match._tier2_batch_match([product], watchlist))
    assert client.generate.await_count == 3
    assert json.loads(cache_path.read_text()) == {}


@pytest.mark.parametrize('title,producer,country,expected', [
    ('Diego Bermúdez - Red Plum', 'Diego Bermudez', 'Colombia', 'Diego Bermudez'),
    ('Finca Deborah - Terroir Gesha', 'Finca Deborah', 'Panama', 'Jamison Savage'),
    ('Iris Estate - Vivid', None, 'Panama', 'Jamison Savage'),
    ('Jhonatan Gasca - Pacamara', 'Jhonatan Gasca', 'Colombia', 'Jhonatan Gasca & Alejandra Muñoz'),
    ('Colombia Coffee', None, 'Colombia', None),
    ('Finca Deborahson', None, 'Panama', None),
    ('El Paraiso', None, 'Honduras', None),
    ('Finca Deborah', None, 'Colombia', None),
    ('Finca Deborah', None, None, None),
    ('Diego Bermudez / Jhonatan Gasca Blend', None, 'Colombia', None),
])
def test_direct_match_requires_unique_explicit_identity(title, producer, country, expected):
    watchlist = [
        {'producer_name':'Diego Bermudez', 'farm_or_station':'Finca El Paraiso', 'country':'Colombia'},
        {'producer_name':'Jamison Savage', 'farm_or_station':'Finca Deborah / Iris Estate', 'country':'Panama'},
        {'producer_name':'Jhonatan Gasca & Alejandra Muñoz', 'farm_or_station':'Finca Zarza', 'country':'Colombia'},
    ]
    product = RoastedCoffeeProduct(roaster_slug='test', roaster_name='Test', product_url='https://example.com/a',
        title=title, handle='a', producer_or_farm=producer, origin_country=country)
    result = match._direct_match(product, watchlist)
    assert (result or {}).get('producer_name') == expected


def test_explicit_name_match_overrides_cached_false_negative(setup, monkeypatch):
    client, product, watchlist, cache_path = setup
    watchlist[0]['tier'] = 'Legend'
    cache_path.write_text(json.dumps({match._content_hash(f'{product.title}|{product.producer_or_farm}'): None}))
    result = asyncio.run(match.match_products([product], watchlist))
    assert result[0].watchlist_match == 'Diego Bermudez'
    assert result[0].watchlist_tier == 'Legend'
    client.generate.assert_not_awaited()
