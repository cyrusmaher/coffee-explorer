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
