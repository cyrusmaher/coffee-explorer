import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import requests

from scraper import llm
from scraper.llm import NvidiaClient, NvidiaRateLimitError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('NVIDIA_API_KEY', 'test-secret')
    c = NvidiaClient()
    c._wait_for_slot = AsyncMock()
    return c


def response(status=200, content='{"ok":true}', finish='stop', retry='0'):
    r = Mock()
    r.status_code = status
    r.headers = {'Retry-After': retry}
    r.json.return_value = {'choices': [{'message': {'content': content}, 'finish_reason': finish}]}
    r.__enter__ = Mock(return_value=r)
    r.__exit__ = Mock(return_value=False)
    return r


def test_missing_key(monkeypatch):
    monkeypatch.delenv('NVIDIA_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='Set NVIDIA_API_KEY'):
        NvidiaClient()


def test_transport_payload(client, monkeypatch):
    post = Mock(return_value=response())
    monkeypatch.setattr(requests, 'post', post)
    assert asyncio.run(client.generate('Extract coffee')) == '{"ok":true}'
    args = post.call_args.kwargs
    assert args['headers']['Authorization'] == 'Bearer test-secret'
    assert args['json']['messages'] == [{'role': 'user', 'content': 'Extract coffee'}]
    assert args['json']['chat_template_kwargs'] == {'enable_thinking': False}
    assert args['allow_redirects'] is False
    assert args['timeout'] == (15, 180)


def test_retry_throttling(client, monkeypatch):
    post = Mock(side_effect=[response(429, retry='7'), response()])
    sleep = AsyncMock()
    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    asyncio.run(client.generate('Extract'))
    assert post.call_count == 2
    sleep.assert_awaited_once_with(60)


def test_auth_failure_is_redacted_and_not_retried(client, monkeypatch):
    post = Mock(return_value=response(401))
    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(RuntimeError, match='HTTP 401') as exc:
        asyncio.run(client.generate('Extract'))
    assert 'test-secret' not in str(exc.value)
    assert post.call_count == 1


@pytest.mark.parametrize('content,finish', [(None, 'stop'), ('{}', 'length'), ('', 'stop')])
def test_reject_incomplete_completion(client, monkeypatch, content, finish):
    monkeypatch.setattr(requests, 'post', Mock(return_value=response(content=content, finish=finish)))
    with pytest.raises(RuntimeError, match='completion'):
        asyncio.run(client.generate('Extract'))


def test_network_retries_are_bounded(client, monkeypatch):
    post = Mock(side_effect=requests.Timeout('test-secret'))
    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock())
    with pytest.raises(RuntimeError, match='4 attempts') as exc:
        asyncio.run(client.generate('Extract'))
    assert 'test-secret' not in str(exc.value)
    assert post.call_count == 4


def test_deepseek_thinking_option(client, monkeypatch):
    client.model = 'deepseek-ai/deepseek-v4-flash-0731'
    post = Mock(return_value=response())
    monkeypatch.setattr(requests, 'post', post)
    asyncio.run(client.generate('Extract'))
    assert post.call_args.kwargs['json']['chat_template_kwargs'] == {'thinking': False}


def test_exhausted_rate_limit_stops_queued_requests(client, monkeypatch):
    post = Mock(return_value=response(429))
    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock())

    with pytest.raises(NvidiaRateLimitError, match='HTTP 429'):
        asyncio.run(client.generate('First product'))
    assert post.call_count == 4
    with pytest.raises(NvidiaRateLimitError):
        asyncio.run(client.generate('Next product'))
    assert post.call_count == 4


def test_long_retry_after_stops_without_retrying_early(client, monkeypatch):
    post = Mock(return_value=response(429, retry='600'))
    monkeypatch.setattr(requests, 'post', post)
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    with pytest.raises(NvidiaRateLimitError):
        asyncio.run(client.generate('Extract'))
    assert post.call_count == 1
    sleep.assert_not_awaited()


def test_waiting_request_observes_new_shared_cooldown(client, monkeypatch):
    now = [0.0]
    delays = []
    monkeypatch.setattr(llm.time, 'monotonic', lambda: now[0])
    client._next_request = 2.0

    async def sleep(delay):
        delays.append(delay)
        now[0] += delay
        if len(delays) == 1:
            # Another in-flight request was throttled while this one waited.
            client._cooldown_until = 60.0

    monkeypatch.setattr(asyncio, 'sleep', sleep)
    asyncio.run(NvidiaClient._wait_for_slot(client))
    assert delays == [2.0, 58.0]
    assert now[0] == 60.0
