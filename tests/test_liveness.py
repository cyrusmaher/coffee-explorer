from unittest.mock import Mock

from scraper import liveness


def test_error_responses_are_never_cached_as_live(monkeypatch):
    session = Mock()
    session.get.side_effect = [Mock(status_code=200), Mock(status_code=503),
                               Mock(status_code=403), Mock(status_code=200, text="Coffee"),
                               Mock(status_code=404)]
    monkeypatch.setattr(liveness.requests, "Session", lambda: session)
    monkeypatch.setattr(liveness.time, "sleep", lambda _: None)
    dead, good = liveness._check_host("example.com", ["error", "blocked", "coffee", "gone"])
    assert dead == ["gone"]
    assert good == ["coffee"]
