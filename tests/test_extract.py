from scraper.extract import _too_many_failures


def test_failure_guard_rejects_total_failure_for_tiny_batch():
    assert _too_many_failures(failed_count=1, work_count=1)


def test_failure_guard_allows_one_failure_in_small_batch():
    assert not _too_many_failures(failed_count=1, work_count=5)


def test_failure_guard_rejects_more_than_ten_percent():
    assert _too_many_failures(failed_count=3, work_count=20)
