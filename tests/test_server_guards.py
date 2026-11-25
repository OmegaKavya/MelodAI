"""The analysis endpoint's refusals, which are what keep a 512 MB instance up.

None of these reach the decoder or the model: each request is refused before
any audio is read, which is the property being tested.
"""
import pytest
from fastapi.testclient import TestClient

from server import main


@pytest.fixture(autouse=True)
def fresh_limiter():
    main._rate_hits.clear()
    yield
    main._rate_hits.clear()


client = TestClient(main.app)


def upload(name, body=b"not really audio"):
    return client.post("/api/analyse", files={"file": (name, body, "application/octet-stream")})


@pytest.mark.parametrize("name", ["notes.txt", "blob", "song.mp3.exe"])
def test_disallowed_or_missing_extensions_are_refused(name):
    assert upload(name).status_code == 415


def test_an_oversized_text_file_is_refused_without_reading_it(monkeypatch):
    # The extension check used to run after up to 25 MB had been buffered.
    def explode(*_args, **_kwargs):
        raise AssertionError("the body was read before the extension check")
    monkeypatch.setattr(main, "_analyse_upload", explode)
    assert upload("big.txt", b"x" * (5 * 1024 * 1024)).status_code == 415


def test_a_second_concurrent_analysis_waits_then_gets_503(monkeypatch):
    monkeypatch.setattr(main, "ANALYSIS_WAIT_SECONDS", 0.05)
    assert main._analysis_slot.acquire(timeout=1)
    try:
        res = upload("song.mp3")
    finally:
        main._analysis_slot.release()
    assert res.status_code == 503
    assert res.headers.get("retry-after")


def test_the_slot_is_released_after_a_failed_analysis():
    # An empty upload fails inside the analysis (400). If the slot leaked,
    # every later request would 503.
    assert upload("song.mp3", b"").status_code == 400
    assert main._analysis_slot.acquire(timeout=0.1)
    main._analysis_slot.release()


def test_rate_limit_applies_per_client():
    codes = [upload("x.txt").status_code for _ in range(main.RATE_MAX + 1)]
    assert codes[:main.RATE_MAX] == [415] * main.RATE_MAX
    assert codes[-1] == 429


def test_tracked_addresses_are_capped(monkeypatch):
    monkeypatch.setattr(main, "RATE_MAX_TRACKED", 50)
    for i in range(200):
        client.post("/api/analyse", files={"file": ("x.txt", b"x")},
                    headers={"x-forwarded-for": f"10.0.{i // 256}.{i % 256}"})
    assert len(main._rate_hits) <= 50
