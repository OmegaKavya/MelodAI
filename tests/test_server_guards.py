"""The analysis endpoint's refusals, which are what keep a 512 MB instance up.

None of these reach the decoder or the model: each request is refused before
any audio is read, which is the property being tested, or meets a stand-in
for the analysis.
"""
import asyncio
import threading
import time

import anyio
import httpx
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


# M4A and AAC included: an upload is decoded from memory, where neither can be.
@pytest.mark.parametrize("name", ["notes.txt", "blob", "song.mp3.exe", "song.m4a", "song.aac"])
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


def test_a_file_over_the_limit_is_refused_before_it_waits_for_the_slot(monkeypatch):
    # Between the file cap and the body cap a file fits in the request, and
    # used to wait its turn for the slot only to be refused once it had it.
    monkeypatch.setattr(main, "MAX_BYTES", 1024)
    monkeypatch.setattr(main, "ANALYSIS_WAIT_SECONDS", 0.05)
    assert main._analysis_slot.acquire(timeout=1)
    try:
        res = upload("song.mp3", b"x" * 1025)
    finally:
        main._analysis_slot.release()
    assert res.status_code == 413


def post_from(http, name, i):
    return asyncio.create_task(http.post(
        "/api/analyse", files={"file": (name, b"x")},
        headers={"x-forwarded-for": f"10.2.0.{i}"}))


def test_waiting_uploads_are_served_in_arrival_order(monkeypatch):
    # Every waiter used to poll for the slot, and whichever woke first after a
    # release took it. Here the later upload arrives just as the slot frees,
    # while the earlier one is still asleep between polls.
    monkeypatch.setattr(main, "ANALYSIS_POLL_SECONDS", 0.5)
    served = []

    def stand_in(file):
        served.append(file.filename)
        return {"ok": True}

    monkeypatch.setattr(main, "_analyse_upload", stand_in)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            assert main._analysis_slot.acquire(timeout=1)
            first = post_from(http, "first.mp3", 1)
            await asyncio.sleep(0.2)
            main._analysis_slot.release()
            second = post_from(http, "second.mp3", 2)
            return await asyncio.gather(first, second)

    results = asyncio.run(scenario())

    assert [r.status_code for r in results] == [200, 200]
    assert served == ["first.mp3", "second.mp3"]
    assert not main._waiting
    assert main._analysis_admitted == 0


@pytest.mark.parametrize("gives_up", ["times out", "is cancelled"])
def test_a_waiter_that_gives_up_leaves_the_line(monkeypatch, gives_up):
    # Only the head of the line may take the slot, so a waiter left in it
    # would block everyone behind it until their deadlines ran out.
    monkeypatch.setattr(main, "ANALYSIS_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(main, "_analyse_upload", lambda file: {"ok": True})

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            assert main._analysis_slot.acquire(timeout=1)
            try:
                waiter = post_from(http, "first.mp3", 1)
                if gives_up == "times out":
                    assert (await waiter).status_code == 503
                else:
                    await asyncio.sleep(0.1)
                    waiter.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiter
            finally:
                main._analysis_slot.release()
            return await post_from(http, "second.mp3", 2)

    assert asyncio.run(scenario()).status_code == 200
    assert not main._waiting
    assert main._analysis_admitted == 0


def test_a_cancelled_analysis_keeps_the_slot_and_its_upload_until_it_ends(monkeypatch):
    # A thread cannot be stopped. Releasing on cancellation let a second
    # analysis start beside the first, and closed the upload under it.
    started = threading.Event()
    seen = {}

    def stand_in(file):
        started.set()
        time.sleep(0.5)
        seen["upload closed"] = file.file.closed
        seen["slot free"] = main._analysis_slot.acquire(blocking=False)
        if seen["slot free"]:
            main._analysis_slot.release()
        return {"ok": True}

    monkeypatch.setattr(main, "_analyse_upload", stand_in)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            request = post_from(http, "song.mp3", 1)
            while not started.is_set():
                await asyncio.sleep(0.01)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            return dict(seen)

    assert asyncio.run(scenario()) == {"upload closed": False, "slot free": False}
    assert main._analysis_slot.acquire(blocking=False)
    main._analysis_slot.release()
    assert main._analysis_admitted == 0


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


def test_a_waiting_upload_holds_no_worker_thread_and_health_still_answers(monkeypatch):
    # Forty queued uploads used to park a worker thread each, and with the
    # pool gone /api/health took thirty seconds. Two threads in all here: one
    # for the running analysis and one spare, which any waiter that parked a
    # thread would take.
    release = threading.Event()
    running, peak = [], []

    def stand_in(file):
        running.append(file)
        peak.append(len(running))
        release.wait(5)
        running.pop()
        return {"ok": True}

    monkeypatch.setattr(main, "_analyse_upload", stand_in)
    admitted = 1 + main.ANALYSIS_QUEUE
    extra = 3

    async def scenario():
        anyio.to_thread.current_default_thread_limiter().total_tokens = 2
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            uploads = [asyncio.create_task(http.post(
                "/api/analyse", files={"file": ("song.mp3", b"x")},
                headers={"x-forwarded-for": f"10.1.0.{i}"}))
                for i in range(admitted + extra)]
            await asyncio.sleep(0.5)
            refused_at_once = [u.result() for u in uploads if u.done()]
            health = await asyncio.wait_for(http.get("/api/health"), timeout=2)
            await asyncio.wait_for(anyio.to_thread.run_sync(lambda: None), timeout=2)
            release.set()
            return refused_at_once, health, await asyncio.gather(*uploads)

    refused_at_once, health, results = asyncio.run(scenario())

    assert health.status_code == 200
    assert [r.status_code for r in refused_at_once] == [503] * extra
    assert all(r.headers.get("retry-after") for r in refused_at_once)
    assert sorted(r.status_code for r in results) == [200] * admitted + [503] * extra
    assert max(peak) == 1
    assert main._analysis_admitted == 0


def test_health_answers_with_every_worker_thread_busy():
    # Render restarts the service on a slow health check, so health must not
    # need a thread at all, however the thread pool comes to be full.
    held = threading.Event()

    async def scenario():
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            occupant = asyncio.create_task(anyio.to_thread.run_sync(held.wait, 5))
            await asyncio.sleep(0.1)
            try:
                return await asyncio.wait_for(http.get("/api/health"), timeout=2)
            finally:
                held.set()
                await occupant

    assert asyncio.run(scenario()).status_code == 200


def multipart(name, data, boundary="b"):
    return (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            ).encode() + data + f"\r\n--{boundary}--\r\n".encode()


def analyse_scope(content_length=None):
    headers = [(b"content-type", b"multipart/form-data; boundary=b")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/api/analyse",
        "raw_path": b"/api/analyse", "root_path": "", "query_string": b"",
        "headers": headers, "client": ("127.0.0.1", 1), "server": ("test", 80),
    }


def raw_post(chunks, content_length=None):
    """Drive the app directly, counting how much of the body it asked for.

    The test client hands over the whole body up front, so it cannot show
    whether a request was refused before or after its body was read.
    """
    scope = analyse_scope(content_length)
    pending = list(chunks)
    consumed = []
    sent = []

    async def receive():
        if not pending:
            return {"type": "http.disconnect"}
        body = pending.pop(0)
        consumed.append(len(body))
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    async def send(message):
        sent.append(message)

    asyncio.run(main.app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], dict(start["headers"]), sum(consumed)


def test_an_oversized_declared_length_is_refused_before_the_body_is_read():
    status, _, consumed = raw_post([multipart("song.mp3", b"x")],
                                   content_length=main.MAX_BODY_BYTES + 1)
    assert status == 413
    assert consumed == 0


def test_an_undeclared_body_is_cut_off_at_the_cap(monkeypatch):
    # A chunked upload carries no Content-Length, so the cap has to hold on
    # the bytes themselves while the multipart parser is spooling them.
    monkeypatch.setattr(main, "MAX_BODY_BYTES", 256 * 1024)
    body = multipart("song.mp3", b"x" * (4 * 1024 * 1024))
    chunks = [body[i:i + 64 * 1024] for i in range(0, len(body), 64 * 1024)]
    status, _, consumed = raw_post(chunks)
    assert status == 413
    assert consumed <= main.MAX_BODY_BYTES + 64 * 1024


@pytest.mark.parametrize("header_chunk", [7, 64 * 1024])
def test_a_disallowed_file_is_refused_at_its_headers_before_its_body(header_chunk):
    # The extension was checked once the form was parsed, by which point up
    # to 25 MB of a .txt had been received and spooled. Headers split across
    # small chunks must be caught as early as headers that arrive whole.
    body = multipart("notes.txt", b"x" * (4 * 1024 * 1024))
    headers_end = body.index(b"\r\n\r\n") + 4
    head, rest = body[:headers_end], body[headers_end:]
    chunks = ([head[i:i + header_chunk] for i in range(0, len(head), header_chunk)]
              + [rest[i:i + 64 * 1024] for i in range(0, len(rest), 64 * 1024)])
    status, _, consumed = raw_post(chunks)
    assert status == 415
    assert consumed == headers_end


def test_a_file_right_at_the_limit_still_fits_in_the_body(monkeypatch):
    monkeypatch.setattr(main, "_analyse_upload",
                        lambda file: {"bytes": len(file.file.read())})
    res = upload("song.mp3", b"x" * main.MAX_BYTES)
    assert res.status_code == 200
    assert res.json() == {"bytes": main.MAX_BYTES}


def test_a_full_queue_refuses_without_reading_the_body(monkeypatch):
    monkeypatch.setattr(main, "_analysis_admitted", 1 + main.ANALYSIS_QUEUE)
    body = multipart("song.mp3", b"x")
    status, headers, consumed = raw_post([body], content_length=len(body))
    assert status == 503
    assert headers.get(b"retry-after")
    assert consumed == 0


def test_a_queue_that_fills_during_an_upload_still_refuses_it(monkeypatch):
    # The early check runs before the body arrives. By the time it has, other
    # uploads may have taken every place, and this one must not jump in.
    def explode(_file):
        raise AssertionError("admitted past a full queue")
    monkeypatch.setattr(main, "_analyse_upload", explode)
    body = multipart("song.mp3", b"x")
    arrived = asyncio.Event()
    sent = []

    async def receive():
        await arrived.wait()
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    async def scenario():
        request = asyncio.create_task(
            main.app(analyse_scope(len(body)), receive, send))
        await asyncio.sleep(0.05)
        monkeypatch.setattr(main, "_analysis_admitted", 1 + main.ANALYSIS_QUEUE)
        arrived.set()
        await request

    asyncio.run(scenario())
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 503
