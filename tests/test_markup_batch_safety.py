"""Batch safety: a Message Batch sent is never paid for and then lost. The keys are written down before the batch is
sent and a send cut off is looked for before anything is sent again; batches nobody runs for are collected by a sweep
(the CLI's `markup collect`, the studio once a day)."""

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")

from test_llm_anthropic import FakeClient, status_error  # noqa: E402
from test_markup_batch import FakeBatches, analysis_items  # noqa: E402

from folivox import markup_cost as mc  # noqa: E402
from folivox import markup_llm as ml  # noqa: E402
from folivox.llm_anthropic import AnthropicLLM  # noqa: E402

DAY = 86400.0
URL = "https://api.anthropic.com/v1/messages/batches"


class Clock:
    """Time that moves only when the run waits."""

    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Account(FakeBatches):
    """Message Batches of one account, with creation times, the list and the ways a send can end: ``lost`` (created,
    the answer never arrives), ``nothing`` (the connection dropped before it was created) or an exception."""

    def __init__(self, owner, clock, polls=1, sends=()):
        super().__init__(owner, polls)
        self.clock, self.sends = clock, list(sends)
        self.attempts: list[float] = []  # when each create was tried
        self.created_at: dict[str, float] = {}
        self.down = False  # the list cannot be reached
        self.gone: set[str] = set()
        self.lag = 0.0  # seconds a create takes before the batch exists

    def create(self, requests):
        self.attempts.append(self.clock.time())
        self.clock.sleep(self.lag)
        how = self.sends.pop(0) if self.sends else None
        if how == "nothing":
            raise anthropic.APIConnectionError(request=httpx.Request("POST", URL))
        if isinstance(how, Exception):
            raise how
        b = super().create(requests)
        self.created_at[b.id] = self.clock.time()
        if how == "lost":
            raise anthropic.APITimeoutError(request=httpx.Request("POST", URL))
        return b

    def list(self, limit=20):
        if self.down:
            raise anthropic.APIConnectionError(request=httpx.Request("GET", URL))
        for bid in reversed(list(self.batches)):  # newest first, as the API lists them
            n, ended = len(self.batches[bid]), self.looks[bid] > self.polls
            yield SimpleNamespace(id=bid, created_at=datetime.fromtimestamp(self.created_at[bid], timezone.utc),
                                  processing_status="ended" if ended else "in_progress",
                                  request_counts=SimpleNamespace(processing=0 if ended else n,
                                                                 succeeded=n if ended else 0, errored=0, canceled=0,
                                                                 expired=0))

    def retrieve(self, bid):
        if bid in self.gone:
            raise status_error(anthropic.NotFoundError, 404)
        return super().retrieve(bid)

    def results(self, bid):
        if bid in self.gone:
            raise status_error(anthropic.NotFoundError, 404)
        return super().results(bid)


def account(clock, **kw):
    c = FakeClient()
    c.messages.batches = Account(c, clock, **kw)
    return c, AnthropicLLM(model="claude-haiku-5-5", client=c)


def runner(llm, cache, clock, cancelled=lambda: False):
    return ml.Runner(llm, llm.model, cache, sleep=clock.sleep, clock=clock.time, cancelled=cancelled)


def test_a_batch_created_while_the_answer_was_lost_is_adopted_by_the_next_run(tmp_path):
    clock = Clock()
    c, llm = account(clock, sends=["lost"])
    batches = c.messages.batches
    batches.down = True  # the connection is gone: the run cannot look for its batch and is stopped
    cache = ml.Cache(tmp_path / "c.sqlite")
    items = analysis_items()
    assert runner(llm, cache, clock, cancelled=lambda: bool(batches.attempts)).ask_batch(items) is None
    assert len(batches.attempts) == 1 and len(batches.created) == 1  # created all the same, and paid for
    assert c.options == [{"max_retries": 0}]  # sent once, without the SDK's retries
    assert len(cache.sending()) == 1  # the write-ahead outlived the run
    batches.down = False
    again = runner(llm, cache, clock)
    out = again.ask_batch(items)
    assert len(batches.attempts) == 1  # nothing sent again: the batch was found and collected
    assert all(isinstance(o, dict) for o in out) and again.calls == 0  # nor asked on its own
    assert cache.sending() == {} and cache.pending(runner(llm, cache, clock).key(*items[0][:2])) is None


def test_a_send_cut_off_before_the_batch_existed_is_sent_again_once_after_the_grace(tmp_path):
    clock = Clock()
    c, llm = account(clock, sends=["nothing"])
    batches = c.messages.batches
    cache = ml.Cache(tmp_path / "c.sqlite")
    items = analysis_items()
    t0 = clock.time()
    stop_at = t0 + ml.RECONCILE_GRACE / 2
    assert runner(llm, cache, clock, cancelled=lambda: clock.time() >= stop_at).ask_batch(items) is None
    assert batches.attempts == [t0] and len(cache.sending()) == 1  # within the grace: looked, sent nothing again
    out = runner(llm, cache, clock).ask_batch(items)
    assert len(batches.attempts) == 2 and batches.attempts[1] > t0 + ml.RECONCILE_GRACE  # once, after the grace
    assert len(batches.created) == 1 and all(isinstance(o, dict) for o in out)
    assert cache.sending() == {}


def test_a_refused_send_leaves_no_write_ahead(tmp_path):
    clock = Clock()
    c, llm = account(clock, sends=[status_error(anthropic.BadRequestError, 400)])
    cache = ml.Cache(tmp_path / "c.sqlite")
    with pytest.raises(ValueError, match="HTTP 400"):
        runner(llm, cache, clock).ask_batch(analysis_items())
    assert cache.sending() == {} and len(c.messages.batches.attempts) == 1


def stopped_run(tmp_path, clock, **kw):
    """A run stopped while its batch runs: the cache file holds one uncollected batch of two requests."""
    c, llm = account(clock, **kw)
    path = tmp_path / "tale.llm-cache.sqlite"
    cache = ml.Cache(path)
    assert runner(llm, cache, clock, cancelled=lambda: bool(c.messages.batches.created)).ask_batch(
        analysis_items()) is None
    cache.close()
    return c, llm, path


def test_the_sweep_collects_a_batch_no_run_waits_for_and_a_later_run_asks_nothing(tmp_path):
    clock = Clock()
    c, llm, path = stopped_run(tmp_path, clock)
    logs = []
    price = mc.Price(1.0, 1.0, 1.0, 1.0)
    swept = ml.sweep([(path, "tale.epub")], llm, lambda level, msg: logs.append((level, msg)), price=price,
                     clock=clock.time)
    assert (swept.collected, swept.waiting, swept.lost, swept.replies) == (1, 0, 0, 2) and swept.usd > 0
    assert (logging.INFO, "markup batches swept: 1 collected, 0 waiting, 0 lost") in logs
    assert any(level == logging.INFO and "outside any run's spending cap" in msg for level, msg in logs)
    cache = ml.Cache(path)
    assert cache.pending_batches(clock.time()) == []
    answered = len(c.requests)  # the fake model's answers so far, batched or not
    later = runner(llm, cache, clock)
    out = later.ask_batch(analysis_items())
    assert all(isinstance(o, dict) for o in out) and later.cached == 2  # every answer came from the cache
    assert len(c.messages.batches.created) == 1 and len(c.requests) == answered and later.calls == 0


def test_an_old_batch_is_reported_every_sweep_and_one_anthropic_no_longer_has_is_crossed_off(tmp_path):
    clock = Clock()
    c, llm, path = stopped_run(tmp_path, clock, polls=10**6)  # still running: nothing to collect yet
    logs = []

    def log(level, msg):
        logs.append((level, msg))

    clock.sleep(15 * DAY)
    for _ in range(2):
        ml.sweep([(path, "tale.epub")], llm, log, clock=clock.time)
    warn = (logging.WARNING, "markup batch uncollected: msgbatch_0, 2 requests, 15 days old, tale.epub")
    assert logs.count(warn) == 2  # on every sweep
    assert logs.count((logging.INFO, "markup batches swept: 0 collected, 1 waiting, 0 lost")) == 2
    logs.clear()
    c.messages.batches.gone.add("msgbatch_0")
    swept = ml.sweep([(path, "tale.epub")], llm, log, clock=clock.time)
    assert (swept.collected, swept.waiting, swept.lost) == (0, 0, 1)
    assert [m for level, m in logs if level == logging.ERROR] == [
        "markup batch lost: msgbatch_0, 2 requests, 15 days old, tale.epub: Anthropic no longer has this batch "
        "(HTTP 404): no"]
    logs.clear()
    ml.sweep([(path, "tale.epub")], llm, log, clock=clock.time)  # crossed off: not looked at again
    assert logs == [(logging.INFO, "markup batches swept: 0 collected, 0 waiting, 0 lost")]
    assert not ml.has_uncollected(path)


def other_book_then_ours(tmp_path, clock):
    """Another book sends a batch of two; two seconds later this book writes ahead a batch of two whose send takes ten
    seconds and loses its answer, and the run stops. In the list the other book's batch is the closer to our send."""
    c, llm = account(clock, polls=0)
    batches = c.messages.batches
    other = ml.Cache(tmp_path / "other.llm-cache.sqlite")
    other_items = [(s, p + " (other book)", v) for s, p, v in analysis_items()]
    assert all(isinstance(o, dict) for o in runner(llm, other, clock).ask_batch(other_items))
    other.close()
    clock.sleep(2)
    batches.sends, batches.lag, batches.down = ["lost"], 10.0, True
    path = tmp_path / "tale.llm-cache.sqlite"
    cache = ml.Cache(path)
    assert runner(llm, cache, clock, cancelled=lambda: len(batches.attempts) > 1).ask_batch(analysis_items()) is None
    cache.close()
    batches.down, batches.lag = False, 0.0
    assert list(batches.batches) == ["msgbatch_0", "msgbatch_1"]  # the other book's, then ours
    return c, llm, path


def test_a_run_rejects_another_books_batch_of_the_same_size_and_collects_its_own(tmp_path):
    clock = Clock()
    c, llm, path = other_book_then_ours(tmp_path, clock)
    cache = ml.Cache(path)
    again = runner(llm, cache, clock)
    out = again.ask_batch(analysis_items())
    assert all(isinstance(o, dict) for o in out) and again.calls == 0  # ours collected, nothing sent or asked again
    assert len(c.messages.batches.attempts) == 2
    assert cache.known("msgbatch_1") and not cache.known("msgbatch_0")  # ours kept, the other book's rejected
    assert cache.sending() == {} and cache.pending_batches(clock.time()) == []


def test_the_sweep_rejects_another_books_batch_of_the_same_size_and_collects_its_own(tmp_path):
    clock = Clock()
    c, llm, path = other_book_then_ours(tmp_path, clock)
    logs = []
    swept = ml.sweep([(path, "tale.epub")], llm, lambda level, msg: logs.append(msg), clock=clock.time)
    assert (swept.collected, swept.waiting, swept.lost) == (1, 0, 0)
    assert "markup batches swept: 1 collected, 0 waiting, 0 lost" in logs
    cache = ml.Cache(path)
    assert cache.known("msgbatch_1") and not cache.known("msgbatch_0") and cache.sending() == {}
    later = runner(llm, cache, clock)
    assert all(isinstance(o, dict) for o in later.ask_batch(analysis_items()))
    assert later.calls == 0 and later.cached == 2 and len(c.messages.batches.attempts) == 2


def test_a_sweep_skips_a_book_a_run_holds_and_changes_nothing(tmp_path):
    clock = Clock()
    _, llm, path = stopped_run(tmp_path, clock)
    run = ml.open_cache(path)
    before = path.read_bytes()
    logs = []
    swept = ml.sweep([(path, "tale.epub")], llm, lambda level, msg: logs.append(msg), clock=clock.time)
    assert (swept.collected, swept.waiting, swept.lost, swept.skipped) == (0, 0, 0, 1)
    assert "markup batches of tale.epub skipped: a markup run is working on this book" in logs
    assert path.read_bytes() == before and ml.has_uncollected(path)
    run.close()


def test_a_run_started_while_a_sweep_holds_the_book_waits_and_asks_nothing_collected(tmp_path):
    import threading

    clock = Clock()
    c, llm, path = stopped_run(tmp_path, clock)
    batches = c.messages.batches
    inside, go, waited = threading.Event(), threading.Event(), threading.Event()
    look = batches.retrieve

    def slow_retrieve(bid):  # the sweep holds the lock while it looks
        inside.set()
        assert go.wait(10)
        return look(bid)

    batches.retrieve = slow_retrieve
    sweeper = threading.Thread(target=ml.sweep, args=([(path, "tale.epub")], llm, lambda level, msg: None),
                               kwargs={"clock": clock.time})
    sweeper.start()
    assert inside.wait(10)
    result = {}

    def run():
        cache = ml.open_cache(path, waiting=waited.set)
        try:
            r = runner(llm, cache, clock)
            result["out"], result["calls"], result["cached"] = r.ask_batch(analysis_items()), r.calls, r.cached
        finally:
            cache.close()

    t = threading.Thread(target=run)
    t.start()
    assert waited.wait(10)  # the run waits for the sweep
    go.set()
    sweeper.join(10)
    t.join(10)
    assert all(isinstance(o, dict) for o in result["out"]) and result["calls"] == 0 and result["cached"] == 2
    assert len(batches.created) == 1


class StoredMeanwhile:
    """A budget whose check, just before a batch is sent, finds that a reply was stored meanwhile (by a sweep)."""

    def __init__(self, cache, key, reply):
        self.cache, self.key, self.reply = cache, key, reply

    def check(self, chars, calls=1, batch=False):
        self.cache.put(self.key, self.reply)

    def record(self, *args, **kwargs):
        pass


def test_a_reply_stored_after_the_run_read_the_cache_is_not_sent_again(tmp_path):
    clock = Clock()
    c, llm = account(clock, polls=0)
    cache = ml.Cache(tmp_path / "c.sqlite")
    items = analysis_items()
    r = runner(llm, cache, clock)
    first = r.key(*items[0][:2])
    reply = FakeClient().model.complete_json(items[0][1], items[0][0])
    r.budget = StoredMeanwhile(cache, first, reply)
    out = r.ask_batch(items)
    assert all(isinstance(o, dict) for o in out) and out[0] == items[0][2](json.loads(reply))
    assert c.messages.batches.created == [[r.key(*items[1][:2])]]  # only the other key was sent


def test_markup_collect_sweeps_one_book_and_prints_the_counts(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from folivox.cli_markup import markup_app

    _, llm, path = stopped_run(tmp_path, Clock())
    monkeypatch.setattr(ml, "make_llm", lambda provider=None, model=None: (llm, llm.model))
    r = CliRunner().invoke(markup_app, ["collect", str(tmp_path / "tale.epub")])
    assert r.exit_code == 0, r.output
    assert "markup batches swept: 1 collected, 0 waiting, 0 lost" in r.output
    assert not ml.has_uncollected(path)


def sdk_llm(handler):
    """The real client over a transport that counts what reaches the network."""
    seen = []

    def count(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        return handler(request)

    client = anthropic.Anthropic(api_key="test-key", max_retries=2,
                                 http_client=httpx.Client(transport=httpx.MockTransport(count)))
    return AnthropicLLM(model="claude-haiku-5-5", client=client), seen


@pytest.mark.parametrize("fail", ["timeout", "drop", "500"])
def test_a_send_that_may_have_reached_anthropic_is_tried_exactly_once(fail):
    def handler(request):
        if fail == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if fail == "drop":
            raise httpx.RemoteProtocolError("connection closed", request=request)
        return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})

    llm, seen = sdk_llm(handler)
    with pytest.raises(ml.BatchSubmitUnknown) as e:
        llm.batch_submit([("k1", "p", "s")])
    assert isinstance(e.value, ConnectionError)  # callers that wait out a busy server still do
    assert [(m, p) for m, p, _ in seen] == [("POST", "/v1/messages/batches")]


def test_a_refused_send_keeps_its_mapping_and_is_not_retried():
    llm, seen = sdk_llm(lambda r: httpx.Response(400, json={"type": "error", "error": {
        "type": "invalid_request_error", "message": "bad"}}))
    with pytest.raises(ValueError, match="HTTP 400"):
        llm.batch_submit([("k1", "p", "s")])
    assert len(seen) == 1


def test_the_batch_list_reads_times_and_counts_and_stops_at_the_first_older_batch():
    t = 1_800_000_000

    def batch(bid, at, counts):
        return {"id": bid, "type": "message_batch", "processing_status": "in_progress",
                "created_at": datetime.fromtimestamp(at, timezone.utc).isoformat().replace("+00:00", "Z"),
                "expires_at": datetime.fromtimestamp(at + DAY, timezone.utc).isoformat().replace("+00:00", "Z"),
                "request_counts": dict(zip(("processing", "succeeded", "errored", "canceled", "expired"), counts,
                                           strict=True))}

    def handler(request):
        return httpx.Response(200, json={"data": [batch("b3", t + 60, (2, 0, 0, 0, 0)),
                                                  batch("b2", t, (0, 3, 1, 1, 0)),
                                                  batch("b1", t - 600, (5, 0, 0, 0, 0))],
                                         "has_more": True, "first_id": "b3", "last_id": "b1"})

    llm, seen = sdk_llm(handler)
    assert llm.batch_list(t - 1) == [("b3", t + 60, 2, "in_progress"), ("b2", t, 5, "in_progress")]
    assert len(seen) == 1  # the next page holds only older batches: not asked for


def test_the_studio_sweep_leaves_a_book_whose_markup_runs_to_its_run(tmp_path, monkeypatch):
    pytest.importorskip("starlette")
    from folivox import config
    from folivox.ui.app import create_app

    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    config.load.cache_clear()
    clock = Clock()
    c, llm = account(clock)
    data = tmp_path / "ui"
    (data / "llm-cache").mkdir(parents=True)
    for bid in ("aaaa", "bbbb"):  # a batch of each book left by a stopped run
        cache = ml.Cache(data / "llm-cache" / f"{bid}.sqlite")
        items = [(s, p + bid, v) for s, p, v in analysis_items()]
        assert runner(llm, cache, clock, cancelled=lambda: True).ask_batch(items) is None
        cache.close()
    (data / "uploads" / "bbbb").mkdir(parents=True)
    (data / "uploads" / "bbbb" / "tale.epub").write_bytes(b"")
    app = create_app(token="t", port=8765, data_dir=data, services={"llm": lambda name: llm})
    run = ml.open_cache(data / "llm-cache" / "aaaa.sqlite")  # a markup run of book aaaa holds it
    sweeper = app.state.batch_sweeper
    lines = []
    sweeper.say = lines.append
    swept = sweeper.sweep_once()
    assert (swept.collected, swept.waiting, swept.lost, swept.skipped) == (1, 0, 0, 1)
    assert ml.has_uncollected(data / "llm-cache" / "aaaa.sqlite")  # left to its run
    assert not ml.has_uncollected(data / "llm-cache" / "bbbb.sqlite")
    assert "folivox INFO: markup batches of aaaa skipped: a markup run is working on this book" in lines
    assert "folivox INFO: markup batches swept: 1 collected, 0 waiting, 0 lost" in lines
    run.close()
    assert sweeper.sweep_once().collected == 1  # the next sweep takes it


def test_the_studio_does_not_sweep_without_a_provider_that_has_batches(tmp_path, monkeypatch):
    pytest.importorskip("starlette")
    from test_markup_llm import FakeLLM

    from folivox import config
    from folivox.ui.app import create_app

    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    config.load.cache_clear()
    app = create_app(token="t", port=8765, data_dir=tmp_path / "ui", services={"llm": lambda name: FakeLLM()})
    assert app.state.batch_sweeper.sweep_once() is None
