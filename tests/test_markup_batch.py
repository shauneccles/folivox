"""Batch mode of the marking pass: windows in strands, one Message Batch per round, against a fake Anthropic client."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("httpx2")

from test_llm_anthropic import FakeClient, book  # noqa: E402

from folivox import markup_cost as mc  # noqa: E402
from folivox import markup_llm as ml  # noqa: E402
from folivox.book import Block, Chapter  # noqa: E402
from folivox.llm_anthropic import AnthropicLLM  # noqa: E402


class FakeBatches:
    """Message Batches over the fake client: a batch runs for ``polls`` looks, then every request is answered as a
    single request would be (or errored / expired, for the ids in ``fail``)."""

    def __init__(self, owner, polls=1, fail=None):
        self.owner, self.polls, self.fail = owner, polls, fail or {}
        self.created, self.looks, self.batches = [], {}, {}

    def create(self, requests):
        bid = f"msgbatch_{len(self.created)}"
        self.created.append([r["custom_id"] for r in requests])
        self.batches[bid] = requests
        self.looks[bid] = 0
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        self.looks[bid] += 1
        return SimpleNamespace(id=bid, processing_status="ended" if self.looks[bid] > self.polls else "in_progress")

    def results(self, bid):
        for n, r in enumerate(self.batches[bid]):
            kind = self.fail.get((bid, n), "succeeded")
            msg = self.owner.answer(r["params"]) if kind == "succeeded" else None
            yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type=kind, message=msg))


def client(**kw):
    c = FakeClient()
    c.messages.batches = FakeBatches(c, **kw)
    return c


def text_of(req):
    return req.get("system", "") + "".join(b["text"] for b in req["messages"][0]["content"])


def run(c, strands, chapters, cache=None, budget=None, cancelled=lambda: False):
    llm = AnthropicLLM(model="claude-haiku-5-5", client=c)
    runner = ml.Runner(llm, llm.model, cache or ml.Cache(None), budget=budget, sleep=lambda s: None,
                       cancelled=cancelled)
    bible = ml.analyse(chapters, "ru", ml.Runner(llm, llm.model, ml.Cache(None)))
    c.requests.clear()
    markup, stats = ml.annotate(chapters, "ru", bible, runner, strands=strands)
    return markup, stats, runner


def test_one_strand_asks_exactly_what_the_one_at_a_time_run_asks():
    chapters = book(3)
    seq_client = client()
    seq, seq_stats, _ = run(seq_client, 0, chapters)
    bat_client = client()
    bat, bat_stats, runner = run(bat_client, 1, chapters)
    sent = [text_of(r) for b in bat_client.messages.batches.batches.values() for r in (x["params"] for x in b)]
    assert sent == [text_of(r) for r in seq_client.requests]
    assert bat.to_dict() == seq.to_dict() and bat_stats == seq_stats
    assert len(bat_client.messages.batches.created) == len(sent) > 1  # one window a round
    assert seq_stats.marked > 0 and all(c.done for c in bat.chapters.values())


def test_strands_send_a_window_of_each_per_round_and_a_mid_book_strand_sees_what_came_before():
    chapters = book(3)
    c = client()
    markup, stats, _ = run(c, 4, chapters)
    created = c.messages.batches.created
    wins = len(ml.windows(chapters, ml.WINDOW_CHARS, only_speech=True))
    assert len(created[0]) == min(4, wins) and sum(map(len, created)) == wins
    assert len(created) == -(-wins // 4)  # rounds: the longest strand
    first_round = [text_of(r["params"]) for r in c.messages.batches.batches["msgbatch_0"]]
    assert "who spoke them is not known yet" not in first_round[0]  # the first strand starts the book
    assert all("who spoke them is not known yet" in t for t in first_round[1:])
    assert stats.failed_windows == 0 and stats.marked > 0 and all(ch.done for ch in markup.chapters.values())


def test_an_errored_or_expired_request_is_asked_again_on_its_own():
    chapters = book(2)
    c = client(fail={("msgbatch_0", 0): "errored", ("msgbatch_0", 1): "expired"})
    markup, stats, runner = run(c, 4, chapters)
    assert c.messages.batches.created == [c.messages.batches.created[0]] and len(c.messages.batches.created[0]) == 2
    assert len(c.requests) == 2  # both asked again one at a time (the fake logs only answered requests)
    assert stats.failed_windows == 0 and stats.marked > 0


def test_a_run_stopped_while_the_batch_runs_collects_it_next_time(tmp_path):
    chapters = book(2)
    cache = ml.Cache(tmp_path / "c.sqlite")
    c = client(polls=3)
    stop = {"now": False}

    def cancelled():
        if c.messages.batches.created:  # stop once the first batch is on its way
            stop["now"] = True
        return stop["now"]

    markup, stats, _ = run(c, 4, chapters, cache=cache, cancelled=cancelled)
    assert len(c.messages.batches.created) == 1 and stats.marked == 0
    again = client()
    again.messages.batches = c.messages.batches  # the same account: the batch is still there
    markup, stats, _ = run(again, 4, chapters, cache=cache)
    assert len(c.messages.batches.created) == 1  # the first round was collected, not sent again
    assert c.messages.batches.looks["msgbatch_0"] > c.messages.batches.polls and stats.marked > 0


def test_batch_replies_count_at_half_price_and_the_cap_stops_before_a_round():
    chapters = book(2)
    price = mc.Price(1.0, 1.0, 1.0, 1.0)
    budget = mc.Budget(price)
    run(client(), 4, chapters, budget=budget)
    assert budget.batched.calls > 0
    assert budget.usd == pytest.approx(price.usd(budget.spent) - 0.5 * price.usd(budget.batched))
    capped = mc.Budget(price, max_cost=price.usd(budget.spent) * 0.01)
    c = client()
    with pytest.raises(ml.CostCapReached):
        run(c, 4, chapters, budget=capped)
    assert c.messages.batches.created == []


def test_the_estimate_prices_the_marking_calls_at_half():
    chapters = book(2)
    bible = ml.Bible()
    price = mc.Price(4.0, 20.0, 0.2, 5.0, "list")
    wins = mc.annotate_windows(chapters)
    full = mc.estimate(chapters, "ru", bible, price, annotate=wins)
    half = mc.estimate(chapters, "ru", bible, price, annotate=wins, batch=True)
    assert half.usd == pytest.approx(full.usd / 2) and half.usd_max == pytest.approx(full.usd_max / 2)
    assert "Message Batches, at half price" in mc.describe(half)


def test_batch_mode_needs_a_provider_with_batches():
    from test_markup_llm import CH, FakeLLM

    bible = ml.analyse(CH, "ru", ml.Runner(FakeLLM(), "m", ml.Cache(None)))
    with pytest.raises(ml.LLMError, match="Message Batches"):
        ml.annotate(CH, "ru", bible, ml.Runner(FakeLLM(), "m", ml.Cache(None)), strands=4)


def reader(c):
    llm = AnthropicLLM(model="claude-haiku-5-5", client=c)
    return ml.Runner(llm, llm.model, ml.Cache(None), sleep=lambda s: None)


def batch_params(c):
    return [r["params"] for b in c.messages.batches.batches.values() for r in b]


def test_one_reading_strand_reads_exactly_as_the_one_at_a_time_run():
    chapters = book(3)
    seq_client, bat_client = client(), client()
    seq = ml.analyse(chapters, "ru", reader(seq_client))
    bat = ml.analyse(chapters, "ru", reader(bat_client), strands=1)
    assert [text_of(p) for p in batch_params(bat_client)] == [text_of(r) for r in seq_client.requests]
    assert all("json_schema" in json.dumps(p) for p in batch_params(bat_client))  # structured outputs, as in real time
    assert len(bat_client.messages.batches.created) == len(seq_client.requests) > 1
    assert bat.to_dict() == seq.to_dict() and bat.done_windows == bat.total_windows


def test_reading_strands_build_their_own_lists_and_are_merged_into_one():
    chapters = book(4)
    seq = ml.analyse(chapters, "ru", reader(client()))
    c = client()
    bat = ml.analyse(chapters, "ru", reader(c), strands=3)
    created = c.messages.batches.created
    assert len(created[0]) == 3 and len(created) == -(-seq.total_windows // 3)  # a window of each strand a round
    later = [text_of(p) for p in batch_params(c)][1:3]  # the first windows of strands 2 and 3: nobody known yet
    assert all("(none yet)" in t for t in later)
    assert sorted(x.name for x in bat.characters) == sorted(x.name for x in seq.characters)  # each one once
    assert {x.name: x.lines for x in bat.characters} == {x.name: x.lines for x in seq.characters}
    assert sorted(x.id for x in bat.speakers()) == list(range(1, len(bat.speakers()) + 1))
    assert bat.narration == "third" and bat.done_windows == bat.total_windows
    assert len(c.requests) == len(batch_params(c))  # two characters: no one to join, nothing asked outside


def test_strands_that_know_one_person_under_two_names_are_joined_as_the_consolidate_pass_joins_them(monkeypatch):
    import test_markup_llm

    monkeypatch.setitem(test_markup_llm.CAST, "Аня", "f")
    chapters = book(4)
    last = chapters[-1]
    chapters[-1] = Chapter(last.title, [Block(b.kind, b.text.replace("Анна", "Аня")) for b in last.blocks])
    c = client()
    bat = ml.analyse(chapters, "ru", reader(c), strands=4)
    asked = [text_of(r) for r in c.requests if ml.CONSOLIDATE_SYSTEM in r["system"]]
    assert len(asked) == 1 and "Аня" in asked[0]  # the one call outside the batches: which names are one person
    assert sorted(x.name for x in bat.characters) == ["Анна", "Борис"] and "Аня" in bat.find("Анна").aliases


def test_a_single_strand_has_nothing_to_merge_and_asks_nothing_more():
    c = client()
    ml.analyse(book(3), "ru", reader(c), strands=1)
    assert len(c.requests) == len(batch_params(c))


def test_merging_strand_bibles_finds_a_character_under_an_alias():
    into = ml.Bible("ru", characters=[ml.Character("Анна", gender="f", lines=3)])
    other = ml.Bible("ru", characters=[ml.Character("Анна Ивановна", aliases=["Анна"], gender="f", lines=2, first=4),
                                       ml.Character("Борис", gender="m", lines=1, first=5)])
    ml.merge_bibles(into, [other])
    assert [(x.name, x.lines) for x in into.characters] == [("Анна", 5), ("Борис", 1)]
    assert "Анна Ивановна" in into.characters[0].aliases


def test_the_estimate_prices_the_reading_calls_at_half_with_batch_reading():
    chapters = book(2)
    bible = ml.Bible()
    price = mc.Price(4.0, 20.0, 0.2, 5.0, "list")
    read, mark = mc.analyse_windows(chapters, bible), mc.annotate_windows(chapters)
    full = mc.estimate(chapters, "ru", bible, price, analyse=read, annotate=mark)
    half = mc.estimate(chapters, "ru", bible, price, analyse=read, annotate=mark, batch=True, batch_reading=True)
    assert half.usd == pytest.approx(full.usd / 2) and half.usd_max == pytest.approx(full.usd_max / 2)
    assert "reading and marking calls go through Message Batches" in mc.describe(half)
