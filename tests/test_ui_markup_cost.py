"""The studio's markup with a paid language model: the cost next to each run, the spending cap, the model list."""

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

import test_ui_markup_llm as studio  # noqa: E402
from test_markup_cost import PaidLLM  # noqa: E402

from folivox import config  # noqa: E402

client = studio.client  # the studio fixture
upload, wait = studio.upload, studio.wait


def use_api(client, model="claude-haiku-5-5", extra=""):
    (client.data_dir.parent / "config.toml").write_text(
        f'[providers]\nllm = "openai"\n[llm.openai]\nmodel = "{model}"\n{extra}', encoding="utf-8")
    config.load.cache_clear()


def test_the_estimate_shows_tokens_and_dollars_for_a_paid_model(client):
    bid = upload(client)
    local = client.get(f"/api/books/{bid}/llm/estimate").json()["cost"]
    assert local["local"] is True and "modes" not in local
    use_api(client)
    cost = client.get(f"/api/books/{bid}/llm/estimate").json()["cost"]
    best, preview = cost["modes"]["best"], cost["modes"]["preview"]
    assert cost["model"] == "claude-haiku-5-5" and not cost["local"]
    assert best["calls"] >= 1 and best["input_tokens"] > 0 and 0 < best["usd"] <= best["usd_max"]
    assert best["extra_calls"] >= 8 and preview["extra_calls"] == 0
    other = client.get(f"/api/books/{bid}/llm/estimate?model=claude-opus-5-5").json()["cost"]["modes"]["best"]
    assert other["usd"] > best["usd"]  # the chosen model's price
    unknown = client.get(f"/api/books/{bid}/llm/estimate?model=some/new-model").json()["cost"]["modes"]["best"]
    assert unknown["usd"] is None and unknown["price_known"] is False and unknown["input_tokens"] > 0
    models = client.get("/api/llm/models").json()
    assert models["local"] is False and models["configured"] == "claude-haiku-5-5" and models["tiers"] == []


def test_the_spending_cap_is_checked_and_stops_a_run_cleanly(client):
    bid = upload(client)
    for bad in ("abc", 0, -1):
        r = client.post(f"/api/books/{bid}/llm/start", json={"task": "auto", "max_cost": bad})
        assert r.status_code == 400 and "spending cap" in r.json()["error"].lower()
    use_api(client, "some/new-model")
    r = client.post(f"/api/books/{bid}/llm/start", json={"task": "auto", "model": "some/new-model", "max_cost": 1})
    assert r.status_code == 400 and "price" in r.json()["error"]  # a cap is never kept on a guessed price
    use_api(client, "some/new-model", "price_in = 1000\nprice_out = 1000\n")
    client.holder["llm"] = PaidLLM(output=400)  # about $0.95 a reply: a $1.50 cap stops before the second
    r = client.post(f"/api/books/{bid}/llm/start", json={"task": "auto", "model": "some/new-model", "max_cost": "1.5"})
    assert r.status_code == 202
    st = wait(client, bid)
    assert st["state"] == "cancelled" and st["reason"] == "cost_cap" and "Spending cap $1.50" in st["message"]
    assert st["spent"]["calls"] == 1 and st["spent"]["cap"] == 1.5 and st["spent"]["usd"] <= 1.5
    # a higher cap carries on where it stopped
    r = client.post(f"/api/books/{bid}/llm/start", json={"task": "auto", "model": "some/new-model", "max_cost": 50})
    st = wait(client, bid)
    assert st["state"] == "done" and st["reason"] == "" and st["spent"]["usd"] <= 50


def test_batch_mode_is_offered_priced_and_run_for_a_provider_with_message_batches(client):
    pytest.importorskip("anthropic")
    pytest.importorskip("httpx2")
    from test_markup_batch import client as batch_client

    from folivox.llm_anthropic import AnthropicLLM

    bid = upload(client)
    use_api(client)  # an OpenAI-compatible API: no Message Batches
    assert client.get("/api/llm/models").json()["batches"] is False
    est = client.get(f"/api/books/{bid}/llm/estimate").json()
    assert "best_batch" not in est["cost"]["modes"]
    r = client.post(f"/api/books/{bid}/llm/start", json={"task": "best", "batch": True})
    assert r.status_code == 400 and "Message Batches" in r.json()["error"]

    (client.data_dir.parent / "config.toml").write_text('[providers]\nllm = "anthropic"\n', encoding="utf-8")
    config.load.cache_clear()
    assert client.get("/api/llm/models").json()["batches"] is True
    est = client.get(f"/api/books/{bid}/llm/estimate").json()
    best, half = est["cost"]["modes"]["best"], est["cost"]["modes"]["best_batch"]
    assert best["usd"] / 2 <= half["usd"] < best["usd"]  # the second looks stay at the full price
    assert half["batch"] and half["batch_reading"]
    assert est["modes"]["best"]["batch"]["rounds"] >= 2 and est["modes"]["best"]["batch"]["secs"] > 0

    fake = batch_client(polls=0)  # ended at the first look: no wait between polls
    client.holder["llm"] = AnthropicLLM(model="claude-haiku-5-5", client=fake)
    r = client.post(f"/api/books/{bid}/llm/start", json={"task": "best", "model": "claude-haiku-5-5", "batch": True})
    assert r.status_code == 202
    st = wait(client, bid)
    assert st["state"] == "done" and st["stats"]["marked"] == 3
    sent = fake.messages.batches.created
    assert len(sent) == 2  # one round reads the book (fewer windows than strands), one marks it
