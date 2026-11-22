"""The studio's Download audio: a book's last finished render, recorded, found again after a move, and served."""
from __future__ import annotations

import os
from pathlib import Path

from folivox.ui import outputs


def _file(path: Path, data: bytes, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (mtime, mtime))
    return path


def test_record_of_a_file(tmp_path):
    m4b = _file(tmp_path / "Book.m4b", b"x" * 10, 1_700_000_000.5)
    rec = outputs.record(m4b, seconds=61.2, chapters=3, now=1_800_000_000.0)
    assert rec == {"path": str(m4b), "dir": False, "bytes": 10, "mtime": 1_700_000_000.5, "seconds": 61.2,
                   "chapters": 3, "at": 1_800_000_000.0}


def test_record_of_an_mp3_folder_sums_its_mp3s(tmp_path):
    folder = tmp_path / "Book"
    _file(folder / "01.mp3", b"a" * 4, 1_700_000_000.0)
    _file(folder / "02.mp3", b"b" * 6, 1_700_000_100.0)
    _file(folder / "cover.jpg", b"c" * 50, 1_700_000_200.0)
    rec = outputs.record(folder, seconds=None, chapters=2, now=1.0)
    assert (rec["dir"], rec["bytes"], rec["mtime"]) == (True, 10, 1_700_000_100.0)


def test_locate_in_place(tmp_path):
    m4b = _file(tmp_path / "Book.m4b", b"x" * 10, 1_700_000_000.5)
    assert outputs.locate(outputs.record(m4b, seconds=1, chapters=1, now=1)) == m4b


def test_locate_refuses_a_file_that_changed_size(tmp_path):
    m4b = _file(tmp_path / "Book.m4b", b"x" * 10, 1_700_000_000.5)
    rec = outputs.record(m4b, seconds=1, chapters=1, now=1)
    _file(m4b, b"x" * 11, 1_700_000_000.5)
    assert outputs.locate(rec) is None


def test_locate_gives_none_when_the_file_is_gone(tmp_path):
    m4b = _file(tmp_path / "Book.m4b", b"x" * 10, 1_700_000_000.5)
    rec = outputs.record(m4b, seconds=1, chapters=1, now=1)
    m4b.unlink()
    assert outputs.locate(rec) is None


def test_summary(tmp_path):
    m4b = _file(tmp_path / "Book.m4b", b"x" * 10, 1_700_000_000.5)
    rec = outputs.record(m4b, seconds=61.2, chapters=3, now=5.0)
    assert outputs.summary(rec, m4b) == {"name": "Book.m4b", "bytes": 10, "seconds": 61.2, "chapters": 3, "at": 5.0,
                                         "zip": False}
    assert outputs.summary(rec, None) is None and outputs.summary(None, None) is None


# ---- the studio: recorded when a render ends done, served per book ---------------------------------------------------

import io  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import pytest  # noqa: E402

from folivox.pipeline import Cancelled  # noqa: E402

TOKEN = "t0ken"
BOOK = ("Chapter 1\n\n" + "It was a quiet morning in the village. " * 12 + "\n\nChapter 2\n\n"
        + "The river was cold and bright. " * 12 + "\n").encode()
OTHER = ("Chapter 1\n\n" + "A different book about the sea and its ships. " * 12 + "\n").encode()


def writing_render(book, opts, events):
    """A render that writes its output (one m4b, or a folder of mp3s) and builds two chapters."""
    out = opts.output
    if opts.mp3:
        out.mkdir(parents=True, exist_ok=True)
        for n in (1, 2):
            (out / f"{n:02d} Chapter {n}.mp3").write_bytes(b"ID3" + bytes([n]) * 5)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"m4b-bytes")
    events.chapter_built(0, "Chapter 1", 2.5)
    events.chapter_built(1, "Chapter 2", 4.0)
    return out


def make_client(tmp_path, monkeypatch, render=writing_render, **kw):
    """The studio app (the ui extra; skipped without it, like the other studio tests, so the outputs tests above run
    in the core-only CI too)."""
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    from folivox.ui.app import create_app

    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    from folivox import config

    config.load.cache_clear()
    app = create_app(token=TOKEN, port=8765, data_dir=tmp_path / "ui", render=render, **kw)
    return TestClient(app, base_url="http://127.0.0.1:8765", headers={"X-Folivox-Token": TOKEN})


@pytest.fixture
def studio(tmp_path, monkeypatch):
    yield make_client(tmp_path, monkeypatch)
    from folivox import config

    config.load.cache_clear()


def upload(client, data=BOOK, name="tale.txt"):
    r = client.post("/api/books", files={"file": (name, data, "text/plain")})
    assert r.status_code == 200, r.text
    return r.json()


def run(client, book_id, out, kind="render", mp3=False):
    r = client.post("/api/jobs", json={"kind": kind, "book_id": book_id, "chapters": [1, 2],
                                       "options": {"output_dir": str(out), "qa": False, "mp3": mp3}})
    assert r.status_code == 202, r.text
    for _ in range(200):
        j = client.get(f"/api/jobs/{r.json()['id']}").json()
        if j["state"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def audio_of(client, book_id):
    return client.post(f"/api/books/{book_id}/open").json()["audio"]


def test_a_finished_render_is_offered_for_download(studio, tmp_path):
    b = upload(studio)
    assert b["audio"] is None
    assert run(studio, b["book_id"], tmp_path / "out")["state"] == "done"
    a = audio_of(studio, b["book_id"])
    assert (a["name"], a["bytes"], a["seconds"], a["chapters"], a["zip"]) == ("tale.m4b", 9, 6.5, 2, False)
    r = studio.get(f"/api/books/{b['book_id']}/audio")
    assert r.status_code == 200 and r.content == b"m4b-bytes"
    assert "attachment" in r.headers["content-disposition"] and "tale.m4b" in r.headers["content-disposition"]
    assert r.headers["content-type"].startswith("audio/mp4")


def test_the_record_survives_reopening_the_listing_and_a_restart(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    b = upload(c)
    run(c, b["book_id"], tmp_path / "out")
    audio_of(c, b["book_id"])                      # opening rewrites the library entry
    assert audio_of(c, b["book_id"])["name"] == "tale.m4b"
    listed = {x["book_id"]: x for x in c.get("/api/books").json()["books"]}
    assert listed[b["book_id"]]["audio"]["name"] == "tale.m4b"
    again = make_client(tmp_path, monkeypatch)    # a new studio on the same data folder
    assert audio_of(again, b["book_id"])["name"] == "tale.m4b"
    assert upload(again)["audio"]["name"] == "tale.m4b"   # the same file is the same book


def test_a_preview_is_not_recorded(studio, tmp_path):
    b = upload(studio)
    run(studio, b["book_id"], tmp_path / "out", kind="preview")
    assert audio_of(studio, b["book_id"]) is None


@pytest.mark.parametrize("error", [Cancelled, RuntimeError])
def test_a_render_that_does_not_finish_is_not_recorded(tmp_path, monkeypatch, error):
    def failing(book, opts, events):
        writing_render(book, opts, events)
        raise error("stop")

    c = make_client(tmp_path, monkeypatch, render=failing)
    b = upload(c)
    assert run(c, b["book_id"], tmp_path / "out")["state"] in ("cancelled", "error")
    assert audio_of(c, b["book_id"]) is None


def test_a_render_whose_output_is_missing_still_finishes(tmp_path, monkeypatch):
    def no_file(book, opts, events):
        return opts.output

    c = make_client(tmp_path, monkeypatch, render=no_file)
    b = upload(c)
    assert run(c, b["book_id"], tmp_path / "out")["state"] == "done"
    assert audio_of(c, b["book_id"]) is None


def test_gone_audio_is_not_offered(studio, tmp_path):
    b = upload(studio)
    run(studio, b["book_id"], tmp_path / "out")
    (tmp_path / "out" / "tale.m4b").unlink()
    assert audio_of(studio, b["book_id"]) is None
    r = studio.get(f"/api/books/{b['book_id']}/audio")
    assert r.status_code == 404 and "No finished audio" in r.json()["error"]


def test_only_the_book_own_audio_is_served(studio, tmp_path):
    b = upload(studio)
    run(studio, b["book_id"], tmp_path / "out")
    other = upload(studio, OTHER, "sea.txt")
    assert studio.get(f"/api/books/{other['book_id']}/audio").status_code == 404
    assert studio.get("/api/books/nope/audio").status_code == 404
    assert studio.get(f"/api/books/{b['book_id']}/audio", headers={"X-Folivox-Token": "wrong"}).status_code == 401


def test_an_mp3_render_downloads_as_a_zip(studio, tmp_path):
    b = upload(studio)
    run(studio, b["book_id"], tmp_path / "out", mp3=True)
    a = audio_of(studio, b["book_id"])
    assert a["zip"] and a["name"] == "tale.zip"
    r = studio.get(f"/api/books/{b['book_id']}/audio")
    assert r.status_code == 200 and "tale.zip" in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert z.namelist() == ["01 Chapter 1.mp3", "02 Chapter 2.mp3"]
    assert z.read("02 Chapter 2.mp3") == b"ID3" + bytes([2]) * 5
    assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


def test_ping_says_whether_the_studio_is_on_the_network(studio, tmp_path, monkeypatch):
    assert studio.get("/api/ping").json()["network"] is False
    net = make_client(tmp_path / "n", monkeypatch, network=True)
    assert net.get("/api/ping").json()["network"] is True


def test_show_in_folder_for_a_book(studio, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda cmd, *a, **k: calls.append(cmd))
    b = upload(studio)
    assert studio.post(f"/api/books/{b['book_id']}/reveal").status_code == 404
    run(studio, b["book_id"], tmp_path / "out")
    assert studio.post(f"/api/books/{b['book_id']}/reveal").status_code == 200 and calls
    net = make_client(tmp_path / "n", monkeypatch, network=True)
    nb = upload(net)
    run(net, nb["book_id"], tmp_path / "n" / "out")
    r = net.post(f"/api/books/{nb['book_id']}/reveal")
    assert r.status_code == 400 and "another machine" in r.json()["error"]
