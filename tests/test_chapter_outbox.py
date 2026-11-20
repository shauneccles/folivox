"""Chapters as they finish: with FOLIVOX_CHAPTER_OUTBOX set, a render writes each chapter as an m4a to
<outbox>/<book stem>/ as soon as its last fragment is done, and book.json beside it lists what is there.

A local patch carried by the homelab repository (ansible/files/am5/folivox/patches), not upstream Folivox.
"""

import json
import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

from folivox import config, pipeline
from folivox import voices as catalog
from folivox.book import Block, BlockKind, Book, Section
from folivox.pipeline import Cancelled, Events, RenderOptions, render
from folivox.text.chunk import ChunkOptions
from folivox.tts import builds, fish_check
from folivox.tts.base import TTSEngine
from folivox.tts.builds import Hardware

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg is needed to build the audiobook")

ONE = "Anna said hello to everyone. Then the door closed behind her."
TWO = "The second chapter is short. It ends here."
FIRST = "001 - One the start.m4a"
SECOND = "002 - Two.m4a"


@pytest.fixture(autouse=True)
def private(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.delenv("FOLIVOX_WORK_DIR", raising=False)
    monkeypatch.setattr(catalog, "user_voices_dir", lambda: tmp_path / "voices")
    config.load.cache_clear()
    yield
    config.load.cache_clear()


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    path = tmp_path / "audiobooks" / ".chapters"
    monkeypatch.setenv("FOLIVOX_CHAPTER_OUTBOX", str(path))
    return path


class Tone(TTSEngine):
    """A local engine that reads every fragment as half a second of tone."""

    name = "tone"

    def synthesize_audio(self, text, voice, seed, speed):
        return (0.3 * np.sin(np.linspace(0, 1000, self.sample_rate // 2))).astype(np.float32), 0.0


@pytest.fixture(autouse=True)
def tone(monkeypatch):
    monkeypatch.setattr(pipeline, "load_engine", lambda opts, events: Tone())


@pytest.fixture
def local_fish(monkeypatch):
    """A Fish render without a Fish build: the build checks pass and the engine is a tone."""
    monkeypatch.setattr(fish_check, "check", lambda opts, events: None)
    monkeypatch.setattr(builds, "detect_hardware", lambda: Hardware("darwin", "arm64", 64.0, 50.0))


def write_wav(path, seconds=8.0, sr=16000):
    pcm = (3000 * np.sin(np.linspace(0, 3000, int(sr * seconds)))).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def make_book(*first):
    paragraphs = first or (ONE,)
    return Book(title="The Book", authors=["Ann Writer"], lang="en", annotation="A short book.",
                sections=[Section(title="One: the start", blocks=[Block(BlockKind.PARAGRAPH, p) for p in paragraphs]),
                          Section(title="Two", blocks=[Block(BlockKind.PARAGRAPH, TWO)])])


def options(tmp_path, tts="cosyvoice3", **kw):
    return RenderOptions(output=tmp_path / "audiobooks" / "Ann Writer - The Book.m4b", tts=tts, qa=False,
                         voice_wav=write_wav(tmp_path / "main.wav"), voice_text="Hello there.", stress="none",
                         work_dir=tmp_path / "work", meta={"series": "Tales", "series_part": "2"}, **kw)


def book_json(folder):
    return json.loads((folder / "book.json").read_text(encoding="utf-8"))


def tags(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    fmt = json.loads(out)["format"]
    return {k.lower(): v for k, v in fmt.get("tags", {}).items()} | {"duration": float(fmt["duration"])}


class Watch(Events):
    """Notes what the outbox holds when the first fragment of chapter 2 starts; with ``stop``, stops the render
    (as the studio's Stop does) once chapter 1 is in the outbox."""

    def __init__(self, folder, stop=False):
        self.folder, self.stop, self.seen = folder, stop, None

    def chunk_started(self, chunk):
        if chunk.chapter == 1 and self.seen is None:
            self.seen = (book_json(self.folder), sorted(os.listdir(self.folder)))

    def cancelled(self):
        return self.stop and (self.folder / FIRST).exists()


def test_chapter_one_is_written_before_chapter_two_starts(tmp_path, outbox):
    folder = outbox / "Ann Writer - The Book"
    watch = Watch(folder)
    out = render(make_book(), options(tmp_path), watch)

    early, files = watch.seen
    assert files == [FIRST, "book.json"]
    assert early["done"] is False and early["chapter_count"] == 2
    assert [(c["n"], c["title"], c["file"]) for c in early["chapters"]] == [(1, "One: the start", FIRST)]

    assert out.exists()  # the whole book is still written as before
    done = book_json(folder)
    assert {k: done[k] for k in ("title", "author", "series", "chapter_count", "done")} == {
        "title": "The Book", "author": "Ann Writer", "series": "Tales", "chapter_count": 2, "done": True}
    assert [(c["n"], c["title"], c["file"]) for c in done["chapters"]] == [(1, "One: the start", FIRST),
                                                                           (2, "Two", SECOND)]
    assert sorted(os.listdir(folder)) == [FIRST, SECOND, "book.json"]  # no temporary file left behind
    for n, c in enumerate(done["chapters"], 1):
        t = tags(folder / c["file"])
        assert (t["title"], t["album"], t["artist"], t["album_artist"], t["track"]) == (
            c["title"], "The Book", "Ann Writer", "Ann Writer", f"{n}/2")
        assert t["comment"] == "A short book."
        assert abs(t["duration"] - c["duration"]) < 0.2


def test_fish_chapters_carry_the_attribution(tmp_path, outbox, local_fish):
    from folivox.tts.licences import ATTRIBUTION

    render(make_book(), options(tmp_path, "fish-drama3"))
    folder = outbox / "Ann Writer - The Book"
    for c in book_json(folder)["chapters"]:
        comment = tags(folder / c["file"])["comment"]
        assert comment.startswith("A short book.") and comment.endswith(ATTRIBUTION)


def test_rerendered_chapter_keeps_its_file_name(tmp_path, outbox):
    folder = outbox / "Ann Writer - The Book"
    render(make_book(), options(tmp_path))
    before = book_json(folder)["chapters"]
    unchanged = (folder / SECOND).stat().st_mtime_ns

    render(make_book(ONE, "A new paragraph was added.", "And another one, to make it longer."), options(tmp_path))
    after = book_json(folder)
    assert sorted(os.listdir(folder)) == [FIRST, SECOND, "book.json"]
    assert after["done"] is True and [c["file"] for c in after["chapters"]] == [FIRST, SECOND]
    assert after["chapters"][0]["duration"] > before[0]["duration"] + 0.5
    assert abs(tags(folder / FIRST)["duration"] - after["chapters"][0]["duration"]) < 0.2
    assert after["chapters"][1] == before[1]
    assert (folder / SECOND).stat().st_mtime_ns == unchanged  # nothing to deliver again


def test_cancelled_render_leaves_the_book_not_done(tmp_path, outbox):
    folder = outbox / "Ann Writer - The Book"
    with pytest.raises(Cancelled):
        render(make_book(), options(tmp_path), Watch(folder, stop=True))
    left = book_json(folder)
    assert left["done"] is False and [c["file"] for c in left["chapters"]] == [FIRST]
    assert sorted(os.listdir(folder)) == [FIRST, "book.json"]


def test_rendering_again_marks_a_done_book_not_done_until_it_ends(tmp_path, outbox):
    folder = outbox / "Ann Writer - The Book"
    render(make_book(), options(tmp_path))
    with pytest.raises(Cancelled):
        render(make_book(ONE, "A new paragraph."), options(tmp_path), Watch(folder, stop=True))
    assert book_json(folder)["done"] is False


def test_nothing_is_written_without_the_setting(tmp_path, monkeypatch):
    monkeypatch.delenv("FOLIVOX_CHAPTER_OUTBOX", raising=False)
    out = render(make_book(), options(tmp_path))
    assert out.exists()
    written = {p.name for p in tmp_path.rglob("*")}
    assert ".chapters" not in written and "book.json" not in written
    assert not any(name.endswith(".m4a") for name in written)


def test_a_section_with_nothing_to_read_is_not_a_chapter(tmp_path, outbox):
    book = make_book()
    book.sections.insert(1, Section(title="Part Two"))  # a heading only, and headings are not read
    render(book, options(tmp_path, chunk=ChunkOptions(speak_headings=False)))
    folder = outbox / "Ann Writer - The Book"
    done = book_json(folder)
    assert done["done"] is True and done["chapter_count"] == 2
    assert [(c["n"], c["title"], c["file"]) for c in done["chapters"]] == [(1, "One: the start", FIRST),
                                                                           (2, "Two", SECOND)]
    assert sorted(os.listdir(folder)) == [FIRST, SECOND, "book.json"]
    assert [tags(folder / name)["track"] for name in (FIRST, SECOND)] == ["1/2", "2/2"]


class Count(Events):
    def __init__(self):
        self.done = 0

    def chunk_done(self, chunk, take, attempts):
        self.done += 1


def test_without_the_setting_a_bad_cover_fails_at_assembly_as_upstream(tmp_path, monkeypatch):
    monkeypatch.delenv("FOLIVOX_CHAPTER_OUTBOX", raising=False)
    count = Count()
    with pytest.raises(FileNotFoundError):
        render(make_book(), options(tmp_path, cover=tmp_path / "gone.jpg"), count)
    assert count.done  # the whole book was read first, and its chapters built: assembly is where it stops
    assert sorted(p.name for p in (tmp_path / "work" / "chapters").iterdir()) == ["001.flac", "002.flac"]


class Warnings(Events):
    def __init__(self):
        self.seen = []

    def warning(self, message):
        self.seen.append(message)


def test_a_chapter_that_cannot_be_written_is_a_warning_and_the_render_goes_on(tmp_path, outbox, monkeypatch):
    from folivox import outbox as box

    encode = box.run

    def disk_full_for_chapter_two(args):
        if args[-1].endswith(f"/.{SECOND}.part"):
            raise RuntimeError("ffmpeg failed (the disk may be full):\nNo space left on device")
        encode(args)

    monkeypatch.setattr(box, "run", disk_full_for_chapter_two)
    events = Warnings()
    out = render(make_book(), options(tmp_path), events)
    assert out.exists()  # the m4b is still made
    folder = outbox / "Ann Writer - The Book"
    done = book_json(folder)
    assert done["done"] is True and done["chapter_count"] == 2  # abs-deliver: done but 1 of 2 chapters rendered
    assert [c["file"] for c in done["chapters"]] == [FIRST]
    assert sorted(os.listdir(folder)) == [FIRST, "book.json"]  # nor any half-written file
    assert any(w.startswith("Chapter 2 (Two) was not written for delivery") and w.endswith("No space left on device")
               for w in events.seen), events.seen


class Snapshot(Warnings):
    """Also notes what the outbox holds when each warning is given."""

    def __init__(self, folder):
        super().__init__()
        self.folder, self.held = folder, []

    def warning(self, message):
        super().warning(message)
        self.held.append((book_json(self.folder), sorted(os.listdir(self.folder))))


def test_a_retake_that_cannot_be_written_takes_the_old_chapter_out_of_delivery(tmp_path, outbox, monkeypatch):
    """The chapter changed, so the file already written is the old recording: it goes, with its book.json entry, so a
    delivery service sees the book done with 1 of 2 chapters and says so, rather than delivering the old audio."""
    from folivox import outbox as box

    folder = outbox / "Ann Writer - The Book"
    render(make_book(), options(tmp_path))
    assert [c["file"] for c in book_json(folder)["chapters"]] == [FIRST, SECOND]
    encode = box.run

    def disk_full_for_chapter_one(args):
        if args[-1].endswith(f"/.{FIRST}.part"):
            raise RuntimeError("ffmpeg failed (the disk may be full):\nNo space left on device")
        encode(args)

    monkeypatch.setattr(box, "run", disk_full_for_chapter_one)
    events = Snapshot(folder)
    out = render(make_book(ONE, "A new paragraph was added."), options(tmp_path), events)
    assert out.exists()  # the render goes on, and the m4b holds the new chapter 1
    warned = [w for w in events.seen if w.startswith("Chapter 1 (One: the start) was not written for delivery")]
    assert warned and warned[0].endswith("No space left on device"), events.seen
    # When the warning is given the old chapter 1 is gone from both book.json and the folder.
    listed, files = events.held[events.seen.index(warned[0])]
    assert [c["file"] for c in listed["chapters"]] == [SECOND] and listed["done"] is False
    assert files == [SECOND, "book.json"]
    done = book_json(folder)
    assert done["done"] is True and done["chapter_count"] == 2  # abs-deliver: done but 1 of 2 chapters rendered
    assert [c["file"] for c in done["chapters"]] == [SECOND]
    assert sorted(os.listdir(folder)) == [SECOND, "book.json"]

    monkeypatch.setattr(box, "run", encode)  # the next render writes it again
    render(make_book(ONE, "A new paragraph was added."), options(tmp_path))
    assert [c["file"] for c in book_json(folder)["chapters"]] == [FIRST, SECOND]


def test_a_preview_is_not_a_book(tmp_path, outbox):
    render(make_book(), options(tmp_path, preview=200))
    assert not outbox.exists()
