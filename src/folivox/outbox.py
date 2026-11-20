"""Chapters as they finish: a local patch carried by the homelab repository, not upstream Folivox.

With ``FOLIVOX_CHAPTER_OUTBOX`` set, a render writes each chapter as soon as its last fragment is done, as
``<outbox>/<book stem>/NNN - <chapter title>.m4a`` (AAC, tagged with the chapter and the book), and rewrites
``book.json`` beside it: ``{"title", "author", "series", "chapter_count", "chapters": [{"n", "title", "file",
"duration"}], "done"}``, ``done`` true once the whole book is written. A delivery service uploads chapters as they
appear. Every file is written under a hidden name in the same folder and renamed, so a reader never sees half of one.
Unset, or for a preview, nothing is written. Writing to the outbox never stops a render: a chapter that cannot be
written (ffmpeg fails, the disk is full) is a warning, and book.json goes on without it (a delivery service then sees
fewer chapters than chapter_count once the book is done), while the m4b is still made. A changed chapter that cannot
be written again takes its earlier file out too, so the old recording is never delivered as the new one.

Chapters are numbered by the render's own plan: only chapters that read something count (a heading-only section is
skipped, as in the m4b), and a render of some chapters numbers those 1..N. So re-rendering a subset of a book before the
earlier render was delivered numbers its files afresh in the same folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from folivox.audio.assemble import COVER_ARGS, BookMeta, ChapterAudio, grouping, run, safe_name
from folivox.pipeline import COMMENT_LIMIT

ENV = "FOLIVOX_CHAPTER_OUTBOX"


class Outbox:
    def __init__(self, folder: Path, journal, meta: BookMeta, cover: bytes | None, bitrate: str, events):
        self.folder, self.journal, self.meta, self.cover, self.bitrate = folder, journal, meta, cover, bitrate
        self.events = events
        # journal chapter index -> file number, 1..N without gaps: a chapter with no fragments makes no audio
        heard = [ci for ci, _ in journal.chapters() if journal.chunks(ci)]
        self.numbers = {ci: n for n, ci in enumerate(heard, 1)}
        self.count = len(heard)
        self.chapters: dict[int, dict] = {}
        try:
            old = json.loads((folder / "book.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for c in old.get("chapters", []):
            if not (folder / c.get("file", "")).is_file():
                continue
            if c.get("n", 0) <= self.count:
                self.chapters[c["n"]] = c
            else:  # the book has fewer chapters now
                (folder / c["file"]).unlink(missing_ok=True)
        try:
            self._write(done=False)  # a book rendered again is not done until this render ends
        except OSError as e:
            self._warn("The chapter outbox's book.json could not be marked unfinished", e)

    @classmethod
    def open(cls, journal, opts, details, events) -> Outbox | None:
        """The render's outbox (after the plan, which sets the chapters), or None when chapters are not written.
        ``details()`` gives the book's (BookMeta, cover), read only when chapters are written; the render's ``events``
        take its warnings."""
        root = os.environ.get(ENV, "").strip()
        if not root or opts.preview:
            return None
        return cls(Path(root).expanduser() / opts.output.stem, journal, *details(), opts.bitrate, events)

    def _warn(self, what: str, error: Exception) -> None:
        self.events.warning(f"{what}: {' '.join(str(error).split())[:500]}")

    def publish(self, ci: int, chapter: ChapterAudio, sig: str) -> None:
        """Journal chapter ``ci`` into the outbox; an unchanged chapter that is already there is left alone."""
        n = self.numbers[ci]
        name = f"{n:03d} - {safe_name(chapter.title) or f'Chapter {n}'}.m4a"
        cover = hashlib.sha1(self.cover).hexdigest() if self.cover else ""
        same = json.dumps([sig, name, self.count, asdict(self.meta), self.bitrate, cover])
        key = hashlib.sha1(same.encode()).hexdigest()
        old = self.chapters.get(n)
        if old and old["file"] == name and self.journal.meta(f"outbox_{n}") == key:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._encode(n, chapter, self.folder / name)
        except (OSError, RuntimeError) as e:  # ffmpeg failed, or the disk is full: the render goes on without it
            if old:  # the file there is the chapter before this render changed it: not to be delivered as this one
                self._withdraw(n)
            self._warn(f"Chapter {n} ({chapter.title}) was not written for delivery; the audiobook still is", e)
            return
        if old and old["file"] != name:
            (self.folder / old["file"]).unlink(missing_ok=True)
        self.chapters[n] = {"n": n, "title": chapter.title, "file": name, "duration": round(chapter.duration, 3)}
        self.journal.set_meta(**{f"outbox_{n}": key})
        try:
            self._write(done=False)
        except OSError as e:  # listed by the next write that succeeds
            self._warn(f"The chapter outbox's book.json could not list chapter {n}", e)

    def _withdraw(self, n: int) -> None:
        """Chapter ``n`` out of delivery: book.json stops listing it (by rename), then its file goes."""
        gone = self.chapters.pop(n)
        self.journal.set_meta(**{f"outbox_{n}": ""})
        try:
            self._write(done=False)
            (self.folder / gone["file"]).unlink(missing_ok=True)
        except OSError as e:  # still listed: the next write that succeeds drops it, and finish() removes the file
            self._warn(f"The chapter outbox could not take out the old chapter {n}", e)

    def finish(self, built: set[int]) -> None:
        """The book is written: only journal chapters ``built`` remain listed, and book.json says done."""
        try:
            for n in set(self.chapters) - {self.numbers[ci] for ci in built}:
                (self.folder / self.chapters.pop(n)["file"]).unlink(missing_ok=True)
            self._write(done=True)
        except OSError as e:
            self._warn("The chapter outbox's book.json could not be marked done; the book is delivered once it is", e)

    def _encode(self, n: int, chapter: ChapterAudio, out: Path) -> None:
        m = self.meta
        tags = {"title": chapter.title, "album": m.title, "artist": m.author, "album_artist": m.author,
                "track": f"{n}/{self.count}", "genre": m.genre or "Audiobook", "composer": m.narrator, "date": m.year,
                "comment": m.comment[:COMMENT_LIMIT], "grouping": grouping(m)}
        with tempfile.TemporaryDirectory() as tmp:
            args, maps = ["-i", str(chapter.path)], ["-map", "0:a"]
            if self.cover:
                cov = Path(tmp) / "cover.img"
                cov.write_bytes(self.cover)
                args += ["-i", str(cov)]
                maps += ["-map", "1:v", *COVER_ARGS]
            meta = [a for k, v in tags.items() if v for a in ("-metadata", f"{k}={v}")]
            part = out.with_name(f".{out.name}.part")
            try:
                run([*args, *maps, "-c:a", "aac", "-b:a", self.bitrate, "-ac", "1", *meta, "-movflags", "+faststart",
                     "-f", "mp4", str(part)])
                if not part.exists() or part.stat().st_size == 0:
                    raise RuntimeError("ffmpeg produced an empty file")
                os.replace(part, out)
            finally:
                part.unlink(missing_ok=True)

    def _write(self, done: bool) -> None:
        book = {"title": self.meta.title, "author": self.meta.author, "series": self.meta.series,
                "chapter_count": self.count, "chapters": [self.chapters[n] for n in sorted(self.chapters)],
                "done": done}
        self.folder.mkdir(parents=True, exist_ok=True)
        part = self.folder / ".book.json.part"
        part.write_text(json.dumps(book, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(part, self.folder / "book.json")
