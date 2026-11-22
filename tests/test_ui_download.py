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
