"""A book's finished audio: what a render produced, recorded so the studio can offer it again later."""
from __future__ import annotations

from pathlib import Path


def _mp3s(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.mp3") if p.is_file())


def _size_mtime(path: Path) -> tuple[int, float]:
    if path.is_dir():
        files = _mp3s(path)
        return sum(p.stat().st_size for p in files), max((p.stat().st_mtime for p in files), default=0.0)
    st = path.stat()
    return st.st_size, st.st_mtime


def record(path: Path, seconds: float | None, chapters: int | None, now: float) -> dict:
    """What a finished render wrote: one m4b, or a folder of mp3s (its bytes and newest mtime are its mp3s')."""
    size, mtime = _size_mtime(path)
    return {"path": str(path), "dir": path.is_dir(), "bytes": size, "mtime": mtime, "seconds": seconds,
            "chapters": chapters, "at": now}


def locate(rec: dict) -> Path | None:
    """The recorded output where it was written, unchanged in size; None when it is gone or was replaced."""
    path = Path(rec["path"])
    try:
        if path.exists() and _size_mtime(path)[0] == rec["bytes"]:
            return path
    except OSError:
        pass
    return None


def summary(rec: dict | None, found: Path | None) -> dict | None:
    """What the page shows: the file's name as found, never the server's path."""
    if rec is None or found is None:
        return None
    return {"name": found.name + (".zip" if found.is_dir() else ""), "bytes": rec["bytes"], "seconds": rec["seconds"],
            "chapters": rec["chapters"], "at": rec["at"], "zip": found.is_dir()}
