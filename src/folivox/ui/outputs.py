"""A book's finished audio: what a render produced, recorded so the studio can offer it again later, and found again
when something else moved it (an uploader that files finished books away, named in FOLIVOX_MOVED_OUTPUTS)."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

MOVED_ENV = "FOLIVOX_MOVED_OUTPUTS"
_SAME_MTIME = 0.01  # seconds: a rename keeps the mtime; a copy that keeps it (cp -p) matches too


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


def moved_dirs(env: Mapping[str, str] = os.environ) -> list[Path]:
    return [Path(p.strip()) for p in env.get(MOVED_ENV, "").split(",") if p.strip()]


def locate(rec: dict, moved: Sequence[Path] = ()) -> Path | None:
    """The recorded output where it was written, unchanged (same size and mtime); else a file directly in one of
    ``moved`` with the same size and mtime (newest first), whatever its name now. A folder of mp3s is only ever found
    where it was written."""
    path = Path(rec["path"])
    try:
        if path.exists():
            size, mtime = _size_mtime(path)
            if size == rec["bytes"] and abs(mtime - rec["mtime"]) <= _SAME_MTIME:
                return path
    except OSError:
        pass
    if rec.get("dir"):
        return None
    found = []
    for folder in moved:
        try:
            for p in folder.iterdir():
                st = p.stat()
                if p.is_file() and st.st_size == rec["bytes"] and abs(st.st_mtime - rec["mtime"]) <= _SAME_MTIME:
                    found.append((st.st_mtime, p))
        except OSError:
            continue
    return max(found)[1] if found else None


def summary(rec: dict | None, found: Path | None) -> dict | None:
    """What the page shows: the name the render wrote it under (a mover may have renamed it), not the server path."""
    if rec is None or found is None:
        return None
    return {"name": Path(rec["path"]).name + (".zip" if found.is_dir() else ""), "bytes": rec["bytes"],
            "seconds": rec["seconds"], "chapters": rec["chapters"], "at": rec["at"], "zip": found.is_dir()}
