"""Consent guard: a public figure's voice renders only with CosyVoice 3, on this computer.

The voice catalog has no tags, so a voice whose name ends in ``(public)`` (any case, surrounding spaces ignored) is a
public figure's. Their voice must never reach a cloud provider, and the Fish licences forbid imitating a person without
consent, so every engine other than CosyVoice 3 is refused, for the narrator and for each speaker of a multi-voice book.
"""

from __future__ import annotations

import re
from pathlib import Path

from folivox.providers import ProviderError

ENGINE = "cosyvoice3"
REFUSED = "A public figure's voice renders only with CosyVoice 3, on this computer."
_PUBLIC = re.compile(r"\(public\)\s*$", re.IGNORECASE)
_CLONE = "fish-clone:"  # a speaker cloned online from a catalog voice or a recording (pipeline.SPEAKER_VOICE_KINDS)


def is_public(voice_name: str | None) -> bool:
    return bool(voice_name) and _PUBLIC.search(voice_name) is not None


def check(voice_name: str | None, engine: str) -> None:
    if engine != ENGINE and is_public(voice_name):
        raise ProviderError(REFUSED)


def voice_name(ref: str | Path | None) -> str | None:
    """The catalog name of a voice given by its id or by the path of its sample; None when the catalog has none."""
    from folivox import voices as catalog

    if not ref or not str(ref).strip():
        return None
    ref = str(ref).strip()
    ref = ref[len(_CLONE):].strip() if ref.startswith(_CLONE) else ref
    sample = Path(ref).expanduser().resolve()  # the same file by another path (relative, a symlink) is the same voice
    entry = catalog.find(ref) or next((v for v in catalog.all_voices() if Path(v.path).resolve() == sample), None)
    return entry.name if entry is not None else None


def check_main(opts) -> None:
    """The book's own voice: the recording a local engine reads with, or the one a cloud engine clones."""
    for ref in (opts.voice_wav, opts.voice_ref):
        check(voice_name(ref), opts.tts)


def check_speakers(markup, engine: str) -> None:
    """Every speaker of a multi-voice book that reads something in a voice of their own."""
    if markup is None:
        return
    for sid in sorted(markup.used_speakers() - {0}):
        sp = markup.speaker(sid)
        if sp is None or not sp.voice:
            continue
        try:
            check(voice_name(sp.voice), engine)
        except ProviderError as e:
            raise ProviderError(f"Speaker {sp.name or sid}: {e}") from None
