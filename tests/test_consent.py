"""Consent guard: a voice whose name ends in "(public)" is a public figure's, and renders only with CosyVoice 3.

A local patch carried by the homelab repository (ansible/files/am5/folivox/patches), not upstream Folivox.
"""

import shutil
import types
import wave

import numpy as np
import pytest

from folivox import config, pipeline
from folivox import markup as mk
from folivox import voices as catalog
from folivox.book import Block, BlockKind, Book, Section
from folivox.pipeline import RenderOptions, render
from folivox.providers import ProviderError
from folivox.tts import builds, consent, fish_check
from folivox.tts.base import TTSEngine
from folivox.tts.builds import Hardware

TEXT = "Anna said hello to everyone. Then the door closed behind her."


@pytest.fixture(autouse=True)
def private(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIVOX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(catalog, "user_voices_dir", lambda: tmp_path / "voices")
    config.load.cache_clear()
    yield
    config.load.cache_clear()


def write_wav(path, seconds=8.0, sr=16000):
    pcm = (3000 * np.sin(np.linspace(0, 3000, int(sr * seconds)))).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


class Tone(TTSEngine):
    """A local engine that reads every fragment as half a second of tone."""

    name = "tone"

    def synthesize_audio(self, text, voice, seed, speed):
        return (0.3 * np.sin(np.linspace(0, 1000, self.sample_rate // 2))).astype(np.float32), 0.0


def nothing_synthesised(opts, events):
    raise AssertionError("the engine was loaded: the render got past the consent guard")


def make_book(annotation=""):
    return Book(title="T", lang="en", annotation=annotation,
                sections=[Section(title="One", blocks=[Block(BlockKind.PARAGRAPH, TEXT)])])


def options(tmp_path, tts, voice_wav, **kw):
    return RenderOptions(output=tmp_path / "b.m4b", tts=tts, voice_wav=voice_wav, voice_text="Hello there.", qa=False,
                         stress="none", work_dir=tmp_path / "w", **kw)


def speaker_markup(tmp_path, book, voice_id):
    m = mk.Markup()
    m.ensure_speaker(1, "Anna").voice = voice_id
    a = TEXT.index("Anna said")
    m.assign(0, 0, a, a + len("Anna said hello to everyone."), 1)
    mk.stamp(m, book.chapters(m.depth))
    path = tmp_path / "b.markup.json"
    m.save(path)
    return path


@pytest.fixture
def local_fish(monkeypatch):
    """A Fish render without a Fish build: the build checks pass and the engine is a tone."""
    monkeypatch.setattr(fish_check, "check", lambda opts, events: None)
    monkeypatch.setattr(builds, "detect_hardware", lambda: Hardware("darwin", "arm64", 64.0, 50.0))
    monkeypatch.setattr(pipeline, "load_engine", lambda opts, events: Tone())


@pytest.mark.parametrize("name", ["Narrator X (public)", "Narrator X (Public) ", "x(PUBLIC)"])
def test_public_names(name):
    assert consent.is_public(name)


def test_family_name_is_not_public():
    assert not consent.is_public("Mum") and not consent.is_public(None)
    assert not consent.is_public("Public speaking (practice)")


@pytest.mark.parametrize("engine", ["fish-drama3", "fish-s2", "elevenlabs", "fish"])
def test_public_voice_refused(engine):
    with pytest.raises(ProviderError, match="renders only with CosyVoice 3, on this computer"):
        consent.check("X (public)", engine)


def test_public_voice_allowed_on_cosyvoice3():
    consent.check("X (public)", "cosyvoice3")


def test_family_voice_allowed_everywhere():
    for engine in ("fish-drama3", "fish-s2", "elevenlabs", "fish", "cosyvoice3"):
        consent.check("Mum", engine)
        consent.check(None, engine)


@pytest.mark.parametrize("engine", ["fish-s2", "elevenlabs"])
def test_render_refuses_public_main_voice(tmp_path, monkeypatch, engine):
    x = catalog.add("x", write_wav(tmp_path / "x.wav"), "Hello there.", name="Narrator X (public)", lang="en")
    monkeypatch.setattr(pipeline, "load_engine", nothing_synthesised)
    with pytest.raises(ProviderError, match="CosyVoice 3"):
        render(make_book(), options(tmp_path, engine, x.path, accept_cloud=True))


def test_render_refuses_public_speaker_in_multivoice_book(tmp_path, monkeypatch, local_fish):
    y = catalog.add("y", write_wav(tmp_path / "y.wav"), "Hello there.", name="Y (public)", lang="en")
    book = make_book()
    markup = speaker_markup(tmp_path, book, y.id)
    monkeypatch.setattr(pipeline, "load_engine", nothing_synthesised)
    with pytest.raises(ProviderError, match="CosyVoice 3"):
        render(book, options(tmp_path, "fish-drama3", write_wav(tmp_path / "main.wav"), markup=markup))


def test_render_refuses_public_speaker_cloned_online(tmp_path, monkeypatch):
    y = catalog.add("y", write_wav(tmp_path / "y.wav"), "Hello there.", name="Y (public)", lang="en")
    book = make_book()
    markup = speaker_markup(tmp_path, book, f"fish-clone:{y.id}")
    monkeypatch.setenv("FISH_API_KEY", "not-used")
    narrator = types.SimpleNamespace(speaker="library-voice", warnings_for=lambda tts: [])
    monkeypatch.setattr(pipeline, "_cloud_voice", lambda *a: (narrator, False))  # a Fish library voice: nothing cloned
    monkeypatch.setattr(pipeline, "cloud_speaker_voices", lambda *a: pytest.fail("a speaker's voice reached Fish"))
    with pytest.raises(ProviderError, match="Speaker Anna: .*CosyVoice 3"):
        render(book, options(tmp_path, "fish", None, markup=markup, accept_cloud=True))


def test_a_sample_given_by_another_path_is_still_its_voice(tmp_path):
    x = catalog.add("x", write_wav(tmp_path / "x.wav"), "Hello there.", name="Narrator X (public)", lang="en")
    link = tmp_path / "linked.flac"
    link.symlink_to(x.path)
    assert consent.voice_name(x.path.parent / ".." / "x" / "voice.flac") == "Narrator X (public)"
    assert consent.voice_name(link) == "Narrator X (public)"
    with pytest.raises(ProviderError, match="CosyVoice 3"):
        consent.check_main(types.SimpleNamespace(voice_wav=link, voice_ref=None, tts="fish-s2"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed to build the audiobook")
def test_render_reads_public_speaker_with_cosyvoice3(tmp_path, monkeypatch):
    y = catalog.add("y", write_wav(tmp_path / "y.wav"), "Hello there.", name="Y (public)", lang="en")
    book = make_book()
    markup = speaker_markup(tmp_path, book, y.id)
    monkeypatch.setattr(pipeline, "load_engine", lambda opts, events: Tone())
    out = render(book, options(tmp_path, "cosyvoice3", write_wav(tmp_path / "main.wav"), markup=markup))
    assert out.exists()

