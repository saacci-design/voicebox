"""Google TTS backend using the gTTS library (cloud-based, no local model)."""

from __future__ import annotations

import asyncio
import os
import tempfile

import numpy as np

GOOGLE_TTS_SAMPLE_RATE = 24_000

# gTTS uses BCP-47 tags for some languages; map our ISO codes that differ.
GTTS_LANG_MAP: dict[str, str] = {
    "zh": "zh-CN",
}


def _get_gtts():
    try:
        from gtts import gTTS  # type: ignore[import-untyped]
        return gTTS
    except ImportError as exc:
        raise ImportError(
            "gTTS is required for the Google TTS backend. "
            "Install it with: pip install gtts>=2.5.0"
        ) from exc


def _synthesise_sync(text: str, lang: str) -> np.ndarray:
    """Run gTTS in a temp file and return a float32 array at GOOGLE_TTS_SAMPLE_RATE."""
    import librosa  # already a project dep

    gTTS = _get_gtts()
    gtts_lang = GTTS_LANG_MAP.get(lang, lang)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tts = gTTS(text=text, lang=gtts_lang)
        tts.save(tmp_path)
        audio, _ = librosa.load(tmp_path, sr=GOOGLE_TTS_SAMPLE_RATE, mono=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return audio.astype(np.float32)


class GoogleTTSBackend:
    """Thin wrapper around gTTS that matches the backend interface."""

    def __init__(self) -> None:
        self._loaded = False

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:  # noqa: D401
        return self._loaded

    def _get_model_path(self, model_size: str) -> str:  # noqa: ARG002
        return ""

    def _is_model_cached(self, model_size: str = "default") -> bool:  # noqa: ARG002
        return True

    async def load_model(self, model_size: str = "default") -> None:  # noqa: ARG002
        _get_gtts()  # raises ImportError early if missing
        self._loaded = True

    def unload_model(self) -> None:
        self._loaded = False

    # ------------------------------------------------------------------
    # Voice-prompt stubs (Google TTS is text-only; we ignore reference audio)
    # ------------------------------------------------------------------

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,  # noqa: ARG002
    ):
        return {"gtts": True}, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ):
        from .base import combine_voice_prompts as _combine_voice_prompts

        return await _combine_voice_prompts(
            audio_paths,
            reference_texts,
            sample_rate=GOOGLE_TTS_SAMPLE_RATE,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        text: str,
        voice_prompt,  # ignored — Google TTS is text-only
        language: str = "en",
        seed: int | None = None,  # noqa: ARG002
        instruct: str | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, int]:
        audio = await asyncio.to_thread(_synthesise_sync, text, language)
        return audio, GOOGLE_TTS_SAMPLE_RATE
