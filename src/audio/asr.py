"""ASR (automatic speech recognition) wrapper skeleton.

This module provides a minimal, importable stub for transcription so the
rest of the project can call into `transcribe_chunk` without requiring
heavy ASR dependencies during development.

Later this will be replaced with a real integration (e.g. Whisper local
model, cloud ASR API, or another on-device model). The public API is a
single function `transcribe_chunk(wav_bytes, language)` which accepts raw
WAV bytes and returns a string transcription.

The implementation below intentionally returns a fixed placeholder string.
"""

from __future__ import annotations

from typing import Optional


def transcribe_chunk(wav_bytes: bytes, language: Optional[str] = "en") -> str:
    """Transcribe a single chunk of audio.

    Args:
        wav_bytes: Audio bytes containing a WAV-encoded or raw PCM chunk.
            The exact encoding will be documented and enforced when a real
            ASR backend is wired in.
        language: Optional language hint (e.g. "en", "es"). Backends may
            use this to choose models or decoding options.

    Returns:
        A transcription string. Currently this is a placeholder value:
        "[ASR transcription placeholder]".

    Notes:
        - This function is intentionally dependency-free and safe to import
          in lightweight environments. Replace the body with a real ASR
          call when ready (for example, a Whisper wrapper, an HTTP API,
          or an on-device model invocation).
    """
    # Placeholder implementation: return a fixed string so callers can
    # rely on a consistent return type during development and testing.
    return "[ASR transcription placeholder]"


__all__ = ["transcribe_chunk"]
