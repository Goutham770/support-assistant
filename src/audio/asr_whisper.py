"""Whisper-based ASR helper.

This module provides a simple wrapper around the OpenAI Whisper
transcription API. It is intentionally small: callers provide a file
path and receive back the transcribed text.

Note: The OpenAI Python client must be installed and configured with
credentials in the environment for this to work (OPENAI_API_KEY or
equivalent). This function will raise whatever errors the client raises
on network/auth problems; callers should handle them as needed.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI


def transcribe_file(path: str) -> str:
    """Transcribe an audio file using OpenAI Whisper.

    Args:
        path: Path to a local audio file (WAV, MP3, etc.).

    Returns:
        The transcription as a plain string.
    """
    client = OpenAI()

    with open(path, "rb") as audio_file:
        resp: Any = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    # The SDK may return a simple string or an object containing the text.
    # Prefer common fields, fall back to string conversion.
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "text"):
        return getattr(resp, "text")
    if isinstance(resp, dict) and "text" in resp:
        return resp["text"]

    return str(resp)


__all__ = ["transcribe_file"]
