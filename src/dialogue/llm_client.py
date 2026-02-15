"""Simple Ollama chat wrapper used by the support assistant.

Provides:
- generate_bot_reply(model, system_prompt, history, user_message) -> str

Behavior:
- Builds a messages list from optional system prompt, history and the new user
  message.
- Calls `ollama.chat(model=model, messages=messages)` and extracts the
  assistant content. On error, returns a friendly fallback string.

Note: this module intentionally keeps the wrapper small and testable — no
streaming or advanced error-retry logic here.
"""
from __future__ import annotations

from typing import Any

import ollama


def generate_bot_reply(model: str, system_prompt: str, history: list[dict], user_message: str) -> str:
    """Generate a single assistant reply via the local Ollama API.

    Args:
        model: Ollama model name (e.g. "llama2" or similar).
        system_prompt: Optional system-level prompt (empty string to omit).
        history: List of {"role": "user"|"assistant", "content": str}.
        user_message: The new user message to append.

    Returns:
        Assistant reply as plain string. On failure returns a short fallback
        message.
    """
    # Build messages for Ollama
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Append history (assume caller passes valid role/content dicts)
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if not role or not content:
            # skip malformed history entries rather than failing hard
            continue
        messages.append({"role": role, "content": content})

    # Add the current user message at the end
    messages.append({"role": "user", "content": user_message})

    try:
        response: Any = ollama.chat(model=model, messages=messages)
        # Return only the assistant text from the expected response shape.
        return response["message"]["content"]

    except Exception as e:
        # Include exception message in the fallback for easier debugging.
        return f"Sorry, I had an issue generating a response: {e}"


__all__ = ["generate_bot_reply"]
