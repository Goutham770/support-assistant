import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"


def generate_with_ollama(system_prompt: str, history: list[dict], user_message: str) -> str:
    """Call the local Ollama /api/chat endpoint and return assistant text.

    Non-streaming, simple JSON request/response.
    """
    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Expect history items already shaped as {"role": ..., "content": ...}
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


__all__ = ["generate_with_ollama"]
