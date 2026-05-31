"""
Ollama utilities — load/unload/generate/stream.
Direct port from v4 notebook with added SSE streaming for the server.
"""
import time
import requests
from typing import List, Optional, Generator
from engine_v4.config import CFG


def ollama_loaded_models() -> list:
    """Return model names currently loaded in Ollama VRAM."""
    try:
        r = requests.get(f"{CFG.ollama_base}/api/ps", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_unload_all():
    """Unload all Ollama models from VRAM (keep_alive=0)."""
    for model in ollama_loaded_models():
        try:
            requests.post(
                f"{CFG.ollama_base}/api/generate",
                json={"model": model, "keep_alive": 0, "prompt": ""},
                timeout=10,
            )
            print(f"  [Ollama] Unloaded: {model}")
        except Exception:
            pass
    time.sleep(1)


def ollama_generate(
    model: str,
    prompt: str,
    images_b64: Optional[List[str]] = None,
    timeout: int = 90,
) -> str:
    """Single-shot Ollama generate. Returns response string."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if images_b64:
        payload["images"] = images_b64
    try:
        r = requests.post(
            f"{CFG.ollama_base}/api/generate", json=payload, timeout=timeout
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[ollama error: {e}]"


def ollama_chat_stream(
    model: str,
    prompt: str,
    system_prompt: str = "",
    chat_history: Optional[list] = None,
) -> Generator[str, None, None]:
    """
    Streaming Ollama chat. Yields token strings as they arrive.
    Used for SSE /generate/stream endpoint.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in (chat_history or []):
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })
    messages.append({"role": "user", "content": prompt})

    try:
        r = requests.post(
            f"{CFG.ollama_base}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        )
        r.raise_for_status()
        import json
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done", False):
                    break
    except Exception as e:
        yield f"[ollama error: {e}]"


def ollama_chat(
    model: str,
    prompt: str,
    system_prompt: str = "",
    chat_history: Optional[list] = None,
) -> str:
    """Non-streaming Ollama chat. Returns full response string."""
    tokens = list(ollama_chat_stream(model, prompt, system_prompt, chat_history))
    return "".join(tokens)
