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


def groq_chat_stream(
    model: str,
    prompt: str,
    chat_history: Optional[list] = None,
) -> Generator[str, None, None]:
    """Streaming Groq chat completions using requests REST API."""
    if not CFG.groq_api_key:
        yield "[Groq API key not set. Please configure it in settings.]"
        return

    messages = []
    for msg in (chat_history or []):
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })
    messages.append({"role": "user", "content": prompt})

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CFG.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    try:
        r = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        import json
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    data_content = line_str[6:]
                    if data_content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_content)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        pass
    except Exception as e:
        yield f"[Groq error: {e}]"


def gemini_chat_stream(
    model: str,
    prompt: str,
    chat_history: Optional[list] = None,
) -> Generator[str, None, None]:
    """Streaming Gemini completions using Google Generative Language REST API."""
    if not CFG.gemini_api_key:
        yield "[Gemini API key not set. Please configure it in settings.]"
        return

    contents = []
    for msg in (chat_history or []):
        role = "user"
        if msg.get("role") in ["assistant", "model"]:
            role = "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    gemini_model_name = model if "gemini" in model else CFG.gemini_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:streamGenerateContent?key={CFG.gemini_api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents}

    try:
        r = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        import json
        buffer = ""
        for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                buffer += chunk
                while True:
                    start_idx = buffer.find("{")
                    if start_idx == -1:
                        buffer = ""
                        break
                    brace_count = 0
                    end_idx = -1
                    for i in range(start_idx, len(buffer)):
                        if buffer[i] == "{":
                            brace_count += 1
                        elif buffer[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i
                                break
                    if end_idx == -1:
                        break
                    json_str = buffer[start_idx:end_idx+1]
                    buffer = buffer[end_idx+1:]
                    try:
                        data = json.loads(json_str)
                        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            yield text
                    except Exception:
                        pass
    except Exception as e:
        yield f"[Gemini error: {e}]"


def llm_chat_stream(
    model: str,
    prompt: str,
    system_prompt: str = "",
    chat_history: Optional[list] = None,
) -> Generator[str, None, None]:
    """Unified LLM router: routes stream requests to Ollama, Groq, or Gemini."""
    model_lower = model.lower()
    if "groq" in model_lower or model_lower in ["llama-3.3-70b-specdec", "gemma2-9b-it", "mixtral-8x7b-32768"]:
        return groq_chat_stream(model, prompt, chat_history)
    elif "gemini" in model_lower:
        return gemini_chat_stream(model, prompt, chat_history)
    else:
        return ollama_chat_stream(model, prompt, system_prompt, chat_history)


def llm_chat(
    model: str,
    prompt: str,
    system_prompt: str = "",
    chat_history: Optional[list] = None,
) -> str:
    """Unified LLM router: returns full string response."""
    tokens = list(llm_chat_stream(model, prompt, system_prompt, chat_history))
    return "".join(tokens)
