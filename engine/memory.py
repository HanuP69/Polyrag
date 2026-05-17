"""
memory.py -- Conversation memory compression for PolyRAG.

Strategy: Rolling summary + last N verbatim turns.
- If history has <= VERBATIM_WINDOW messages, pass all verbatim
- If history has > VERBATIM_WINDOW messages, compress older turns into a summary
- Summary is generated via the same LLM used for rewriting (fast, cheap)
"""

import requests
from typing import Optional

VERBATIM_WINDOW = 6  # last 3 pairs (user+assistant)
MAX_SUMMARY_TOKENS = 200


def build_memory_context(
    chat_history: list[dict],
    rolling_summary: str = None
) -> tuple[str, list[dict]]:
    """
    Returns (memory_prefix, recent_turns).

    memory_prefix: compressed summary of older turns (or empty string)
    recent_turns: last VERBATIM_WINDOW messages to include verbatim
    """
    if not chat_history:
        return "", []

    if len(chat_history) <= VERBATIM_WINDOW:
        return rolling_summary or "", chat_history

    # Split: older turns → compress, recent turns → keep verbatim
    older = chat_history[:-VERBATIM_WINDOW]
    recent = chat_history[-VERBATIM_WINDOW:]

    # Build summary of older turns
    old_text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in older)

    if rolling_summary:
        old_text = f"Previous summary: {rolling_summary}\n\nNew turns:\n{old_text}"

    return old_text, recent


def compress_summary(old_context: str, model: str = None) -> str:
    """
    Use LLM to compress old conversation context into a brief summary.
    Called asynchronously after each response when history is long.
    """
    from rewrite import _resolve
    from config import (
        OLLAMA_BASE_URL, GROQ_API_KEY,
        GEMINI_API_KEY, GEMINI_BASE_URL
    )

    prompt = (
        "Summarize this conversation context in 2-3 sentences. "
        "Focus on: what document/topic is being discussed, what questions were asked, "
        "and what key facts were established. Be extremely concise.\n\n"
        f"{old_context}"
    )

    provider, api_name = _resolve(model)

    try:
        if provider == "gemini":
            url = f"{GEMINI_BASE_URL}/models/{api_name}:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": MAX_SUMMARY_TOKENS},
            }, timeout=10)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "ollama":
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                "model": api_name, "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": MAX_SUMMARY_TOKENS},
            }, timeout=15)
            resp.raise_for_status()
            return resp.json()["response"].strip()

        else:  # groq
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": MAX_SUMMARY_TOKENS,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"[Memory] Summary compression failed: {e}")
        # Fallback: truncate old context
        return old_context[:300] + "..."
