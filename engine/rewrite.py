import requests
from engine.config import (
    TESTING, OLLAMA_BASE_URL, OLLAMA_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_BASE_URL,
    MODEL_REGISTRY
)


def _resolve(model_key=None):
    if model_key and model_key in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_key]
        return entry["provider"], entry["api_name"]
    if model_key:
        if "groq" in model_key.lower() or "gemma2" in model_key.lower():
            return "groq", model_key
        if "gemini" in model_key.lower() or "gemma-3" in model_key.lower() or "gemma-4" in model_key.lower():
            return "gemini", model_key
        return "ollama", model_key
    return ("ollama", OLLAMA_MODEL) if TESTING else ("groq", GROQ_MODEL)


def rewrite_query(query: str, model: str = None, chat_history: list = None) -> str:
    # Build context from conversation history for coreference resolution
    history_context = ""
    if chat_history:
        recent = chat_history[-4:]  # last 2 pairs
        history_context = "Recent conversation:\n"
        history_context += "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in recent
        )
        history_context += "\n\n"

    prompt = (
        f"{history_context}"
        "You are an AI search assistant. Your ONLY job is to rewrite the user's latest query to make it fully self-contained based on the recent conversation.\n"
        "RULES:\n"
        "1. DO NOT answer the question.\n"
        "2. DO NOT change the core intent.\n"
        "3. DO NOT drop important keywords.\n"
        "4. DO NOT add conversational filler (like 'Here is the rewritten query:').\n"
        "5. ONLY return the rewritten query text.\n\n"
        f"Original Query: {query}"
    )


    provider, api_name = _resolve(model)

    if provider == "ollama":
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": api_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 128},
                },
                timeout=30,
            )
            response.raise_for_status()
            rewritten = response.json()["response"].strip()
            rewritten = rewritten.strip('"').strip("'")
            if len(rewritten) < 5 or len(rewritten) > 500:
                return query
            print(f'[Rewrite] ({api_name}) "{query}" -> "{rewritten}"')
            return rewritten
        except Exception as e:
            print(f"[Rewrite] Failed: {e}")
            return query

    elif provider == "gemini":
        try:
            url = f"{GEMINI_BASE_URL}/models/{api_name}:generateContent?key={GEMINI_API_KEY}"
            response = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 128},
                },
                timeout=15,
            )
            response.raise_for_status()
            rewritten = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            rewritten = rewritten.strip('"').strip("'")
            if len(rewritten) < 5 or len(rewritten) > 500:
                return query
            print(f'[Rewrite] ({api_name}) "{query}" -> "{rewritten}"')
            return rewritten
        except Exception as e:
            print(f"[Rewrite] Failed: {e}")
            return query

    else:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": api_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 128,
                },
                timeout=10,
            )
            response.raise_for_status()
            rewritten = response.json()["choices"][0]["message"]["content"].strip()
            rewritten = rewritten.strip('"').strip("'")
            print(f'[Rewrite] ({api_name}) "{query}" -> "{rewritten}"')
            return rewritten
        except Exception as e:
            print(f"[Rewrite] Failed: {e}")
            return query
