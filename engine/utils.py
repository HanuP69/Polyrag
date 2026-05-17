from __future__ import annotations


def resolve_model(model_key: str | None = None) -> tuple[str, str]:
    from engine.config import MODEL_REGISTRY, TESTING, OLLAMA_MODEL, GROQ_MODEL

    if model_key and model_key in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_key]
        return entry["provider"], entry["api_name"]
    if model_key:
        if "groq" in model_key.lower() or "gemma2" in model_key.lower():
            return "groq", model_key
        return "ollama", model_key
    return ("ollama", OLLAMA_MODEL) if TESTING else ("groq", GROQ_MODEL)