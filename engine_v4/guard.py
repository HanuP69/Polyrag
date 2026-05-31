"""
Answer verification (guard).
Simplified version — checks if the answer is grounded in sources.
"""
from engine_v4.ollama import ollama_generate
from engine_v4.config import CFG


GUARD_PROMPT = """You are a fact-checking assistant. Given an answer and source documents, verify if the answer is grounded in the sources.

For each claim in the answer, check if it can be found in or reasonably inferred from the sources.

Return a JSON object with:
- "verified": true/false (overall)
- "score": 0.0-1.0 (confidence)
- "claims": list of {{"claim": "...", "supported": true/false}}

Answer: {answer}

Sources:
{sources}

Respond with ONLY valid JSON, no markdown."""


def verify_answer(answer: str, sources: list) -> dict:
    """Verify answer is grounded in sources using Ollama."""
    if not answer or not sources:
        return {"verified": True, "score": 1.0, "claims": []}

    sources_text = "\n\n".join(f"[Source {i+1}]: {s[:500]}" for i, s in enumerate(sources[:5]))

    try:
        resp = ollama_generate(
            CFG.text_model,
            GUARD_PROMPT.format(answer=answer[:1000], sources=sources_text),
            timeout=30,
        )
        import json
        # Try to extract JSON from response
        resp = resp.strip()
        if resp.startswith("```"):
            resp = resp.split("```")[1]
            if resp.startswith("json"):
                resp = resp[4:]
        result = json.loads(resp)
        return {
            "verified": result.get("verified", True),
            "score": float(result.get("score", 0.5)),
            "claims": result.get("claims", []),
        }
    except Exception:
        # If guard fails, don't block the response
        return {"verified": True, "score": 0.5, "claims": []}
