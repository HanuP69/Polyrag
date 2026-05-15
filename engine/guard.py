from sentence_transformers import CrossEncoder
import numpy as np

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_nli_model = None


def _get_nli():
    global _nli_model
    if _nli_model is None:
        print(f"[Guard] Loading NLI model: {NLI_MODEL}")
        _nli_model = CrossEncoder(NLI_MODEL, max_length=512)
        print("[Guard] [OK] NLI model loaded")
    return _nli_model


def _extract_claims(answer: str) -> list[str]:
    sentences = answer.replace("?\n", "? ").replace(".\n", ". ").split(". ")
    claims = []
    for s in sentences:
        s = s.strip().rstrip(".")
        if len(s) > 20 and not s.startswith("[") and not s.startswith("Source"):
            claims.append(s)
    return claims


def verify_answer(answer: str, sources: list[str]) -> dict:
    if not answer or not sources:
        return {"verified": False, "claims": [], "score": 0.0}

    model = _get_nli()
    claims = _extract_claims(answer)

    if not claims:
        return {"verified": True, "claims": [], "score": 1.0}

    source_text = " ".join(s[:500] for s in sources[:5])
    claims = claims[:10]

    pairs = [(claim, source_text) for claim in claims]
    all_scores = model.predict(pairs)

    results = []
    for i, claim in enumerate(claims):
        scores = all_scores[i] if all_scores.ndim > 1 else all_scores

        if isinstance(scores, np.ndarray) and scores.ndim > 0 and len(scores) > 1:
            entailment = float(scores[1])
            contradiction = float(scores[0])
        else:
            entailment = float(scores) if float(scores) > 0 else 0.5
            contradiction = 0.0

        grounded = entailment > 0.5 and contradiction < 0.5

        results.append({
            "claim": claim[:200],
            "grounded": grounded,
            "confidence": round(entailment, 3),
        })

    verified_count = sum(1 for r in results if r["grounded"])
    total = len(results)
    avg_score = sum(r["confidence"] for r in results) / total if total > 0 else 0.0

    print(f"[Guard] {verified_count}/{total} claims verified (avg confidence: {avg_score:.3f})")

    return {
        "verified": verified_count == total,
        "claims": results,
        "score": round(avg_score, 3),
        "verified_count": verified_count,
        "total_claims": total,
    }
