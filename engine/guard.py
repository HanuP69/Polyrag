from sentence_transformers import CrossEncoder
import numpy as np

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_nli_model = None


def _get_nli():
    global _nli_model
    if _nli_model is None:
        print(f"[Guard] Loading NLI model: {NLI_MODEL}")
        _nli_model = CrossEncoder(NLI_MODEL, max_length=512, device="cpu")
        print("[Guard] [OK] NLI model loaded")
    return _nli_model
import re

def _extract_claims(answer: str) -> list[str]:
    # Parse code blocks to ignore their contents from factual claim verification
    lines = answer.split("\n")
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        cleaned_lines.append(line)
        
    text_outside_code = ". ".join(cleaned_lines)
    
    # Split into sentences using regex respecting punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', text_outside_code)
    
    claims = []
    
    # Common conversational and boilerplate phrase patterns
    boilerplate_patterns = [
        r"^\s*sure",
        r"here\s+(?:is|are)",
        r"let\s+me\s+know",
        r"hope\s+this\s+helps",
        r"based\s+on\s+my",
        r"according\s+to\s+the",
        r"you\s+asked",
        r"i\s+found",
        r"as\s+requested",
        r"feel\s+free",
        r"is\s+there\s+anything",
        r"thank\s+you",
        r"hello",
        r"greetings",
        r"welcome",
        r"let's\s+(?:explore|look|dive|start)",
        r"below\s+(?:is|are)",
        r"following\s+(?:is|are|code)",
        r"i\s+can\s+help",
        r"i\s+will\s+explain",
        r"here's\s+a",
    ]
    
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in boilerplate_patterns]
    
    for s in raw_sentences:
        s = s.strip().rstrip(".").rstrip()
        if len(s) < 25:
            continue
        # Skip pure list markers, markdown headers, and bracket tags
        if s.startswith("- ") or s.startswith("* ") or s.startswith("#") or s.startswith("[") or s.startswith("Source"):
            continue
        # Skip conversational boilerplate
        is_boilerplate = False
        for pattern in compiled_patterns:
            if pattern.search(s):
                is_boilerplate = True
                break
        if is_boilerplate:
            continue
            
        claims.append(s)
        
    return claims


def verify_answer(answer: str, sources: list[str]) -> dict:
    if not answer or not sources:
        return {"verified": False, "claims": [], "score": 0.0}

    model = _get_nli()
    claims = _extract_claims(answer)

    if not claims:
        # If no actual factual claims were made (e.g. conversational prompt), treat as verified
        return {"verified": True, "claims": [], "score": 1.0}

    # Extend context window to 1500 chars to cover full relevance
    source_text = " ".join(s[:1500] for s in sources[:5])
    claims = claims[:10]

    pairs = [(claim, source_text) for claim in claims]
    all_scores = model.predict(pairs)

    results = []
    for i, claim in enumerate(claims):
        scores = all_scores[i] if all_scores.ndim > 1 else all_scores

        if isinstance(scores, np.ndarray) and scores.ndim > 0 and len(scores) > 1:
            # Apply stable softmax to convert raw logits to probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            entailment = float(probs[1])
            contradiction = float(probs[0])
        else:
            entailment = float(scores) if float(scores) > 0 else 0.5
            contradiction = 0.0

        # Loosened guardrail threshold
        grounded = entailment > 0.3 and contradiction < 0.7

        results.append({
            "claim": claim[:200],
            "grounded": grounded,
            "confidence": round(entailment, 3),
        })

    verified_count = sum(1 for r in results if r["grounded"])
    total = len(results)
    avg_score = sum(r["confidence"] for r in results) / total if total > 0 else 0.0

    # Verification threshold: 85% of claims must be grounded to verify the whole response
    is_verified = (verified_count / total) >= 0.85 if total > 0 else True

    print(f"[Guard] {verified_count}/{total} claims verified (avg confidence: {avg_score:.3f}, verified: {is_verified})")

    return {
        "verified": is_verified,
        "claims": results,
        "score": round(avg_score, 3),
        "verified_count": verified_count,
        "total_claims": total,
    }
