from sentence_transformers import CrossEncoder
import numpy as np

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_nli_model = None
_label_map = None   # maps semantic role -> column index: "entailment" -> int


def _get_nli():
    global _nli_model, _label_map
    if _nli_model is None:
        print(f"[Guard] Loading NLI model: {NLI_MODEL}")
        _nli_model = CrossEncoder(NLI_MODEL, max_length=512, device="cpu")

        # Resolve label order from model config automatically.
        # cross-encoder/nli-deberta-v3-small: {0: contradiction, 1: entailment, 2: neutral}
        id2label = getattr(_nli_model.config, "id2label", {})
        if id2label:
            _label_map = {v.lower(): int(k) for k, v in id2label.items()}
            print(f"[Guard] Label map resolved from config: {_label_map}")
        else:
            # Known fallback for this model family
            _label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
            print("[Guard] Label map defaulted:", _label_map)

        print("[Guard] [OK] NLI model loaded")
    return _nli_model, _label_map


import re


def _extract_claims(answer: str) -> list[str]:
    """Extract verifiable factual sentences from the answer."""
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

    text = ". ".join(cleaned_lines)
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    boilerplate_patterns = [
        r"^\s*sure", r"here\s+(?:is|are)", r"let\s+me\s+know",
        r"hope\s+this\s+helps", r"based\s+on\s+my", r"according\s+to\s+the",
        r"you\s+asked", r"i\s+found", r"as\s+requested", r"feel\s+free",
        r"is\s+there\s+anything", r"thank\s+you", r"hello", r"greetings",
        r"welcome", r"let's\s+(?:explore|look|dive|start)",
        r"below\s+(?:is|are)", r"following\s+(?:is|are|code)",
        r"i\s+can\s+help", r"i\s+will\s+explain", r"here's\s+a",
    ]
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in boilerplate_patterns]

    claims = []
    for s in raw_sentences:
        s = s.strip().rstrip(".").rstrip()
        if len(s) < 25:
            continue
        if s.startswith(("- ", "* ", "#", "[", "Source")):
            continue
        if any(p.search(s) for p in compiled_patterns):
            continue
        claims.append(s)

    return claims


def _build_source_text(sources: list[str]) -> str:
    """Compact source context - up to 5 sources, 800 chars each."""
    return " ".join(s[:800].strip() for s in sources[:5])


def verify_answer(answer: str, sources: list[str]) -> dict:
    if not answer or not sources:
        return {"verified": False, "claims": [], "score": 0.0}

    model, label_map = _get_nli()
    claims = _extract_claims(answer)

    if not claims:
        return {"verified": True, "claims": [], "score": 1.0}

    source_text = _build_source_text(sources)
    claims = claims[:10]

    # -----------------------------------------------------------------------
    # CRITICAL FIX: NLI premise/hypothesis ordering.
    # Question: "Does the SOURCE TEXT entail the CLAIM?"
    # Correct order: pair = (premise=source_text, hypothesis=claim)
    # The original code had (claim, source_text) which inverts the semantics
    # and causes systematic near-random verification scores.
    # -----------------------------------------------------------------------
    pairs = [(source_text, claim) for claim in claims]
    all_scores = model.predict(pairs)

    ent_idx = label_map.get("entailment", 1)
    con_idx = label_map.get("contradiction", 0)

    results = []
    for i, claim in enumerate(claims):
        if all_scores.ndim == 2:
            raw = all_scores[i]
        elif all_scores.ndim == 1 and len(all_scores) == 3:
            raw = all_scores
        else:
            raw = all_scores[i] if i < len(all_scores) else np.zeros(3)

        # Stable softmax
        exp_s = np.exp(raw - np.max(raw))
        probs = exp_s / (np.sum(exp_s) + 1e-9)

        entailment    = float(probs[ent_idx])
        contradiction = float(probs[con_idx])

        # Grounded if entailment probability is above threshold and
        # contradiction is not dominant.
        grounded = entailment > 0.30 and contradiction < 0.65

        results.append({
            "claim": claim[:200],
            "grounded": grounded,
            "confidence": round(entailment, 3),
        })

    verified_count = sum(1 for r in results if r["grounded"])
    total = len(results)
    avg_score = sum(r["confidence"] for r in results) / total if total > 0 else 0.0

    # 80% of claims must be grounded (was 85%, loosened to 80% for paraphrase gap)
    is_verified = (verified_count / total) >= 0.80 if total > 0 else True

    print(
        f"[Guard] {verified_count}/{total} claims verified "
        f"(avg confidence: {avg_score:.3f}, verified: {is_verified})"
    )

    return {
        "verified": is_verified,
        "claims": results,
        "score": round(avg_score, 3),
        "verified_count": verified_count,
        "total_claims": total,
    }