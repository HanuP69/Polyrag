"""engine/eval.py

Utilities to evaluate RAG outputs for faithfulness using the existing
`engine.guard.verify_answer` NLI-based verifier.
"""
from typing import List, Dict, Iterable, Any
import json

def compute_faithfulness(answer: str, sources: List[str]) -> Dict[str, Any]:
    """Return per-answer verification details by calling `engine.guard.verify_answer`.

    Import the verifier lazily to avoid loading heavy NLI libraries at module import time.
    """
    try:
        from engine.guard import verify_answer as _verify
    except Exception as e:
        raise RuntimeError("NLI verifier could not be imported") from e
    return _verify(answer, sources)


def evaluate_records(records: Iterable[Dict], answer_key: str = "answer", sources_key: str = "sources") -> Dict[str, Any]:
    """Evaluate an iterable of records and return an aggregated summary.

    Each record is expected to contain an `answer` (string) and `sources`
    (list of strings) under the given keys. Records with `sources` as a
    single string are accepted and coerced to a list.
    """
    results = []
    total_claims = 0
    total_verified_claims = 0
    verified_examples = 0
    score_sum = 0.0

    for rec in records:
        answer = rec.get(answer_key, "")
        sources = rec.get(sources_key, [])
        if isinstance(sources, str):
            sources = [sources]

        res = compute_faithfulness(answer, sources)
        results.append(res)

        total_claims += int(res.get("total_claims", 0))
        total_verified_claims += int(res.get("verified_count", 0))
        score_sum += float(res.get("score", 0.0))
        if res.get("verified"):
            verified_examples += 1

    n = len(results)
    percent_verified = (verified_examples / n * 100.0) if n else 0.0
    avg_score = (score_sum / n) if n else 0.0
    claim_verification_rate = (total_verified_claims / total_claims * 100.0) if total_claims else 100.0

    return {
        "n_examples": n,
        "percent_verified": round(percent_verified, 2),
        "avg_score": round(avg_score, 3),
        "claim_verification_rate": round(claim_verification_rate, 2),
        "total_claims": total_claims,
        "total_verified_claims": total_verified_claims,
        "details": results,
    }


def load_records(path: str) -> Iterable[Dict]:
    """Load records from a JSON or JSONL file.

    Supported formats:
      - JSONL: one JSON object per line
      - JSON list: top-level list of objects
      - JSON dict containing a single list-of-dicts value (first such list)
    """
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, "r", encoding="utf8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                for rec in obj:
                    yield rec
                return
            if isinstance(obj, dict):
                # Find first list-of-dicts in the values
                for v in obj.values():
                    if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                        for rec in v:
                            yield rec
                        return
            raise ValueError("Unsupported JSON format for evaluation. Use JSON list or JSONL of records.")
