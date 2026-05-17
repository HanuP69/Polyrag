from sentence_transformers import CrossEncoder
import numpy as np
from experts.base import Chunk

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"[Rerank] Loading cross-encoder: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        print("[Rerank] [OK] Reranker loaded")
    return _reranker


def rerank(query: str, chunks: list[Chunk], top_n: int = 12) -> list[Chunk]:
    if not chunks:
        return []

    model = get_reranker()

    pairs = [(query, c.content[:512]) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    result = [c for c, _ in scored[:top_n]]
    # Log top scores for debugging
    top3 = [(f"p{c.metadata.get('page','?')}", f"{s:.2f}") for c, s in scored[:3]]
    print(f"[Rerank] {len(chunks)} -> {len(result)} | top3: {top3}")
    return result
