from FlagEmbedding import FlagReranker
import numpy as np
from engine.experts.base import Chunk
from typing import Optional

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"[Rerank] Loading FlagReranker: {RERANKER_MODEL} on CPU")
        # Ensure we run on CPU to avoid VRAM collision
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False, device="cpu")
        print("[Rerank] [OK] Reranker loaded")
    return _reranker


def rerank(query: str, chunks: list[Chunk], top_n: int = 12, gate_weights: Optional[dict[str, float]] = None) -> list[Chunk]:
    if not chunks:
        return []

    model = get_reranker()

    # Dedup input chunks to prevent duplicate reranking
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique_chunks.append(c)

    # Use chunk text/content; bge-reranker-v2-m3 takes pair list [[query, text], ...]
    # Truncate content to 512 chars to prevent CPU OOM/slowness
    pairs = [[query, getattr(c, "content", getattr(c, "text", ""))[:512]] for c in unique_chunks]
    scores = model.compute_score(pairs, normalize=True)

    # Ensure scores is a list/numpy array
    if not isinstance(scores, list) and not isinstance(scores, np.ndarray):
        scores = list(scores) if hasattr(scores, '__iter__') else [scores]

    scored = []
    for chunk, score in zip(unique_chunks, scores):
        adjusted_score = float(score)
        # Apply gate weights if provided, to boost specific expert modalities
        if gate_weights:
            # Fallback to a baseline probability of 0.15 if the expert was not predicted active,
            # ensuring a baseline boost (+1.275 for table, +1.5 for image) to counteract reranker bias.
            prob = gate_weights.get(chunk.expert_id, 0.15)
            if chunk.expert_id == "text":
                adjusted_score += prob * 1.5
            elif chunk.expert_id == "table":
                adjusted_score += prob * 8.5
            elif chunk.expert_id == "image":
                adjusted_score += prob * 10.0
            elif chunk.expert_id == "code":
                adjusted_score += prob * 7.5
        scored.append((chunk, adjusted_score))

    scored.sort(key=lambda x: x[1], reverse=True)

    result = [c for c, _ in scored[:top_n]]
    
    # Log top scores for debugging
    top3 = [(f"p{c.metadata.get('page','?') or c.metadata.get('page_no','?')}({c.expert_id})", f"{s:.2f}") for c, s in scored[:3]]
    print(f"[Rerank] {len(chunks)} -> {len(result)} | top3: {top3}")
    return result
