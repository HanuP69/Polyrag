"""
fuse.py — Reciprocal Rank Fusion (RRF).

Pure math, no ML. Merges results from multiple experts into
a single ranked list using RRF scoring.
"""

from engine.experts.base import Chunk


def rrf_fuse(
    expert_results: dict[str, list[Chunk]],
    weights: dict[str, float],
    k: int = 60,
    top_n: int = 10
) -> list[Chunk]:
    """
    Reciprocal Rank Fusion.
    
    Args:
        expert_results: {expert_id: [chunks in rank order]}
        weights: {expert_id: gate_weight}  — from the gate classifier
        k: RRF constant (default 60 per original paper)
        top_n: number of results to return
    
    Returns:
        Fused list of chunks, sorted by RRF score descending.
    
    Formula per chunk:
        score(chunk) = Σ_expert  (weight_expert / (k + rank + 1))
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}
    
    for expert_id, chunks in expert_results.items():
        weight = weights.get(expert_id, 1.0)
        
        for rank, chunk in enumerate(chunks):
            cid = chunk.chunk_id
            
            if cid not in scores:
                scores[cid] = 0.0
                chunk_map[cid] = chunk
            
            scores[cid] += weight / (k + rank + 1)
    
    # Sort by RRF score descending
    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    
    # Attach RRF score to metadata
    results = []
    for cid in ranked_ids[:top_n]:
        chunk = chunk_map[cid]
        chunk.metadata["rrf_score"] = scores[cid]
        results.append(chunk)
    
    return results
