from experts.base import Chunk


def rrf_fuse(
    expert_results: dict[str, list[Chunk]],
    weights: dict[str, float],
    k: int = 60,
    top_n: int = 10
) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for expert_id, chunks in expert_results.items():
        base_expert = expert_id.split("_")[0]
        weight = weights.get(base_expert, 1.0)

        for rank, chunk in enumerate(chunks):
            cid = chunk.chunk_id
            if cid not in scores:
                scores[cid] = 0.0
                chunk_map[cid] = chunk
            scores[cid] += weight / (k + rank + 1)

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    results = []
    for cid in ranked_ids[:top_n]:
        chunk = chunk_map[cid]
        chunk.metadata["rrf_score"] = scores[cid]
        results.append(chunk)

    return results