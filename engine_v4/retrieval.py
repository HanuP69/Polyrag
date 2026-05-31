"""
Retrieval pipeline — direct port from v4 notebook.
Dense (pgvector) + Sparse (BM25) + RRF fusion + cross-modal fusion.
"""
import re
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from engine_v4.config import CFG
from engine_v4.models import embedder, reranker
from engine_v4.chunker import Chunk
from engine_v4 import db


# ── BM25 in-memory cache (loaded at startup, rebuilt on ingest) ──────────────

_bm25_cache: Dict[str, dict] = {}   # org_id -> {"indexes": {...}, "chunk_ids": {...}}
_chunk_cache: Dict[str, dict] = {}   # org_id -> {"chunk_lookup": {...}, "modal_chunks": {...}}


def load_indexes(org_id: str = "default"):
    """Load BM25 indexes and chunk data from DB into memory."""
    global _bm25_cache, _chunk_cache
    _bm25_cache[org_id] = db.load_bm25(org_id)
    _chunk_cache[org_id] = db.load_chunks(org_id)
    n_chunks = sum(len(v) for v in _chunk_cache[org_id]["modal_chunks"].values())
    n_bm25 = len(_bm25_cache[org_id]["indexes"])
    print(f"[Retrieval] Loaded indexes for org={org_id}: {n_chunks} chunks, {n_bm25} BM25 indexes")


def reload_indexes(org_id: str = "default"):
    """Reload after ingestion."""
    load_indexes(org_id)


# ── Tokenizer (from v4 notebook) ────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.findall(r'[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*', text.lower())


# ── RRF Fusion (from v4 notebook) ───────────────────────────────────────────

def rrf_fuse(ranked_lists: List[List], weights: Optional[List[float]] = None,
             k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion. Returns fused indices."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, idx in enumerate(ranked):
            scores[idx] += w / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── Per-modality hybrid retrieval ───────────────────────────────────────────

def retrieve_modality_db(query: str, qvec: np.ndarray, modality: str,
                         org_id: str = "default",
                         dense_k: int = 30, sparse_k: int = 30,
                         rrf_k: int = 60) -> List[dict]:
    """
    Per-modality hybrid retrieval:
    1. Dense: pgvector cosine search
    2. Sparse: BM25 from in-memory index
    3. Fuse with RRF
    """
    # 1. Dense retrieval via pgvector
    dense_results = db.dense_search(qvec, modality, org_id, dense_k)
    dense_ids = [r["chunk_id"] for r in dense_results]
    chunk_map = {r["chunk_id"]: r for r in dense_results}

    # 2. Sparse retrieval via BM25
    bm25_data = _bm25_cache.get(org_id, {"indexes": {}, "chunk_ids": {}})
    bm25_idx = bm25_data["indexes"].get(modality)
    bm25_cids = bm25_data["chunk_ids"].get(modality, [])

    sparse_ids = []
    if bm25_idx is not None and bm25_cids:
        tokens = tokenize(query)
        if tokens:
            scores = bm25_idx.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:sparse_k]
            sparse_ids = [bm25_cids[i] for i in top_indices if i < len(bm25_cids)]

    # Build unified chunk map (sparse results may not be in dense results)
    chunk_data = _chunk_cache.get(org_id, {}).get("chunk_lookup", {})
    for cid in sparse_ids:
        if cid not in chunk_map and cid in chunk_data:
            ch = chunk_data[cid]
            chunk_map[cid] = ch.to_dict()

    # 3. RRF fusion
    all_ids = list(dict.fromkeys(dense_ids + sparse_ids))  # dedupe, preserve order
    if not all_ids:
        return []

    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    dense_ranked = [id_to_idx[cid] for cid in dense_ids if cid in id_to_idx]
    sparse_ranked = [id_to_idx[cid] for cid in sparse_ids if cid in id_to_idx]

    fused_indices = rrf_fuse([dense_ranked, sparse_ranked], [1.0, 1.0], rrf_k)
    return [chunk_map.get(all_ids[i], {}) for i in fused_indices if all_ids[i] in chunk_map]


# ── Intent classification (from v4 notebook) ────────────────────────────────

INTENT_PATTERNS = {
    "lookup":     r"what is|who is|when did|where is|define|meaning of|what does",
    "comparison": r"compare|difference|versus|vs[.]?|better|worse|differ",
    "procedural": r"how to|steps to|process of|guide|tutorial|implement",
    "analytical": r"why|explain|reason|cause|effect|impact|relationship|how does",
}

def classify_intent(query: str) -> str:
    for intent, pat in INTENT_PATTERNS.items():
        if re.search(pat, query.lower()):
            return intent
    return "general"


def table_query_weight(query: str) -> float:
    signals = [
        r"\d", r"how many|how much|count|total|sum",
        r"compare|versus|vs[.]?|differ",
        r"list|which|highest|lowest|best|worst|most|least|top|bottom",
        r"average|mean|rate|percent|ratio|score",
    ]
    hits = sum(1 for p in signals if re.search(p, query.lower()))
    return min(1.0 + hits * 0.12, 1.5)


# ── HyDE expansion ──────────────────────────────────────────────────────────

HYDE_PROMPT = (
    "Write a short factual passage (2-3 sentences) that would directly answer "
    "the following question. Write it as if extracted from a document. "
    "Do not say 'the answer is' — just write the passage.\n\n"
    "Question: {query}\n\nPassage:"
)

def hyde_expand(query: str) -> str:
    from engine_v4.ollama import ollama_generate
    hyp = ollama_generate(CFG.text_model, HYDE_PROMPT.format(query=query), timeout=30)
    if hyp.startswith("[ollama error"):
        return query
    return f"{query} {hyp}"


# ── Full retrieve pipeline (from v4 notebook) ───────────────────────────────

def retrieve(query: str, org_id: str = "default", top_k: int = 8,
             file_ids: Optional[List[str]] = None) -> List[dict]:
    """
    Full v4 retrieval pipeline:
    1. Intent → HyDE expansion (if analytical/general)
    2. Embed query
    3. Per-modality hybrid (dense + sparse + RRF)
    4. Cross-modal RRF fusion
    5. Reranker: top_k results
    """
    # 1. HyDE
    embed_query = query
    if CFG.use_hyde and classify_intent(query) in ("analytical", "general"):
        embed_query = hyde_expand(query)

    # 2. Embed
    qvec = embedder.embed([embed_query])[0]

    # 3. Per-modality hybrid retrieval
    modal_results = {}
    for m in ("text", "table", "image"):
        tw = table_query_weight(query) if m == "table" else 1.0
        modal_results[m] = retrieve_modality_db(
            query, qvec, m, org_id,
            dense_k=int(CFG.dense_top_k.get(m, 30) * tw),
            sparse_k=int(CFG.sparse_top_k.get(m, 30) * tw),
            rrf_k=CFG.rrf_k,
        )

    # 4. Cross-modal RRF
    cross_weights = {"text": 1.0, "table": table_query_weight(query), "image": 0.8}
    all_chunks_map = {}
    ranked_per_modal = {}

    for m, results in modal_results.items():
        for r in results:
            cid = r.get("chunk_id", "")
            if cid and cid not in all_chunks_map:
                all_chunks_map[cid] = r

    all_ids = list(all_chunks_map.keys())
    if not all_ids:
        return []

    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    ranked_lists = []
    weights = []
    for m, results in modal_results.items():
        r_indices = [id_to_idx[r["chunk_id"]] for r in results if r.get("chunk_id") in id_to_idx]
        if r_indices:
            ranked_lists.append(r_indices)
            weights.append(cross_weights.get(m, 1.0))

    fused_indices = rrf_fuse(ranked_lists, weights, CFG.rrf_k)
    candidates = [all_chunks_map[all_ids[i]] for i in fused_indices[:CFG.rerank_top_n]]

    if not candidates:
        return []

    # 5. Rerank
    texts = [c.get("content", "") for c in candidates]
    top_idxs = reranker.rerank(query, texts, min(top_k, len(candidates)))
    return [candidates[i] for i in top_idxs]
