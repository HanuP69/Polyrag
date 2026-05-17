import os
import sys

# Ensure project root is on sys.path so `engine.*` imports resolve
# regardless of how this file is launched (python -m, subprocess, etc.)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import json
import shutil
import hashlib
import asyncio
import uuid as uuid_lib
from typing import Optional
from contextlib import asynccontextmanager
import threading
from engine.utils import resolve_model
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import numpy as np

from engine.config import (
    TESTING, OLLAMA_BASE_URL, OLLAMA_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_BASE_URL,
    DEFAULT_TOP_K, GATE_THRESHOLD,
    UPLOAD_DIR, EXPERT_IDS, EMBEDDING_MODEL, MODEL_REGISTRY,
    EXPERT_MODEL_MAP, CASCADE_THRESHOLD, CASCADE_SMALL_MODEL, CASCADE_BIG_MODEL,
)
from engine.experts.base import Chunk
from engine.fuse import rrf_fuse
from engine.rerank import rerank
from engine.guard import verify_answer
from engine.heal import get_pipeline_health, should_retrain_gate

_gate = None
_experts = {}
_embed_model = None
_query_cache = {}
_ingestion_status = {}
MAX_CACHE = 500
_io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="polyrag-io")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gate, _experts

    print("=" * 60)
    print("  PolyRAG Engine Starting")
    print(f"  Mode: {'TESTING (Ollama)' if TESTING else 'PRODUCTION (Groq)'}")
    print("=" * 60)

    try:
        from engine.db import init_db, ensure_org
        init_db()
        ensure_org("default", "Default Organization")
        print("[Main] [OK] Database initialized")
    except Exception as e:
        print(f"[Main] [WARN] Database init failed: {e}")
        print("[Main]   Running without database -- some features will be unavailable")

    try:
        from engine.gate.gate import get_gate
        _gate = get_gate()
        print("[Main] [OK] Gate loaded")
    except FileNotFoundError:
        print("[Main] [WARN] Gate model not found -- run generate_data.py + train.py first")
        print("[Main]   Gate will default to 'text' expert for all queries")
    except Exception as e:
        print(f"[Main] [WARN] Gate load failed: {e}")

    from engine.experts.text import TextExpert
    from engine.experts.table import TableExpert
    from engine.experts.image import ImageExpert
    from engine.experts.code import CodeExpert
    _experts["text"] = TextExpert()
    _experts["table"] = TableExpert()
    _experts["image"] = ImageExpert()
    _experts["code"] = CodeExpert()
    print("[Main] [OK] Text expert registered")
    print("[Main] [OK] Table expert registered")
    print("[Main] [OK] Image expert registered")
    print("[Main] [OK] Code expert registered")

    global _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        print(f"[Main] [OK] Shared embedding model loaded: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"[Main] [WARN] Shared embedding model failed: {e}")

    yield

    print("[Main] Shutting down...")


app = FastAPI(
    title="PolyRAG Engine",
    description="Multimodal RAG with MoE-style expert routing",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GateRequest(BaseModel):
    query: str

class RetrieveRequest(BaseModel):
    query: str
    expert_id: str
    org_id: str = "default"
    top_k: int = 10
    file_ids: Optional[list[str]] = None

class RerankRequest(BaseModel):
    query: str
    chunks: list[dict]

class QueryRequest(BaseModel):
    query: str
    org_id: str = "default"
    top_k: int = DEFAULT_TOP_K
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None
    chat_history: Optional[list[dict]] = None
    file_ids: Optional[list[str]] = None

class EmbedRequest(BaseModel):
    chunks: list[dict]
    expert_id: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "testing" if TESTING else "production",
        "llm": OLLAMA_MODEL if TESTING else GROQ_MODEL,
        "experts_loaded": list(_experts.keys()),
        "gate_loaded": _gate is not None,
    }


_resolve_model = resolve_model


@app.get("/models")
async def list_models():
    return {"models": MODEL_REGISTRY}


@app.post("/gate")
async def gate_route(req: GateRequest):
    if _gate is None:
        return {"weights": {"text": 1.0}, "raw": {"text": 1.0, "table": 0.0, "image": 0.0}}

    raw = _gate.route_raw(req.query)
    active = _gate.route(req.query)

    return {"weights": active, "raw": raw}


def _safe_file_path(filename: str) -> str:
    safe_name = f"{uuid_lib.uuid4()}_{os.path.basename(filename)}"
    return os.path.join(UPLOAD_DIR, safe_name)


@app.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    org_id: str = Form("default"),
    file_id: str = Form("")
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = _safe_file_path(file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    ext = os.path.splitext(file.filename)[1].lower()

    all_chunks = []
    experts_used = []

    if ext in [".pdf", ".txt", ".md"] and "text" in _experts:
        chunks = _experts["text"].parse(file_path, file_id=file_id, org_id=org_id)
        all_chunks.extend(chunks)
        if chunks:
            experts_used.append("text")

    if ext in [".csv"] and "table" in _experts:
        chunks = _experts["table"].parse(file_path, file_id=file_id, org_id=org_id)
        all_chunks.extend(chunks)
        if chunks:
            experts_used.append("table")
    if ext == ".pdf" and "table" in _experts:
        chunks = _experts["table"].parse(file_path, file_id=file_id, org_id=org_id)
        all_chunks.extend(chunks)
        if chunks and "table" not in experts_used:
            experts_used.append("table")

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"] and "image" in _experts:
        chunks = _experts["image"].parse(file_path, file_id=file_id, org_id=org_id)
        all_chunks.extend(chunks)
        if chunks:
            experts_used.append("image")
    if ext == ".pdf" and "image" in _experts:
        chunks = _experts["image"].parse(file_path, file_id=file_id, org_id=org_id)
        all_chunks.extend(chunks)
        if chunks and "image" not in experts_used:
            experts_used.append("image")

    return {
        "file_path": file_path,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "expert_id": c.expert_id,
                "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                "metadata": c.metadata,
            }
            for c in all_chunks
        ],
        "total_chunks": len(all_chunks),
        "experts_used": experts_used,
    }


@app.post("/embed")
async def embed_chunks(req: EmbedRequest):
    expert_id = req.expert_id
    if expert_id not in _experts:
        raise HTTPException(status_code=400, detail=f"Expert '{expert_id}' not registered")

    expert = _experts[expert_id]

    chunks = [
        Chunk(
            chunk_id=c.get("chunk_id", ""),
            org_id=c.get("org_id", "default"),
            file_id=c.get("file_id", ""),
            expert_id=expert_id,
            content=c["content"],
            metadata=c.get("metadata", {}),
        )
        for c in req.chunks
    ]

    chunks = expert.embed(chunks)

    try:
        from engine.db import upsert_chunks
        upsert_chunks(chunks)
    except Exception as e:
        print(f"[Main] [WARN] DB upsert failed: {e}")
        return {"status": "embedded_not_stored", "count": len(chunks), "error": str(e)}

    return {"status": "ok", "count": len(chunks)}


def _shared_embed_query(query: str) -> np.ndarray:
    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not ready yet")
    return _embed_model.encode(query, normalize_embeddings=True)

_embed_cache = {}
_EMBED_CACHE_MAX = 1000
_embed_cache_lock = threading.Lock()

def _cached_embed_query(query: str) -> np.ndarray:
    with _embed_cache_lock:
        if query in _embed_cache:
            return _embed_cache[query]
    vec = _shared_embed_query(query)
    with _embed_cache_lock:
        if len(_embed_cache) < _EMBED_CACHE_MAX:
            _embed_cache[query] = vec
        else:
            oldest = next(iter(_embed_cache))
            del _embed_cache[oldest]
            _embed_cache[query] = vec
    return vec


@app.post("/retrieve")
async def retrieve_endpoint(req: RetrieveRequest):
    if req.expert_id not in _experts:
        raise HTTPException(status_code=404, detail=f"Expert {req.expert_id} not found")

    expert = _experts[req.expert_id]
    try:
        query_vec = _cached_embed_query(req.query)
        # Pass file_ids if provided
        if req.file_ids:
            from engine.db import search_chunks
            chunks = await asyncio.to_thread(search_chunks, query_vec, req.org_id, req.expert_id, req.top_k, req.file_ids)
        else:
            chunks = await asyncio.to_thread(expert.retrieve, query_vec, req.org_id, req.top_k)
        return {
            "expert_id": req.expert_id,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "expert_id": c.expert_id,
                    "content": c.content,
                    "metadata": c.metadata,
                }
                for c in chunks
            ],
            "count": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
async def rerank_endpoint(req: RerankRequest):
    try:
        chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                org_id=c.get("org_id", "default"),
                file_id=c.get("file_id", "unknown"),
                expert_id=c["expert_id"],
                content=c["content"],
                metadata=c.get("metadata", {}),
            )
            for c in req.chunks
        ]
        reranked = rerank(req.query, chunks)
        return [
            {
                "chunk_id": c.chunk_id,
                "expert_id": c.expert_id,
                "content": c.content,
                "metadata": c.metadata,
            }
            for c in reranked
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    org_id: str = Form("default"),
):
    start = time.time()

    try:
        from engine.db import ensure_org, create_file, update_file_status, upsert_chunks
        ensure_org(org_id)
        db_available = True
    except Exception as e:
        print(f"[Main] [WARN] DB not available: {e}")
        db_available = False

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = _safe_file_path(file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    ext = os.path.splitext(file.filename)[1].lower()

    file_id = ""
    if db_available:
        file_id = create_file(org_id, file.filename, ext.lstrip("."))
        update_file_status(file_id, "parsing")

    parse_tasks = []
    experts_to_check = []

    if ext in [".pdf", ".txt", ".md"] and "text" in _experts:
        parse_tasks.append(asyncio.to_thread(_experts["text"].parse, file_path, file_id, org_id))
        experts_to_check.append("text")

    if (ext in [".csv"] or ext == ".pdf") and "table" in _experts:
        parse_tasks.append(asyncio.to_thread(_experts["table"].parse, file_path, file_id, org_id))
        experts_to_check.append("table")

    if (ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"] or ext == ".pdf") and "image" in _experts:
        parse_tasks.append(asyncio.to_thread(_experts["image"].parse, file_path, file_id, org_id))
        experts_to_check.append("image")

    parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)

    all_chunks = []
    experts_used = []

    for i, result in enumerate(parse_results):
        expert_id = experts_to_check[i]
        if isinstance(result, Exception):
            print(f"[Ingest] [WARN] Expert {expert_id} failed: {result}")
            continue
        if result:
            all_chunks.extend(result)
            experts_used.append(expert_id)

    if not all_chunks:
        if db_available:
            update_file_status(file_id, "failed")
        return {"status": "no_chunks", "file_id": file_id}

    if db_available:
        update_file_status(file_id, "embedding", chunk_count=len(all_chunks))

    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not ready")

    import torch
    BATCH_SIZE = 8
    total = len(all_chunks)
    print(f"[Ingest] Embedding {total} chunks (batch_size={BATCH_SIZE})...")

    def _do_embed():
        all_embs = []
        for i in range(0, total, BATCH_SIZE):
            batch = [c.content for c in all_chunks[i:i+BATCH_SIZE]]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            embs = _embed_model.encode(
                batch, batch_size=BATCH_SIZE,
                show_progress_bar=False, normalize_embeddings=True
            )
            all_embs.extend(embs)
        return all_embs

    embeddings = await asyncio.to_thread(_do_embed)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb

    print(f"[Ingest] [OK] Embedded all chunks")

    if db_available:
        upsert_chunks(all_chunks)
        update_file_status(
            file_id, "indexed",
            chunk_count=len(all_chunks),
            experts_used=experts_used
        )

    elapsed = int((time.time() - start) * 1000)

    return {
        "status": "indexed",
        "file_id": file_id,
        "file_name": file.filename,
        "total_chunks": len(all_chunks),
        "experts_used": experts_used,
        "latency_ms": elapsed,
    }


@app.post("/ingest/github")
async def ingest_github(
    repo_url: str = Form(...),
    org_id: str = Form("default"),
):
    import zipfile
    import tempfile

    start = time.time()

    try:
        from engine.db import ensure_org, create_file, update_file_status, upsert_chunks
        ensure_org(org_id)
        db_available = True
    except Exception as e:
        print(f"[Main] [WARN] DB not available: {e}")
        db_available = False

    clean_url = repo_url.rstrip("/")
    parts = clean_url.split("/")
    if len(parts) >= 2:
        owner = parts[-2]
        repo_name = parts[-1]
        zip_url = f"https://api.github.com/repos/{owner}/{repo_name}/zipball"
    else:
        repo_name = clean_url
        zip_url = clean_url

    file_id = ""
    if db_available:
        file_id = create_file(org_id, f"github_{repo_name}", "zip")
        update_file_status(file_id, "parsing")

    try:
        resp = requests.get(zip_url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        if db_available:
            update_file_status(file_id, "failed")
        raise HTTPException(status_code=400, detail=f"Failed to download repo: {e}")

    all_chunks = []
    experts_used = ["code"]

    from experts.code import CODE_EXTENSIONS

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "repo.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)

        extract_dir = os.path.join(tmpdir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        for root, _, files in os.walk(extract_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in CODE_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, extract_dir)

                    if "code" in _experts:
                        try:
                            chunks = await asyncio.to_thread(_experts["code"].parse, file_path, file_id, org_id)
                            for c in chunks:
                                c.metadata["file_path"] = rel_path
                                c.content = c.content.replace(os.path.basename(file_path), rel_path)
                            all_chunks.extend(chunks)
                        except Exception as e:
                            print(f"[Ingest] [WARN] Code expert failed on {rel_path}: {e}")

    if not all_chunks:
        if db_available:
            update_file_status(file_id, "failed")
        return {"status": "no_chunks", "file_id": file_id}

    if db_available:
        update_file_status(file_id, "embedding", chunk_count=len(all_chunks))

    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not ready")

    import torch
    BATCH_SIZE = 8
    total = len(all_chunks)
    print(f"[Ingest] Embedding {total} chunks from GitHub repo...")

    def _do_embed():
        all_embs = []
        for i in range(0, total, BATCH_SIZE):
            batch = [c.content for c in all_chunks[i:i+BATCH_SIZE]]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            embs = _embed_model.encode(
                batch, batch_size=BATCH_SIZE,
                show_progress_bar=False, normalize_embeddings=True
            )
            all_embs.extend(embs)
        return all_embs

    embeddings = await asyncio.to_thread(_do_embed)
    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb

    if db_available:
        upsert_chunks(all_chunks)
        update_file_status(
            file_id, "indexed",
            chunk_count=len(all_chunks),
            experts_used=experts_used
        )

    elapsed = int((time.time() - start) * 1000)
    return {
        "status": "indexed",
        "file_id": file_id,
        "repo": repo_name,
        "total_chunks": len(all_chunks),
        "experts_used": experts_used,
        "latency_ms": elapsed,
    }


@app.get("/file/{file_id}")
async def get_file(file_id: str):
    try:
        from engine.db import get_file_status
        status = get_file_status(file_id)
        if status is None:
            raise HTTPException(status_code=404, detail="File not found")
        for k, v in status.items():
            if hasattr(v, 'isoformat'):
                status[k] = v.isoformat()
            elif hasattr(v, 'hex'):
                status[k] = str(v)
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_pipeline(req: QueryRequest):
    start = time.time()

    _hist_hash = hashlib.md5(str(req.chat_history).encode()).hexdigest()[:8]
    cache_key = f"{req.org_id}:{req.model}:{_hist_hash}:{hashlib.md5(req.query.encode()).hexdigest()}"
    if cache_key in _query_cache:
        cached = _query_cache[cache_key]
        cached["cached"] = True
        return cached

    loop = asyncio.get_event_loop()

    # Gate only — no rewriter
    gate_task = loop.run_in_executor(_io_pool,
        lambda: _gate.route(req.query) if _gate else {"text": 1.0})
    gate_raw_task = loop.run_in_executor(_io_pool,
        lambda: _gate.route_raw(req.query) if _gate else {"text": 1.0, "table": 0.0, "image": 0.0})

    gate_result, gate_raw = await asyncio.gather(gate_task, gate_raw_task)

    final_query = req.query
    max_gate_conf = max(gate_raw.values()) if gate_raw else 0

    t_gate = time.time()
    print(f"[Query] Gate: {int((t_gate-start)*1000)}ms (conf={max_gate_conf:.2f}, experts={list(gate_result.keys())})")

    query_vec = await loop.run_in_executor(_io_pool, _cached_embed_query, final_query)

    t_embed = time.time()
    print(f"[Query] Embed: {int((t_embed-t_gate)*1000)}ms")

    from engine.db import search_bm25
    retrieve_tasks = []
    retrieve_keys = []

    for expert_id, weight in gate_result.items():
        if expert_id not in _experts:
            continue
        expert = _experts[expert_id]
        retrieve_tasks.append(loop.run_in_executor(_io_pool,
            expert.retrieve, query_vec, req.org_id, req.top_k))
        retrieve_keys.append(expert_id)
        retrieve_tasks.append(loop.run_in_executor(_io_pool,
            search_bm25, final_query, req.org_id, expert_id, 15))
        retrieve_keys.append(f"{expert_id}_bm25")

    retrieval_results = await asyncio.gather(*retrieve_tasks, return_exceptions=True)

    expert_results = {}
    for key, result in zip(retrieve_keys, retrieval_results):
        if isinstance(result, Exception):
            print(f"[Query] [WARN] {key} failed: {result}")
            continue
        if result:
            expert_results[key] = result

    t_retrieve = time.time()
    for key, chunks in expert_results.items():
        top_sim = max((c.metadata.get("similarity", c.metadata.get("bm25_score", 0)) for c in chunks), default=0)
        print(f"[Query] {key}: {len(chunks)} chunks (top_sim={top_sim:.3f})")
    print(f"[Query] Parallel retrieve ({len(retrieve_tasks)} tasks): {int((t_retrieve-t_embed)*1000)}ms")

    if expert_results:
        all_sims = [c.metadata.get("similarity", c.metadata.get("bm25_score", 0))
                    for chunks in expert_results.values() for c in chunks]
        avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0

        if avg_sim < 0.3 and len(gate_result) < len(_experts):
            print(f"[Query] Fallback cascade (avg_sim={avg_sim:.3f})")
            fallback_tasks = []
            fallback_keys = []
            for expert_id, expert in _experts.items():
                if expert_id not in gate_result:
                    fallback_tasks.append(loop.run_in_executor(_io_pool,
                        expert.retrieve, query_vec, req.org_id, req.top_k))
                    fallback_keys.append(f"{expert_id}_fallback")

            fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)
            for key, result in zip(fallback_keys, fallback_results):
                if not isinstance(result, Exception) and result:
                    expert_results[key] = result
                    gate_result[key.replace("_fallback", "")] = 0.3

    if expert_results:
        fused_chunks = rrf_fuse(expert_results, gate_result, top_n=40)
    else:
        fused_chunks = []

    if fused_chunks:
        fused_chunks = await loop.run_in_executor(_io_pool,
            rerank, final_query, fused_chunks, 12)

    t_rerank = time.time()
    print(f"[Query] Fuse+Rerank: {int((t_rerank-t_retrieve)*1000)}ms")

    context = ""
    sources = []
    for i, chunk in enumerate(fused_chunks):
        chunk_content = chunk.content[:1500] if len(chunk.content) > 1500 else chunk.content
        context += f"\n[Source {i+1} ({chunk.expert_id})]:\n{chunk_content}\n"
        sources.append({
            "chunk_id": chunk.chunk_id,
            "expert_id": chunk.expert_id,
            "content": chunk.content[:1000],
            "metadata": chunk.metadata,
        })

    history_block = ""
    if req.chat_history:
        from engine.memory import build_memory_context
        summary, recent = build_memory_context(req.chat_history)
        if summary:
            history_block += f"\n--- Conversation Summary ---\n{summary}\n"
        if recent:
            history_block += "\n--- Recent Messages ---\n"
            for m in recent:
                history_block += f"{m['role'].upper()}: {m['content'][:500]}\n"

    system_prompt = req.system_prompt or (
        "You are a document Q&A assistant. You MUST answer ONLY using the provided sources. "
        "Do NOT use your own knowledge. Cite sources using [Source N] notation. "
        "If sources don't contain the answer, say so clearly. "
        "Use the conversation history for context about what was previously discussed."
    )
    full_prompt = f"{system_prompt}\n{history_block}\n--- Sources ---\n{context}\n\n--- Question ---\n{final_query}"

    # ── Cascade: small local model first, escalate to big cloud if low confidence ──
    fired_experts = list(gate_result.keys())
    # Determine model: user override > cascade logic > expert-specific default
    if req.model:
        answer_model = req.model
    elif max_gate_conf < CASCADE_THRESHOLD:
        # Low confidence → fast local pass first, then escalate
        print(f"[Query] Cascade: conf={max_gate_conf:.2f} < {CASCADE_THRESHOLD}, escalating to {CASCADE_BIG_MODEL}")
        answer_model = CASCADE_BIG_MODEL
    else:
        # MoE: pick the model for the top-weighted expert
        top_expert = max(gate_result, key=gate_result.get) if gate_result else "text"
        expert_cfg = EXPERT_MODEL_MAP.get(top_expert, {})
        answer_model = expert_cfg.get("model", CASCADE_SMALL_MODEL)
        print(f"[Query] MoE routing: top_expert={top_expert} → model={answer_model}")

    answer = await _generate_answer(full_prompt, req.query, answer_model)

    t_llm = time.time()
    print(f"[Query] LLM ({answer_model}): {int((t_llm-t_rerank)*1000)}ms")

    guard_task = loop.run_in_executor(_io_pool,
        lambda: verify_answer(answer, [s["content"] for s in sources]))

    elapsed = int((time.time() - start) * 1000)

    query_log_id = None
    try:
        from engine.db import log_query
        query_log_id = log_query(
            org_id=req.org_id,
            query=req.query,
            gate_weights=gate_result,
            experts_fired=fired_experts,
            chunk_ids=[c.chunk_id for c in fused_chunks],
            latency_ms=elapsed
        )
    except Exception as e:
        print(f"[Query] [WARN] Logging failed: {e}")

    guard_result = None
    try:
        guard_result = await guard_task
    except Exception as e:
        print(f"[Query] [WARN] Guard failed: {e}")

    t_end = time.time()
    total_ms = int((t_end - start) * 1000)
    print(f"[Query] TOTAL: {total_ms}ms")

    result = {
        "answer": answer,
        "sources": sources,
        "gate_weights": gate_result,
        "experts_fired": fired_experts,
        "latency_ms": total_ms,
        "cached": False,
        "query_log_id": query_log_id,
        "rewritten_query": None,   # rewriter removed
        "guard": guard_result,
        "model_used": answer_model,
    }

    if len(_query_cache) < MAX_CACHE:
        _query_cache[cache_key] = result

    return result


async def _generate_answer(prompt: str, query: str, model: Optional[str] = None) -> str:
    provider, api_name = _resolve_model(model)
    loop = asyncio.get_event_loop()
    if provider == "ollama":
        return await loop.run_in_executor(_io_pool, _generate_ollama, prompt, api_name)
    elif provider == "gemini":
        return await loop.run_in_executor(_io_pool, _generate_gemini, prompt, api_name)
    else:
        return await loop.run_in_executor(_io_pool, _generate_groq, prompt, api_name)


def _generate_ollama(prompt: str, model: Optional[str] = None) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model or OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,   # evict from VRAM immediately after response
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048,
                }
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"[LLM Error] Failed to generate answer: {e}"


def _generate_groq(prompt: str, model: Optional[str] = None) -> str:
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model or GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Error] Failed to generate answer: {e}"


def _stream_ollama(prompt: str, model: Optional[str] = None):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model or OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "keep_alive": 0,   # evict from VRAM after stream completes
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048,
                },
            },
            timeout=180,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except Exception as e:
        yield f"[LLM Error] {e}"


def _generate_gemini(prompt: str, model: str = "gemma-3-27b-it") -> str:
    try:
        url = f"{GEMINI_BASE_URL}/models/{model}:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2048,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[Gemini Error] {e}"


def _stream_gemini(prompt: str, model: str = "gemma-3-27b-it"):
    try:
        url = f"{GEMINI_BASE_URL}/models/{model}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
        response = requests.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2048,
                },
            },
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                    except Exception:
                        pass
    except Exception as e:
        yield f"[Gemini Error] {e}"


class GenerateRequest(BaseModel):
    prompt: str
    query: str = ""
    model: Optional[str] = None
    chat_history: Optional[list[dict]] = None

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    prompt = req.prompt

    if req.chat_history:
        from engine.memory import build_memory_context
        summary, recent = build_memory_context(req.chat_history)
        history_block = ""
        if summary:
            history_block += f"\n--- Conversation Summary ---\n{summary}\n"
        if recent:
            history_block += "\n--- Recent Messages ---\n"
            for m in recent:
                history_block += f"{m['role'].upper()}: {m['content'][:500]}\n"

        if history_block:
            if "--- Sources ---" in prompt:
                prompt = prompt.replace("--- Sources ---", f"{history_block}\n--- Sources ---")
            else:
                prompt = f"{history_block}\n{prompt}"

    def event_stream():
        provider, api_name = _resolve_model(req.model)

        if provider == "ollama":
            for token in _stream_ollama(prompt, api_name):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        elif provider == "gemini":
            for token in _stream_gemini(prompt, api_name):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        else:
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": api_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 2048,
                        "stream": True,
                    },
                    timeout=60,
                    stream=True,
                )
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                        if line_str.startswith("data: ") and line_str != "data: [DONE]":
                            try:
                                data = json.loads(line_str[6:])
                                token = data["choices"][0].get("delta", {}).get("content", "")
                                if token:
                                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                            except Exception:
                                pass
            except Exception as e:
                yield f"data: {json.dumps({'type': 'token', 'content': f'[LLM Error] {e}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    start = time.time()
    loop = asyncio.get_event_loop()

    # Gate only — rewriter removed
    gate_task = loop.run_in_executor(_io_pool,
        lambda: _gate.route(req.query) if _gate else {"text": 1.0})
    gate_raw_task = loop.run_in_executor(_io_pool,
        lambda: _gate.route_raw(req.query) if _gate else {"text": 1.0, "table": 0.0, "image": 0.0})

    gate_result, gate_raw = await asyncio.gather(gate_task, gate_raw_task)

    final_query = req.query
    max_gate_conf = max(gate_raw.values()) if gate_raw else 0

    query_vec = await loop.run_in_executor(_io_pool, _cached_embed_query, final_query)

    from engine.db import search_bm25
    retrieve_tasks = []
    retrieve_keys = []

    for expert_id, weight in gate_result.items():
        if expert_id not in _experts:
            continue
        expert = _experts[expert_id]
        if req.file_ids:
            from engine.db import search_chunks as _sc
            retrieve_tasks.append(loop.run_in_executor(_io_pool,
                _sc, query_vec, req.org_id, expert_id, req.top_k, req.file_ids))
        else:
            retrieve_tasks.append(loop.run_in_executor(_io_pool,
                expert.retrieve, query_vec, req.org_id, req.top_k))
        retrieve_keys.append(expert_id)
        retrieve_tasks.append(loop.run_in_executor(_io_pool,
            search_bm25, final_query, req.org_id, expert_id, 15))
        retrieve_keys.append(f"{expert_id}_bm25")

    retrieval_results = await asyncio.gather(*retrieve_tasks, return_exceptions=True)

    expert_results = {}
    for key, result in zip(retrieve_keys, retrieval_results):
        if isinstance(result, Exception):
            continue
        if result:
            expert_results[key] = result

    if expert_results:
        fused_chunks = rrf_fuse(expert_results, gate_result, top_n=40)
    else:
        fused_chunks = []

    if fused_chunks:
        fused_chunks = await loop.run_in_executor(_io_pool,
            rerank, final_query, fused_chunks, 12)

    query_log_id = None
    try:
        from engine.db import log_query
        query_log_id = log_query(
            org_id=req.org_id,
            query=req.query,
            gate_weights=gate_result,
            experts_fired=list(gate_result.keys()),
            chunk_ids=[c.chunk_id for c in fused_chunks],
            latency_ms=int((time.time() - start) * 1000)
        )
    except Exception as e:
        print(f"[QueryStream] Logging failed: {e}")

    context = ""
    sources = []
    for i, chunk in enumerate(fused_chunks):
        chunk_content = chunk.content[:1500] if len(chunk.content) > 1500 else chunk.content
        context += f"\n[Source {i+1} ({chunk.expert_id})]:\n{chunk_content}\n"
        sources.append({
            "chunk_id": chunk.chunk_id,
            "expert_id": chunk.expert_id,
            "content": chunk.content[:300],
            "metadata": chunk.metadata,
        })

    history_block = ""
    if req.chat_history:
        from engine.memory import build_memory_context
        summary, recent = build_memory_context(req.chat_history)
        if summary:
            history_block += f"\n--- Conversation Summary ---\n{summary}\n"
        if recent:
            history_block += "\n--- Recent Messages ---\n"
            for m in recent:
                history_block += f"{m['role'].upper()}: {m['content'][:500]}\n"

    system_prompt = req.system_prompt or (
        "You are a document Q&A assistant. You MUST answer ONLY using the provided sources. "
        "Do NOT use your own knowledge. Cite sources using [Source N] notation. "
        "If sources don't contain the answer, say so clearly. "
        "Use the conversation history for context about what was previously discussed."
    )
    full_prompt = f"{system_prompt}\n{history_block}\n--- Sources ---\n{context}\n\n--- Question ---\n{final_query}"

    # ── MoE model selection + cascade ──
    if req.model:
        answer_model = req.model
    elif max_gate_conf < CASCADE_THRESHOLD:
        print(f"[QueryStream] Cascade: conf={max_gate_conf:.2f} → escalating to {CASCADE_BIG_MODEL}")
        answer_model = CASCADE_BIG_MODEL
    else:
        top_expert = max(gate_result, key=gate_result.get) if gate_result else "text"
        expert_cfg = EXPERT_MODEL_MAP.get(top_expert, {})
        answer_model = expert_cfg.get("model", CASCADE_SMALL_MODEL)
        print(f"[QueryStream] MoE: top_expert={top_expert} → model={answer_model}")

    active_experts = list(gate_result.keys())

    def sse_generator():
        yield f"data: {json.dumps({'type': 'meta', 'gate': gate_result, 'sources': sources, 'active_experts': active_experts, 'rewritten_query': None, 'query_log_id': query_log_id, 'model_used': answer_model})}\n\n"

        provider, api_name = _resolve_model(answer_model)

        if provider == "ollama":
            for token in _stream_ollama(full_prompt, api_name):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        elif provider == "gemini":
            for token in _stream_gemini(full_prompt, api_name):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        else:
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": api_name,
                        "messages": [{"role": "user", "content": full_prompt}],
                        "temperature": 0.3,
                        "max_tokens": 2048,
                        "stream": True,
                    },
                    timeout=60,
                    stream=True,
                )
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                        if line_str.startswith("data: ") and line_str != "data: [DONE]":
                            try:
                                data = json.loads(line_str[6:])
                                token = data["choices"][0].get("delta", {}).get("content", "")
                                if token:
                                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                            except Exception:
                                pass
            except Exception as e:
                yield f"data: {json.dumps({'type': 'token', 'content': f'[LLM Error] {e}'})}\n\n"

        elapsed = int((time.time() - start) * 1000)
        yield f"data: {json.dumps({'type': 'done', 'latency_ms': elapsed})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


class OrgConfigUpdate(BaseModel):
    name: str = ""
    config: dict = {}

@app.get("/config/{org_id}")
async def get_org_config(org_id: str):
    try:
        from engine.db import get_org_config
        config = get_org_config(org_id)
        if config is None:
            raise HTTPException(status_code=404, detail="Org not found")
        return config
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/config/{org_id}")
async def update_org_config(org_id: str, req: OrgConfigUpdate):
    try:
        from engine.db import update_org_config
        update_org_config(org_id, req.name, req.config)
        return {"status": "ok", "org_id": org_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _ingest_background(file_path: str, file_id: str, org_id: str, ext: str):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    expert_status = {}
    expert_names = []

    def _update(status, progress, **extra):
        _ingestion_status[file_id] = {
            "status": status, "progress": progress,
            "experts": dict(expert_status), "expert_names": list(expert_names),
            **extra,
        }

    _update("parsing", 0)

    try:
        from engine.db import update_file_status, upsert_chunks

        parse_tasks = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            if ext in [".pdf", ".txt", ".md"] and "text" in _experts:
                parse_tasks["text"] = pool.submit(
                    _experts["text"].parse, file_path, file_id, org_id)
                expert_names.append("text")
                expert_status["text"] = {"state": "running", "chunks": 0}

            if (ext in [".csv"] or ext == ".pdf") and "table" in _experts:
                parse_tasks["table"] = pool.submit(
                    _experts["table"].parse, file_path, file_id, org_id)
                expert_names.append("table")
                expert_status["table"] = {"state": "running", "chunks": 0}

            if (ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"] or ext == ".pdf") and "image" in _experts:
                parse_tasks["image"] = pool.submit(
                    _experts["image"].parse, file_path, file_id, org_id)
                expert_names.append("image")
                expert_status["image"] = {"state": "running", "chunks": 0}

            _update("parsing", 5)

            all_chunks = []
            experts_used = []
            total_tasks = len(parse_tasks)
            future_to_expert = {v: k for k, v in parse_tasks.items()}

            for future in as_completed(parse_tasks.values()):
                expert_id = future_to_expert[future]
                try:
                    chunks = future.result(timeout=600)
                    count = len(chunks) if chunks else 0
                    if chunks:
                        all_chunks.extend(chunks)
                        experts_used.append(expert_id)
                    expert_status[expert_id] = {"state": "done", "chunks": count}
                    print(f"[Ingest] {expert_id} -> {count} chunks")
                except Exception as e:
                    expert_status[expert_id] = {"state": "failed", "error": str(e)[:100]}
                    print(f"[Ingest] [WARN] {expert_id} parse failed: {e}")

                done_count = sum(1 for v in expert_status.values() if v["state"] != "running")
                progress = int(5 + (done_count / total_tasks) * 40)
                _update("parsing", progress)

        if not all_chunks:
            update_file_status(file_id, "failed")
            _update("failed", 100)
            return

        _update("embedding", 50, total_chunks=len(all_chunks))
        update_file_status(file_id, "embedding", chunk_count=len(all_chunks))

        if _embed_model is None:
            update_file_status(file_id, "failed")
            _update("failed", 100, error="Embedding model not ready")
            return

        import torch
        BATCH_SIZE = 8
        total = len(all_chunks)
        print(f"[Ingest] Embedding {total} chunks (batch_size={BATCH_SIZE})...")

        all_embeddings = []
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_texts = [c.content for c in all_chunks[batch_start:batch_end]]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_embs = _embed_model.encode(
                batch_texts, batch_size=BATCH_SIZE,
                show_progress_bar=False, normalize_embeddings=True
            )
            all_embeddings.extend(batch_embs)

            done_pct = int(50 + (batch_end / total) * 30)
            _update("embedding", done_pct, total_chunks=total, embedded=batch_end)
            if batch_end % (BATCH_SIZE * 10) == 0 or batch_end == total:
                print(f"[Ingest] Embedded {batch_end}/{total}")

        for chunk, emb in zip(all_chunks, all_embeddings):
            chunk.embedding = emb
        print(f"[Ingest] [OK] Embedded all {total} chunks")

        _update("indexing", 85, total_chunks=len(all_chunks))
        upsert_chunks(all_chunks)
        update_file_status(file_id, "indexed", chunk_count=len(all_chunks), experts_used=experts_used)
        _update("indexed", 100, total_chunks=len(all_chunks), experts_used=experts_used)
        print(f"[Ingest] [OK] {file_id} indexed: {len(all_chunks)} chunks, experts: {experts_used}")
    except Exception as e:
        print(f"[Main] [WARN] Background ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        _update("failed", 100, error=str(e))


@app.post("/ingest/async")
async def ingest_file_async(
    file: UploadFile = File(...),
    org_id: str = Form("default"),
):
    try:
        from engine.db import ensure_org, create_file
        ensure_org(org_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = _safe_file_path(file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    ext = os.path.splitext(file.filename)[1].lower()
    file_id = create_file(org_id, file.filename, ext.lstrip("."))

    _ingestion_status[file_id] = {"status": "queued", "progress": 0}

    import threading
    t = threading.Thread(
        target=_ingest_background,
        args=(file_path, file_id, org_id, ext),
        daemon=True,
    )
    t.start()

    return {"status": "queued", "file_id": file_id}


@app.get("/ingest/status/{file_id}")
async def ingest_status(file_id: str):
    if file_id in _ingestion_status:
        return {"file_id": file_id, **_ingestion_status[file_id]}
    raise HTTPException(status_code=404, detail="File ID not found in ingestion queue")


class FeedbackRequest(BaseModel):
    query_log_id: Optional[str] = None
    rating: int
    correct_expert: Optional[str] = None

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    if not req.query_log_id:
        return {"status": "ok", "feedback_id": None, "note": "no query_log_id provided"}
    try:
        from engine.db import save_feedback
        fb_id = save_feedback(req.query_log_id, req.rating, req.correct_expert)
        return {"status": "ok", "feedback_id": fb_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/pipeline")
async def pipeline_health(org_id: Optional[str] = None):
    health = get_pipeline_health(org_id)
    health["retrain_recommended"] = should_retrain_gate()
    return health


class GuardRequest(BaseModel):
    answer: str
    sources: list[str]

@app.post("/guard")
async def guard_endpoint(req: GuardRequest):
    result = verify_answer(req.answer, req.sources)
    return result


class BM25Request(BaseModel):
    query: str
    expert_id: str
    org_id: str = "default"
    top_k: int = 5
    file_ids: Optional[list[str]] = None

@app.post("/retrieve/bm25")
async def retrieve_bm25(req: BM25Request):
    try:
        from engine.db import search_bm25
        chunks = search_bm25(req.query, req.org_id, req.expert_id, req.top_k, file_ids=req.file_ids)
        return {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "expert_id": c.expert_id,
                    "content": c.content[:300],
                    "metadata": c.metadata,
                }
                for c in chunks
            ],
            "total": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{org_id}")
async def get_org_files(org_id: str):
    """Get all files for an org."""
    try:
        from engine.db import get_files_by_org
        files = get_files_by_org(org_id)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{org_id}/{file_id}")
async def delete_org_file(org_id: str, file_id: str):
    """Delete a file and its chunks."""
    try:
        from engine.db import delete_file
        success = delete_file(org_id, file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "engine.main:app", host="0.0.0.0", port=8000, reload=True,
        reload_excludes=["scripts/*", "tests/*", "client/*", "server/*", "data/*", "uploads/*"]
    )