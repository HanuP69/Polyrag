"""
main.py -- FastAPI app for the PolyRAG engine.

All ML lives here. Routes:
  POST /parse      -- detect file type → call expert.parse() → return chunks
  POST /embed      -- call expert.embed() → upsert to pgvector
  POST /gate       -- gate.py inference → return expert weights
  POST /retrieve   -- embed query → pgvector cosine search → return top-k chunks
  POST /ingest     -- full pipeline: upload → parse → embed → index
  POST /query      -- full query pipeline: gate → retrieve → fuse → generate
  GET  /health     -- health check
  GET  /file/{id}  -- get file status
"""

import os
import sys
import time
import json
import shutil
import hashlib
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import numpy as np

from engine.config import (
    TESTING, OLLAMA_BASE_URL, OLLAMA_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    DEFAULT_TOP_K, GATE_THRESHOLD,
    UPLOAD_DIR, EXPERT_IDS
)
from engine.experts.base import Chunk
from engine.fuse import rrf_fuse


# ──────────────────────────── Globals ────────────────────────────────
# Lazy-loaded singletons
_gate = None
_experts = {}
_query_cache = {}
MAX_CACHE = 500


# ──────────────────────────── Startup / Shutdown ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown."""
    global _gate, _experts
    
    print("=" * 60)
    print("  PolyRAG Engine Starting")
    print(f"  Mode: {'TESTING (Ollama)' if TESTING else 'PRODUCTION (Groq)'}")
    print("=" * 60)
    
    # Initialize database
    try:
        from engine.db import init_db, ensure_org
        init_db()
        ensure_org("default", "Default Organization")
        print("[Main] [OK] Database initialized")
    except Exception as e:
        print(f"[Main] [WARN] Database init failed: {e}")
        print("[Main]   Running without database -- some features will be unavailable")
    
    # Load gate (lazy -- will load on first request if model exists)
    try:
        from engine.gate.gate import get_gate
        _gate = get_gate()
        print("[Main] [OK] Gate loaded")
    except FileNotFoundError:
        print("[Main] [WARN] Gate model not found -- run generate_data.py + train.py first")
        print("[Main]   Gate will default to 'text' expert for all queries")
    except Exception as e:
        print(f"[Main] [WARN] Gate load failed: {e}")
    
    # Register experts
    from engine.experts.text import TextExpert
    _experts["text"] = TextExpert()
    print("[Main] [OK] Text expert registered")
    
    # TODO: Phase 2 -- register table expert
    # TODO: Phase 3 -- register image expert
    
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


# ──────────────────────────── Request Models ─────────────────────────
class GateRequest(BaseModel):
    query: str
    
class RetrieveRequest(BaseModel):
    query: str
    expert_id: str
    org_id: str = "default"
    top_k: int = DEFAULT_TOP_K

class QueryRequest(BaseModel):
    query: str
    org_id: str = "default"
    top_k: int = DEFAULT_TOP_K
    system_prompt: str = ""

class EmbedRequest(BaseModel):
    chunks: list[dict]
    expert_id: str


# ──────────────────────────── Routes ─────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "mode": "testing" if TESTING else "production",
        "llm": OLLAMA_MODEL if TESTING else GROQ_MODEL,
        "experts_loaded": list(_experts.keys()),
        "gate_loaded": _gate is not None,
    }


@app.post("/gate")
async def gate_route(req: GateRequest):
    """
    Route a query to expert(s) via the gate classifier.
    Returns expert weights dict.
    """
    if _gate is None:
        # Fallback: route everything to text
        return {"weights": {"text": 1.0}, "raw": {"text": 1.0, "table": 0.0, "image": 0.0}}
    
    raw = _gate.route_raw(req.query)
    active = _gate.route(req.query)
    
    return {"weights": active, "raw": raw}


@app.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    org_id: str = Form("default"),
    file_id: str = Form("")
):
    """
    Parse an uploaded file into chunks.
    Detects file type and routes to appropriate expert(s).
    """
    # Save uploaded file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Detect file type and route to expert(s)
    ext = os.path.splitext(file.filename)[1].lower()
    
    all_chunks = []
    experts_used = []
    
    # Text expert handles PDFs, txt, md
    if ext in [".pdf", ".txt", ".md"] and "text" in _experts:
        chunks = _experts["text"].parse(file_path, file_id=file_id, org_id=org_id)
        all_chunks.extend(chunks)
        if chunks:
            experts_used.append("text")
    
    # TODO: Phase 2 -- table expert for CSV, and table regions in PDF
    # TODO: Phase 3 -- image expert for images in PDF and standalone images
    
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
    """
    Embed chunks and upsert to pgvector.
    """
    expert_id = req.expert_id
    if expert_id not in _experts:
        raise HTTPException(status_code=400, detail=f"Expert '{expert_id}' not registered")
    
    expert = _experts[expert_id]
    
    # Reconstruct Chunk objects
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
    
    # Embed
    chunks = expert.embed(chunks)
    
    # Upsert to database
    try:
        from engine.db import upsert_chunks
        upsert_chunks(chunks)
    except Exception as e:
        print(f"[Main] [WARN] DB upsert failed: {e}")
        return {"status": "embedded_not_stored", "count": len(chunks), "error": str(e)}
    
    return {"status": "ok", "count": len(chunks)}


@app.post("/retrieve")
async def retrieve_chunks(req: RetrieveRequest):
    """
    Embed query and search pgvector for similar chunks.
    """
    expert_id = req.expert_id
    if expert_id not in _experts:
        raise HTTPException(status_code=400, detail=f"Expert '{expert_id}' not registered")
    
    expert = _experts[expert_id]
    
    # Embed query
    query_vec = expert.embed_query(req.query)
    
    # Search
    chunks = expert.retrieve(query_vec, req.org_id, req.top_k)
    
    return {
        "expert_id": expert_id,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata,
                "expert_id": c.expert_id,
            }
            for c in chunks
        ],
        "count": len(chunks),
    }


@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    org_id: str = Form("default"),
):
    """
    Full ingestion pipeline:
    1. Save file
    2. Create file record in DB
    3. Parse → chunks
    4. Embed chunks
    5. Upsert to pgvector
    6. Update file status → indexed
    """
    start = time.time()
    
    # Ensure org exists
    try:
        from engine.db import ensure_org, create_file, update_file_status, upsert_chunks
        ensure_org(org_id)
        db_available = True
    except Exception as e:
        print(f"[Main] [WARN] DB not available: {e}")
        db_available = False
    
    # Save file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    ext = os.path.splitext(file.filename)[1].lower()
    
    # Create file record
    file_id = ""
    if db_available:
        file_id = create_file(org_id, file.filename, ext.lstrip("."))
        update_file_status(file_id, "parsing")
    
    # Parse
    all_chunks = []
    experts_used = []
    
    if ext in [".pdf", ".txt", ".md"] and "text" in _experts:
        chunks = _experts["text"].parse(file_path, file_id=file_id, org_id=org_id)
        all_chunks.extend(chunks)
        if chunks:
            experts_used.append("text")
    
    if not all_chunks:
        if db_available:
            update_file_status(file_id, "failed")
        return {"status": "no_chunks", "file_id": file_id}
    
    # Embed
    if db_available:
        update_file_status(file_id, "embedding")
    
    for expert_id in experts_used:
        expert_chunks = [c for c in all_chunks if c.expert_id == expert_id]
        _experts[expert_id].embed(expert_chunks)
    
    # Upsert
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


@app.get("/file/{file_id}")
async def get_file(file_id: str):
    """Get file status."""
    try:
        from engine.db import get_file_status
        status = get_file_status(file_id)
        if status is None:
            raise HTTPException(status_code=404, detail="File not found")
        # Convert UUID and datetime to string for JSON serialization
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
    """
    Full query pipeline:
    1. Check cache
    2. Gate → active experts
    3. Retrieve from each active expert (parallel in concept, sequential here)
    4. RRF fusion
    5. LLM generation
    6. Return answer + sources
    """
    start = time.time()
    
    # Cache check
    cache_key = f"{req.org_id}:{hashlib.md5(req.query.encode()).hexdigest()}"
    if cache_key in _query_cache:
        cached = _query_cache[cache_key]
        cached["cached"] = True
        return cached
    
    # Gate
    if _gate is not None:
        gate_result = _gate.route(req.query)
    else:
        gate_result = {"text": 1.0}
    
    # Retrieve from each active expert
    expert_results = {}
    for expert_id, weight in gate_result.items():
        if expert_id not in _experts:
            continue
        
        expert = _experts[expert_id]
        query_vec = expert.embed_query(req.query)
        
        try:
            chunks = expert.retrieve(query_vec, req.org_id, req.top_k)
            if chunks:
                expert_results[expert_id] = chunks
        except Exception as e:
            print(f"[Main] [WARN] Retrieve from {expert_id} failed: {e}")
    
    # RRF Fusion
    if expert_results:
        fused_chunks = rrf_fuse(expert_results, gate_result, top_n=8)
    else:
        fused_chunks = []
    
    # Build prompt
    context = ""
    sources = []
    for i, chunk in enumerate(fused_chunks):
        context += f"\n[Source {i+1} ({chunk.expert_id})]:\n{chunk.content}\n"
        sources.append({
            "chunk_id": chunk.chunk_id,
            "expert_id": chunk.expert_id,
            "content": chunk.content[:300],
            "metadata": chunk.metadata,
        })
    
    system_prompt = req.system_prompt or (
        "You are a helpful assistant. Answer the user's question based on the provided sources. "
        "Cite sources using [Source N] notation. If the sources don't contain relevant information, "
        "say so clearly."
    )
    
    full_prompt = f"{system_prompt}\n\n--- Sources ---\n{context}\n\n--- Question ---\n{req.query}"
    
    # LLM Generation
    answer = await _generate_answer(full_prompt, req.query)
    
    elapsed = int((time.time() - start) * 1000)
    
    # Log query
    try:
        from engine.db import log_query
        log_query(
            org_id=req.org_id,
            query=req.query,
            gate_weights=gate_result,
            experts_fired=list(gate_result.keys()),
            chunk_ids=[c.chunk_id for c in fused_chunks],
            latency_ms=elapsed
        )
    except Exception as e:
        print(f"[Main] [WARN] Query logging failed: {e}")
    
    result = {
        "answer": answer,
        "sources": sources,
        "gate_weights": gate_result,
        "experts_fired": list(gate_result.keys()),
        "latency_ms": elapsed,
        "cached": False,
    }
    
    # Cache result
    if len(_query_cache) < MAX_CACHE:
        _query_cache[cache_key] = result
    
    return result


# ──────────────────────────── LLM Generation ─────────────────────────
async def _generate_answer(prompt: str, query: str) -> str:
    """Generate answer using Ollama (testing) or Groq (production)."""
    
    if TESTING:
        return _generate_ollama(prompt)
    else:
        return _generate_groq(prompt)


def _generate_ollama(prompt: str) -> str:
    """Generate via Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"[LLM Error] Failed to generate answer: {e}"


def _generate_groq(prompt: str) -> str:
    """Generate via Groq API."""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
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


# ──────────────────────────── Entry Point ────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("engine.main:app", host="0.0.0.0", port=8000, reload=True)
