"""
PolyRAG v4 Engine — FastAPI server.
Implements all endpoints that the Node.js orchestrator expects.
Architecture: BGE-M3 + BM25 + RRF + BGE-reranker + Ollama generation.
"""
import os
import sys
import json
import uuid
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from engine_v4.config import CFG
from engine_v4 import db
from engine_v4.models import embedder, reranker
from engine_v4.retrieval import load_indexes, retrieve as v4_retrieve
from engine_v4.ollama import ollama_chat_stream, ollama_chat, ollama_unload_all
from engine_v4.guard import verify_answer
from engine_v4.ingest import ingest_file

# Thread pools
_ingest_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
_io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="io")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  PolyRAG v4 Engine Starting")
    print(f"  Embedder: {CFG.embedder_model} (lazy-loaded)")
    print(f"  Reranker: {CFG.reranker_model} (lazy-loaded)")
    print(f"  Text LLM: {CFG.text_model} (Ollama)")
    print(f"  Caption:  {CFG.caption_model} (Ollama)")
    print(f"  Postgres: {CFG.pg_conn}")
    print("=" * 60)

    # Init DB
    try:
        db.init_db()
        db.ensure_org("default", "Default Organization")
        print("[Main] DB initialized")
    except Exception as e:
        print(f"[Main] DB init failed: {e}")

    # Models are lazy-loaded on first query (to avoid OOM at startup)
    print("[Main] Models will load on first query")

    # Load in-memory BM25 indexes from DB (lightweight)
    try:
        load_indexes("default")
    except Exception as e:
        print(f"[Main] Index loading skipped (OK if no data yet): {e}")

    print("[Main] Ready!")
    yield
    print("[Main] Shutting down...")


app = FastAPI(
    title="PolyRAG v4 Engine",
    description="Multimodal RAG: BGE-M3 + BM25 + RRF + Reranker + Ollama",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ───────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str
    org_id: str = "default"
    top_k: int = 10
    file_ids: Optional[List[str]] = None

class RerankRequest(BaseModel):
    query: str
    chunks: list

class GenerateRequest(BaseModel):
    prompt: str
    query: str = ""
    model: Optional[str] = None
    chat_history: Optional[list] = None

class GuardRequest(BaseModel):
    answer: str
    sources: list

class FeedbackRequest(BaseModel):
    query_log_id: str = ""
    rating: int = 0


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "v4",
        "llm": CFG.text_model,
        "embedder": CFG.embedder_model,
        "reranker": CFG.reranker_model,
    }


@app.get("/health/pipeline")
async def pipeline_health():
    return {
        "status": "healthy",
        "components": {
            "embedder": embedder.loaded,
            "reranker": reranker.loaded,
            "database": True,
        },
    }


@app.get("/models")
async def list_models():
    """Return model registry in the format the frontend expects:
    { models: { model_key: { display, group, caps, type } } }
    """
    models = {
        # ── Local Ollama models ──────────────────────────────────────────
        CFG.text_model: {
            "type": "ollama",
            "display": CFG.text_model,
            "group": "Local (Ollama)",
            "caps": ["text"],
        },
        CFG.caption_model: {
            "type": "ollama",
            "display": f"{CFG.caption_model} (Vision)",
            "group": "Local (Ollama)",
            "caps": ["text", "vision"],
        },
    }

    # ── Gemini (if API key set) ──────────────────────────────────────
    if CFG.gemini_api_key:
        models[CFG.gemini_model] = {
            "type": "gemini",
            "display": f"{CFG.gemini_model} ☁",
            "group": "Cloud (Gemini)",
            "caps": ["text", "vision"],
        }

    return {"models": models}


# ── Retrieval ────────────────────────────────────────────────────────────────

@app.post("/retrieve")
async def retrieve_endpoint(req: RetrieveRequest):
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(
        _io_pool,
        lambda: v4_retrieve(req.query, req.org_id, req.top_k, req.file_ids)
    )
    return {"chunks": chunks[:req.top_k]}


# ── Rerank ───────────────────────────────────────────────────────────────────

@app.post("/rerank")
async def rerank_endpoint(req: RerankRequest):
    texts = [c.get("content", "") for c in req.chunks]
    if not texts:
        return {"chunks": []}

    loop = asyncio.get_event_loop()
    top_idxs = await loop.run_in_executor(
        _io_pool,
        lambda: reranker.rerank(req.query, texts, min(8, len(texts)))
    )
    return {"chunks": [req.chunks[i] for i in top_idxs]}


# ── Guard ────────────────────────────────────────────────────────────────────

@app.post("/guard")
async def guard_endpoint(req: GuardRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _io_pool,
        lambda: verify_answer(req.answer, req.sources)
    )
    return result


# ── Generation (streaming + non-streaming) ───────────────────────────────────

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    model = req.model or CFG.text_model

    async def event_generator():
        loop = asyncio.get_event_loop()

        def run_stream():
            tokens = []
            for token in ollama_chat_stream(
                model, req.prompt,
                chat_history=req.chat_history or [],
            ):
                tokens.append(token)
                yield token
            return tokens

        # Run streaming in sync, yield SSE events
        for token in ollama_chat_stream(
            model, req.prompt,
            chat_history=req.chat_history or [],
        ):
            event = json.dumps({"type": "token", "content": token})
            yield f"data: {event}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate")
async def generate(req: GenerateRequest):
    model = req.model or CFG.text_model
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _io_pool,
        lambda: ollama_chat(model, req.prompt, chat_history=req.chat_history or [])
    )
    return {"response": result}


# ── Ingestion ────────────────────────────────────────────────────────────────

_ingestion_status = {}

@app.post("/ingest/async")
async def ingest_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    org_id: str = Form("default"),
    models: str = Form("{}"),
):
    os.makedirs(CFG.upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(CFG.upload_dir, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create file record
    ext = os.path.splitext(file.filename)[1].lower()
    db.create_file(file_id, org_id, file.filename, ext)
    _ingestion_status[file_id] = {"status": "processing", "file_id": file_id}

    def run_ingest():
        try:
            result = ingest_file(file_path, file_id, org_id)
            # Normalize: ingest returns "completed", frontend expects "indexed"
            final_status = "indexed" if result.get("status") == "completed" else result.get("status", "indexed")
            result["status"] = final_status
            result["file_id"] = file_id
            result["id"] = file_id  # frontend uses file.id
            _ingestion_status[file_id] = result
            db.update_file_status(file_id, final_status,
                                  result.get("chunk_count", 0))
        except Exception as e:
            print(f"[Ingest] Error: {e}")
            _ingestion_status[file_id] = {"status": "error", "error": str(e)}
            db.update_file_status(file_id, "error", error=str(e))

    background_tasks.add_task(run_ingest)

    return {"status": "processing", "file_id": file_id, "filename": file.filename}


@app.get("/file/{file_id}")
async def get_file_status(file_id: str):
    # Check in-memory status first (for active ingestions)
    if file_id in _ingestion_status:
        return _ingestion_status[file_id]
    # Check DB
    f = db.get_file(file_id)
    if f:
        return f
    raise HTTPException(404, "File not found")


# ── Files CRUD ───────────────────────────────────────────────────────────────

@app.get("/files/{org_id}")
async def list_files(org_id: str):
    return db.get_org_files(org_id)


@app.delete("/files/{org_id}/{file_id}")
async def delete_file(org_id: str, file_id: str):
    db.delete_file_and_chunks(org_id, file_id)
    # Rebuild BM25 indexes after deletion
    from engine_v4.retrieval import reload_indexes
    reload_indexes(org_id)
    return {"status": "ok", "file_id": file_id}


# ── Config ───────────────────────────────────────────────────────────────────

@app.get("/config/{org_id}")
async def get_config(org_id: str):
    cfg = db.get_org_config(org_id)
    if cfg:
        return cfg
    raise HTTPException(404, "Org not found")


@app.put("/config/{org_id}")
async def update_config(org_id: str, body: dict):
    return db.update_org_config(org_id, body.get("name", ""), body.get("config", {}))


# ── Feedback ─────────────────────────────────────────────────────────────────

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    # Simplified: just acknowledge
    return {"status": "ok", "query_log_id": req.query_log_id}


# ── Chat Sessions ───────────────────────────────────────────────────────────

@app.get("/chat/sessions/{org_id}")
async def get_sessions(org_id: str):
    return db.get_chat_sessions(org_id)


@app.post("/chat/sessions")
async def create_session(body: dict):
    return db.create_chat_session(
        body["session_id"], body.get("org_id", "default"), body.get("title", "New Chat")
    )


@app.delete("/chat/sessions/{org_id}/{session_id}")
async def delete_session(org_id: str, session_id: str):
    return db.delete_chat_session(org_id, session_id)


@app.get("/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    return db.get_chat_messages(session_id)


@app.post("/chat/sessions/{session_id}/messages")
async def add_message(session_id: str, body: dict):
    return db.add_chat_message(
        session_id, body["message_id"], body["role"], body["content"],
        body.get("sources", []), body.get("org_id", "default"),
    )


@app.get("/chat/sessions/{session_id}/owner")
async def get_owner(session_id: str):
    owner = db.get_session_owner(session_id)
    return {"org_id": owner}


@app.post("/chat/logout")
async def chat_logout(body: dict = {}):
    org_id = body.get("org_id", "default")
    deleted = db.delete_all_chat_sessions(org_id)
    return {"deleted_sessions": deleted}


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CFG.host, port=CFG.port)
