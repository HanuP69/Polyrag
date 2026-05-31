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
import unicodedata
import re
import io
import base64
from typing import Optional, List, Dict, Any
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
_ingestion_status = {}

# LRU query cache with TTL (auto-evicts after 300s, max 500 entries)
try:
    from cachetools import TTLCache
    _query_cache = TTLCache(maxsize=500, ttl=300)
except ImportError:
    # Fallback: plain dict (no eviction, but won't crash)
    _query_cache = {}

_io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="polyrag-io")    # HTTP calls to Groq/Gemini/Ollama
_cpu_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="polyrag-cpu")  # Embedding, reranking, gate inference
_ingest_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="polyrag-ingest") # Orchestrates ingestion jobs
_parse_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="polyrag-parse")   # PDF/CSV/Image parsing


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

    # Gate loading disabled for unified flat non-MoE architecture
    _gate = None
    print("[Main] Gate loading disabled (Unified Flat RAG Architecture)")

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

    # Eager-load reranker at startup to prevent 5-15s block on first query
    try:
        from engine.rerank import get_reranker
        get_reranker()
        print("[Main] [OK] Reranker pre-loaded at startup")
    except Exception as e:
        print(f"[Main] [WARN] Reranker pre-load failed: {e}")

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


from fastapi import Request
from engine.db import tenant_context

@app.middleware("http")
async def add_tenant_context(request: Request, call_next):
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        tenant_id = request.query_params.get("org_id", "default")
    token = tenant_context.set(tenant_id)
    try:
        response = await call_next(request)
        return response
    finally:
        tenant_context.reset(token)


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
    mock_weights = {"text": 0.25, "table": 0.25, "image": 0.25, "code": 0.25}
    return {"weights": mock_weights, "raw": mock_weights}


def _safe_file_path(filename: str) -> str:
    safe_name = f"{uuid_lib.uuid4()}_{os.path.basename(filename)}"
    return os.path.join(UPLOAD_DIR, safe_name)


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED PDF LAYOUT PARSER & IMAGE CAPTIONING FALLBACK CHAIN
# ──────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class IRBlock:
    block_id:           str
    pdf_id:             str
    page_no:            int
    modality:           str          # 'text' | 'code' | 'image' | 'table'
    bbox:               tuple
    block_index:        int
    raw_content:        str
    processed_content:  str
    image_b64:          Optional[str] = None
    table_json:         Optional[Dict] = None
    neighboring_blocks: List[str] = field(default_factory=list)
    metadata:           Dict[str, Any] = field(default_factory=dict)

CODE_KEYWORDS = re.compile(
    r'\b(def |class |import |return |for |while |if |else|elif |print\(|'
    r'#include|void |int |float |public |private |function |=>|\{\}|->)'
)
SYMBOL_RE = re.compile(r'[=\[\](){};<>!&|^~%]')

def _clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def _is_code(text: str, font_name: str = "") -> bool:
    mono_hints = any(k in font_name.lower() for k in ('mono','courier','consol','code'))
    kw_hits    = len(CODE_KEYWORDS.findall(text)) >= 2
    sym_ratio  = len(SYMBOL_RE.findall(text)) / max(len(text), 1)
    indent     = bool(re.search(r'^    |\t', text, re.M))
    return mono_hints or (kw_hits and (sym_ratio > 0.05 or indent))

def _img_to_b64(pixmap) -> str:
    img_bytes = pixmap.tobytes(output="png")
    return base64.b64encode(img_bytes).decode()

def _table_to_markdown(tbl) -> str:
    try:
        rows = tbl.extract()
        if not rows:
            return ""
        header = rows[0]
        sep    = ["---"] * len(header)
        body   = rows[1:]
        def fmt_row(r):
            return "| " + " | ".join(str(c or "").replace('\n', ' ') for c in r) + " |"
        lines  = [fmt_row(header), fmt_row(sep)] + [fmt_row(r) for r in body]
        return "\n".join(lines)
    except Exception:
        return ""

def _table_to_json(tbl) -> dict:
    try:
        rows    = tbl.extract()
        if not rows:
            return {}
        headers = [str(c or "") for c in rows[0]]
        data    = [[str(c or "") for c in r] for r in rows[1:]]
        return {"headers": headers, "rows": data}
    except Exception:
        return {}

def _table_to_natural(tbl_json: dict) -> str:
    if not tbl_json:
        return ""
    parts = []
    hdrs  = tbl_json.get("headers", [])
    for row in tbl_json.get("rows", []):
        pairs = ", ".join(f"{h}: {v}" for h, v in zip(hdrs, row) if h)
        if pairs:
            parts.append(pairs)
    return ". ".join(parts)

def _approx_tokens(text: str) -> int:
    return len(text.split())

def _caption_image_optional(b64_image: str, config: dict = None) -> str:
    prompt = (
        "Describe this image in detail. Include all visible text, labels, "
        "data values, diagram elements, and any structural information. "
        "Be thorough and factual."
    )
    # 1. Try Gemini
    gemini_key = (config or {}).get("geminiApiKey") or GEMINI_API_KEY
    if gemini_key:
        try:
            model_name = "gemini-2.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
            resp = requests.post(
                url,
                json={
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/png", "data": b64_image}}
                        ]
                    }],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
                },
                timeout=30
            )
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text:
                    print(f"[Caption] Gemini caption success: {text[:50]}...")
                    return text
            else:
                print(f"[Caption] Gemini caption status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[Caption] Gemini caption error: {e}")

    # 2. Try Ollama Vision
    try:
        from engine.config import OLLAMA_VISION_MODEL
        model_name = OLLAMA_VISION_MODEL or "llava:latest"
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                },
            },
            timeout=45,
        )
        if resp.status_code == 200:
            text = resp.json().get("response", "").strip()
            if text:
                print(f"[Caption] Ollama caption success: {text[:50]}...")
                return text
    except Exception as e:
        print(f"[Caption] Ollama caption error: {e}")

    return ""

def extract_pdf(pdf_path: str, pdf_id: Optional[str] = None) -> list[IRBlock]:
    import fitz
    import uuid
    from pathlib import Path
    
    doc    = fitz.open(pdf_path)
    pdf_id = pdf_id or Path(pdf_path).stem
    blocks: list[IRBlock] = []
    block_idx = 0

    for page_no, page in enumerate(doc):
        page_blocks: list[IRBlock] = []

        # 1. Text/Code extraction
        raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        for rb in raw_blocks:
            if rb.get("type") != 0:
                continue
            text = " ".join(
                span["text"]
                for line in rb["lines"]
                for span in line["spans"]
            )
            text = _clean_text(text)
            if not text:
                continue

            font_name = ""
            try:
                font_name = rb["lines"][0]["spans"][0].get("font", "")
            except Exception:
                pass

            modality = "code" if _is_code(text, font_name) else "text"
            bid      = str(uuid.uuid4())
            blk      = IRBlock(
                block_id=bid, pdf_id=pdf_id, page_no=page_no,
                modality=modality, bbox=tuple(rb["bbox"]),
                block_index=-1, raw_content=text,
                processed_content=text,
                metadata={"font": font_name},
            )
            page_blocks.append(blk)

        # 2. Image extraction
        img_list = page.get_images(full=True)
        for img_info in img_list:
            xref     = img_info[0]
            try:
                pix  = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 32 or pix.height < 32:
                    continue
                
                # Resize large images
                img_bytes = pix.tobytes(output="png")
                w, h = pix.width, pix.height
                if pix.width > 800 or pix.height > 800:
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes))
                    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    w, h = img.width, img.height
                b64 = base64.b64encode(img_bytes).decode()

                img_rect  = page.get_image_rects(xref)
                bbox      = tuple(img_rect[0]) if img_rect else (0,0,0,0)
                ocr_clip  = page.get_textbox(fitz.Rect(bbox)) if bbox != (0,0,0,0) else ""
                ocr_text  = _clean_text(ocr_clip)

                bid = str(uuid.uuid4())
                blk = IRBlock(
                    block_id=bid, pdf_id=pdf_id, page_no=page_no,
                    modality="image", bbox=bbox,
                    block_index=-1, raw_content=ocr_text,
                    processed_content=ocr_text, image_b64=b64,
                    metadata={"xref": xref, "w": w, "h": h},
                )
                page_blocks.append(blk)
            except Exception as e:
                print(f"[Parser] Failed to extract image: {e}")
                continue

        # 3. Table extraction
        try:
            tables = page.find_tables()
            for tbl in tables.tables:
                md      = _table_to_markdown(tbl)
                tj      = _table_to_json(tbl)
                natural = _table_to_natural(tj)
                content = f"{md}\n\n{natural}".strip() if natural else md
                if not content:
                    continue

                bid = str(uuid.uuid4())
                blk = IRBlock(
                    block_id=bid, pdf_id=pdf_id, page_no=page_no,
                    modality="table", bbox=tuple(tbl.bbox),
                    block_index=-1, raw_content=md,
                    processed_content=content, table_json=tj,
                    metadata={"rows": len(tj.get("rows",[])), "cols": len(tj.get("headers",[]))},
                )
                page_blocks.append(blk)
        except Exception as e:
            print(f"[Parser] Failed to extract table: {e}")
            pass

        # Sort all blocks on this page by y0, then x0 to establish reading order
        page_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        # Now, process sorted blocks on the page
        for local_idx, blk in enumerate(page_blocks):
            blk.block_index = block_idx
            block_idx += 1

            # Locate surrounding text paragraphs for images
            if blk.modality == "image":
                prec_text = ""
                for prev_blk in reversed(page_blocks[:local_idx]):
                    if prev_blk.modality in ("text", "code"):
                        prec_text = prev_blk.processed_content
                        break
                
                succ_text = ""
                for next_blk in page_blocks[local_idx+1:]:
                    if next_blk.modality in ("text", "code"):
                        succ_text = next_blk.processed_content
                        break

                fig_captions = []
                for other_blk in page_blocks:
                    if other_blk.modality in ("text", "code") and other_blk.block_id != blk.block_id:
                        iy0, iy1 = blk.bbox[1], blk.bbox[3]
                        oy0, oy1 = other_blk.bbox[1], other_blk.bbox[3]
                        v_dist = min(abs(oy0 - iy1), abs(iy0 - oy1))
                        if v_dist <= 250:
                            if re.search(r"(?i)\b(fig(ure)?|chart|diagram|table)\b", other_blk.processed_content):
                                fig_captions.append(other_blk.processed_content)

                blk.metadata["preceding_paragraph"] = prec_text
                blk.metadata["succeeding_paragraph"] = succ_text
                blk.metadata["figure_captions"] = fig_captions

            blocks.append(blk)

    doc.close()
    return blocks

def build_chunks(blocks: list[IRBlock], embed_fn, org_id: str = "default", file_id: str = "", config: Optional[Dict] = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    sim_threshold = 0.65
    CHUNK_MAX_TOKENS = 512
    
    from collections import defaultdict
    page_blocks = defaultdict(list)
    for b in blocks:
        page_blocks[b.page_no].append(b)
        
    for page_no in sorted(page_blocks.keys()):
        blks = page_blocks[page_no]
        i = 0
        while i < len(blks):
            b = blks[i]
            
            if b.modality == 'image':
                ocr_text = b.processed_content
                
                # Optional captioning using VLLM/Gemini
                caption = _caption_image_optional(b.image_b64, config=config)
                
                prec_text = b.metadata.get("preceding_paragraph", "")
                succ_text = b.metadata.get("succeeding_paragraph", "")
                fig_caps = b.metadata.get("figure_captions", [])
                
                fig_caps_str = "\n".join(f"- [Figure Caption]: {fc}" for fc in fig_caps) if fig_caps else "- [Figure Caption]: None"
                
                repr_parts = []
                repr_parts.append("[Image OCR Text]:")
                repr_parts.append(ocr_text if ocr_text else "None")
                repr_parts.append("")
                repr_parts.append("[Image Caption]:")
                repr_parts.append(caption if caption else "None")
                repr_parts.append("")
                repr_parts.append("[Nearby Context]:")
                repr_parts.append(fig_caps_str)
                repr_parts.append(f"- [Preceding Paragraph]: {prec_text}" if prec_text else "- [Preceding Paragraph]: None")
                repr_parts.append(f"- [Succeeding Paragraph]: {succ_text}" if succ_text else "- [Succeeding Paragraph]: None")
                
                content = "\n".join(repr_parts)
                
                meta = dict(b.metadata)
                meta["page"] = page_no + 1
                meta["type"] = "pdf_image"
                meta["image_b64"] = b.image_b64
                
                chunks.append(Chunk(
                    chunk_id=b.block_id,
                    org_id=org_id,
                    file_id=file_id,
                    expert_id="image",
                    content=content,
                    metadata=meta
                ))
                i += 1
                
            elif b.modality == 'table':
                meta = dict(b.metadata)
                meta["page"] = page_no + 1
                meta["type"] = "pdf_table"
                meta["table_json"] = b.table_json
                
                chunks.append(Chunk(
                    chunk_id=b.block_id,
                    org_id=org_id,
                    file_id=file_id,
                    expert_id="table",
                    content=b.processed_content,
                    metadata=meta
                ))
                i += 1
                
            elif b.modality == 'code':
                toks = _approx_tokens(b.processed_content)
                meta = dict(b.metadata)
                meta["page"] = page_no + 1
                meta["type"] = "code"
                
                if toks <= CHUNK_MAX_TOKENS:
                    chunks.append(Chunk(
                        chunk_id=b.block_id,
                        org_id=org_id,
                        file_id=file_id,
                        expert_id="code",
                        content=b.processed_content,
                        metadata=meta
                    ))
                else: 
                    lines = b.processed_content.split('\n')
                    sub_chunks = []
                    curr_lines = []
                    curr_toks = 0
                    for line in lines:
                        line_toks = _approx_tokens(line)
                        if curr_toks + line_toks > CHUNK_MAX_TOKENS:
                            if curr_lines:
                                sub_chunks.append("\n".join(curr_lines))
                            curr_lines = [line]
                            curr_toks = line_toks
                        else:
                            curr_lines.append(line)
                            curr_toks += line_toks
                    if curr_lines:
                        sub_chunks.append("\n".join(curr_lines))
                        
                    for s_idx, sc_text in enumerate(sub_chunks):
                        sub_meta = dict(meta)
                        sub_meta["split_index"] = s_idx
                        chunks.append(Chunk(
                            chunk_id=f"{b.block_id}_sub_{s_idx}",
                            org_id=org_id,
                            file_id=file_id,
                            expert_id="code",
                            content=sc_text,
                            metadata=sub_meta
                        ))
                i += 1
                
            elif b.modality == 'text':
                text_seq = []
                j = i
                while j < len(blks) and blks[j].modality == 'text':
                    text_seq.append(blks[j])
                    j += 1
                
                texts = [tb.processed_content for tb in text_seq]
                embeddings = None
                if texts and embed_fn:
                    try:
                        embeddings = embed_fn(texts)
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        norms[norms == 0] = 1e-12
                        embeddings = embeddings / norms
                    except Exception as e:
                        print(f"Error embedding text blocks for semantic merging: {e}")
                        embeddings = None
                
                curr_chunk_text = ""
                curr_chunk_blocks = []
                curr_chunk_toks = 0
                
                for idx, tb in enumerate(text_seq):
                    tb_toks = _approx_tokens(tb.processed_content)
                    
                    if tb_toks > CHUNK_MAX_TOKENS:
                        if curr_chunk_blocks:
                            meta = {"page": page_no + 1, "type": "text"}
                            chunks.append(Chunk(
                                chunk_id=curr_chunk_blocks[0].block_id,
                                org_id=org_id,
                                file_id=file_id,
                                expert_id="text",
                                content=curr_chunk_text.strip(),
                                metadata=meta
                            ))
                            curr_chunk_text = ""
                            curr_chunk_blocks = []
                            curr_chunk_toks = 0
                        
                        sentences = re.split(r'(?<=[.!?])\s+', tb.processed_content)
                        curr_split_text = ""
                        curr_split_toks = 0
                        split_idx = 0
                        
                        for sent in sentences:
                            sent_toks = _approx_tokens(sent)
                            if curr_split_toks + sent_toks > CHUNK_MAX_TOKENS:
                                if curr_split_text:
                                    meta = {"page": page_no + 1, "type": "text", "split": True, "split_index": split_idx}
                                    chunks.append(Chunk(
                                        chunk_id=f"{tb.block_id}_split_{split_idx}",
                                        org_id=org_id,
                                        file_id=file_id,
                                        expert_id="text",
                                        content=curr_split_text.strip(),
                                        metadata=meta
                                    ))
                                    split_idx += 1
                                curr_split_text = sent + " "
                                curr_split_toks = sent_toks
                            else:
                                curr_split_text += sent + " "
                                curr_split_toks += sent_toks
                        
                        if curr_split_text:
                            meta = {"page": page_no + 1, "type": "text", "split": True, "split_index": split_idx}
                            chunks.append(Chunk(
                                chunk_id=f"{tb.block_id}_split_{split_idx}",
                                org_id=org_id,
                                file_id=file_id,
                                expert_id="text",
                                content=curr_split_text.strip(),
                                metadata=meta
                            ))
                        continue
                    
                    if not curr_chunk_blocks:
                        curr_chunk_text = tb.processed_content + "\n\n"
                        curr_chunk_blocks = [tb]
                        curr_chunk_toks = tb_toks
                    else:
                        similarity = 0.0
                        if (embeddings is not None and idx > 0 and 
                            idx < len(embeddings) and embeddings[idx] is not None and embeddings[idx-1] is not None):
                            v1 = embeddings[idx-1]
                            v2 = embeddings[idx]
                            similarity = float(np.dot(v1, v2))
                            
                        if similarity >= sim_threshold and (curr_chunk_toks + tb_toks <= CHUNK_MAX_TOKENS):
                            curr_chunk_text += tb.processed_content + "\n\n"
                            curr_chunk_blocks.append(tb)
                            curr_chunk_toks += tb_toks
                        else:
                            meta = {"page": page_no + 1, "type": "text"}
                            chunks.append(Chunk(
                                chunk_id=curr_chunk_blocks[0].block_id,
                                org_id=org_id,
                                file_id=file_id,
                                expert_id="text",
                                content=curr_chunk_text.strip(),
                                metadata=meta
                            ))
                            curr_chunk_text = tb.processed_content + "\n\n"
                            curr_chunk_blocks = [tb]
                            curr_chunk_toks = tb_toks
                            
                if curr_chunk_blocks:
                    meta = {"page": page_no + 1, "type": "text"}
                    chunks.append(Chunk(
                        chunk_id=curr_chunk_blocks[0].block_id,
                        org_id=org_id,
                        file_id=file_id,
                        expert_id="text",
                        content=curr_chunk_text.strip(),
                        metadata=meta
                    ))
                i = j
                
    return chunks


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

    if ext == ".pdf":
        try:
            print(f"[Parse] Parsing PDF with layout-aware unified parser: {file_path}")
            embed_fn = None
            if _embed_model is not None:
                embed_fn = lambda texts: _embed_model.encode(texts, show_progress_bar=False)
            
            cfg = None
            try:
                from engine.db import get_org_config
                cfg_data = get_org_config(org_id)
                if cfg_data:
                    cfg = cfg_data.get("config")
            except Exception:
                pass

            blocks = extract_pdf(file_path, file_id)
            all_chunks = build_chunks(blocks, embed_fn, org_id, file_id, config=cfg)
            experts_used = list(set(c.expert_id for c in all_chunks))
        except Exception as e:
            print(f"[Parse] [WARN] Layout-aware unified parser failed: {e}. Falling back to modular experts.")
            all_chunks = []
            experts_used = []

    if not all_chunks:
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

from collections import OrderedDict
_embed_cache = OrderedDict()
_EMBED_CACHE_MAX = 1000
_embed_cache_lock = threading.Lock()

def _normalize_query(q: str) -> str:
    """Normalize query for cache: lowercase, strip, collapse whitespace."""
    import re
    return re.sub(r'\s+', ' ', q.lower().strip())

def _cached_embed_query(query: str) -> np.ndarray:
    key = _normalize_query(query)
    with _embed_cache_lock:
        if key in _embed_cache:
            _embed_cache.move_to_end(key)  # LRU: mark as recently used
            return _embed_cache[key]
    vec = _shared_embed_query(query)
    with _embed_cache_lock:
        if len(_embed_cache) >= _EMBED_CACHE_MAX:
            _embed_cache.popitem(last=False)  # Evict oldest
        _embed_cache[key] = vec
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

    all_chunks = []
    experts_used = []

    if ext == ".pdf":
        try:
            print(f"[Ingest] Parsing PDF with layout-aware unified parser: {file_path}")
            embed_fn = None
            if _embed_model is not None:
                embed_fn = lambda texts: _embed_model.encode(texts, show_progress_bar=False)
            
            cfg = None
            try:
                from engine.db import get_org_config
                cfg_data = get_org_config(org_id)
                if cfg_data:
                    cfg = cfg_data.get("config")
            except Exception:
                pass

            blocks = extract_pdf(file_path, file_id)
            all_chunks = build_chunks(blocks, embed_fn, org_id, file_id, config=cfg)
            experts_used = list(set(c.expert_id for c in all_chunks))
        except Exception as e:
            print(f"[Ingest] [WARN] Layout-aware unified parser failed: {e}. Falling back to modular experts.")
            all_chunks = []
            experts_used = []

    if not all_chunks:
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

    from engine.experts.code import CODE_EXTENSIONS

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
def get_file(file_id: str):
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


async def _run_retrieval_pipeline(req: QueryRequest, loop, start: float):
    # Bypass gate routing model execution completely
    gate_result = {"text": 0.25, "table": 0.25, "image": 0.25, "code": 0.25}
    gate_raw = {"text": 0.25, "table": 0.25, "image": 0.25, "code": 0.25}

    final_query = req.query
    max_gate_conf = 0.25

    t_gate = time.time()
    print(f"[Query] Gate (bypassed, mock equal weights): {int((t_gate-start)*1000)}ms (conf={max_gate_conf:.2f})")

    query_vec = await loop.run_in_executor(_cpu_pool, _cached_embed_query, final_query)

    t_embed = time.time()
    print(f"[Query] Embed: {int((t_embed-t_gate)*1000)}ms")

    # Unified Flat RAG: Query all chunks globally in parallel
    from engine.db import search_chunks_all, search_bm25_all
    
    dense_task = loop.run_in_executor(_io_pool,
        search_chunks_all, query_vec, req.org_id, 40, req.file_ids)
    sparse_task = loop.run_in_executor(_io_pool,
        search_bm25_all, final_query, req.org_id, 20, req.file_ids)

    dense_chunks, sparse_chunks = await asyncio.gather(dense_task, sparse_task)

    t_retrieve = time.time()
    print(f"[Query] Flat retrieve: {int((t_retrieve-t_embed)*1000)}ms (dense={len(dense_chunks or [])}, sparse={len(sparse_chunks or [])})")

    # Perform RRF fusion on global lists with equal weights (weights=None)
    if dense_chunks or sparse_chunks:
        fused_chunks = rrf_fuse(
            {"global_dense": dense_chunks, "global_sparse": sparse_chunks},
            weights=None,
            top_n=40
        )
    else:
        fused_chunks = []

    # Cross-Encoder reranking without modality boosts (gate_weights=None)
    if fused_chunks:
        fused_chunks = await loop.run_in_executor(_cpu_pool,
            rerank, final_query, fused_chunks, 12, None)

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

    # Extract unique base64 images from retrieved metadata
    image_b64s = []
    for chunk in fused_chunks:
        b64 = chunk.metadata.get("image_b64")
        if b64 and b64 not in image_b64s:
            image_b64s.append(b64)
            if len(image_b64s) >= 5:
                break

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

    if req.model:
        answer_model = req.model
    else:
        # Default to correct provider based on TESTING mode
        answer_model = CASCADE_SMALL_MODEL if TESTING else CASCADE_BIG_MODEL

    return full_prompt, answer_model, gate_result, sources, fused_chunks, t_rerank, image_b64s

@app.post("/query")
async def query_pipeline(req: QueryRequest):
    start = time.time()

    _hist_hash = hashlib.md5(str(req.chat_history).encode()).hexdigest()[:8]
    cache_key = f"{req.org_id}:{req.model}:{_hist_hash}:{hashlib.md5(req.query.encode()).hexdigest()}"
    if cache_key in _query_cache:
        cached = _query_cache[cache_key]
        cached["cached"] = True
        return cached

    loop = asyncio.get_running_loop()
    
    full_prompt, answer_model, gate_result, sources, fused_chunks, t_rerank, image_b64s = await _run_retrieval_pipeline(req, loop, start)

    # Record which experts were fired for this query (used in logging and response)
    fired_experts = list(gate_result.keys()) if gate_result else []

    answer = await _generate_answer(full_prompt, req.query, answer_model, image_b64s)

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

    _query_cache[cache_key] = result

    return result


async def _generate_answer(prompt: str, query: str, model: Optional[str] = None, image_b64s: Optional[list[str]] = None) -> str:
    provider, api_name = _resolve_model(model)
    loop = asyncio.get_event_loop()
    if provider == "ollama":
        return await loop.run_in_executor(_io_pool, _generate_ollama, prompt, api_name, image_b64s)
    elif provider == "gemini":
        return await loop.run_in_executor(_io_pool, _generate_gemini, prompt, api_name, image_b64s)
    else:
        return await loop.run_in_executor(_io_pool, _generate_groq, prompt, api_name)


def _generate_ollama(prompt: str, model: Optional[str] = None, image_b64s: Optional[list[str]] = None) -> str:
    try:
        payload = {
            "model": model or OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,   # evict from VRAM immediately after response
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
            }
        }
        if image_b64s:
            payload["images"] = image_b64s

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
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
        if 'response' in locals() and hasattr(response, 'text'):
            return f"[LLM Error] Failed to generate answer: {e} | Response: {response.text}"
        return f"[LLM Error] Failed to generate answer: {e}"


def _stream_ollama(prompt: str, model: Optional[str] = None, image_b64s: Optional[list[str]] = None):
    try:
        payload = {
            "model": model or OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": 0,   # evict from VRAM after stream completes
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
            },
        }
        if image_b64s:
            payload["images"] = image_b64s

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
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


def _generate_gemini(prompt: str, model: str = "gemma-3-27b-it", image_b64s: Optional[list[str]] = None) -> str:
    try:
        url = f"{GEMINI_BASE_URL}/models/{model}:generateContent?key={GEMINI_API_KEY}"
        parts = [{"text": prompt}]
        if image_b64s:
            for b64 in image_b64s:
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
        response = requests.post(
            url,
            json={
                "contents": [{"parts": parts}],
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


def _stream_gemini(prompt: str, model: str = "gemma-3-27b-it", image_b64s: Optional[list[str]] = None):
    try:
        url = f"{GEMINI_BASE_URL}/models/{model}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
        parts = [{"text": prompt}]
        if image_b64s:
            for b64 in image_b64s:
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
        response = requests.post(
            url,
            json={
                "contents": [{"parts": parts}],
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
    loop = asyncio.get_running_loop()
    
    full_prompt, answer_model, gate_result, sources, fused_chunks, t_rerank, image_b64s = await _run_retrieval_pipeline(req, loop, start)

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

    active_experts = list(gate_result.keys())

    def sse_generator():
        yield f"data: {json.dumps({'type': 'meta', 'gate': gate_result, 'sources': sources, 'active_experts': active_experts, 'rewritten_query': None, 'query_log_id': query_log_id, 'model_used': answer_model})}\n\n"

        provider, api_name = _resolve_model(answer_model)

        if provider == "ollama":
            for token in _stream_ollama(full_prompt, api_name, image_b64s):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        elif provider == "gemini":
            for token in _stream_gemini(full_prompt, api_name, image_b64s):
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
def get_org_config(org_id: str):
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
def update_org_config(org_id: str, req: OrgConfigUpdate):
    try:
        from engine.db import update_org_config
        update_org_config(org_id, req.name, req.config)
        return {"status": "ok", "org_id": org_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _ingest_background(file_path: str, file_id: str, org_id: str, ext: str):
    from engine.db import tenant_context
    token = tenant_context.set(org_id)
    try:
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

            all_chunks = []
            experts_used = []

            if ext == ".pdf":
                try:
                    print(f"[IngestBackground] Running unified PDF layout parser on {file_path}")
                    embed_fn = None
                    if _embed_model is not None:
                        embed_fn = lambda texts: _embed_model.encode(texts, show_progress_bar=False)
                    
                    cfg = None
                    try:
                        from engine.db import get_org_config
                        cfg_data = get_org_config(org_id)
                        if cfg_data:
                            cfg = cfg_data.get("config")
                    except Exception:
                        pass

                    blocks = extract_pdf(file_path, file_id)
                    all_chunks = build_chunks(blocks, embed_fn, org_id, file_id, config=cfg)
                    experts_used = list(set(c.expert_id for c in all_chunks))
                    
                    for expert_id in ["text", "table", "image", "code"]:
                        cnt = sum(1 for c in all_chunks if c.expert_id == expert_id)
                        if cnt > 0:
                            expert_status[expert_id] = {"state": "done", "chunks": cnt}
                            expert_names.append(expert_id)
                    
                    _update("parsing", 45)
                except Exception as e:
                    print(f"[IngestBackground] [WARN] Unified PDF layout parser failed: {e}. Falling back to modular experts.")
                    all_chunks = []
                    experts_used = []

            if not all_chunks:
                parse_tasks = {}
                if ext in [".pdf", ".txt", ".md"] and "text" in _experts:
                    parse_tasks["text"] = _parse_pool.submit(
                        _experts["text"].parse, file_path, file_id, org_id)
                    expert_names.append("text")
                    expert_status["text"] = {"state": "running", "chunks": 0}

                if (ext in [".csv"] or ext == ".pdf") and "table" in _experts:
                    parse_tasks["table"] = _parse_pool.submit(
                        _experts["table"].parse, file_path, file_id, org_id)
                    expert_names.append("table")
                    expert_status["table"] = {"state": "running", "chunks": 0}

                if (ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"] or ext == ".pdf") and "image" in _experts:
                    parse_tasks["image"] = _parse_pool.submit(
                        _experts["image"].parse, file_path, file_id, org_id)
                    expert_names.append("image")
                    expert_status["image"] = {"state": "running", "chunks": 0}

                _update("parsing", 5)

                total_tasks = len(parse_tasks)
                if total_tasks > 0:
                    future_to_expert = {v: k for k, v in parse_tasks.items()}
                    for future in as_completed(parse_tasks.values()):
                        expert_id = future_to_expert[future]
                        try:
                            chunks = future.result(timeout=600)
                            count = len(chunks) if chunks else 0
                            if chunks:
                                all_chunks.extend(chunks)
                                if expert_id not in experts_used:
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
    finally:
        tenant_context.reset(token)


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

    _ingest_pool.submit(_ingest_background, file_path, file_id, org_id, ext)

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
def pipeline_health(org_id: Optional[str] = None):
    health = get_pipeline_health(org_id)
    health["retrain_recommended"] = should_retrain_gate()
    return health


class GuardRequest(BaseModel):
    answer: str
    sources: list[str]

@app.post("/guard")
def guard_endpoint(req: GuardRequest):
    result = verify_answer(req.answer, req.sources)
    return result


class BM25Request(BaseModel):
    query: str
    expert_id: str
    org_id: str = "default"
    top_k: int = 5
    file_ids: Optional[list[str]] = None

@app.post("/retrieve/bm25")
def retrieve_bm25(req: BM25Request):
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
def get_org_files(org_id: str):
    """Get all files for an org."""
    try:
        from engine.db import get_files_by_org
        files = get_files_by_org(org_id)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{org_id}/{file_id}")
def delete_org_file(org_id: str, file_id: str):
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
class ChatSessionCreate(BaseModel):
    session_id: str
    org_id: str
    title: str

class ChatMessageCreate(BaseModel):
    message_id: str
    role: str
    content: str
    sources: Optional[list] = []

@app.get("/chat/sessions/{org_id}")
def get_sessions(org_id: str):
    try:
        from engine.db import get_chat_sessions
        return get_chat_sessions(org_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/sessions")
def create_session(data: ChatSessionCreate):
    try:
        from engine.db import create_chat_session
        create_chat_session(data.session_id, data.org_id, data.title)
        return {"status": "success", "session_id": data.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/sessions/{org_id}/{session_id}")
def delete_session(org_id: str, session_id: str):
    try:
        from engine.db import delete_chat_session
        delete_chat_session(session_id, org_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/{session_id}/messages")
def get_messages(session_id: str):
    try:
        from engine.db import get_chat_messages
        return get_chat_messages(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions/{session_id}/owner")
def get_session_owner(session_id: str):
    try:
        from engine.db import get_session_org
        owner = get_session_org(session_id)
        return {"org_id": owner}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/logout")
def logout():
    try:
        from engine.db import delete_all_chat_sessions, tenant_context
        org_id = tenant_context.get()
        deleted = delete_all_chat_sessions(org_id)
        return {"deleted_sessions": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/sessions/{session_id}/messages")
def add_message(session_id: str, data: ChatMessageCreate):
    try:
        from engine.db import add_chat_message
        add_chat_message(data.message_id, session_id, data.role, data.content, data.sources)
        return {"status": "success", "message_id": data.message_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "engine.main:app", host="0.0.0.0", port=8000, reload=True,
        reload_excludes=["scripts/*", "tests/*", "client/*", "server/*", "data/*", "uploads/*"]
    )