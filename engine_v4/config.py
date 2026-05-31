"""
PolyRAG v4 Configuration — mirrors the notebook Config dataclass.
"""
import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # ── Ollama ────────────────────────────────────────────────────────────
    ollama_base:    str = os.getenv("OLLAMA_BASE", "http://localhost:11434")
    caption_model:  str = os.getenv("CAPTION_MODEL", "llava:latest")
    text_model:     str = os.getenv("TEXT_MODEL", "qwen2.5:7b-instruct-q4_K_M")

    # ── Gemini (optional fallback) ────────────────────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model:   str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ── Local HF models ──────────────────────────────────────────────────
    embedder_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ── PostgreSQL ───────────────────────────────────────────────────────
    pg_conn: str = os.getenv(
        "PG_CONN", "postgresql://postgres:tan69@localhost:5433/polyrag"
    )

    # ── Devices ──────────────────────────────────────────────────────────
    embed_device:  str = os.getenv("EMBED_DEVICE", "cpu")
    rerank_device: str = os.getenv("RERANK_DEVICE", "cpu")
    embed_batch:   int = 32

    # ── Retrieval (from v4 notebook) ─────────────────────────────────────
    dense_top_k:  dict = field(default_factory=lambda: {"text": 30, "table": 30, "image": 20})
    sparse_top_k: dict = field(default_factory=lambda: {"text": 30, "table": 30, "image": 20})
    rrf_k:        int  = 60
    rerank_top_n: int  = 50
    final_top_k:  int  = 8

    # ── Features ─────────────────────────────────────────────────────────
    use_hyde:               bool = True
    skip_decorative_images: bool = True
    max_images_per_section: int  = 3

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")


CFG = Config()
