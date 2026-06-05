"""
PolyRAG v4 Configuration — loads from polyrag.config.json (single source of truth).
Falls back to env vars → defaults if config file is missing.
"""
import os
import json
from dataclasses import dataclass, field
from pathlib import Path

# ── Load config file ─────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "polyrag.config.json"

_file_cfg: dict = {}
if _CONFIG_PATH.exists():
    try:
        with open(_CONFIG_PATH, "r") as f:
            _file_cfg = json.load(f)
        print(f"[Config] Loaded from {_CONFIG_PATH}")
    except Exception as e:
        print(f"[Config] Warning: could not parse {_CONFIG_PATH}: {e}")
else:
    print(f"[Config] No config file at {_CONFIG_PATH}, using env/defaults")

# Also load .env for backward compat
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


def _get(section: str, key: str, env_var: str = "", default=""):
    """Resolve config value: env var → config file → default."""
    # 1. Environment variable
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)
    # 2. Config file (nested: section.key)
    section_data = _file_cfg.get(section, {})
    if key in section_data:
        return section_data[key]
    # 3. Default
    return default


@dataclass
class Config:
    # ── Ollama ────────────────────────────────────────────────────────────
    ollama_base:    str = _get("ollama", "base_url", "OLLAMA_BASE", "http://localhost:11434")
    caption_model:  str = _get("models", "caption_model", "CAPTION_MODEL", "llava:latest")
    text_model:     str = _get("models", "text_model", "TEXT_MODEL", "qwen2.5:7b-instruct-q4_K_M")

    # ── Gemini (optional cloud fallback) ──────────────────────────────────
    gemini_api_key: str = _get("cloud", "gemini_api_key", "GEMINI_API_KEY", "")
    gemini_model:   str = _get("cloud", "gemini_model", "GEMINI_MODEL", "gemini-2.5-flash")
    groq_api_key:   str = _get("cloud", "groq_api_key", "GROQ_API_KEY", "")

    # ── Local HF models ──────────────────────────────────────────────────
    embedder_model: str = _get("models", "embedder_model", "", "BAAI/bge-m3")
    reranker_model: str = _get("models", "reranker_model", "", "BAAI/bge-reranker-v2-m3")

    # ── PostgreSQL ───────────────────────────────────────────────────────
    pg_conn: str = _get("database", "pg_conn", "PG_CONN", "postgresql://postgres:tan69@localhost:5433/polyrag")

    # ── Devices ──────────────────────────────────────────────────────────
    embed_device:  str = _get("devices", "embed_device", "EMBED_DEVICE", "cpu")
    rerank_device: str = _get("devices", "rerank_device", "RERANK_DEVICE", "cpu")
    embed_batch:   int = int(_get("devices", "embed_batch_size", "", 32))

    # ── Retrieval (from v4 notebook) ─────────────────────────────────────
    dense_top_k:  dict = field(default_factory=lambda: _get("retrieval", "dense_top_k", "", {"text": 30, "table": 30, "image": 20}))
    sparse_top_k: dict = field(default_factory=lambda: _get("retrieval", "sparse_top_k", "", {"text": 30, "table": 30, "image": 20}))
    rrf_k:        int  = int(_get("retrieval", "rrf_k", "", 60))
    rerank_top_n: int  = int(_get("retrieval", "rerank_top_n", "", 50))
    final_top_k:  int  = int(_get("retrieval", "final_top_k", "", 8))

    # ── Features ─────────────────────────────────────────────────────────
    use_hyde:               bool = bool(_get("retrieval", "use_hyde", "", True))
    skip_decorative_images: bool = bool(_get("ingestion", "skip_decorative_images", "", True))
    max_images_per_section: int  = int(_get("ingestion", "max_images_per_section", "", 3))

    # ── Server ───────────────────────────────────────────────────────────
    host: str = _get("server", "engine_host", "", "0.0.0.0")
    port: int = int(_get("server", "engine_port", "", 8000))
    upload_dir: str = _get("ingestion", "upload_dir", "UPLOAD_DIR", "./data/uploads")


CFG = Config()
