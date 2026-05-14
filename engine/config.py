"""
PolyRAG global configuration.

TESTING = True  → use Ollama (local) for all LLM calls
TESTING = False → use Groq API for LLM calls
"""

# ──────────────────────────── Global Mode ────────────────────────────
TESTING = True

# ──────────────────────────── Ollama (testing) ───────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# ──────────────────────────── Groq (production) ──────────────────────
GROQ_API_KEY = ""          # set via env var in production
GROQ_MODEL = "llama-3.1-70b-versatile"

# ──────────────────────────── Embedding ──────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024       # BGE-M3 actually outputs 1024-dim

# ──────────────────────────── Database ───────────────────────────────
# For testing: local PostgreSQL. For prod: Supabase.
DATABASE_URL = "postgresql://postgres:tan69@localhost:5432/polyrag"

# ──────────────────────────── Retrieval defaults ─────────────────────
DEFAULT_TOP_K = 10
GATE_THRESHOLD = 0.4       # experts with score > this are fired
CHUNK_SIZE = 512            # tokens per chunk
CHUNK_OVERLAP = 64          # token overlap between chunks

# ──────────────────────────── Expert registry ────────────────────────
EXPERT_IDS = ["text", "table", "image"]

# ──────────────────────────── Paths ──────────────────────────────────
import os

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ENGINE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
GATE_MODEL_PATH = os.path.join(DATA_DIR, "gate_model.pt")
GATE_TRAINING_DATA_PATH = os.path.join(DATA_DIR, "gate_training.json")
UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
