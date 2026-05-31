"""
PolyRAG global configuration.

TESTING = True  → use Ollama (local) for all LLM calls
TESTING = False → use Groq API for LLM calls
"""
from dotenv import load_dotenv
import os
load_dotenv()
# ──────────────────────────── Global Mode ────────────────────────────
TESTING = False

# ──────────────────────────── Ollama (local) ─────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_VISION_MODEL = "llava:latest"

# ──────────────────────────── Groq (cloud) ───────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ──────────────────────────── Gemini (cloud) ─────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# ──────────────────────────── Model Registry ─────────────────────────
MODEL_REGISTRY = {
    # ── Local (Ollama) ──
    "qwen2.5:7b-instruct-q4_K_M": {
        "provider": "ollama",
        "api_name": "qwen2.5:7b-instruct-q4_K_M",
        "display": "Qwen 2.5 7B Instruct (Local) 🧠",
        "group": "Local (Ollama)",
        "caps": ["text"],
    },
    "llama3.2:3b": {
        "provider": "ollama",
        "api_name": "llama3.2:3b",
        "display": "Llama 3.2 3B",
        "group": "Local (Ollama)",
        "caps": ["text"],
    },
    "gemma3:4b": {
        "provider": "ollama",
        "api_name": "gemma3:4b",
        "display": "Gemma 3 4B",
        "group": "Local (Ollama)",
        "caps": ["text"],
    },
    "gemma3:12b": {
        "provider": "ollama",
        "api_name": "gemma3:12b",
        "display": "Gemma 3 12B",
        "group": "Local (Ollama)",
        "caps": ["text"],
    },
    "llava:latest": {
        "provider": "ollama",
        "api_name": "llava:latest",
        "display": "LLaVA (Vision)",
        "group": "Local (Ollama)",
        "caps": ["vision"],
    },
    # ── Cloud (Groq) ──
    "deepseek-r1-distill-llama-70b": {
        "provider": "groq",
        "api_name": "deepseek-r1-distill-llama-70b",
        "display": "DeepSeek R1 70B (Groq) 💭",
        "group": "Cloud (Groq)",
        "caps": ["text"],
    },
    "llama-3.3-70b-specdec": {
        "provider": "groq",
        "api_name": "llama-3.3-70b-versatile",
        "display": "Llama 3.3 70B SpecDec (Redirected to Versatile) ⚡",
        "group": "Cloud (Groq)",
        "caps": ["text"],
    },
    "llama-3.1-70b-versatile": {
        "provider": "groq",
        "api_name": "llama-3.3-70b-versatile",
        "display": "Llama 3.1 70B (Redirected to 3.3)",
        "group": "Cloud (Groq)",
        "caps": ["text"],
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "api_name": "llama-3.3-70b-versatile",
        "display": "Llama 3.3 70B Versatile ⚡",
        "group": "Cloud (Groq)",
        "caps": ["text"],
    },
    "gemma2-9b-it": {
        "provider": "groq",
        "api_name": "gemma2-9b-it",
        "display": "Gemma 2 9B",
        "group": "Cloud (Groq)",
        "caps": ["text"],
    },
    "mixtral-8x7b-32768": {
        "provider": "groq",
        "api_name": "mixtral-8x7b-32768",
        "display": "Mixtral 8x7B",
        "group": "Cloud (Groq)",
        "caps": ["text"],
    },
    # ── Cloud (Gemini) ──
    "gemini-2.0-pro-exp": {
        "provider": "gemini",
        "api_name": "gemini-2.0-pro-exp-02-05",
        "display": "Gemini 2.0 Pro Exp 🚀",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemini-2.0-flash-thinking": {
        "provider": "gemini",
        "api_name": "gemini-2.0-flash-thinking-exp-01-21",
        "display": "Gemini 2.0 Flash Thinking 🧠",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "api_name": "gemini-2.5-flash",
        "display": "Gemini 2.5 Flash ⚡",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "api_name": "gemini-2.5-pro",
        "display": "Gemini 2.5 Pro",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemini-2.0-flash": {
        "provider": "gemini",
        "api_name": "gemini-2.0-flash",
        "display": "Gemini 2.0 Flash",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemini-1.5-pro": {
        "provider": "gemini",
        "api_name": "gemini-1.5-pro",
        "display": "Gemini 1.5 Pro (2M Context)",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemini-1.5-flash": {
        "provider": "gemini",
        "api_name": "gemini-1.5-flash",
        "display": "Gemini 1.5 Flash",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemma-4-27b": {
        "provider": "gemini",
        "api_name": "gemma-4-26b-a4b-it",
        "display": "Gemma 4 27B (MoE)",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
    "gemma-4-31b": {
        "provider": "gemini",
        "api_name": "gemma-4-31b-it",
        "display": "Gemma 4 31B (Dense)",
        "group": "Cloud (Gemini)",
        "caps": ["text", "vision"],
    },
}

# ──────────────────────────── Embedding ──────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024       # BGE-M3 actually outputs 1024-dim

# ──────────────────────────── Database ───────────────────────────────
# For testing: local PostgreSQL. For prod: Supabase.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:tan69@localhost:5433/polyrag")

# ──────────────────────────── Retrieval defaults ─────────────────────
DEFAULT_TOP_K = 30
GATE_THRESHOLD = 0.4       # experts with score > this are fired
CHUNK_SIZE = 512            # tokens per chunk
CHUNK_OVERLAP = 64          # token overlap between chunks

# ──────────────────────────── Expert registry ────────────────────────
EXPERT_IDS = ["text", "table", "image", "code"]

if TESTING:
    EXPERT_MODEL_MAP = {
        "text":  {"provider": "ollama", "model": "llama3.2:3b"},
        "table": {"provider": "ollama", "model": "llama3.2:3b"},   # good at structured reasoning
        "image": {"provider": "ollama", "model": "llava:latest"},   # vision model
        "code":  {"provider": "ollama", "model": "llama3.2:3b"},   # swap to codellama if you have it
    }
else:
    EXPERT_MODEL_MAP = {
        "text":  {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "table": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "image": {"provider": "gemini", "model": "gemini-2.5-flash"},  # Gemini has excellent vision capability
        "code":  {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    }

# Cascade thresholds
CASCADE_THRESHOLD   = 0.45   # gate max_conf below this → escalate
CASCADE_SMALL_MODEL = "llama3.2:3b"          # local fast pass
CASCADE_BIG_MODEL   = "llama-3.3-70b-versatile"  # groq escalation

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
