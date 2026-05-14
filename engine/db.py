"""
db.py — Database helpers for PolyRAG.

TESTING mode: SQLite + numpy cosine similarity (zero infrastructure)
PRODUCTION mode: PostgreSQL + pgvector (Supabase)

All vector search is done via numpy cosine similarity in testing mode.
Chunks and embeddings are stored in SQLite with embeddings serialized as blobs.
"""

import json
import os
import sqlite3
import uuid
import struct
from typing import Optional
import numpy as np

from engine.config import TESTING, DATABASE_URL, EMBEDDING_DIM, DATA_DIR
from engine.experts.base import Chunk


# ──────────────────────────── SQLite Path ────────────────────────────
SQLITE_PATH = os.path.join(DATA_DIR, "polyrag.db")


# ──────────────────────────── Connection ─────────────────────────────
_connection = None


def _serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for SQLite storage."""
    return vec.astype(np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def get_conn() -> sqlite3.Connection:
    """Get or create SQLite connection."""
    global _connection
    if _connection is None:
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        _connection = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        _connection.execute("PRAGMA journal_mode=WAL;")
        _connection.execute("PRAGMA synchronous=NORMAL;")
    return _connection


# ──────────────────────────── Schema ─────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    
    # Organizations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orgs (
            org_id      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            config      TEXT NOT NULL DEFAULT '{}'
        );
    """)
    
    # Files
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id     TEXT PRIMARY KEY,
            org_id      TEXT REFERENCES orgs(org_id),
            name        TEXT NOT NULL,
            type        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'uploading',
            chunk_count INTEGER DEFAULT 0,
            experts_used TEXT DEFAULT '[]',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Chunks — embeddings stored as BLOBs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            org_id      TEXT REFERENCES orgs(org_id),
            file_id     TEXT REFERENCES files(file_id),
            expert_id   TEXT NOT NULL,
            content     TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}',
            embedding   BLOB,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Indexes for filtered retrieval
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_org_expert
        ON chunks (org_id, expert_id);
    """)
    
    # Query logs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            log_id        TEXT PRIMARY KEY,
            org_id        TEXT REFERENCES orgs(org_id),
            query         TEXT NOT NULL,
            gate_weights  TEXT,
            experts_fired TEXT,
            chunk_ids     TEXT,
            latency_ms    INTEGER,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    print(f"[DB] [OK] SQLite schema initialized at {SQLITE_PATH}")


# ──────────────────────────── Org Management ─────────────────────────
def ensure_org(org_id: str, name: str = "default") -> str:
    """Create org if not exists, return org_id."""
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO orgs (org_id, name) VALUES (?, ?)",
        (org_id, name)
    )
    conn.commit()
    return org_id


# ──────────────────────────── File Management ────────────────────────
def create_file(org_id: str, name: str, file_type: str) -> str:
    """Create a file record, return file_id."""
    conn = get_conn()
    file_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO files (file_id, org_id, name, type, status) VALUES (?, ?, ?, ?, 'uploading')",
        (file_id, org_id, name, file_type)
    )
    conn.commit()
    return file_id


def update_file_status(file_id: str, status: str, chunk_count: int = None, experts_used: list = None):
    """Update file processing status."""
    conn = get_conn()
    
    updates = ["status = ?"]
    params = [status]
    
    if chunk_count is not None:
        updates.append("chunk_count = ?")
        params.append(chunk_count)
    
    if experts_used is not None:
        updates.append("experts_used = ?")
        params.append(json.dumps(experts_used))
    
    params.append(file_id)
    conn.execute(
        f"UPDATE files SET {', '.join(updates)} WHERE file_id = ?",
        params
    )
    conn.commit()


def get_file_status(file_id: str) -> Optional[dict]:
    """Get file status."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


# ──────────────────────────── Chunk Operations ───────────────────────
def upsert_chunks(chunks: list[Chunk]):
    """
    Insert chunks with embeddings into SQLite.
    Embeddings are stored as BLOBs (serialized float32 arrays).
    """
    if not chunks:
        return
    
    conn = get_conn()
    
    inserted = 0
    for chunk in chunks:
        if chunk.embedding is None:
            print(f"[DB] WARNING: Chunk {chunk.chunk_id} has no embedding, skipping")
            continue
        
        embedding_blob = _serialize_vector(chunk.embedding)
        metadata_str = json.dumps(chunk.metadata) if isinstance(chunk.metadata, dict) else chunk.metadata
        
        conn.execute(
            """INSERT OR REPLACE INTO chunks 
               (chunk_id, org_id, file_id, expert_id, content, metadata, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id,
                chunk.org_id,
                chunk.file_id,
                chunk.expert_id,
                chunk.content,
                metadata_str,
                embedding_blob
            )
        )
        inserted += 1
    
    conn.commit()
    print(f"[DB] [OK] Upserted {inserted} chunks")


def search_chunks(
    query_vec: np.ndarray,
    org_id: str,
    expert_id: str,
    top_k: int = 10
) -> list[Chunk]:
    """
    Cosine similarity search using numpy.
    
    1. Load all embeddings for (org_id, expert_id) from SQLite
    2. Compute cosine similarity with numpy
    3. Return top_k results
    
    This is O(n) but fast enough for <100k chunks on a single machine.
    For production, use pgvector.
    """
    conn = get_conn()
    
    rows = conn.execute(
        """SELECT chunk_id, org_id, file_id, expert_id, content, metadata, embedding
           FROM chunks
           WHERE org_id = ? AND expert_id = ?""",
        (org_id, expert_id)
    ).fetchall()
    
    if not rows:
        return []
    
    # Deserialize embeddings and compute similarities
    query_vec = query_vec.astype(np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    query_vec_normalized = query_vec / query_norm
    
    results = []
    for row in rows:
        embedding = _deserialize_vector(row["embedding"])
        emb_norm = np.linalg.norm(embedding)
        if emb_norm == 0:
            continue
        
        similarity = float(np.dot(query_vec_normalized, embedding / emb_norm))
        
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        chunk = Chunk(
            chunk_id=row["chunk_id"],
            org_id=row["org_id"],
            file_id=row["file_id"],
            expert_id=row["expert_id"],
            content=row["content"],
            metadata=metadata,
        )
        chunk.metadata["similarity"] = similarity
        results.append((similarity, chunk))
    
    # Sort by similarity descending, return top_k
    results.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in results[:top_k]]


# ──────────────────────────── Query Logs ─────────────────────────────
def log_query(
    org_id: str,
    query: str,
    gate_weights: dict,
    experts_fired: list[str],
    chunk_ids: list[str],
    latency_ms: int
):
    """Log a query for future gate retraining."""
    conn = get_conn()
    conn.execute(
        """INSERT INTO query_logs (log_id, org_id, query, gate_weights, experts_fired, chunk_ids, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            org_id,
            query,
            json.dumps(gate_weights),
            json.dumps(experts_fired),
            json.dumps(chunk_ids),
            latency_ms
        )
    )
    conn.commit()


# ──────────────────────────── Utilities ──────────────────────────────
def get_chunk_count(org_id: str, expert_id: str = None) -> int:
    """Count chunks for an org, optionally filtered by expert."""
    conn = get_conn()
    
    if expert_id:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE org_id = ? AND expert_id = ?",
            (org_id, expert_id)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE org_id = ?",
            (org_id,)
        ).fetchone()
    
    return row["cnt"] if row else 0


def delete_file_chunks(file_id: str):
    """Delete all chunks for a file."""
    conn = get_conn()
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    conn.commit()
