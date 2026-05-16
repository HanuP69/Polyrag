import json
import os
import uuid
from typing import Optional
import numpy as np

try:
    from psycopg2.extras import execute_values
except ImportError:
    execute_values = None

from engine.config import TESTING, DATABASE_URL, EMBEDDING_DIM, DATA_DIR
from engine.experts.base import Chunk


SQLITE_PATH = os.path.join(DATA_DIR, "polyrag.db")

_sqlite_conn = None
_pg_pool = None


def _serialize_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def get_conn():
    if TESTING:
        return _get_sqlite()
    return _get_pg()


def _get_sqlite():
    import sqlite3
    global _sqlite_conn
    if _sqlite_conn is None:
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        _sqlite_conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        _sqlite_conn.row_factory = sqlite3.Row
        _sqlite_conn.execute("PRAGMA journal_mode=WAL;")
        _sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
    return _sqlite_conn


def _get_pg():
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = psycopg2.connect(DATABASE_URL)
        _pg_pool.autocommit = False
        with _pg_pool.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        _pg_pool.commit()
        register_vector(_pg_pool)
    return _pg_pool


def init_db():
    if TESTING:
        _init_sqlite()
    else:
        _init_pg()


def _init_sqlite():
    conn = _get_sqlite()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orgs (
            org_id      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            config      TEXT NOT NULL DEFAULT '{}'
        );
    """)
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
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_org_expert
        ON chunks (org_id, expert_id);
    """)
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id   TEXT PRIMARY KEY,
            query_log_id  TEXT REFERENCES query_logs(log_id),
            rating        INTEGER,
            correct_expert TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    print(f"[DB] [OK] SQLite schema initialized at {SQLITE_PATH}")


def _init_pg():
    conn = _get_pg()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orgs (
            org_id      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            config      JSONB NOT NULL DEFAULT '{}'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id     TEXT PRIMARY KEY,
            org_id      TEXT REFERENCES orgs(org_id),
            name        TEXT NOT NULL,
            type        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'uploading',
            chunk_count INTEGER DEFAULT 0,
            experts_used TEXT[] DEFAULT '{}',
            created_at  TIMESTAMPTZ DEFAULT now()
        )
    """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            org_id      TEXT REFERENCES orgs(org_id),
            file_id     TEXT REFERENCES files(file_id),
            expert_id   TEXT NOT NULL,
            content     TEXT NOT NULL,
            metadata    JSONB NOT NULL DEFAULT '{{}}',
            embedding   vector({EMBEDDING_DIM}),
            tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            created_at  TIMESTAMPTZ DEFAULT now()
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_org_expert
        ON chunks (org_id, expert_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_tsv
        ON chunks USING gin(tsv)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            log_id        TEXT PRIMARY KEY,
            org_id        TEXT REFERENCES orgs(org_id),
            query         TEXT NOT NULL,
            gate_weights  JSONB,
            experts_fired TEXT[],
            chunk_ids     TEXT[],
            latency_ms    INTEGER,
            created_at    TIMESTAMPTZ DEFAULT now()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id   TEXT PRIMARY KEY,
            query_log_id  TEXT REFERENCES query_logs(log_id),
            rating        INTEGER,
            correct_expert TEXT,
            created_at    TIMESTAMPTZ DEFAULT now()
        )
    """)
    conn.commit()
    print(f"[DB] [OK] pgvector schema initialized at {DATABASE_URL.split('@')[1]}")


def ensure_org(org_id: str, name: str = "default") -> str:
    conn = get_conn()
    if TESTING:
        conn.execute(
            "INSERT OR IGNORE INTO orgs (org_id, name) VALUES (?, ?)",
            (org_id, name)
        )
        conn.commit()
    else:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO orgs (org_id, name) VALUES (%s, %s) ON CONFLICT (org_id) DO NOTHING",
            (org_id, name)
        )
        conn.commit()
    return org_id


def get_org_config(org_id: str) -> Optional[dict]:
    conn = get_conn()
    if TESTING:
        row = conn.execute("SELECT name, config FROM orgs WHERE org_id = ?", (org_id,)).fetchone()
        if not row:
            return None
        return {"name": row["name"], "config": json.loads(row["config"])}
    else:
        cur = conn.cursor()
        cur.execute("SELECT name, config FROM orgs WHERE org_id = %s", (org_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"name": row[0], "config": row[1] if isinstance(row[1], dict) else json.loads(row[1])}


def update_org_config(org_id: str, name: str, config: dict):
    conn = get_conn()
    ensure_org(org_id)
    if TESTING:
        conn.execute(
            "UPDATE orgs SET name = ?, config = ? WHERE org_id = ?",
            (name, json.dumps(config), org_id)
        )
        conn.commit()
    else:
        cur = conn.cursor()
        cur.execute(
            "UPDATE orgs SET name = %s, config = %s WHERE org_id = %s",
            (name, json.dumps(config), org_id)
        )
        conn.commit()


def create_file(org_id: str, name: str, file_type: str) -> str:
    conn = get_conn()
    file_id = str(uuid.uuid4())
    if TESTING:
        conn.execute(
            "INSERT INTO files (file_id, org_id, name, type, status) VALUES (?, ?, ?, ?, 'uploading')",
            (file_id, org_id, name, file_type)
        )
        conn.commit()
    else:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO files (file_id, org_id, name, type, status) VALUES (%s, %s, %s, %s, 'uploading')",
            (file_id, org_id, name, file_type)
        )
        conn.commit()
    return file_id


def update_file_status(file_id: str, status: str, chunk_count: int = None, experts_used: list = None):
    conn = get_conn()
    if TESTING:
        updates = ["status = ?"]
        params = [status]
        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)
        if experts_used is not None:
            updates.append("experts_used = ?")
            params.append(json.dumps(experts_used))
        params.append(file_id)
        conn.execute(f"UPDATE files SET {', '.join(updates)} WHERE file_id = ?", params)
        conn.commit()
    else:
        cur = conn.cursor()
        updates = ["status = %s"]
        params = [status]
        if chunk_count is not None:
            updates.append("chunk_count = %s")
            params.append(chunk_count)
        if experts_used is not None:
            updates.append("experts_used = %s")
            params.append(experts_used)
        params.append(file_id)
        cur.execute(f"UPDATE files SET {', '.join(updates)} WHERE file_id = %s", params)
        conn.commit()


def get_file_status(file_id: str) -> Optional[dict]:
    conn = get_conn()
    if TESTING:
        row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return dict(row) if row else None
    else:
        cur = conn.cursor()
        cur.execute("SELECT file_id, org_id, name, type, status, chunk_count, experts_used, created_at FROM files WHERE file_id = %s", (file_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"file_id": row[0], "org_id": row[1], "name": row[2], "type": row[3], "status": row[4], "chunk_count": row[5], "experts_used": row[6], "created_at": str(row[7])}


def upsert_chunks(chunks: list[Chunk]):
    if not chunks:
        return

    conn = get_conn()
    inserted = 0

    valid_chunks = [c for c in chunks if c.embedding is not None]
    if not valid_chunks:
        print("[DB] [WARN] No valid chunks with embeddings to upsert")
        return

    if TESTING:
        data = [
            (c.chunk_id, c.org_id, c.file_id, c.expert_id, c.content, 
             json.dumps(c.metadata) if isinstance(c.metadata, dict) else c.metadata,
             _serialize_vector(c.embedding))
            for c in valid_chunks
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, org_id, file_id, expert_id, content, metadata, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            data
        )
        conn.commit()
        inserted = len(data)
    else:
        cur = conn.cursor()
        data = [
            (c.chunk_id, c.org_id, c.file_id, c.expert_id, c.content, 
             json.dumps(c.metadata) if isinstance(c.metadata, dict) else c.metadata,
             c.embedding.tolist())
            for c in valid_chunks
        ]
        
        if execute_values:
            execute_values(
                cur,
                """INSERT INTO chunks
                   (chunk_id, org_id, file_id, expert_id, content, metadata, embedding)
                   VALUES %s
                   ON CONFLICT (chunk_id) DO UPDATE SET
                   content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                data
            )
        else:
            # Fallback if execute_values is missing
            cur.executemany(
                """INSERT INTO chunks
                   (chunk_id, org_id, file_id, expert_id, content, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET
                   content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                data
            )
        conn.commit()
        inserted = len(data)

    print(f"[DB] [OK] Bulk upserted {inserted} chunks")


def search_chunks(
    query_vec: np.ndarray,
    org_id: str,
    expert_id: str,
    top_k: int = 10
) -> list[Chunk]:
    conn = get_conn()

    if TESTING:
        return _search_sqlite(conn, query_vec, org_id, expert_id, top_k)
    else:
        return _search_pgvector(conn, query_vec, org_id, expert_id, top_k)


def _search_sqlite(conn, query_vec, org_id, expert_id, top_k):
    rows = conn.execute(
        """SELECT chunk_id, org_id, file_id, expert_id, content, metadata, embedding
           FROM chunks WHERE org_id = ? AND expert_id = ?""",
        (org_id, expert_id)
    ).fetchall()
    if not rows:
        return []

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
            chunk_id=row["chunk_id"], org_id=row["org_id"], file_id=row["file_id"],
            expert_id=row["expert_id"], content=row["content"], metadata=metadata,
        )
        chunk.metadata["similarity"] = similarity
        results.append((similarity, chunk))

    results.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in results[:top_k]]


def _search_pgvector(conn, query_vec, org_id, expert_id, top_k):
    cur = conn.cursor()
    embedding_list = query_vec.tolist()
    cur.execute(
        """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                  1 - (embedding <=> %s::vector) as similarity
           FROM chunks
           WHERE org_id = %s AND expert_id = %s
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (embedding_list, org_id, expert_id, embedding_list, top_k)
    )
    rows = cur.fetchall()
    results = []
    for row in rows:
        metadata = row[5] if isinstance(row[5], dict) else json.loads(row[5])
        metadata["similarity"] = float(row[6])
        chunk = Chunk(
            chunk_id=row[0], org_id=row[1], file_id=row[2],
            expert_id=row[3], content=row[4], metadata=metadata,
        )
        results.append(chunk)
    return results


def search_bm25(
    query: str,
    org_id: str,
    expert_id: str,
    top_k: int = 5
) -> list[Chunk]:
    conn = get_conn()

    if TESTING:
        return _search_bm25_sqlite(conn, query, org_id, expert_id, top_k)
    else:
        return _search_bm25_pg(conn, query, org_id, expert_id, top_k)


def _search_bm25_sqlite(conn, query, org_id, expert_id, top_k):
    keywords = query.lower().split()
    if not keywords:
        return []
    rows = conn.execute(
        """SELECT chunk_id, org_id, file_id, expert_id, content, metadata
           FROM chunks WHERE org_id = ? AND expert_id = ?""",
        (org_id, expert_id)
    ).fetchall()
    if not rows:
        return []

    scored = []
    for row in rows:
        content_lower = row["content"].lower()
        score = sum(1 for kw in keywords if kw in content_lower) / len(keywords)
        if score > 0:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata["bm25_score"] = score
            chunk = Chunk(
                chunk_id=row["chunk_id"], org_id=row["org_id"], file_id=row["file_id"],
                expert_id=row["expert_id"], content=row["content"], metadata=metadata,
            )
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def _search_bm25_pg(conn, query, org_id, expert_id, top_k):
    cur = conn.cursor()
    cur.execute(
        """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                  ts_rank(tsv, plainto_tsquery('english', %s)) as rank
           FROM chunks
           WHERE org_id = %s AND expert_id = %s
             AND tsv @@ plainto_tsquery('english', %s)
           ORDER BY rank DESC
           LIMIT %s""",
        (query, org_id, expert_id, query, top_k)
    )
    rows = cur.fetchall()
    results = []
    for row in rows:
        metadata = row[5] if isinstance(row[5], dict) else json.loads(row[5])
        metadata["bm25_score"] = float(row[6])
        chunk = Chunk(
            chunk_id=row[0], org_id=row[1], file_id=row[2],
            expert_id=row[3], content=row[4], metadata=metadata,
        )
        results.append(chunk)
    return results


def log_query(
    org_id: str,
    query: str,
    gate_weights: dict,
    experts_fired: list[str],
    chunk_ids: list[str],
    latency_ms: int
):
    conn = get_conn()
    log_id = str(uuid.uuid4())
    if TESTING:
        conn.execute(
            """INSERT INTO query_logs (log_id, org_id, query, gate_weights, experts_fired, chunk_ids, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (log_id, org_id, query, json.dumps(gate_weights),
             json.dumps(experts_fired), json.dumps(chunk_ids), latency_ms)
        )
        conn.commit()
    else:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO query_logs (log_id, org_id, query, gate_weights, experts_fired, chunk_ids, latency_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (log_id, org_id, query, json.dumps(gate_weights),
             experts_fired, chunk_ids, latency_ms)
        )
        conn.commit()
    return log_id


def save_feedback(query_log_id: str, rating: int, correct_expert: str = None):
    conn = get_conn()
    feedback_id = str(uuid.uuid4())
    if TESTING:
        conn.execute(
            """INSERT INTO user_feedback (feedback_id, query_log_id, rating, correct_expert)
               VALUES (?, ?, ?, ?)""",
            (feedback_id, query_log_id, rating, correct_expert)
        )
        conn.commit()
    else:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO user_feedback (feedback_id, query_log_id, rating, correct_expert)
               VALUES (%s, %s, %s, %s)""",
            (feedback_id, query_log_id, rating, correct_expert)
        )
        conn.commit()
    return feedback_id


def get_chunk_count(org_id: str, expert_id: str = None) -> int:
    conn = get_conn()
    if TESTING:
        if expert_id:
            row = conn.execute("SELECT COUNT(*) as cnt FROM chunks WHERE org_id = ? AND expert_id = ?", (org_id, expert_id)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as cnt FROM chunks WHERE org_id = ?", (org_id,)).fetchone()
        return row["cnt"] if row else 0
    else:
        cur = conn.cursor()
        if expert_id:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE org_id = %s AND expert_id = %s", (org_id, expert_id))
        else:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE org_id = %s", (org_id,))
        return cur.fetchone()[0]


def delete_file_chunks(file_id: str):
    conn = get_conn()
    if TESTING:
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.commit()
    else:
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE file_id = %s", (file_id,))
        conn.commit()
