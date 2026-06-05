"""
PostgreSQL + pgvector database layer.
Schema from v4 notebook + file/chat/org tables for server compatibility.
"""
import json
import pickle
import psycopg2
from psycopg2.extras import execute_values, Json, RealDictCursor
from typing import List, Dict, Optional, Any
import numpy as np

from engine_v4.config import CFG

from psycopg2.pool import ThreadedConnectionPool

# Threaded connection pool for resource reuse and preventing leaks
_pool = None

class PooledConnectionWrapper:
    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        if self._conn is not None:
            try:
                self._pool.putconn(self._conn)
            except Exception:
                pass
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            if exc_type is not None:
                self._conn.rollback()
            else:
                self._conn.commit()
            self.close()

class DirectConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            if exc_type is not None:
                self._conn.rollback()
            else:
                self._conn.commit()
            self.close()

def get_conn():
    global _pool
    if _pool is None:
        try:
            # Keep pool limits conservative for free tier poolers (limit 15)
            _pool = ThreadedConnectionPool(1, 4, CFG.pg_conn)
            print("[DB] Threaded connection pool initialized successfully.")
        except Exception as e:
            print(f"[DB] Error initializing connection pool: {e}")
            return DirectConnectionWrapper(psycopg2.connect(CFG.pg_conn))
    try:
        return PooledConnectionWrapper(_pool, _pool.getconn())
    except Exception as e:
        print(f"[DB] Connection pool acquisition failed: {e}. Falling back to direct connection.")
        return DirectConnectionWrapper(psycopg2.connect(CFG.pg_conn))


# ── Schema ───────────────────────────────────────────────────────────────────

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- v4 core tables
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    section_id  INTEGER NOT NULL DEFAULT 0,
    modality    TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    org_id      TEXT NOT NULL DEFAULT 'default',
    file_id     TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc      ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_modality ON chunks(modality);
CREATE INDEX IF NOT EXISTS idx_chunks_org      ON chunks(org_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file     ON chunks(file_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding   vector(1024)
);
CREATE INDEX IF NOT EXISTS idx_embed_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m=32, ef_construction=200);

CREATE TABLE IF NOT EXISTS bm25_store (
    modality    TEXT NOT NULL,
    org_id      TEXT NOT NULL DEFAULT 'default',
    chunk_ids   TEXT[],
    index_blob  BYTEA NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (modality, org_id)
);

-- File tracking (server compat)
CREATE TABLE IF NOT EXISTS files (
    file_id     TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL DEFAULT 'default',
    filename    TEXT NOT NULL,
    file_type   TEXT DEFAULT '',
    status      TEXT DEFAULT 'processing',
    chunk_count INTEGER DEFAULT 0,
    error       TEXT DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_files_org ON files(org_id);

-- Org config (server compat)
CREATE TABLE IF NOT EXISTS orgs (
    org_id      TEXT PRIMARY KEY,
    name        TEXT DEFAULT '',
    config      JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Chat sessions (server compat)
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL DEFAULT 'default',
    title       TEXT DEFAULT 'New Chat',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_org ON chat_sessions(org_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    message_id  TEXT NOT NULL,
    org_id      TEXT NOT NULL DEFAULT 'default',
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    sources     JSONB DEFAULT '[]',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chatmsg_session ON chat_messages(session_id);
"""



def init_db():
    conn = get_conn()
    with conn.cursor() as cur:
        # Execute each statement separately (psycopg2 needs this for mixed DDL)
        statements = [s.strip() for s in DDL.split(";") if s.strip()]
        for stmt in statements:
            try:
                cur.execute(stmt)
                conn.commit()
            except Exception as e:
                conn.rollback()
                # If table exists with different schema, migrate it
                err_msg = str(e).lower()
                if "already exists" in err_msg or "duplicate" in err_msg:
                    continue
                # Try adding missing columns for existing tables
                if "column" in err_msg and "does not exist" in err_msg:
                    print(f"[DB] Schema migration needed, will handle: {e}")
                    continue
                print(f"[DB] DDL warning: {e}")

    # Migrate existing tables: add missing columns
    migrations = [
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS org_id TEXT NOT NULL DEFAULT 'default'",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS file_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section_id INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE bm25_store ADD COLUMN IF NOT EXISTS org_id TEXT NOT NULL DEFAULT 'default'",
        "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS error TEXT DEFAULT ''",
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS org_id TEXT NOT NULL DEFAULT 'default'",
        "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS sources JSONB DEFAULT '[]'",
    ]
    with conn.cursor() as cur:
        for sql in migrations:
            try:
                cur.execute(sql)
                conn.commit()
            except Exception:
                conn.rollback()

    # Migrate bm25_store: old engine uses PK(modality), v4 needs PK(modality, org_id)
    with conn.cursor() as cur:
        try:
            cur.execute("""
                SELECT array_agg(a.attname ORDER BY x.ordinality)
                FROM pg_constraint con
                JOIN pg_class rel ON rel.oid = con.conrelid
                CROSS JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS x(attnum, ordinality)
                JOIN pg_attribute a ON a.attrelid = rel.oid AND a.attnum = x.attnum
                WHERE rel.relname = 'bm25_store' AND con.contype = 'p'
                GROUP BY con.conname
            """)
            row = cur.fetchone()
            if row and row[0] == ['modality']:
                # Old PK: just (modality). Upgrade to composite (modality, org_id).
                cur.execute("ALTER TABLE bm25_store DROP CONSTRAINT bm25_store_pkey")
                cur.execute("ALTER TABLE bm25_store ADD PRIMARY KEY (modality, org_id)")
                conn.commit()
                print("[DB] Migrated bm25_store PK: (modality) -> (modality, org_id)")
            else:
                conn.rollback()
        except Exception as e:
            conn.rollback()
            # Table may not exist yet (first run), that's fine
            if "does not exist" not in str(e).lower():
                print(f"[DB] bm25_store PK migration skipped: {e}")

    # Upgrade idx_embed_hnsw if parameters are weaker than m=32, ef_construction=200
    with conn.cursor() as cur:
        try:
            cur.execute("""
                SELECT pg_get_indexdef(indexrelid)
                FROM pg_index
                WHERE indexrelid::regclass::text = 'idx_embed_hnsw'
            """)
            row = cur.fetchone()
            if row:
                def_str = row[0].lower()
                def_str_clean = def_str.replace("'", "").replace('"', "")
                if "m=32" not in def_str_clean or "ef_construction=200" not in def_str_clean:
                    print("[DB] Upgrading idx_embed_hnsw index to m=32, ef_construction=200...")
                    cur.execute("DROP INDEX IF EXISTS idx_embed_hnsw")
                    cur.execute("""
                        CREATE INDEX idx_embed_hnsw ON embeddings
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m=32, ef_construction=200)
                    """)
                    conn.commit()
                    print("[DB] Index idx_embed_hnsw successfully upgraded.")
        except Exception as e:
            conn.rollback()
            print(f"[DB] Failed checking/upgrading idx_embed_hnsw parameters: {e}")

    conn.close()
    print("[DB] Initialized.")



def ensure_org(org_id: str, name: str = ""):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO orgs (org_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (org_id, name),
            )
        conn.commit()


# ── Chunk CRUD ───────────────────────────────────────────────────────────────

def store_chunks(chunks, conn=None):
    """Upsert chunks into the chunks table."""
    own_conn = conn is None
    if own_conn:
        conn = get_conn()
    rows = [
        (ch.chunk_id, ch.doc_id, ch.section_id, ch.modality, ch.content,
         Json(ch.metadata), ch.org_id, ch.file_id)
        for ch in chunks
    ]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO chunks (chunk_id, doc_id, section_id, modality, content, metadata, org_id, file_id)
            VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                content  = EXCLUDED.content,
                metadata = EXCLUDED.metadata
        """, rows, page_size=500)
    conn.commit()
    if own_conn:
        conn.close()


def store_embeddings(chunks, conn=None):
    """Upsert embeddings into the embeddings table."""
    own_conn = conn is None
    if own_conn:
        conn = get_conn()
    rows = [
        (ch.chunk_id, ch.embedding.tolist())
        for ch in chunks if ch.embedding is not None
    ]
    if not rows:
        if own_conn:
            conn.close()
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO embeddings (chunk_id, embedding)
            VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding::vector
        """, rows, page_size=200)
    conn.commit()
    if own_conn:
        conn.close()


def store_bm25(chunks, org_id="default", conn=None):
    """Build and store BM25 indexes per modality (from v4 notebook)."""
    import re
    from rank_bm25 import BM25Okapi

    own_conn = conn is None
    if own_conn:
        conn = get_conn()

    def tokenize(text):
        return re.findall(r'[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*', text.lower())

    by_modality = {}
    for ch in chunks:
        by_modality.setdefault(ch.modality, []).append(ch)

    for modality, mod_chunks in by_modality.items():
        chunk_ids = [ch.chunk_id for ch in mod_chunks]
        tokenized = [tokenize(ch.content) for ch in mod_chunks]
        bm25 = BM25Okapi(tokenized)
        blob = pickle.dumps(bm25)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bm25_store (modality, org_id, chunk_ids, index_blob)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (modality, org_id) DO UPDATE SET
                    chunk_ids  = EXCLUDED.chunk_ids,
                    index_blob = EXCLUDED.index_blob,
                    updated_at = NOW()
            """, (modality, org_id, chunk_ids, psycopg2.Binary(blob)))
        conn.commit()
        print(f"  [DB] BM25 stored: {modality} ({len(chunk_ids)} chunks, {len(blob)/1024:.1f}KB)")


def rebuild_bm25(org_id="default"):
    """Rebuild BM25 index from scratch using all remaining chunks for the org."""
    chunks_data = load_chunks(org_id)
    all_chunks = list(chunks_data["chunk_lookup"].values())
    if not all_chunks:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM bm25_store WHERE org_id = %s", (org_id,))
            conn.commit()
        print(f"  [DB] Deleted BM25 indexes for org={org_id} (no chunks left)")
    else:
        # Delete any modalities that are no longer present for this org
        modalities_present = {ch.modality for ch in all_chunks}
        with get_conn() as conn:
            with conn.cursor() as cur:
                if modalities_present:
                    placeholders = ", ".join(["%s"] * len(modalities_present))
                    cur.execute(f"""
                        DELETE FROM bm25_store 
                        WHERE org_id = %s AND modality NOT IN ({placeholders})
                    """, [org_id] + list(modalities_present))
                else:
                    cur.execute("DELETE FROM bm25_store WHERE org_id = %s", (org_id,))
            conn.commit()
        store_bm25(all_chunks, org_id)


# ── Load from DB (for retrieval) ─────────────────────────────────────────────

def load_chunks(org_id="default") -> Dict[str, Any]:
    """Load all chunks from DB, grouped by modality."""
    from engine_v4.chunker import Chunk
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, doc_id, section_id, modality, content, metadata, org_id, file_id "
            "FROM chunks WHERE org_id = %s", (org_id,)
        )
        rows = cur.fetchall()
    conn.close()

    chunk_lookup = {}
    modal_chunks = {"text": [], "table": [], "image": []}
    for r in rows:
        ch = Chunk(
            chunk_id=r[0], doc_id=r[1], section_id=r[2],
            modality=r[3], content=r[4], metadata=r[5] or {},
            org_id=r[6], file_id=r[7], expert_id=r[3],
        )
        chunk_lookup[ch.chunk_id] = ch
        if ch.modality in modal_chunks:
            modal_chunks[ch.modality].append(ch)

    return {"chunk_lookup": chunk_lookup, "modal_chunks": modal_chunks}


def load_embeddings(org_id="default") -> Dict[str, np.ndarray]:
    """Load embeddings from DB, keyed by chunk_id."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.chunk_id, e.embedding::text
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.org_id = %s
        """, (org_id,))
        rows = cur.fetchall()
    conn.close()

    embeddings = {}
    for chunk_id, vec_text in rows:
        vec = np.array(json.loads(vec_text.replace("[", "[").replace("]", "]")), dtype=np.float32)
        embeddings[chunk_id] = vec
    return embeddings


def load_bm25(org_id="default") -> dict:
    """Load BM25 indexes and chunk_id lists from DB."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT modality, chunk_ids, index_blob FROM bm25_store WHERE org_id = %s",
            (org_id,),
        )
        rows = cur.fetchall()
    conn.close()

    bm25_indexes = {}
    bm25_chunk_ids = {}
    for modality, chunk_ids, blob in rows:
        bm25_indexes[modality] = pickle.loads(blob)
        bm25_chunk_ids[modality] = chunk_ids
    return {"indexes": bm25_indexes, "chunk_ids": bm25_chunk_ids}


# ── Dense search via pgvector ────────────────────────────────────────────────

def dense_search(query_vec: np.ndarray, modality: str, org_id: str = "default",
                 top_k: int = 30) -> List[dict]:
    """Cosine similarity search via pgvector <=> operator."""
    conn = get_conn()
    vec_str = "[" + ",".join(str(v) for v in query_vec.tolist()) + "]"
    with conn.cursor() as cur:
        try:
            cur.execute("SET hnsw.ef_search = 64")
        except Exception:
            pass
        cur.execute("""
            SELECT c.chunk_id, c.doc_id, c.section_id, c.modality,
                   c.content, c.metadata, c.org_id, c.file_id,
                   1 - (e.embedding <=> %s::vector) as similarity
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.modality = %s AND c.org_id = %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """, (vec_str, modality, org_id, vec_str, top_k))
        rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "chunk_id": r[0], "doc_id": r[1], "section_id": r[2],
            "modality": r[3], "content": r[4], "metadata": r[5] or {},
            "org_id": r[6], "file_id": r[7], "similarity": float(r[8]),
            "expert_id": r[3],
        })
    return results


# ── File CRUD ────────────────────────────────────────────────────────────────
# Note: old engine uses columns (name, type, experts_used) instead of (filename, file_type, error, updated_at).
# We detect schema at runtime and adapt.

_files_schema = None

def _detect_files_schema():
    global _files_schema
    if _files_schema is not None:
        return _files_schema
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'files'")
        cols = {r[0] for r in cur.fetchall()}
    conn.close()
    _files_schema = cols
    return cols


def create_file(file_id: str, org_id: str, filename: str, file_type: str = ""):
    cols = _detect_files_schema()
    conn = get_conn()
    with conn.cursor() as cur:
        if "filename" in cols:
            cur.execute(
                "INSERT INTO files (file_id, org_id, filename, file_type, status) VALUES (%s, %s, %s, %s, 'processing') "
                "ON CONFLICT (file_id) DO UPDATE SET status='processing'",
                (file_id, org_id, filename, file_type),
            )
        else:
            # Old schema: name, type
            cur.execute(
                "INSERT INTO files (file_id, org_id, name, type, status) VALUES (%s, %s, %s, %s, 'processing') "
                "ON CONFLICT (file_id) DO UPDATE SET status='processing'",
                (file_id, org_id, filename, file_type),
            )
    conn.commit()
    conn.close()


def update_file_status(file_id: str, status: str, chunk_count: int = 0, error: str = ""):
    cols = _detect_files_schema()
    conn = get_conn()
    with conn.cursor() as cur:
        if "error" in cols and "updated_at" in cols:
            cur.execute(
                "UPDATE files SET status=%s, chunk_count=%s, error=%s, updated_at=NOW() WHERE file_id=%s",
                (status, chunk_count, error, file_id),
            )
        else:
            cur.execute(
                "UPDATE files SET status=%s, chunk_count=%s WHERE file_id=%s",
                (status, chunk_count, file_id),
            )
    conn.commit()
    conn.close()


def _normalize_file(row_dict: dict) -> dict:
    """Normalize DB file row to match frontend expectations.
    Frontend uses: file.id, file.status === 'indexed', file.filename
    DB stores:     file_id, status = 'completed', name
    """
    d = dict(row_dict)
    # file_id → id (frontend uses file.id everywhere)
    if "file_id" in d and "id" not in d:
        d["id"] = d["file_id"]
    # name → filename
    if "name" in d and "filename" not in d:
        d["filename"] = d["name"]
    # completed → indexed (frontend checks file.status === 'indexed')
    if d.get("status") == "completed":
        d["status"] = "indexed"
    return d


def get_file(file_id: str) -> Optional[dict]:
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM files WHERE file_id = %s", (file_id,))
        row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return _normalize_file(row)


def get_org_files(org_id: str) -> List[dict]:
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM files WHERE org_id = %s ORDER BY created_at DESC", (org_id,))
        rows = cur.fetchall()
    conn.close()
    return [_normalize_file(r) for r in rows]


def delete_file_and_chunks(org_id: str, file_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Embeddings cascade-deleted via FK
            cur.execute("DELETE FROM chunks WHERE file_id = %s AND org_id = %s", (file_id, org_id))
            cur.execute("DELETE FROM files WHERE file_id = %s AND org_id = %s", (file_id, org_id))
        conn.commit()


# ── Org Config ───────────────────────────────────────────────────────────────

def get_org_config(org_id: str) -> Optional[dict]:
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM orgs WHERE org_id = %s", (org_id,))
        row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_org_config(org_id: str, name: str = "", config: dict = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO orgs (org_id, name, config) VALUES (%s, %s, %s) "
                "ON CONFLICT (org_id) DO UPDATE SET name=EXCLUDED.name, config=EXCLUDED.config",
                (org_id, name or org_id, Json(config or {})),
            )
        conn.commit()
    return {"status": "ok"}


# ── Chat CRUD ────────────────────────────────────────────────────────────────

def get_chat_sessions(org_id: str) -> List[dict]:
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM chat_sessions WHERE org_id = %s ORDER BY created_at DESC",
            (org_id,),
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_chat_session(session_id: str, org_id: str, title: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (session_id, org_id, title) VALUES (%s, %s, %s) "
                "ON CONFLICT (session_id) DO UPDATE SET title=EXCLUDED.title, updated_at=NOW()",
                (session_id, org_id, title),
            )
        conn.commit()
    return {"status": "ok", "session_id": session_id}


def delete_chat_session(org_id: str, session_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_sessions WHERE session_id = %s AND org_id = %s",
                (session_id, org_id),
            )
        conn.commit()
    return {"status": "ok"}


def get_chat_messages(session_id: str) -> List[dict]:
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY created_at",
            (session_id,),
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_chat_message(session_id: str, message_id: str, role: str, content: str,
                     sources: list = None, org_id: str = "default"):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (session_id, message_id, org_id, role, content, sources) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (session_id, message_id, org_id, role, content, Json(sources or [])),
            )
            try:
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE session_id = %s",
                    (session_id,),
                )
            except Exception:
                pass  # updated_at column may not exist in old schema
        conn.commit()
    return {"status": "ok"}


def get_session_owner(session_id: str) -> Optional[str]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT org_id FROM chat_sessions WHERE session_id = %s", (session_id,))
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def delete_all_chat_sessions(org_id: str) -> int:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT session_id FROM chat_sessions WHERE org_id = %s", (org_id,))
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            cur.execute("DELETE FROM chat_sessions WHERE org_id = %s", (org_id,))
    conn.commit()
    conn.close()
    return len(ids)
