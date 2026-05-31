import json
import os
import uuid
import threading
from typing import Optional
import numpy as np
import contextvars

try:
    from psycopg2.extras import execute_values
except ImportError:
    execute_values = None

from engine.config import TESTING, DATABASE_URL, EMBEDDING_DIM, DATA_DIR
from engine.experts.base import Chunk

# Context variable to hold the current request's tenant (org_id)
tenant_context = contextvars.ContextVar("tenant_context", default="default")

# In-memory connection pool caches to maintain high performance/concurrency
_sqlite_conns = {}      # db_path -> connection
_sqlite_lock = threading.Lock()
_sqlite_conn_locks = {}  # db_path -> threading.Lock()
_pg_pools = {}          # tenant db_name -> ThreadedConnectionPool
_pg_pools_lock = threading.Lock()
_conn_to_db_name = {}
_conn_to_db_name_lock = threading.Lock()


def _serialize_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _get_db_name_for_org(org_id: str) -> str:
    if org_id == "default":
        _, default_name = DATABASE_URL.rsplit('/', 1)
        return default_name
    clean_org = org_id.lower().replace('-', '_')
    return f"polyrag_user_{clean_org}"


def get_conn():
    if TESTING:
        return _get_sqlite()
    return _get_pg()


def _get_sqlite():
    import sqlite3
    global _sqlite_conns
    org_id = tenant_context.get()
    clean_org = org_id.lower().replace('-', '_')
    db_path = os.path.join(DATA_DIR, f"polyrag_{clean_org}.db")
    
    if db_path not in _sqlite_conns:
        with _sqlite_lock:
            if db_path not in _sqlite_conns:
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                _init_sqlite_schema(conn)
                _sqlite_conns[db_path] = conn
                # Create a per-database lock so different tenant DBs
                # don't contend on a single global lock.
                _sqlite_conn_locks[db_path] = threading.Lock()
    return _sqlite_conns[db_path]


def _get_sqlite_lock_for_conn(conn):
    """Return the per-database lock for a sqlite connection, falling
    back to the global _sqlite_lock if not found."""
    for path, c in _sqlite_conns.items():
        if c is conn:
            return _sqlite_conn_locks.get(path, _sqlite_lock)
    return _sqlite_lock


def _get_pg():
    """Borrow a connection from the tenant-specific connection pool."""
    import psycopg2
    from psycopg2 import pool as pg_pool
    from pgvector.psycopg2 import register_vector
    import time
    
    org_id = tenant_context.get()
    db_name = _get_db_name_for_org(org_id)
    
    if db_name not in _pg_pools:
        with _pg_pools_lock:
            if db_name not in _pg_pools:
                # Lazy provision the database & schema
                if db_name != _get_db_name_for_org("default"):
                    _init_tenant_db(db_name, org_id)
                else:
                    _init_default_db_schema_only()
                
                base_url, _ = DATABASE_URL.rsplit('/', 1)
                tenant_url = f"{base_url}/{db_name}"
                
                _pg_pools[db_name] = pg_pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=5,
                    dsn=tenant_url,
                )
                print(f"[DB] Connection pool created for tenant database: {db_name}")
                
    pool = _pg_pools[db_name]
    
    conn = None
    for _ in range(100): # Wait up to 10s
        try:
            conn = pool.getconn()
            if conn.closed:
                pool.putconn(conn, close=True)
                continue
            break
        except pg_pool.PoolError:
            time.sleep(0.1)
            
    if conn is None:
        raise Exception(f"Database connection pool exhausted for tenant database '{db_name}'.")
        
    # Some DB driver connection objects (psycopg2) don't allow setting
    # arbitrary attributes. Track the mapping separately using the connection id.
    with _conn_to_db_name_lock:
        _conn_to_db_name[id(conn)] = db_name
    
    try:
        conn.autocommit = False
        register_vector(conn)
    except Exception:
        pass
    return conn


def _return_pg(conn):
    """Return a borrowed connection back to its specific pool."""
    if conn is None:
        return

    db_name = None
    with _conn_to_db_name_lock:
        db_name = _conn_to_db_name.pop(id(conn), None)

    if db_name:
        pool = _pg_pools.get(db_name)
        if pool is not None:
            try:
                pool.putconn(conn)
            except Exception:
                pass


def init_db():
    if TESTING:
        token = tenant_context.set("default")
        try:
            _get_sqlite()
        finally:
            tenant_context.reset(token)
        print("[DB] [OK] SQLite default database initialized.")
    else:
        _init_default_db_schema_only()


def _init_sqlite_schema(conn):
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id  TEXT PRIMARY KEY,
            org_id      TEXT REFERENCES orgs(org_id),
            title       TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id  TEXT PRIMARY KEY,
            session_id  TEXT REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            sources     TEXT NOT NULL DEFAULT '[]',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()


def _init_default_db_schema_only():
    """Ensure the default database has the schema loaded."""
    import psycopg2
    base_url, default_db = DATABASE_URL.rsplit('/', 1)
    
    admin_conn = None
    try:
        admin_conn = psycopg2.connect(f"{base_url}/postgres")
        admin_conn.autocommit = True
        cur = admin_conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (default_db,))
        exists = cur.fetchone()
        if not exists:
            print(f"[DB] Creating default database {default_db}...")
            cur.execute(f'CREATE DATABASE "{default_db}"')
            
        default_conn = psycopg2.connect(DATABASE_URL)
        try:
            _init_tenant_db_schema(default_conn)
        finally:
            default_conn.close()
    except Exception as e:
        print(f"[DB] Error provisioning default database: {e}")
        raise e
    finally:
        if admin_conn:
            try:
                admin_conn.close()
            except Exception:
                pass


def _init_tenant_db(db_name: str, org_id: str):
    """Ensure the tenant database exists and has schema loaded."""
    import psycopg2
    base_url, _ = DATABASE_URL.rsplit('/', 1)
    
    admin_conn = None
    try:
        admin_conn = psycopg2.connect(f"{base_url}/postgres")
        admin_conn.autocommit = True
        cur = admin_conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        if not exists:
            print(f"[DB] Creating separate database {db_name} for tenant {org_id}...")
            cur.execute(f'CREATE DATABASE "{db_name}"')
            
            tenant_conn = psycopg2.connect(f"{base_url}/{db_name}")
            try:
                _init_tenant_db_schema(tenant_conn)
                print(f"[DB] Database {db_name} schema initialized.")
            finally:
                tenant_conn.close()
    except Exception as e:
        print(f"[DB] Error provisioning database {db_name}: {e}")
        raise e
    finally:
        if admin_conn:
            try:
                admin_conn.close()
            except Exception:
                pass


def _init_tenant_db_schema(conn):
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
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id  TEXT PRIMARY KEY,
            org_id      TEXT REFERENCES orgs(org_id),
            title       TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT now()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id  TEXT PRIMARY KEY,
            session_id  TEXT REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            sources     JSONB NOT NULL DEFAULT '[]',
            created_at  TIMESTAMPTZ DEFAULT now()
        )
    """)
    conn.commit()


def ensure_org(org_id: str, name: str = "default") -> str:
    conn = get_conn()
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(
                "INSERT OR IGNORE INTO orgs (org_id, name) VALUES (?, ?)",
                (org_id, name)
            )
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO orgs (org_id, name) VALUES (%s, %s) ON CONFLICT (org_id) DO NOTHING",
                (org_id, name)
            )
            conn.commit()
        finally:
            _return_pg(conn)
    return org_id


def get_org_config(org_id: str) -> Optional[dict]:
    conn = get_conn()
    if TESTING:
        row = conn.execute("SELECT name, config FROM orgs WHERE org_id = ?", (org_id,)).fetchone()
        if not row:
            return None
        return {"name": row["name"], "config": json.loads(row["config"])}
    else:
        try:
            cur = conn.cursor()
            cur.execute("SELECT name, config FROM orgs WHERE org_id = %s", (org_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {"name": row[0], "config": row[1] if isinstance(row[1], dict) else json.loads(row[1])}
        finally:
            _return_pg(conn)


def update_org_config(org_id: str, name: str, config: dict):
    ensure_org(org_id)
    conn = get_conn()
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(
                "UPDATE orgs SET name = ?, config = ? WHERE org_id = ?",
                (name, json.dumps(config), org_id)
            )
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE orgs SET name = %s, config = %s WHERE org_id = %s",
                (name, json.dumps(config), org_id)
            )
            conn.commit()
        finally:
            _return_pg(conn)


def get_files_by_org(org_id: str) -> list:
    conn = get_conn()
    if TESTING:
        rows = conn.execute(
            "SELECT file_id, name, type, status, chunk_count, experts_used, created_at FROM files WHERE org_id = ? ORDER BY created_at DESC",
            (org_id,)
        ).fetchall()
        return [
            {
                "id": r["file_id"],
                "name": r["name"],
                "type": r["type"],
                "status": r["status"],
                "total_chunks": r["chunk_count"],
                "experts_used": json.loads(r["experts_used"]) if r["experts_used"] else [],
                "created_at": r["created_at"]
            } for r in rows
        ]
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT file_id, name, type, status, chunk_count, experts_used, created_at FROM files WHERE org_id = %s ORDER BY created_at DESC",
                (org_id,)
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "type": r[2],
                    "status": r[3],
                    "total_chunks": r[4],
                    "experts_used": r[5] if isinstance(r[5], list) else [],
                    "created_at": r[6].isoformat() if hasattr(r[6], 'isoformat') else str(r[6])
                } for r in rows
            ]
        finally:
            _return_pg(conn)


def delete_file(org_id: str, file_id: str) -> bool:
    conn = get_conn()
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            row = conn.execute("SELECT file_id FROM files WHERE org_id = ? AND file_id = ?", (org_id, file_id)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
            conn.commit()
        return True
    else:
        try:
            cur = conn.cursor()
            cur.execute("SELECT file_id FROM files WHERE org_id = %s AND file_id = %s", (org_id, file_id))
            if not cur.fetchone():
                return False
            cur.execute("DELETE FROM chunks WHERE file_id = %s", (file_id,))
            cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            conn.commit()
            return True
        finally:
            _return_pg(conn)


def create_file(org_id: str, name: str, file_type: str) -> str:
    conn = get_conn()
    file_id = str(uuid.uuid4())
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(
                "INSERT INTO files (file_id, org_id, name, type, status) VALUES (?, ?, ?, ?, 'uploading')",
                (file_id, org_id, name, file_type)
            )
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO files (file_id, org_id, name, type, status) VALUES (%s, %s, %s, %s, 'uploading')",
                (file_id, org_id, name, file_type)
            )
            conn.commit()
        finally:
            _return_pg(conn)
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
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(f"UPDATE files SET {', '.join(updates)} WHERE file_id = ?", params)
            conn.commit()
    else:
        try:
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
        finally:
            _return_pg(conn)


def get_file_status(file_id: str) -> Optional[dict]:
    conn = get_conn()
    if TESTING:
        row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return dict(row) if row else None
    else:
        try:
            cur = conn.cursor()
            cur.execute("SELECT file_id, org_id, name, type, status, chunk_count, experts_used, created_at FROM files WHERE file_id = %s", (file_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {"file_id": row[0], "org_id": row[1], "name": row[2], "type": row[3], "status": row[4], "chunk_count": row[5], "experts_used": row[6], "created_at": str(row[7])}
        finally:
            _return_pg(conn)


def upsert_chunks(chunks: list[Chunk]):
    if not chunks:
        return

    valid_chunks = [c for c in chunks if c.embedding is not None]
    if not valid_chunks:
        print("[DB] [WARN] No valid chunks with embeddings to upsert")
        return

    conn = get_conn()

    if TESTING:
        data = [
            (c.chunk_id, c.org_id, c.file_id, c.expert_id, c.content,
             json.dumps(c.metadata) if isinstance(c.metadata, dict) else c.metadata,
             _serialize_vector(c.embedding))
            for c in valid_chunks
        ]
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, org_id, file_id, expert_id, content, metadata, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                data
            )
            conn.commit()
    else:
        try:
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
                cur.executemany(
                    """INSERT INTO chunks
                       (chunk_id, org_id, file_id, expert_id, content, metadata, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (chunk_id) DO UPDATE SET
                       content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                    data
                )
            conn.commit()
        finally:
            _return_pg(conn)

    print(f"[DB] [OK] Bulk upserted {len(valid_chunks)} chunks")


def search_chunks(
    query_vec: np.ndarray,
    org_id: str,
    expert_id: Optional[str] = None,
    top_k: int = 10,
    file_ids: list = None
) -> list[Chunk]:
    conn = get_conn()
    if TESTING:
        return _search_sqlite(conn, query_vec, org_id, expert_id, top_k, file_ids=file_ids)
    else:
        try:
            return _search_pgvector(conn, query_vec, org_id, expert_id, top_k, file_ids=file_ids)
        finally:
            _return_pg(conn)


def _search_sqlite(conn, query_vec, org_id, expert_id, top_k, file_ids=None):
    if file_ids:
        placeholders = ",".join("?" * len(file_ids))
        if expert_id:
            rows = conn.execute(
                f"""SELECT chunk_id, org_id, file_id, expert_id, content, metadata, embedding
                   FROM chunks WHERE org_id = ? AND expert_id = ? AND file_id IN ({placeholders})""",
                (org_id, expert_id, *file_ids)
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT chunk_id, org_id, file_id, expert_id, content, metadata, embedding
                   FROM chunks WHERE org_id = ? AND file_id IN ({placeholders})""",
                (org_id, *file_ids)
            ).fetchall()
    else:
        if expert_id:
            rows = conn.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata, embedding
                   FROM chunks WHERE org_id = ? AND expert_id = ?""",
                (org_id, expert_id)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata, embedding
                   FROM chunks WHERE org_id = ?""",
                (org_id,)
            ).fetchall()
    if not rows:
        return []

    query_vec = query_vec.astype(np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    query_vec_normalized = query_vec / query_norm

    chunk_list = []
    embeddings = []
    for row in rows:
        emb = _deserialize_vector(row["embedding"])
        if emb is None or len(emb) == 0:
            continue
        chunk_list.append(row)
        embeddings.append(emb)

    if not embeddings:
        return []

    emb_matrix = np.stack(embeddings).astype(np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    similarities = emb_matrix @ query_vec_normalized

    top_indices = np.argpartition(similarities, max(-top_k, -len(similarities)))[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        row = chunk_list[idx]
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        metadata["similarity"] = float(similarities[idx])
        chunk = Chunk(
            chunk_id=row["chunk_id"], org_id=row["org_id"], file_id=row["file_id"],
            expert_id=row["expert_id"], content=row["content"], metadata=metadata,
        )
        results.append(chunk)

    return results


def _search_pgvector(conn, query_vec, org_id, expert_id, top_k, file_ids=None):
    cur = conn.cursor()
    embedding_list = query_vec.tolist()
    if file_ids:
        if expert_id:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          1 - (embedding <=> %s::vector) as similarity
                   FROM chunks
                   WHERE org_id = %s AND expert_id = %s AND file_id = ANY(%s)
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (embedding_list, org_id, expert_id, file_ids, embedding_list, top_k)
            )
        else:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          1 - (embedding <=> %s::vector) as similarity
                   FROM chunks
                   WHERE org_id = %s AND file_id = ANY(%s)
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (embedding_list, org_id, file_ids, embedding_list, top_k)
            )
    else:
        if expert_id:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          1 - (embedding <=> %s::vector) as similarity
                   FROM chunks
                   WHERE org_id = %s AND expert_id = %s
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (embedding_list, org_id, expert_id, embedding_list, top_k)
            )
        else:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          1 - (embedding <=> %s::vector) as similarity
                   FROM chunks
                   WHERE org_id = %s
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (embedding_list, org_id, embedding_list, top_k)
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
    expert_id: Optional[str] = None,
    top_k: int = 5,
    file_ids: list = None
) -> list[Chunk]:
    conn = get_conn()
    if TESTING:
        return _search_bm25_sqlite(conn, query, org_id, expert_id, top_k, file_ids=file_ids)
    else:
        try:
            return _search_bm25_pg(conn, query, org_id, expert_id, top_k, file_ids=file_ids)
        finally:
            _return_pg(conn)


def _search_bm25_sqlite(conn, query, org_id, expert_id, top_k, file_ids=None):
    keywords = query.lower().split()
    if not keywords:
        return []

    if file_ids:
        placeholders = ",".join("?" * len(file_ids))
        if expert_id:
            rows = conn.execute(
                f"""SELECT chunk_id, org_id, file_id, expert_id, content, metadata
                   FROM chunks WHERE org_id = ? AND expert_id = ? AND file_id IN ({placeholders})""",
                (org_id, expert_id, *file_ids)
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT chunk_id, org_id, file_id, expert_id, content, metadata
                   FROM chunks WHERE org_id = ? AND file_id IN ({placeholders})""",
                (org_id, *file_ids)
            ).fetchall()
    else:
        if expert_id:
            rows = conn.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata
                   FROM chunks WHERE org_id = ? AND expert_id = ?""",
                (org_id, expert_id)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata
                   FROM chunks WHERE org_id = ?""",
                (org_id,)
            ).fetchall()
    if not rows:
        return []

    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [row["content"].lower().split() for row in rows]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(keywords)
    except ImportError:
        scores = _tfidf_scores(keywords, [row["content"] for row in rows])

    scored = []
    for i, row in enumerate(rows):
        if scores[i] > 0:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata["bm25_score"] = float(scores[i])
            chunk = Chunk(
                chunk_id=row["chunk_id"], org_id=row["org_id"], file_id=row["file_id"],
                expert_id=row["expert_id"], content=row["content"], metadata=metadata,
            )
            scored.append((float(scores[i]), chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def _tfidf_scores(keywords: list[str], docs: list[str]) -> list[float]:
    import math
    N = len(docs)
    tokenized = [d.lower().split() for d in docs]
    df = {}
    for tokens in tokenized:
        for kw in set(keywords):
            if kw in tokens:
                df[kw] = df.get(kw, 0) + 1

    scores = []
    for tokens in tokenized:
        total = len(tokens) or 1
        score = 0.0
        for kw in keywords:
            tf = tokens.count(kw)
            idf = math.log((N + 1) / (df.get(kw, 0) + 1))
            score += tf * idf
        scores.append(score)
    return scores


def _build_tsquery_string(query: str) -> str:
    import re
    words = re.findall(r'\w+', query.lower())
    clean_words = [w for w in words if len(w) > 1]
    if not clean_words:
        clean_words = words if words else ["*"]
    return " | ".join(clean_words[:15])


def _search_bm25_pg(conn, query, org_id, expert_id, top_k, file_ids=None):
    cur = conn.cursor()
    tsquery_str = _build_tsquery_string(query)
    if file_ids:
        if expert_id:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          ts_rank(tsv, to_tsquery('english', %s)) as rank
                   FROM chunks
                   WHERE org_id = %s AND expert_id = %s AND file_id = ANY(%s)
                     AND tsv @@ to_tsquery('english', %s)
                   ORDER BY rank DESC
                   LIMIT %s""",
                (tsquery_str, org_id, expert_id, file_ids, tsquery_str, top_k)
            )
        else:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          ts_rank(tsv, to_tsquery('english', %s)) as rank
                   FROM chunks
                   WHERE org_id = %s AND file_id = ANY(%s)
                     AND tsv @@ to_tsquery('english', %s)
                   ORDER BY rank DESC
                   LIMIT %s""",
                (tsquery_str, org_id, file_ids, tsquery_str, top_k)
            )
    else:
        if expert_id:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          ts_rank(tsv, to_tsquery('english', %s)) as rank
                   FROM chunks
                   WHERE org_id = %s AND expert_id = %s
                     AND tsv @@ to_tsquery('english', %s)
                   ORDER BY rank DESC
                   LIMIT %s""",
                (tsquery_str, org_id, expert_id, tsquery_str, top_k)
            )
        else:
            cur.execute(
                """SELECT chunk_id, org_id, file_id, expert_id, content, metadata,
                          ts_rank(tsv, to_tsquery('english', %s)) as rank
                   FROM chunks
                   WHERE org_id = %s
                     AND tsv @@ to_tsquery('english', %s)
                   ORDER BY rank DESC
                   LIMIT %s""",
                (tsquery_str, org_id, tsquery_str, top_k)
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


def search_chunks_all(
    query_vec: np.ndarray,
    org_id: str,
    top_k: int = 10,
    file_ids: list = None
) -> list[Chunk]:
    return search_chunks(query_vec, org_id, expert_id=None, top_k=top_k, file_ids=file_ids)


def search_bm25_all(
    query: str,
    org_id: str,
    top_k: int = 5,
    file_ids: list = None
) -> list[Chunk]:
    return search_bm25(query, org_id, expert_id=None, top_k=top_k, file_ids=file_ids)


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
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(
                """INSERT INTO query_logs (log_id, org_id, query, gate_weights, experts_fired, chunk_ids, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (log_id, org_id, query, json.dumps(gate_weights),
                 json.dumps(experts_fired), json.dumps(chunk_ids), latency_ms)
            )
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO query_logs (log_id, org_id, query, gate_weights, experts_fired, chunk_ids, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (log_id, org_id, query, json.dumps(gate_weights),
                 experts_fired, chunk_ids, latency_ms)
            )
            conn.commit()
        finally:
            _return_pg(conn)
    return log_id


def save_feedback(query_log_id: str, rating: int, correct_expert: str = None):
    conn = get_conn()
    feedback_id = str(uuid.uuid4())
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(
                """INSERT INTO user_feedback (feedback_id, query_log_id, rating, correct_expert)
                   VALUES (?, ?, ?, ?)""",
                (feedback_id, query_log_id, rating, correct_expert)
            )
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO user_feedback (feedback_id, query_log_id, rating, correct_expert)
                   VALUES (%s, %s, %s, %s)""",
                (feedback_id, query_log_id, rating, correct_expert)
            )
            conn.commit()
        finally:
            _return_pg(conn)
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
        try:
            cur = conn.cursor()
            if expert_id:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE org_id = %s AND expert_id = %s", (org_id, expert_id))
            else:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE org_id = %s", (org_id,))
            return cur.fetchone()[0]
        finally:
            _return_pg(conn)


def delete_file_chunks(file_id: str):
    conn = get_conn()
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM chunks WHERE file_id = %s", (file_id,))
            conn.commit()
        finally:
            _return_pg(conn)


def get_chat_sessions(org_id: str) -> list[dict]:
    conn = get_conn()
    if TESTING:
        rows = conn.execute(
            "SELECT session_id, title, created_at FROM chat_sessions WHERE org_id = ? ORDER BY created_at DESC",
            (org_id,)
        ).fetchall()
        return [{"session_id": r["session_id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT session_id, title, created_at FROM chat_sessions WHERE org_id = %s ORDER BY created_at DESC",
                (org_id,)
            )
            rows = cur.fetchall()
            return [{"session_id": r[0], "title": r[1], "created_at": r[2].isoformat() if r[2] else None} for r in rows]
        finally:
            _return_pg(conn)


def create_chat_session(session_id: str, org_id: str, title: str) -> str:
    ensure_org(org_id)
    conn = get_conn()
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(
                "INSERT INTO chat_sessions (session_id, org_id, title) VALUES (?, ?, ?)",
                (session_id, org_id, title)
            )
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO chat_sessions (session_id, org_id, title) VALUES (%s, %s, %s)",
                (session_id, org_id, title)
            )
            conn.commit()
        finally:
            _return_pg(conn)
    return session_id


def delete_chat_session(session_id: str, org_id: str):
    conn = get_conn()
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ? AND org_id = ?", (session_id, org_id))
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM chat_sessions WHERE session_id = %s AND org_id = %s", (session_id, org_id))
            conn.commit()
        finally:
            _return_pg(conn)


def get_chat_messages(session_id: str) -> list[dict]:
    conn = get_conn()
    if TESTING:
        rows = conn.execute(
            "SELECT message_id, role, content, sources, created_at FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        ).fetchall()
        return [{
            "message_id": r["message_id"],
            "role": r["role"],
            "content": r["content"],
            "sources": json.loads(r["sources"]),
            "created_at": r["created_at"]
        } for r in rows]
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT message_id, role, content, sources, created_at FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,)
            )
            rows = cur.fetchall()
            return [{
                "message_id": r[0],
                "role": r[1],
                "content": r[2],
                "sources": r[3] if isinstance(r[3], list) else json.loads(r[3]),
                "created_at": r[4].isoformat() if r[4] else None
            } for r in rows]
        finally:
            _return_pg(conn)


def get_session_org(session_id: str) -> Optional[str]:
    """Return the org_id that owns the given session_id, or None if not found.

    This is tenant-aware: `get_conn()` will target the tenant DB identified
    by the current `tenant_context`. If the session does not belong to the
    current tenant, None is returned.
    """
    conn = get_conn()
    if TESTING:
        row = conn.execute("SELECT org_id FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["org_id"] if row else None
    else:
        try:
            cur = conn.cursor()
            cur.execute("SELECT org_id FROM chat_sessions WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            _return_pg(conn)


def delete_all_chat_sessions(org_id: str) -> int:
    """Delete all chat sessions and their messages for an org. Returns number of sessions deleted."""
    conn = get_conn()
    deleted = 0
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            # Delete messages belonging to sessions for this org, then delete the sessions
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id IN (SELECT session_id FROM chat_sessions WHERE org_id = ?)",
                (org_id,)
            )
            cur = conn.execute("SELECT COUNT(*) as cnt FROM chat_sessions WHERE org_id = ?", (org_id,))
            row = cur.fetchone()
            deleted = int(row["cnt"] or 0)
            conn.execute("DELETE FROM chat_sessions WHERE org_id = ?", (org_id,))
            conn.commit()
        return deleted
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM chat_messages WHERE session_id IN (SELECT session_id FROM chat_sessions WHERE org_id = %s)",
                (org_id,)
            )
            cur.execute("SELECT COUNT(*) FROM chat_sessions WHERE org_id = %s", (org_id,))
            deleted = cur.fetchone()[0]
            cur.execute("DELETE FROM chat_sessions WHERE org_id = %s", (org_id,))
            conn.commit()
            return deleted
        finally:
            _return_pg(conn)


def add_chat_message(message_id: str, session_id: str, role: str, content: str, sources: list) -> str:
    conn = get_conn()
    sources_str = json.dumps(sources)
    if TESTING:
        lock = _get_sqlite_lock_for_conn(conn)
        with lock:
            conn.execute(
                "INSERT INTO chat_messages (message_id, session_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, sources_str)
            )
            conn.commit()
    else:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO chat_messages (message_id, session_id, role, content, sources) VALUES (%s, %s, %s, %s, %s)",
                (message_id, session_id, role, content, sources_str)
            )
            conn.commit()
        finally:
            _return_pg(conn)
    return message_id