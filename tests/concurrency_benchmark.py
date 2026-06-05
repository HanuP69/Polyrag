import time
import numpy as np
import sys
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v4.config import CFG
from engine_v4.db import get_conn, store_chunks, store_embeddings, dense_search, ensure_org, create_file
from engine_v4.chunker import Chunk

EMBEDDING_DIM = 1024
org_id = "load_bench_org"
file_id = str(uuid.uuid4())

VECTOR_COUNT = 10000
CONCURRENT_USERS = 250
TOTAL_QUERIES = 1000

print("=" * 60)
print("  PolyRAG v4 Concurrency & Load Benchmarks")
print("=" * 60)

ensure_org(org_id, "Load Test Org")
create_file(file_id, org_id, "load_test.pdf", "pdf")

print(f"\n[1/5] Ingesting {VECTOR_COUNT} vectors (dim={EMBEDDING_DIM})...", flush=True)
chunks = []
for i in range(VECTOR_COUNT):
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    c = Chunk(
        chunk_id=f"load_{i}",
        doc_id=file_id,
        section_id=i,
        modality="text",
        content=f"Load test chunk {i} with concurrent query scaling data.",
        metadata={"page": i + 1},
        embedding=vec,
        org_id=org_id,
        file_id=file_id,
    )
    chunks.append(c)

store_chunks(chunks)
store_embeddings(chunks)
print(f"  [OK] Ingested {VECTOR_COUNT} vectors", flush=True)

query_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
query_vec = query_vec / np.linalg.norm(query_vec)

conn = get_conn()
with conn.cursor() as cur:
    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()

print("\n[2/5] Flat search — sequential + parallel...", flush=True)

start_seq = time.time()
for _ in range(TOTAL_QUERIES):
    dense_search(query_vec, "text", org_id, top_k=5)
seq_duration_flat = time.time() - start_seq
avg_seq_flat_ms = (seq_duration_flat / TOTAL_QUERIES) * 1000

def run_query():
    dense_search(query_vec, "text", org_id, top_k=5)

start_par = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
    futures = [executor.submit(run_query) for _ in range(TOTAL_QUERIES)]
    for f in futures:
        f.result()
par_duration_flat = time.time() - start_par
throughput_flat = TOTAL_QUERIES / par_duration_flat

print(f"  Seq avg: {avg_seq_flat_ms:.2f} ms", flush=True)
print(f"  Par total ({CONCURRENT_USERS} users, {TOTAL_QUERIES} queries): {par_duration_flat:.2f}s", flush=True)
print(f"  Throughput: {throughput_flat:.2f} q/s", flush=True)

print("\n[3/5] Building HNSW index...", flush=True)
with conn.cursor() as cur:
    start_idx = time.time()
    cur.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw
        ON embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    conn.commit()
    index_build_time = time.time() - start_idx
    cur.execute("SELECT pg_relation_size('idx_chunks_embedding_hnsw')")
    index_size_bytes = cur.fetchone()[0]

print(f"  Built in {index_build_time:.3f}s | size: {index_size_bytes / 1024:.2f} KB", flush=True)

print("\n[4/5] HNSW search — sequential + parallel...", flush=True)

start_seq = time.time()
for _ in range(TOTAL_QUERIES):
    dense_search(query_vec, "text", org_id, top_k=5)
seq_duration_hnsw = time.time() - start_seq
avg_seq_hnsw_ms = (seq_duration_hnsw / TOTAL_QUERIES) * 1000

start_par = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
    futures = [executor.submit(run_query) for _ in range(TOTAL_QUERIES)]
    for f in futures:
        f.result()
par_duration_hnsw = time.time() - start_par
throughput_hnsw = TOTAL_QUERIES / par_duration_hnsw

print(f"  Seq avg: {avg_seq_hnsw_ms:.2f} ms", flush=True)
print(f"  Par total: {par_duration_hnsw:.2f}s | Throughput: {throughput_hnsw:.2f} q/s", flush=True)

print("\n[5/5] Summary")
print("=" * 60)
print(f"RAW_VAL_SEQ_FLAT_MS={avg_seq_flat_ms:.3f}")
print(f"RAW_VAL_PAR_FLAT_TIME={par_duration_flat:.3f}")
print(f"RAW_VAL_THROUGHPUT_FLAT={throughput_flat:.3f}")
print(f"RAW_VAL_INDEX_BUILD_S={index_build_time:.3f}")
print(f"RAW_VAL_INDEX_SIZE_KB={index_size_bytes / 1024:.3f}")
print(f"RAW_VAL_SEQ_HNSW_MS={avg_seq_hnsw_ms:.3f}")
print(f"RAW_VAL_PAR_HNSW_TIME={par_duration_hnsw:.3f}")
print(f"RAW_VAL_THROUGHPUT_HNSW={throughput_hnsw:.3f}")
print("=" * 60)

print("\nCleaning up...", flush=True)
with conn.cursor() as cur:
    cur.execute("DELETE FROM embeddings WHERE chunk_id LIKE 'load_%'")
    cur.execute("DELETE FROM chunks WHERE org_id = %s", (org_id,))
    cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()
conn.close()
print("Done.", flush=True)
