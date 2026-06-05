import time
import numpy as np
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v4.config import CFG
from engine_v4.db import get_conn, store_chunks, store_embeddings, dense_search, ensure_org, create_file
from engine_v4.chunker import Chunk

EMBEDDING_DIM = 1024
DATABASE_URL = CFG.pg_conn

print("=" * 60)
print("  PolyRAG v4 Vector Indexing Benchmarks")
print("=" * 60)

org_id = "bench_org"
file_id = str(uuid.uuid4())
ensure_org(org_id, "Benchmark Organization")
create_file(file_id, org_id, "benchmark.pdf", "pdf")

CHUNK_COUNT = 300
print(f"\n[1/4] Ingesting {CHUNK_COUNT} vectors (dim={EMBEDDING_DIM})...", flush=True)
chunks = []
for i in range(CHUNK_COUNT):
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    c = Chunk(
        chunk_id=f"bench_{i}",
        doc_id=file_id,
        section_id=i,
        modality="text",
        content=f"Benchmark content row {i} covering machine learning and vector storage.",
        metadata={"page": i + 1},
        embedding=vec,
        org_id=org_id,
        file_id=file_id,
    )
    chunks.append(c)

start_ingest = time.time()
store_chunks(chunks)
store_embeddings(chunks)
ingest_duration = time.time() - start_ingest
print(f"  [OK] Ingested {CHUNK_COUNT} vectors in {ingest_duration:.2f}s", flush=True)

query_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
query_vec = query_vec / np.linalg.norm(query_vec)

conn = get_conn()
with conn.cursor() as cur:
    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()

print("\n[2/4] Flat cosine scan (no index)...", flush=True)
non_indexed_times = []
for _ in range(50):
    start = time.time()
    dense_search(query_vec, "text", org_id, top_k=5)
    non_indexed_times.append(time.time() - start)
avg_non_indexed_ms = np.mean(non_indexed_times) * 1000
print(f"  Avg latency (no index): {avg_non_indexed_ms:.3f} ms", flush=True)

print("\n[3/4] Creating HNSW index...", flush=True)
with conn.cursor() as cur:
    start_index = time.time()
    cur.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw
        ON embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
conn.commit()
index_duration = time.time() - start_index
print(f"  [OK] Index built in {index_duration:.3f}s", flush=True)

print("\n[4/4] HNSW indexed scan...", flush=True)
indexed_times = []
for _ in range(50):
    start = time.time()
    dense_search(query_vec, "text", org_id, top_k=5)
    indexed_times.append(time.time() - start)
avg_indexed_ms = np.mean(indexed_times) * 1000
print(f"  Avg latency (HNSW): {avg_indexed_ms:.3f} ms", flush=True)

boost_factor = avg_non_indexed_ms / avg_indexed_ms
print("\n" + "=" * 60)
print("  BENCHMARK SUMMARY")
print("=" * 60)
print(f"  Vectors:          {CHUNK_COUNT}")
print(f"  Flat latency:     {avg_non_indexed_ms:.3f} ms")
print(f"  HNSW latency:     {avg_indexed_ms:.3f} ms")
print(f"  Speedup:          {boost_factor:.2f}x")
print("=" * 60)

print("\nCleaning up...", flush=True)
with conn.cursor() as cur:
    cur.execute("DELETE FROM embeddings WHERE chunk_id LIKE 'bench_%'")
    cur.execute("DELETE FROM chunks WHERE org_id = %s", (org_id,))
    cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()
conn.close()
print("Done.", flush=True)
