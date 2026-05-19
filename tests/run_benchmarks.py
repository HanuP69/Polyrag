import time
import numpy as np
import psycopg2
import sys
import os

# Insert parent dir into sys.path to access engine configs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import DATABASE_URL, EMBEDDING_DIM
from engine.db import get_conn, upsert_chunks, search_chunks, ensure_org, create_file
from engine.experts.base import Chunk

print("=" * 60)
print("  PolyRAG Production Concurrency & Vector Indexing Benchmarks")
print("=" * 60)

# Connect to database
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Ensure clean state
org_id = "bench_org"
ensure_org(org_id, "Benchmark Organization")
file_id = create_file(org_id, "benchmark.pdf", "pdf")

# Generate 300 dense vectors to populate database for a fast, responsive test
CHUNK_COUNT = 300
print(f"\n[1/4] Generating & Ingesting {CHUNK_COUNT} dense vectors (dim={EMBEDDING_DIM})...", flush=True)
chunks = []
for i in range(CHUNK_COUNT):
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    c = Chunk(
        chunk_id=f"bench_{i}",
        org_id=org_id,
        file_id=file_id,
        expert_id="text",
        content=f"Benchmark content row {i} covering computational machine learning and vector data storage indexes.",
        metadata={"page": i + 1}
    )
    c.embedding = vec
    chunks.append(c)

start_ingest = time.time()
upsert_chunks(chunks)
ingest_duration = time.time() - start_ingest
print(f"  [OK] Ingested {CHUNK_COUNT} vectors in {ingest_duration:.2f} seconds.", flush=True)

# Prepare benchmark query vector
query_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
query_vec = query_vec / np.linalg.norm(query_vec)

# Ensure index is REMOVED for unindexed baseline test
cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()

# Test 1: Flat/Linear Cosine Search (No Index)
print("\n[2/4] Benchmarking Test: Raw Linear Cosine Scan (No Index)...", flush=True)
non_indexed_times = []
for _ in range(50):
    start = time.time()
    # Executing raw search_chunks
    search_chunks(query_vec, org_id, "text", top_k=5)
    non_indexed_times.append(time.time() - start)

avg_non_indexed_ms = np.mean(non_indexed_times) * 1000
print(f"  Average Query Latency (No Index): {avg_non_indexed_ms:.3f} ms", flush=True)

# Test 2: Create HNSW Vector Index
print("\n[3/4] Creating HNSW Vector Index dynamically...", flush=True)
start_index = time.time()
cur.execute("""
    CREATE INDEX idx_chunks_embedding_hnsw 
    ON chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")
conn.commit()
index_duration = time.time() - start_index
print(f"  [OK] Index built on {CHUNK_COUNT} rows in {index_duration:.3f} seconds.", flush=True)

# Test 3: Indexed HNSW Cosine Search
print("\n[4/4] Benchmarking Test: Graph-based HNSW Scan...", flush=True)
indexed_times = []
for _ in range(50):
    start = time.time()
    search_chunks(query_vec, org_id, "text", top_k=5)
    indexed_times.append(time.time() - start)

avg_indexed_ms = np.mean(indexed_times) * 1000
print(f"  Average Query Latency (HNSW Index): {avg_indexed_ms:.3f} ms", flush=True)

# Summary Results
boost_factor = avg_non_indexed_ms / avg_indexed_ms
print("\n" + "=" * 60, flush=True)
print("  BENCHMARK SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  - Ingested Vector Count:    {CHUNK_COUNT}", flush=True)
print(f"  - Query Time (Flat):        {avg_non_indexed_ms:.3f} ms", flush=True)
print(f"  - Query Time (HNSW):        {avg_indexed_ms:.3f} ms", flush=True)
print(f"  - Real Optimization Boost:  {boost_factor:.2f}x Faster!", flush=True)
print("=" * 60, flush=True)

# Cleanup benchmark tables
print("\nCleaning up benchmark rows...", flush=True)
cur.execute("DELETE FROM chunks WHERE org_id = %s", (org_id,))
cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()
conn.close()
print("Cleaned up successfully.", flush=True)
