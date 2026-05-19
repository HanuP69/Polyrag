import time
import numpy as np
import psycopg2
import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import DATABASE_URL, EMBEDDING_DIM
from engine.db import get_conn, upsert_chunks, search_chunks, ensure_org, create_file
from engine.experts.base import Chunk

# Setup DB Connection
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

org_id = "load_bench_org"
ensure_org(org_id, "Load Test Org")
file_id = create_file(org_id, "load_test.pdf", "pdf")

VECTOR_COUNT = 10000
CONCURRENT_USERS = 250
TOTAL_QUERIES = 1000

print("=" * 60)
print("  PolyRAG Production Concurrency & Load Handling Benchmarks")
print("=" * 60)

# Generate vectors
print(f"\n[1/5] Ingesting {VECTOR_COUNT} baseline vectors (dim={EMBEDDING_DIM})...", flush=True)
chunks = []
for i in range(VECTOR_COUNT):
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    c = Chunk(
        chunk_id=f"load_{i}",
        org_id=org_id,
        file_id=file_id,
        expert_id="text",
        content=f"Baseline testing chunk row {i} containing system scaling and concurrent query load testing data.",
        metadata={"page": i + 1}
    )
    c.embedding = vec
    chunks.append(c)

upsert_chunks(chunks)

# Test parameters
query_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
query_vec = query_vec / np.linalg.norm(query_vec)

# Ensure no HNSW index exists first
cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()

# Query relation sizes before HNSW
cur.execute("SELECT pg_relation_size('chunks')")
table_size_bytes_before = cur.fetchone()[0]

# --- 1. Sequential vs Parallel on Flat Search ---
print("\n[2/5] Benchmarking Flat Search (No Index)...", flush=True)

# Sequential run
start_seq = time.time()
for _ in range(TOTAL_QUERIES):
    search_chunks(query_vec, org_id, "text", top_k=5)
seq_duration_flat = time.time() - start_seq
avg_seq_flat_ms = (seq_duration_flat / TOTAL_QUERIES) * 1000

# Parallel run (Using thread pool to simulate concurrent users)
def run_query():
    search_chunks(query_vec, org_id, "text", top_k=5)

start_par = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
    futures = [executor.submit(run_query) for _ in range(TOTAL_QUERIES)]
    for f in futures:
        f.result()
par_duration_flat = time.time() - start_par
throughput_flat = TOTAL_QUERIES / par_duration_flat

print(f"  - Sequential Avg Latency: {avg_seq_flat_ms:.2f} ms", flush=True)
print(f"  - Parallel Total Time ({CONCURRENT_USERS} users, {TOTAL_QUERIES} queries): {par_duration_flat:.2f} s", flush=True)
print(f"  - Throughput (Parallel): {throughput_flat:.2f} queries/sec", flush=True)


# --- 2. Creating Index and Measuring Index Space footprint ---
print("\n[3/5] Creating HNSW Index and Measuring Footprint...", flush=True)
start_idx = time.time()
cur.execute("""
    CREATE INDEX idx_chunks_embedding_hnsw 
    ON chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")
conn.commit()
index_build_time = time.time() - start_idx

# Get index size in bytes
cur.execute("SELECT pg_relation_size('idx_chunks_embedding_hnsw')")
index_size_bytes = cur.fetchone()[0]
cur.execute("SELECT pg_relation_size('chunks')")
table_size_bytes_after = cur.fetchone()[0]

print(f"  - Index Build Time: {index_build_time:.3f} s", flush=True)
print(f"  - Index Disk Size: {index_size_bytes / 1024:.2f} KB", flush=True)


# --- 3. Sequential vs Parallel on Indexed HNSW Search ---
print("\n[4/5] Benchmarking Indexed Search (HNSW Graph)...", flush=True)

# Sequential run
start_seq = time.time()
for _ in range(TOTAL_QUERIES):
    search_chunks(query_vec, org_id, "text", top_k=5)
seq_duration_hnsw = time.time() - start_seq
avg_seq_hnsw_ms = (seq_duration_hnsw / TOTAL_QUERIES) * 1000

# Parallel run
start_par = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
    futures = [executor.submit(run_query) for _ in range(TOTAL_QUERIES)]
    for f in futures:
        f.result()
par_duration_hnsw = time.time() - start_par
throughput_hnsw = TOTAL_QUERIES / par_duration_hnsw

print(f"  - Sequential Avg Latency: {avg_seq_hnsw_ms:.2f} ms", flush=True)
print(f"  - Parallel Total Time ({CONCURRENT_USERS} users, {TOTAL_QUERIES} queries): {par_duration_hnsw:.2f} s", flush=True)
print(f"  - Throughput (Parallel): {throughput_hnsw:.2f} queries/sec", flush=True)


# --- 4. Report Final Output ---
print("\n" + "=" * 60)
print("  RAW BENCHMARK NUMBERS FOR TABLE")
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

# Cleanup
print("\nCleaning up benchmarking tables...", flush=True)
cur.execute("DELETE FROM chunks WHERE org_id = %s", (org_id,))
cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
conn.commit()
conn.close()
print("Successfully cleaned up.", flush=True)
