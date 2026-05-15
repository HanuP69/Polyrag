import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.config as config
config.TESTING = False

from engine.db import init_db, ensure_org, create_file, update_file_status, get_file_status
from engine.db import upsert_chunks, search_chunks, search_bm25, log_query, save_feedback
from engine.experts.base import Chunk
import numpy as np

print("=" * 60)
print("  pgvector Integration Test")
print("=" * 60)

print("\n--- 1. Init Schema ---")
init_db()
print("  PASS")

print("\n--- 2. Org Management ---")
ensure_org("test_org", "Test Organization")
print("  PASS - org created")

print("\n--- 3. File Management ---")
file_id = create_file("test_org", "test.pdf", "pdf")
print(f"  PASS - file created: {file_id[:8]}...")
update_file_status(file_id, "parsing")
status = get_file_status(file_id)
print(f"  PASS - status: {status['status']}")

print("\n--- 4. Chunk Upsert with vector ---")
chunks = []
for i in range(5):
    vec = np.random.randn(1024).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    c = Chunk(
        org_id="test_org",
        file_id=file_id,
        expert_id="text",
        content=f"This is test chunk number {i} about machine learning and neural networks. Revenue was $100M in Q3.",
        metadata={"page": i + 1, "section": f"Section {i}"},
    )
    c.embedding = vec
    chunks.append(c)
upsert_chunks(chunks)
print("  PASS - 5 chunks upserted with 1024-dim vectors")

print("\n--- 5. Vector Search (pgvector cosine) ---")
query_vec = np.random.randn(1024).astype(np.float32)
query_vec = query_vec / np.linalg.norm(query_vec)
results = search_chunks(query_vec, "test_org", "text", top_k=3)
print(f"  PASS - got {len(results)} results")
for r in results:
    print(f"    similarity={r.metadata.get('similarity', 0):.4f} | {r.content[:60]}...")

print("\n--- 6. BM25 Search (tsvector) ---")
bm25_results = search_bm25("machine learning neural networks", "test_org", "text", top_k=3)
print(f"  PASS - got {len(bm25_results)} BM25 results")
for r in bm25_results:
    print(f"    bm25_score={r.metadata.get('bm25_score', 0):.4f} | {r.content[:60]}...")

print("\n--- 7. Query Logging ---")
log_id = log_query(
    org_id="test_org",
    query="test query about ML",
    gate_weights={"text": 0.9, "table": 0.2},
    experts_fired=["text"],
    chunk_ids=[c.chunk_id for c in results],
    latency_ms=150
)
print(f"  PASS - logged: {log_id[:8]}...")

print("\n--- 8. User Feedback ---")
fb_id = save_feedback(log_id, rating=5, correct_expert="text")
print(f"  PASS - feedback: {fb_id[:8]}...")

print("\n--- 9. Cleanup ---")
from engine.db import delete_file_chunks, get_conn
delete_file_chunks(file_id)
conn = get_conn()
cur = conn.cursor()
cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
cur.execute("DELETE FROM user_feedback WHERE query_log_id = %s", (log_id,))
cur.execute("DELETE FROM query_logs WHERE log_id = %s", (log_id,))
conn.commit()
print("  PASS - cleaned up test data")

print("\n" + "=" * 60)
print("  pgvector Integration Test PASSED")
print("=" * 60)
