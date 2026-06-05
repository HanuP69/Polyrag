import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_v4.db import (
    init_db, ensure_org, create_file, update_file_status, get_file,
    store_chunks, store_embeddings, dense_search, delete_file_and_chunks, get_conn,
)
from engine_v4.chunker import Chunk
import numpy as np
import uuid

print("=" * 60)
print("  pgvector Integration Test (v4)")
print("=" * 60)

print("\n--- 1. Init Schema ---")
init_db()
print("  PASS")

print("\n--- 2. Org Management ---")
ensure_org("test_org", "Test Organization")
print("  PASS - org created")

print("\n--- 3. File Management ---")
file_id = str(uuid.uuid4())
create_file(file_id, "test_org", "test.pdf", "pdf")
print(f"  PASS - file created: {file_id[:8]}...")
update_file_status(file_id, "processing", 0)
file_rec = get_file(file_id)
print(f"  PASS - status: {file_rec.get('status')}")

print("\n--- 4. Chunk Store with 1024-dim vectors ---")
chunks = []
for i in range(5):
    vec = np.random.randn(1024).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    c = Chunk(
        chunk_id=f"test_chunk_{i}",
        doc_id=file_id,
        section_id=i,
        modality="text",
        content=f"This is test chunk {i} about machine learning and neural networks. Revenue was $100M in Q3.",
        metadata={"page": i + 1},
        embedding=vec,
        org_id="test_org",
        file_id=file_id,
    )
    chunks.append(c)

store_chunks(chunks)
store_embeddings(chunks)
print("  PASS - 5 chunks stored with 1024-dim vectors")

print("\n--- 5. Vector Search (pgvector cosine) ---")
query_vec = np.random.randn(1024).astype(np.float32)
query_vec = query_vec / np.linalg.norm(query_vec)
results = dense_search(query_vec, "text", "test_org", top_k=3)
print(f"  PASS - got {len(results)} results")
for r in results:
    print(f"    score={r.get('similarity', r.get('score', 0)):.4f} | {r.get('content', '')[:60]}...")

print("\n--- 6. Cleanup ---")
delete_file_and_chunks("test_org", file_id)
conn = get_conn()
with conn.cursor() as cur:
    cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
conn.commit()
conn.close()
print("  PASS - cleaned up test data")

print("\n" + "=" * 60)
print("  pgvector Integration Test PASSED")
print("=" * 60)
