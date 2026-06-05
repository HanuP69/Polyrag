import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_PDF = os.path.join(DATA_DIR, "test_contract.pdf")


def test_health():
    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    r = requests.get(f"{BASE_URL}/health", timeout=30)
    data = r.json()
    print(f"  Status: {data['status']}")
    print(f"  Mode: {data['mode']}")
    print(f"  LLM: {data['llm']}")
    print(f"  Embedder: {data['embedder']}")
    print(f"  Reranker: {data['reranker']}")
    assert data["status"] == "ok"
    assert data["mode"] == "v4"
    print("  >> PASSED")


def test_ingest():
    print("\n" + "=" * 60)
    print("TEST 2: File Ingestion")
    print("=" * 60)

    if not os.path.exists(TEST_PDF):
        print(f"  ERROR: Test PDF not found at {TEST_PDF}")
        print("  Run: python tests/create_test_pdf.py first")
        return None

    print(f"  Uploading: {os.path.basename(TEST_PDF)}")

    start = time.time()
    with open(TEST_PDF, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest/async",
            files={"file": ("test_contract.pdf", f, "application/pdf")},
            data={"org_id": "default"},
            timeout=60,
        )
    data = r.json()
    file_id = data.get("file_id")
    print(f"  File ID: {file_id}")
    assert file_id

    for _ in range(120):
        r = requests.get(f"{BASE_URL}/file/{file_id}", timeout=10)
        status_data = r.json()
        print(f"  Status: {status_data.get('status')} | chunks: {status_data.get('chunk_count', '?')}")
        if status_data.get("status") in ("completed", "indexed", "error", "failed"):
            break
        time.sleep(2)

    assert status_data.get("status") not in ("error", "failed"), f"Ingestion failed: {status_data}"
    elapsed = time.time() - start
    print(f"  Latency: {elapsed:.1f}s")
    print("  >> PASSED")
    return file_id


def test_retrieve():
    print("\n" + "=" * 60)
    print("TEST 3: Retrieve")
    print("=" * 60)

    r = requests.post(
        f"{BASE_URL}/retrieve",
        json={"query": "termination conditions", "org_id": "default", "top_k": 5},
        timeout=60,
    )
    data = r.json()
    chunks = data.get("chunks", [])
    print(f"  Chunks returned: {len(chunks)}")
    for c in chunks[:3]:
        print(f"    [{c.get('modality', 'unknown').upper()}] {c.get('content', '')[:100]}...")
    assert len(chunks) > 0
    print("  >> PASSED")


def test_query():
    print("\n" + "=" * 60)
    print("TEST 4: Query Pipeline (Hybrid Retrieval + Rerank + LLM)")
    print("=" * 60)

    queries = [
        "What are the termination conditions in this contract?",
        "What does the indemnification section say?",
        "How long does the confidentiality obligation last?",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        print("  " + "-" * 50)

        start = time.time()
        r = requests.post(
            f"{BASE_URL}/generate",
            json={
                "prompt": query,
                "query": query,
            },
            timeout=120,
        )
        elapsed = time.time() - start

        data = r.json()
        answer = data.get("response", "")
        if len(answer) > 500:
            answer = answer[:500] + "..."
        print(f"  Answer: {answer}")
        print(f"  Latency: {elapsed:.1f}s")

    print("\n  >> Query pipeline test complete")


if __name__ == "__main__":
    print("=" * 60)
    print("  PolyRAG v4 End-to-End Pipeline Test")
    print("=" * 60)

    test_health()
    file_id = test_ingest()
    if file_id:
        test_retrieve()
        test_query()

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)
