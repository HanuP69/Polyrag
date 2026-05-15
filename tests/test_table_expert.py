"""
test_table_expert.py -- Phase 2 test for the Table Expert.

1. Verify health shows table expert loaded
2. Verify gate routes table queries correctly
3. Ingest a CSV file
4. Query the table data
"""

import requests
import os
import time

BASE_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_CSV = os.path.join(DATA_DIR, "test_sales.csv")


def run():
    # 1. Health
    print("=" * 60)
    print("  Phase 2 -- Table Expert Test")
    print("=" * 60)

    r = requests.get(f"{BASE_URL}/health", timeout=30)
    data = r.json()
    print(f"\n[Health] experts_loaded: {data['experts_loaded']}")
    assert "table" in data["experts_loaded"], "Table expert not loaded!"
    print("[Health] PASS -- table expert is registered")

    # 2. Gate
    print("\n--- Gate routing ---")
    table_queries = [
        "what was Q3 revenue for Asia Pacific",
        "compare growth rates across all regions",
        "which region had the highest total units",
    ]
    text_queries = [
        "summarize the indemnity clause",
        "explain the termination conditions",
    ]

    for q in table_queries:
        r = requests.post(f"{BASE_URL}/gate", json={"query": q}, timeout=30)
        top = max(r.json()["raw"], key=r.json()["raw"].get)
        score = r.json()["raw"][top]
        status = "PASS" if top == "table" else "FAIL"
        print(f"  [{status}] \"{q}\" -> {top} ({score:.2f})")

    for q in text_queries:
        r = requests.post(f"{BASE_URL}/gate", json={"query": q}, timeout=30)
        top = max(r.json()["raw"], key=r.json()["raw"].get)
        score = r.json()["raw"][top]
        status = "PASS" if top == "text" else "FAIL"
        print(f"  [{status}] \"{q}\" -> {top} ({score:.2f})")

    # 3. Ingest CSV
    print("\n--- CSV Ingestion ---")
    with open(TEST_CSV, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest",
            files={"file": ("test_sales.csv", f, "text/csv")},
            data={"org_id": "default"},
            timeout=120,
        )
    data = r.json()
    print(f"  Status: {data.get('status')}")
    print(f"  Chunks: {data.get('total_chunks')}")
    print(f"  Experts: {data.get('experts_used')}")
    print(f"  Latency: {data.get('latency_ms')}ms")
    assert data["status"] == "indexed"
    assert "table" in data["experts_used"]
    print("  PASS -- CSV ingested via table expert")

    # 4. Query the table
    print("\n--- Table Queries ---")
    queries = [
        "What was the Q3 revenue for Asia Pacific?",
        "Which region had the highest growth rate?",
        "Compare Q1 and Q4 revenue for North America",
    ]
    for q in queries:
        print(f"\n  Q: \"{q}\"")
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/query",
            json={"query": q, "org_id": "default", "top_k": 5},
            timeout=120,
        )
        elapsed = time.time() - start
        data = r.json()

        answer = data.get("answer", "")[:400]
        print(f"  Gate: {data.get('gate_weights')}")
        print(f"  Experts fired: {data.get('experts_fired')}")
        print(f"  Sources: {len(data.get('sources', []))}")
        for s in data.get("sources", [])[:2]:
            print(f"    [{s['expert_id'].upper()}] {s['content'][:120]}...")
        print(f"  Answer: {answer}")
        print(f"  Latency: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("  Phase 2 test complete")
    print("=" * 60)


if __name__ == "__main__":
    run()
