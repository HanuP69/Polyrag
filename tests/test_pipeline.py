"""
test_pipeline.py -- End-to-end test for PolyRAG.

Tests:
1. Health check
2. Gate routing
3. File ingestion (parse + embed + store)
4. Query pipeline (gate + retrieve + fuse + LLM generate)
"""

import requests
import json
import time
import os
import sys

BASE_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_PDF = os.path.join(DATA_DIR, "test_contract.pdf")


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    data = r.json()
    print(f"  Status: {data['status']}")
    print(f"  Mode: {data['mode']}")
    print(f"  LLM: {data['llm']}")
    print(f"  Experts: {data['experts_loaded']}")
    print(f"  Gate: {data['gate_loaded']}")
    assert data["status"] == "ok"
    print("  >> PASSED")


def test_gate():
    """Test gate routing."""
    print("\n" + "=" * 60)
    print("TEST 2: Gate Routing")
    print("=" * 60)
    
    test_cases = [
        ("summarize the indemnity clause", "text"),
        ("how many rows have revenue > 100k", "table"),
        ("describe the architecture diagram", "image"),
        ("what are the termination conditions", "text"),
    ]
    
    for query, expected in test_cases:
        r = requests.post(f"{BASE_URL}/gate", json={"query": query}, timeout=30)
        data = r.json()
        top_expert = max(data["raw"], key=data["raw"].get)
        raw_str = " | ".join(f"{k}: {v:.2f}" for k, v in data["raw"].items())
        status = "PASS" if top_expert == expected else "FAIL"
        print(f"  [{status}] \"{query}\"")
        print(f"         Raw: {raw_str}")
        print(f"         Top: {top_expert} (expected: {expected})")
    
    print("  >> Gate routing test complete")


def test_ingest():
    """Test file ingestion."""
    print("\n" + "=" * 60)
    print("TEST 3: File Ingestion")
    print("=" * 60)
    
    if not os.path.exists(TEST_PDF):
        print(f"  ERROR: Test PDF not found at {TEST_PDF}")
        print("  Run: python tests/create_test_pdf.py first")
        return None
    
    print(f"  Uploading: {os.path.basename(TEST_PDF)}")
    
    start = time.time()
    with open(TEST_PDF, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest",
            files={"file": ("test_contract.pdf", f, "application/pdf")},
            data={"org_id": "default"},
            timeout=300
        )
    elapsed = time.time() - start
    
    data = r.json()
    print(f"  Status: {data.get('status')}")
    print(f"  File ID: {data.get('file_id')}")
    print(f"  Chunks: {data.get('total_chunks')}")
    print(f"  Experts used: {data.get('experts_used')}")
    print(f"  Latency: {elapsed:.1f}s")
    
    assert data["status"] == "indexed", f"Expected 'indexed', got '{data['status']}'"
    assert data["total_chunks"] > 0, "No chunks extracted"
    print("  >> PASSED")
    
    return data.get("file_id")


def test_query(queries=None):
    """Test query pipeline."""
    print("\n" + "=" * 60)
    print("TEST 4: Query Pipeline (Gate + Retrieve + Fuse + LLM)")
    print("=" * 60)
    
    if queries is None:
        queries = [
            "What are the termination conditions in this contract?",
            "What does the indemnification section say?",
            "How long does the confidentiality obligation last?",
            "What is the monthly fee for the services?",
        ]
    
    for query in queries:
        print(f"\n  Query: \"{query}\"")
        print("  " + "-" * 50)
        
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/query",
            json={
                "query": query,
                "org_id": "default",
                "top_k": 5,
            },
            timeout=120
        )
        elapsed = time.time() - start
        
        data = r.json()
        
        # Print answer (truncated)
        answer = data.get("answer", "")
        if len(answer) > 500:
            answer = answer[:500] + "..."
        print(f"  Answer: {answer}")
        print(f"\n  Gate weights: {data.get('gate_weights')}")
        print(f"  Experts fired: {data.get('experts_fired')}")
        print(f"  Sources: {len(data.get('sources', []))}")
        for i, src in enumerate(data.get("sources", [])[:3]):
            content_preview = src["content"][:100] + "..." if len(src["content"]) > 100 else src["content"]
            print(f"    [{i+1}] ({src['expert_id']}) page {src['metadata'].get('page', '?')}: {content_preview}")
        print(f"  Latency: {elapsed:.1f}s")
        print(f"  Cached: {data.get('cached', False)}")
    
    print("\n  >> Query pipeline test complete")


if __name__ == "__main__":
    print("=" * 60)
    print("  PolyRAG End-to-End Pipeline Test")
    print("=" * 60)
    
    test_health()
    test_gate()
    file_id = test_ingest()
    if file_id:
        test_query()
    
    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)
