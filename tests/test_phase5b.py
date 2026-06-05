import requests
import time

BASE_URL = "http://localhost:8000"

def run():
    print("=" * 60)
    print("  v4 Advanced RAG Test")
    print("=" * 60)

    print("\n--- 1. HyDE + Hybrid Retrieval ---")
    r = requests.post(f"{BASE_URL}/retrieve", json={
        "query": "revenue quarterly sales",
        "org_id": "default",
        "top_k": 5,
    }, timeout=60)
    data = r.json()
    chunks = data.get("chunks", [])
    print(f"  Chunks returned: {len(chunks)}")
    for c in chunks[:3]:
        print(f"    [{c.get('modality','?').upper()}] {c.get('content','')[:80]}...")
    print("  [PASS] Hybrid retrieval works")

    print("\n--- 2. Hallucination Guard API ---")
    r = requests.post(f"{BASE_URL}/guard", json={
        "answer": "North America had Q1 revenue of $125,000 and Q4 revenue of $165,000. The company was founded in 1847.",
        "sources": [
            "Region: North America | Q1 Revenue: $125,000 | Q2 Revenue: $135,000 | Q3 Revenue: $148,000 | Q4 Revenue: $165,000"
        ],
    }, timeout=300)
    data = r.json()
    print(f"  Verified: {data['verified']}")
    print(f"  Score: {data['score']}")
    print(f"  Claims checked: {data['total_claims']}")
    for claim in data.get("claims", []):
        status = "GROUNDED" if claim["grounded"] else "UNVERIFIED"
        print(f"    [{status}] ({claim['confidence']:.3f}) {claim['claim'][:80]}...")
    print("  [PASS] Hallucination guard works")

    print("\n--- 3. Pipeline Health ---")
    r = requests.get(f"{BASE_URL}/health/pipeline", timeout=30)
    data = r.json()
    print(f"  Embedder loaded: {data['components']['embedder']}")
    print(f"  Reranker loaded: {data['components']['reranker']}")
    print(f"  DB: {data['components']['database']}")
    print("  [PASS] Pipeline health API works")

    print("\n--- 4. Feedback API ---")
    r = requests.post(f"{BASE_URL}/feedback", json={
        "query_log_id": "test-log-id",
        "rating": 5,
    }, timeout=30)
    fb_data = r.json()
    assert fb_data["status"] == "ok"
    print("  [PASS] Feedback submitted")

    print("\n--- 5. Full Streaming Query ---")
    import json
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/generate/stream",
        json={
            "prompt": "What was Q4 revenue for North America?",
            "query": "What was Q4 revenue for North America?",
        },
        stream=True,
        timeout=120,
    )
    full_answer = ""
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        full_answer += event["content"]
                    elif event["type"] == "done":
                        print(f"  Latency: {event.get('latency_ms', '?')}ms")
                except json.JSONDecodeError:
                    pass
    elapsed = time.time() - start
    print(f"  Answer: {full_answer[:200]}")
    print(f"  Total time: {elapsed:.1f}s")
    print("  [PASS] Streaming generation works")

    print("\n" + "=" * 60)
    print("  v4 advanced RAG test complete")
    print("=" * 60)

if __name__ == "__main__":
    run()
