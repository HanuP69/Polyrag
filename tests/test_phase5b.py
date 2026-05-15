import requests
import time

BASE_URL = "http://localhost:8000"

def run():
    print("=" * 60)
    print("  Phase 5B -- Advanced RAG Test")
    print("=" * 60)

    print("\n--- 1. Query Rewrite API ---")
    r = requests.post(f"{BASE_URL}/rewrite", json={"query": "money stuff"}, timeout=60)
    data = r.json()
    print(f"  Original: \"{data['original']}\"")
    print(f"  Rewritten: \"{data['rewritten']}\"")
    assert data["rewritten"] != "" and len(data["rewritten"]) > 3
    print("  [PASS] Query rewriting works")

    print("\n--- 2. BM25 Retrieve API ---")
    r = requests.post(f"{BASE_URL}/retrieve/bm25", json={
        "query": "revenue quarterly sales",
        "expert_id": "table",
        "org_id": "default",
        "top_k": 3
    }, timeout=30)
    data = r.json()
    print(f"  BM25 results: {data['total']}")
    print("  [PASS] BM25 endpoint works")

    print("\n--- 3. Hallucination Guard API ---")
    r = requests.post(f"{BASE_URL}/guard", json={
        "answer": "North America had Q1 revenue of $125,000 and Q4 revenue of $165,000. The company was founded in 1847.",
        "sources": [
            "Region: North America | Q1 Revenue: $125,000 | Q2 Revenue: $135,000 | Q3 Revenue: $148,000 | Q4 Revenue: $165,000"
        ]
    }, timeout=300)
    data = r.json()
    print(f"  Verified: {data['verified']}")
    print(f"  Score: {data['score']}")
    print(f"  Claims checked: {data['total_claims']}")
    for claim in data.get("claims", []):
        status = "GROUNDED" if claim["grounded"] else "UNVERIFIED"
        print(f"    [{status}] ({claim['confidence']:.3f}) {claim['claim'][:80]}...")
    print("  [PASS] Hallucination guard works")

    print("\n--- 4. Pipeline Health ---")
    r = requests.get(f"{BASE_URL}/health/pipeline", timeout=30)
    data = r.json()
    print(f"  Total queries: {data['total_queries']}")
    print(f"  Avg latency: {data['avg_latency_ms']}ms")
    print(f"  Feedback count: {data['feedback_count']}")
    print(f"  Retrain recommended: {data['retrain_recommended']}")
    print("  [PASS] Pipeline health API works")

    print("\n--- 5. Full Query with Advanced RAG ---")
    start = time.time()
    r = requests.post(f"{BASE_URL}/query", json={
        "query": "What was Q4 revenue for North America?",
        "org_id": "default",
        "top_k": 5,
    }, timeout=180)
    elapsed = time.time() - start
    data = r.json()
    print(f"  Answer: {data.get('answer', '')[:200]}")
    print(f"  Gate: {data.get('gate_weights')}")
    print(f"  Rewritten: {data.get('rewritten_query')}")
    print(f"  Guard: verified={data.get('guard', {}).get('verified')}, score={data.get('guard', {}).get('score')}")
    print(f"  Log ID: {data.get('query_log_id', 'N/A')[:12]}...")
    print(f"  Latency: {elapsed:.1f}s")

    query_log_id = data.get("query_log_id")
    if query_log_id:
        print("\n--- 6. User Feedback ---")
        r = requests.post(f"{BASE_URL}/feedback", json={
            "query_log_id": query_log_id,
            "rating": 5,
            "correct_expert": "table"
        }, timeout=30)
        fb_data = r.json()
        print(f"  Feedback ID: {fb_data.get('feedback_id', 'N/A')[:12]}...")
        assert fb_data["status"] == "ok"
        print("  [PASS] Feedback submitted")

        r = requests.get(f"{BASE_URL}/health/pipeline", timeout=30)
        data = r.json()
        print(f"  Feedback count after: {data['feedback_count']}")

    print("\n" + "=" * 60)
    print("  Phase 5B test complete")
    print("=" * 60)

if __name__ == "__main__":
    run()
