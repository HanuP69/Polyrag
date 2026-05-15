import requests
import os
import time

BASE_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_IMG = os.path.join(DATA_DIR, "test_chart.png")


def run():
    print("=" * 60)
    print("  Phase 3 -- Image Expert Test")
    print("=" * 60)

    r = requests.get(f"{BASE_URL}/health", timeout=30)
    data = r.json()
    print(f"\n[Health] experts_loaded: {data['experts_loaded']}")
    assert "image" in data["experts_loaded"], "Image expert not loaded!"
    print("[Health] PASS -- image expert is registered")

    print("\n--- Gate routing ---")
    image_queries = [
        "describe the architecture diagram on page 3",
        "what does the pie chart show",
        "explain the image of the workflow",
    ]
    for q in image_queries:
        r = requests.post(f"{BASE_URL}/gate", json={"query": q}, timeout=30)
        top = max(r.json()["raw"], key=r.json()["raw"].get)
        score = r.json()["raw"][top]
        status = "PASS" if top == "image" else "FAIL"
        print(f"  [{status}] \"{q}\" -> {top} ({score:.2f})")

    print("\n--- Image Ingestion ---")
    with open(TEST_IMG, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest",
            files={"file": ("test_chart.png", f, "image/png")},
            data={"org_id": "default"},
            timeout=180,
        )
    data = r.json()
    print(f"  Status: {data.get('status')}")
    print(f"  Chunks: {data.get('total_chunks')}")
    print(f"  Experts: {data.get('experts_used')}")
    print(f"  Latency: {data.get('latency_ms')}ms")
    assert data["status"] == "indexed"
    assert "image" in data["experts_used"]
    print("  PASS -- Image ingested via image expert")

    print("\n--- Image Queries ---")
    queries = [
        "What does the chart show about revenue trends?",
        "What was the highest quarterly revenue?",
    ]
    for q in queries:
        print(f"\n  Q: \"{q}\"")
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/query",
            json={"query": q, "org_id": "default", "top_k": 5},
            timeout=180,
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
    print("  Phase 3 test complete")
    print("=" * 60)


if __name__ == "__main__":
    run()
