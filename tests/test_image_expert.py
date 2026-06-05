import requests
import os
import time

BASE_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_IMG = os.path.join(DATA_DIR, "test_chart.png")


def run():
    print("=" * 60)
    print("  v4 Image Modality Test")
    print("=" * 60)

    r = requests.get(f"{BASE_URL}/health", timeout=30)
    data = r.json()
    print(f"\n[Health] mode: {data['mode']}")
    assert data["mode"] == "v4"
    print("[Health] PASS")

    print("\n--- Image Ingestion ---")
    with open(TEST_IMG, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest/async",
            files={"file": ("test_chart.png", f, "image/png")},
            data={"org_id": "default"},
            timeout=60,
        )
    data = r.json()
    file_id = data.get("file_id")
    print(f"  File ID: {file_id}")
    assert file_id

    for i in range(120):
        r = requests.get(f"{BASE_URL}/file/{file_id}", timeout=10)
        status_data = r.json()
        print(f"  Poll {i}: {status_data.get('status')} | chunks: {status_data.get('chunk_count', '?')}")
        if status_data.get("status") in ("completed", "indexed", "error", "failed"):
            break
        time.sleep(2)

    assert status_data.get("status") not in ("error", "failed"), f"Ingestion failed: {status_data}"
    print("  PASS -- Image ingested")

    print("\n--- Image Queries via Hybrid Retrieval ---")
    queries = [
        "What does the chart show about revenue trends?",
        "What was the highest quarterly revenue?",
    ]
    for q in queries:
        print(f"\n  Q: \"{q}\"")
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/retrieve",
            json={"query": q, "org_id": "default", "top_k": 5},
            timeout=60,
        )
        elapsed = time.time() - start
        data = r.json()
        chunks = data.get("chunks", [])
        print(f"  Chunks: {len(chunks)}")
        for c in chunks[:2]:
            print(f"    [{c.get('modality', '?').upper()}] {c.get('content', '')[:120]}...")
        print(f"  Latency: {elapsed:.1f}s")
        assert len(chunks) > 0

    print("\n" + "=" * 60)
    print("  Image modality test complete")
    print("=" * 60)


if __name__ == "__main__":
    run()
