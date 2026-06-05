import requests
import os
import time
import json

BASE_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_PDF = os.path.join(DATA_DIR, "test_contract.pdf")

def run():
    print("=" * 60)
    print("  v4 Hardening Test")
    print("=" * 60)

    print("\n--- 1. Org Config API ---")
    config_data = {
        "name": "Acme Corp",
        "config": {
            "top_k": 5,
            "system_prompt": "You are Acme Corp's legal assistant. Answer politely.",
        },
    }
    r = requests.put(f"{BASE_URL}/config/acme_org", json=config_data)
    print(f"  PUT /config/acme_org: {r.status_code}")
    assert r.status_code == 200

    r = requests.get(f"{BASE_URL}/config/acme_org")
    data = r.json()
    print(f"  GET /config/acme_org: {data.get('name')}")
    assert data["name"] == "Acme Corp"
    assert data["config"]["top_k"] == 5
    print("  [PASS] Org Config API works")

    print("\n--- 2. Async Ingestion ---")
    with open(TEST_PDF, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest/async",
            files={"file": ("test_contract.pdf", f, "application/pdf")},
            data={"org_id": "acme_org"},
        )
    data = r.json()
    print(f"  POST /ingest/async status: {data.get('status')}")
    file_id = data.get("file_id")
    assert file_id

    for i in range(120):
        r = requests.get(f"{BASE_URL}/file/{file_id}")
        status_data = r.json()
        print(f"  Poll {i}: {status_data['status']} | chunks: {status_data.get('chunk_count', '?')}")
        if status_data["status"] in ("completed", "indexed", "error", "failed"):
            break
        time.sleep(2)

    assert status_data["status"] not in ("error", "failed"), f"Ingestion failed: {status_data}"
    print("  [PASS] Async ingestion works")

    print("\n--- 3. SSE Streaming Query ---")
    query_payload = {
        "query": "What are the termination conditions for convenience?",
        "org_id": "acme_org",
        "top_k": 5,
        "system_prompt": "Answer very briefly.",
    }

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/generate/stream",
        json={"prompt": query_payload["query"], "query": query_payload["query"]},
        stream=True,
    )

    print("  Streaming response...")
    full_answer = ""
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        full_answer += event["content"]
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "done":
                        print(f"\n  [Done] Latency: {event.get('latency_ms', '?')}ms")
                except json.JSONDecodeError:
                    pass

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.2f}s")
    assert len(full_answer) > 10
    print("  [PASS] SSE Streaming works")

    print("\n--- 4. Rerank endpoint ---")
    chunks = [
        {"chunk_id": "c1", "content": "The contract may be terminated for convenience with 30 days notice.", "modality": "text"},
        {"chunk_id": "c2", "content": "Payment terms are net 30.", "modality": "text"},
        {"chunk_id": "c3", "content": "Termination for cause requires written notice and 15-day cure period.", "modality": "text"},
    ]
    r = requests.post(
        f"{BASE_URL}/rerank",
        json={"query": "termination conditions", "chunks": chunks},
        timeout=60,
    )
    data = r.json()
    print(f"  Reranked {len(data['chunks'])} chunks")
    for c in data["chunks"]:
        print(f"    {c['chunk_id']}: {c['content'][:60]}...")
    assert len(data["chunks"]) > 0
    print("  [PASS] Reranker works")

    print("\n" + "=" * 60)
    print("  v4 hardening test complete")
    print("=" * 60)

if __name__ == "__main__":
    run()
