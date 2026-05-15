import requests
import os
import time
import json

BASE_URL = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEST_PDF = os.path.join(DATA_DIR, "test_contract.pdf")

def run():
    print("=" * 60)
    print("  Phase 4 -- Hardening Test")
    print("=" * 60)

    # 1. Org Config API
    print("\n--- 1. Org Config API ---")
    
    # Put config
    config_data = {
        "name": "Acme Corp",
        "config": {
            "active_experts": ["text", "table"],
            "top_k": 5,
            "system_prompt": "You are Acme Corp's legal assistant. Answer politely."
        }
    }
    r = requests.put(f"{BASE_URL}/config/acme_org", json=config_data)
    print(f"  PUT /config/acme_org: {r.status_code}")
    assert r.status_code == 200
    
    # Get config
    r = requests.get(f"{BASE_URL}/config/acme_org")
    data = r.json()
    print(f"  GET /config/acme_org: {data.get('name')}")
    assert data["name"] == "Acme Corp"
    assert data["config"]["top_k"] == 5
    print("  [PASS] Org Config API works")

    # 2. Async Ingestion
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
    
    # Poll status
    max_polls = 60
    polls = 0
    while polls < max_polls:
        r = requests.get(f"{BASE_URL}/ingest/status/{file_id}")
        status_data = r.json()
        print(f"  Polling {polls}: {status_data['status']} ({status_data.get('progress')}%)")
        if status_data["status"] in ["indexed", "failed"]:
            break
        time.sleep(1)
        polls += 1
        
    assert status_data["status"] == "indexed"
    print("  [PASS] Async ingestion works")

    # 3. SSE Streaming + Reranker (Reranker is implicit in the query pipeline)
    print("\n--- 3. SSE Streaming Query ---")
    query_payload = {
        "query": "What are the termination conditions for convenience?",
        "org_id": "acme_org",
        "top_k": 5,
        "system_prompt": "Answer very briefly."
    }
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/query/stream", 
        json=query_payload, 
        stream=True
    )
    
    print(f"  Streaming response...")
    full_answer = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    event = json.loads(data_str)
                    if event["type"] == "meta":
                        print(f"  [Meta] Gate Weights: {event['gate']}")
                        print(f"  [Meta] Sources retrieved: {len(event['sources'])}")
                        assert len(event['sources']) > 0
                    elif event["type"] == "token":
                        full_answer += event["content"]
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "done":
                        print(f"\n  [Done] Latency: {event['latency_ms']}ms")
                except json.JSONDecodeError:
                    pass

    elapsed = time.time() - start
    print(f"\n  Full Answer: {full_answer}")
    print(f"  Total time: {elapsed:.2f}s")
    assert len(full_answer) > 10
    print("  [PASS] SSE Streaming and Query Pipeline work")
    
    print("\n" + "=" * 60)
    print("  Phase 4 test complete")
    print("=" * 60)

if __name__ == "__main__":
    run()
