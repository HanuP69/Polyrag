"""Warm benchmark — different query to avoid cache."""
import requests, time

s = time.time()
r = requests.post("http://localhost:8000/query", json={
    "query": "what are the topics in unit 1 about agents",
    "org_id": "default",
    "top_k": 5
}, timeout=120)
e = time.time()

d = r.json()
print(f"Total round-trip: {int((e-s)*1000)}ms")
print(f"Server latency:   {d.get('latency_ms')}ms")
print(f"Sources:          {len(d.get('sources', []))}")
guard = d.get('guard', {})
if guard:
    print(f"Guard score:      {guard.get('score', 'N/A')}")
    print(f"Guard verified:   {guard.get('verified_count')}/{guard.get('total_claims')}")
print(f"Answer preview:   {d.get('answer', '')[:200]}...")
