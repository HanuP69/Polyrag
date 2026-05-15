import sqlite3
import json
import numpy as np
import sys
import os

# Add current directory to path so engine imports work
sys.path.append(os.getcwd())

from engine.db import search_chunks, search_bm25
from engine.experts.text import TextExpert

query = 'whats condition of himachal'
expert = TextExpert()
query_vec = expert.embed_query(query)
org_id = 'default'

print(f"Testing Query: {query}")

vec_results = expert.retrieve(query_vec, org_id, 30)
print(f"\nVector Results: {len(vec_results)}")
for c in vec_results[:10]:
    sim = c.metadata.get("similarity", 0)
    print(f"[{sim:.3f}] (Page {c.metadata.get('page')}) {c.content[:150]}...")

bm25_results = search_bm25(query, org_id, 'text', 15)
print(f"\nBM25 Results: {len(bm25_results)}")
for c in bm25_results[:10]:
    score = c.metadata.get("bm25_score", 0)
    print(f"[{score:.3f}] (Page {c.metadata.get('page')}) {c.content[:150]}...")
