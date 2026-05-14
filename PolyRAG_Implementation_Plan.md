# PolyRAG — Zero to Hero Implementation Plan

> A domain-agnostic, multimodal RAG engine with MoE-style expert routing.
> Built solo. No deadlines. Done right.

---

## Table of Contents

1. [What We're Building](#1-what-were-building)
2. [Final Stack & Why](#2-final-stack--why)
3. [Project Structure](#3-project-structure)
4. [Database Schema](#4-database-schema)
5. [API Contracts](#5-api-contracts)
6. [Phase 0 — Gate Classifier](#6-phase-0--gate-classifier)
7. [Phase 1 — Text Expert, Full Pipeline](#7-phase-1--text-expert-full-pipeline)
8. [Phase 2 — Table Expert](#8-phase-2--table-expert)
9. [Phase 3 — Image Expert](#9-phase-3--image-expert)
10. [Phase 4 — Harden](#10-phase-4--harden)
11. [Query Flow (Final)](#11-query-flow-final)
12. [Ingestion Flow (Final)](#12-ingestion-flow-final)
13. [Cost Profile](#13-cost-profile)
14. [Build Order & Milestones](#14-build-order--milestones)
15. [Resume Framing](#15-resume-framing)

---

## 1. What We're Building

Standard RAG is monomodal. One vector space, one retriever, hardcoded pipeline.
Real-world documents are heterogeneous — a legal PDF has prose, tables, and diagrams.
Searching all of them through a single text index produces garbage results.

**PolyRAG solves this with MoE-style routing applied to retrieval:**

- N expert retrievers, one per modality (text, table, image)
- A learned gate that routes each query to the right experts
- Per-modality vector indexes so embeddings never pollute each other
- RRF fusion to merge results across experts into a single ranked list
- One LLM call per query for generation — everything else is local and free

The architecture is modular by design. Adding a new modality means writing one new expert class and registering it. Nothing else changes.

---

## 2. Final Stack & Why

| Layer | Tool | Why |
|---|---|---|
| Vector + metadata DB | PostgreSQL + pgvector | Replaces FAISS + SQLite in one tool. Free on Supabase. |
| Embeddings | BGE-M3 | Local, free, 768-dim, strong on multilingual and technical text. |
| Gate | BGE-M3 + PyTorch classifier | Tiny 3-layer net. Runs in ~5ms. Zero API cost at inference. |
| Image captioning | BLIP-2 | Half the weight of LLaVA. Good enough for retrieval. |
| Reranker | MiniLM cross-encoder | Local, free. Added in Phase 4 only — no premature optimization. |
| Orchestration | LangGraph | Models the query flow as a graph. Handles parallel retrieval natively. Added in Phase 4. |
| LLM | Groq free tier | Llama 3.1 70B. Fast inference. 1 API call per unique query. |
| Python backend | FastAPI | Async, typed, lightweight. All ML lives here. |
| Node backend | Express | Orchestration, SSE streaming, Groq calls. Never does ML. |
| Frontend | React | Simple upload + query UI. No extra UI libraries. |
| Hosting DB | Supabase free tier | Managed Postgres + pgvector. No ops overhead. |
| Queue | None until Phase 4 | Add Bull + Redis when pain is felt. Not before. |

**Tools explicitly cut:**
- LangChain — too much abstraction over things we want control over
- LlamaIndex — same problem
- FAISS standalone — pgvector does the same job without a separate service
- LLaVA — too heavy for MVP captioning
- MongoDB — SQL schema is clear, flexible schema not needed
- Redis — deferred until Bull is actually needed

---

## 3. Project Structure

```
polyrag/
│
├── engine/                         # Python — all ML lives here
│   ├── experts/
│   │   ├── base.py                 # Abstract expert class (embed, chunk, retrieve)
│   │   ├── text.py                 # Text expert
│   │   ├── table.py                # Table expert
│   │   └── image.py                # Image expert
│   │
│   ├── gate/
│   │   ├── generate_data.py        # One-time: generate synthetic training data via Groq
│   │   ├── train.py                # Train BGE-M3 + Linear classifier, serialize weights
│   │   └── gate.py                 # Inference: embed query → load weights → return expert weights
│   │
│   ├── fuse.py                     # RRF fusion math (pure Python, no ML)
│   ├── rerank.py                   # MiniLM cross-encoder (Phase 4)
│   ├── db.py                       # pgvector insert + search helpers
│   └── main.py                     # FastAPI app + all routes
│
├── server/                         # Node — orchestration only
│   ├── routes/
│   │   ├── ingest.js               # POST /api/ingest
│   │   ├── query.js                # POST /api/query
│   │   └── config.js               # GET/PUT /api/config
│   ├── workers/
│   │   └── ingestWorker.js         # Bull worker (Phase 4)
│   ├── llm.js                      # Groq client + SSE streaming
│   ├── cache.js                    # In-memory query cache (JS Map)
│   └── index.js                    # Express app entry
│
└── client/                         # React
    ├── components/
    │   ├── Upload.jsx               # File upload + ingestion progress
    │   ├── Query.jsx                # Query input + streaming answer
    │   └── Sources.jsx              # Citation cards with expert badge
    └── App.jsx
```

---

## 4. Database Schema

Single Postgres database on Supabase. pgvector extension enabled.

```sql
-- Organizations (multi-tenancy foundation)
CREATE TABLE orgs (
  org_id      TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  config      JSONB NOT NULL DEFAULT '{}'   -- gating mode, top_k, system prompt
);

-- Files ingested per org
CREATE TABLE files (
  file_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id      TEXT REFERENCES orgs(org_id),
  name        TEXT NOT NULL,
  type        TEXT NOT NULL,               -- pdf, csv, png, jpg
  status      TEXT NOT NULL DEFAULT 'uploading',
                                           -- uploading | parsing | captioning | embedding | indexed | failed
  chunk_count INTEGER DEFAULT 0,
  experts_used TEXT[],                     -- ['text', 'table', 'image']
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Chunks — one row per chunk across all experts
CREATE TABLE chunks (
  chunk_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id      TEXT REFERENCES orgs(org_id),
  file_id     UUID REFERENCES files(file_id),
  expert_id   TEXT NOT NULL,              -- 'text' | 'table' | 'image'
  content     TEXT NOT NULL,             -- raw text or caption
  metadata    JSONB NOT NULL DEFAULT '{}',
                                         -- text: {page, section}
                                         -- table: {row_count, col_names, sheet}
                                         -- image: {caption, width, height, page}
  embedding   vector(768),               -- BGE-M3 output
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Index for fast filtered vector search
CREATE INDEX ON chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Query logs — used to retrain gate over time
CREATE TABLE query_logs (
  log_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id        TEXT REFERENCES orgs(org_id),
  query         TEXT NOT NULL,
  gate_weights  JSONB,                   -- {text: 0.8, table: 0.6, image: 0.1}
  experts_fired TEXT[],
  chunk_ids     UUID[],                  -- chunks retrieved
  latency_ms    INTEGER,
  created_at    TIMESTAMPTZ DEFAULT now()
);
```

**Why one chunks table for all experts?**
The `expert_id` column filters at query time. `metadata` is JSONB so each expert can store
what it needs without schema changes. Adding a new expert = no migration.

---

## 5. API Contracts

### Node → React (external)

```
POST   /api/ingest             Upload files → triggers ingestion, returns file_ids
GET    /api/ingest/:file_id    Poll file status (status field from files table)
POST   /api/query              Query → SSE stream of answer tokens + sources
GET    /api/config             Get org config
PUT    /api/config             Update org config (experts, top_k, system prompt)
```

### Node → Python engine (internal)

```
POST   /parse                  file_path + org_id → chunks per modality
POST   /embed                  chunks → vectors → upserted to pgvector
POST   /gate                   query → {text: 0.8, table: 0.6, image: 0.1}
POST   /retrieve               query + expert_id + org_id + top_k → chunks[]
POST   /rerank                 query + chunks[] → reranked chunks[] (Phase 4)
```

**Note:** Fusion (RRF) happens in Node, not Python. It's pure math — no reason for a round-trip.

---

## 6. Phase 0 — Gate Classifier

**Build this before anything else. It's isolated, testable immediately, and the most novel part.**

### Why a learned classifier instead of LLM gating

LLM zero-shot gating costs one API call per query, adds 1-2 seconds of latency, and hits
rate limits fast during development. A local PyTorch classifier costs nothing at inference,
runs in ~5ms, and gets better as real query logs accumulate.

### Cold start: synthetic data generation

No query logs on day 1. Solution: generate them offline with one Groq call, never repeat.

**generate_data.py — what it does:**

Sends a single prompt to Groq asking for 150 example queries per class
(text, table, image), seeded with 5-6 hand-written examples per class to ensure quality.
Saves output to `data/gate_training.json`. This file is committed. Never regenerated.

Prompt structure:
```
Given these example queries that need TEXT retrieval:
  - "summarize the indemnity clause"
  - "explain how the approval process works"
  - "what does the contract say about termination"
Generate 150 more diverse queries that clearly need text retrieval.
Do the same for TABLE queries and IMAGE queries.
Return JSON only: {"text": [...], "table": [...], "image": [...]}
```

Total: 450 labeled samples. Enough for a 3-class classifier.

**train.py — what it does:**

1. Load BGE-M3, embed all 450 queries → 768-dim vectors
2. Build dataset: (vector, label) pairs, label ∈ {0, 1, 2}
3. Train classifier:
   ```
   Linear(768 → 128) → ReLU → Dropout(0.2) → Linear(128 → 3) → Sigmoid
   ```
4. Loss: BCEWithLogitsLoss (multi-label — a query can route to multiple experts)
5. Train for 20 epochs, 80/20 split, save best weights to `gate_model.pt`

**gate.py — inference:**

1. Load BGE-M3 + `gate_model.pt` once at startup (not per request)
2. Per query: embed → forward pass → sigmoid → threshold at 0.4
3. Return active experts: `{text: 0.82, table: 0.61, image: 0.09}`
4. Threshold mask: experts with score > 0.4 are fired

**Phase 0 milestone:**

Gate correctly routes:
- "how many rows have revenue > 100k" → table
- "describe the architecture diagram on page 3" → image
- "summarize the force majeure clause" → text
- "compare Q1 and Q2 sales figures from the chart" → table + image (multi-label)

**Gate retraining (ongoing, automatic):**

Once real query logs accumulate (target: 200+ real queries), retrain on real data.
`train.py` already handles this — just point it at query_logs table instead of synthetic file.
Gate silently improves over time with zero extra work.

---

## 7. Phase 1 — Text Expert, Full Pipeline

**Goal: entire system working end-to-end for text only. Nothing else matters until this milestone is hit.**

### Why text first

Proves the full architecture — gate, retrieve, fuse, generate, stream — before adding
complexity. If the pipeline is broken, you want to know with the simplest expert, not
after building all three.

### base.py — abstract expert

Every expert implements:
```python
class BaseExpert(ABC):
    expert_id: str          # 'text' | 'table' | 'image'
    embed_dim: int          # must be 768 for all experts (pgvector schema fixed)

    @abstractmethod
    def parse(self, file_path: str) -> list[Chunk]:
        """Extract chunks from file. Returns list of Chunk objects."""

    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[np.ndarray]:
        """Embed chunks. Returns list of 768-dim vectors."""

    def retrieve(self, query_vec: np.ndarray, org_id: str, top_k: int) -> list[Chunk]:
        """pgvector cosine search filtered by org_id + expert_id. Shared by all experts."""
```

Retrieve is implemented once in base — it's the same pgvector query for every expert,
just filtered by `expert_id`. Experts only override parse and embed.

### text.py

**Parse:**
- PyMuPDF (`fitz`) extracts text blocks from PDF
- Chunk at 512 tokens, 64 token overlap
- Respect sentence boundaries — don't cut mid-sentence
- Each chunk tagged with page number and section heading (detected from font size)

**Embed:**
- BGE-M3 via HuggingFace `sentence-transformers`
- Batch embed (32 chunks per batch) for speed
- Returns 768-dim vectors

**Metadata stored:**
```json
{ "page": 4, "section": "Indemnity Clause", "char_offset": 1240 }
```

### FastAPI routes (main.py)

```
POST /parse     → detect file type → call expert.parse() → return chunks
POST /embed     → call expert.embed() → upsert to pgvector via db.py
POST /gate      → gate.py inference → return expert weights dict
POST /retrieve  → embed query → pgvector cosine search → return top-k chunks
```

### Node orchestration (query.js)

```
1. Load org config from Postgres
2. Check in-memory cache: hash(query + org_id) → return cached if hit
3. POST /gate → get active experts
4. Promise.all: POST /retrieve for each active expert (parallel)
5. RRF fusion in Node (pure JS math, no round-trip to Python)
6. Build prompt: system prompt from config + query + formatted chunks with sources
7. Groq streaming call → pipe SSE to React
8. Save query_log async (don't block response)
9. Store result in cache
```

### RRF fusion in Node

```javascript
// Reciprocal Rank Fusion — no ML, pure math
function rrf(expertResults, weights, k = 60) {
  const scores = {};
  for (const [expertId, chunks] of Object.entries(expertResults)) {
    const weight = weights[expertId];
    chunks.forEach((chunk, rank) => {
      scores[chunk.chunk_id] = (scores[chunk.chunk_id] || 0)
        + weight / (k + rank + 1);
    });
  }
  return Object.entries(scores)
    .sort(([,a], [,b]) => b - a)
    .map(([id]) => chunkById[id]);
}
```

### In-memory query cache

```javascript
const cache = new Map();  // resets on restart, fine for dev
const MAX_CACHE = 500;

function getCached(query, orgId) {
  return cache.get(`${orgId}:${hashQuery(query)}`);
}
```

No Redis. No external service. Works for dev and light production.

### React (Phase 1 UI)

- File upload input → POST /api/ingest → poll /api/ingest/:file_id for status
- Status bar: `uploading → parsing → embedding → indexed`
- Query input → POST /api/query → render streaming tokens as they arrive
- Source cards below answer: chunk content + page number + expert badge `[TEXT]`

### Phase 1 milestone

Upload a 20-page PDF. Ask "what are the termination conditions?".
Get a streamed, grounded answer citing specific pages. One Groq call. Total latency < 2s.

---

## 8. Phase 2 — Table Expert

**Proves modularity. Adding this expert should touch minimal existing code.**

### Why tables need a separate expert

Table content embedded as raw text ("Revenue Q1 Q2 Q3 Sales 100 120 140") produces
poor semantic vectors. Linearization — "Sales | Q1: 100 | Q2: 120 | Q3: 140" — gives
the model structured context it can reason about. This also means table queries
("what was Q2 revenue?") map to a different embedding space than prose queries.
Mixing them in one index degrades retrieval for both.

### table.py

**Parse:**
- PyMuPDF layout heuristics detect table regions (text blocks with grid alignment)
- Camelot as fallback for complex bordered tables
- Linearization per row: `Col1: val | Col2: val | Col3: val`
- Whole table = one chunk (not split across chunks — context must be preserved)
- CSV files go directly to table expert (no layout detection needed)

**Embed:**
- Same BGE-M3 as text expert — linearized text is just text
- No new embedding model. No new dependency.

**Metadata stored:**
```json
{ "row_count": 12, "col_names": ["Quarter", "Revenue", "Units"], "page": 7 }
```

### Changes to existing code

- `main.py`: register table expert in expert registry (2 lines)
- `query.js`: no changes — already fires all active experts in parallel
- `gate.py`: no changes — already returns table weight
- React: add `[TABLE]` badge to source cards

That's it. Everything else is handled by the base class.

### Phase 2 milestone

Upload a PDF with embedded financial tables. Ask "what was the average revenue in Q3 across all regions?".
Gate routes to table expert. Answer uses table data. Text chunks are not retrieved.

---

## 9. Phase 3 — Image Expert

**Heaviest phase. Built last when everything else is stable and proven.**

### Why BLIP-2 over LLaVA

LLaVA requires running a 7B+ parameter model — either locally (heavy GPU requirement)
or via an API (cost). BLIP-2 is significantly lighter, runs on CPU for inference,
and produces captions good enough for retrieval matching. Switching to LLaVA later
is a one-line model swap in image.py.

### image.py

**Parse:**
- PyMuPDF `.get_images()` extracts embedded images from PDF
- PIL loads each image, saves to temp file
- BLIP-2 generates caption: "A bar chart showing quarterly revenue by region for 2023"
- Caption = the chunk content stored in Postgres
- CLIP ViT-B/32 generates 512-dim visual embedding — stored in metadata as JSON
  (not used for retrieval yet, reserved for Phase 4 visual similarity search)

**Embed:**
- BGE-M3 embeds the BLIP-2 caption → 768-dim vector
- Same column, same index, same retrieve logic as text and table
- No schema changes

**Metadata stored:**
```json
{
  "caption": "Bar chart showing quarterly revenue by region for 2023",
  "page": 11,
  "width": 800,
  "height": 600,
  "clip_embed": [0.23, -0.41, ...]
}
```

### Captioning is slow — handle it right

BLIP-2 on CPU takes 2-10 seconds per image. Don't block ingestion.

File status flow with images:
```
uploading → parsing → captioning → embedding → indexed
```

React shows current stage. User knows it's working, not hung.

### Phase 3 milestone

Upload a PDF with architecture diagrams. Ask "describe the system architecture shown in the document".
Gate routes to image expert. Answer describes the diagram using BLIP-2 caption. Correct page cited.

---

## 10. Phase 4 — Harden

**System works. Now make it production-shaped.**

### LangGraph for query orchestration

Replace ad-hoc Promise.all + manual orchestration with a proper LangGraph graph.

Nodes:
```
gate_node → parallel_retrieve_node → fuse_node → rerank_node → generate_node
```

Benefits:
- State is explicit and inspectable
- Parallel branches are declared, not manual Promise.all
- Easy to add new nodes (e.g., query rewriting, fallback logic) without restructuring
- LangGraph handles errors per node, not try/catch soup

### MiniLM reranker

After RRF fusion, top-20 chunks go through MiniLM cross-encoder.
Cross-encoder sees (query, chunk) jointly — much better relevance signal than vector cosine.
Returns top-8 to LLM.

Why Phase 4 and not earlier: reranking only helps if retrieval is already working well.
Adding it before the pipeline is stable hides retrieval bugs under reranking corrections.

### Bull + Redis for ingestion

Now that ingestion is proven synchronous, add async queue:
- Bull job per file, persistent in Redis
- Multiple workers process files in parallel
- SSE progress events per job (parsing %, embedding %)
- Retry on failure with backoff
- This is the right time — you'll have felt the pain of slow synchronous ingestion

### Multi-tenancy

Already in schema (org_id on every table). Wire it up:
- Every pgvector query: `WHERE org_id = $1 AND expert_id = $2`
- Every file: tagged with org_id at upload
- Config per org: active experts, top_k, system prompt, Groq model override
- `GET/PUT /api/config` — live config without redeploy

### Gate retraining

By Phase 4, query_logs has hundreds of real entries.
Retrain gate classifier on real data:
```
python engine/gate/train.py --source postgres --org_id all
```
Gate silently improves. Ship new `gate_model.pt`. No other changes.

### CLIP visual retrieval (optional Phase 4)

If CLIP embeds are stored in metadata from Phase 3, enable visual similarity search:
- For image queries that include an uploaded image, embed with CLIP
- Cosine similarity against stored CLIP embeds in metadata
- Secondary signal alongside caption-based retrieval

---

## 11. Query Flow (Final)

```
React
  │
  └─ POST /api/query { query, org_id }
          │
       Node
          ├─ check in-memory cache → return immediately if hit
          ├─ load org config from Postgres
          │
          └─ POST /gate (Python, ~5ms)
                  │
                  └─ { text: 0.8, table: 0.6, image: 0.1 }
                          │
                  Promise.all (parallel):
                  ├─ POST /retrieve { expert: text,  top_k: 10 }  → 10 text chunks
                  ├─ POST /retrieve { expert: table, top_k: 10 }  → 10 table chunks
                  └─ POST /retrieve { expert: image, top_k: 5  }  → 5 image chunks
                          │
                  RRF fusion in Node (~1ms)
                          │
                  POST /rerank (Phase 4, ~100ms)
                          │
                  top-8 chunks → build prompt
                          │
                  Groq streaming call (Llama 3.1 70B)
                          │
                  SSE stream → React renders tokens
                          │
                  save query_log async (non-blocking)
                  store in cache
```

**Latency budget:**
| Step | Time |
|---|---|
| Gate | ~5ms |
| BGE-M3 query embed | ~50ms |
| Parallel pgvector search | ~20ms |
| RRF fusion | ~1ms |
| MiniLM rerank (Phase 4) | ~100ms |
| Groq (streaming, first token) | ~300ms |
| **Total to first token** | **~480ms** |

---

## 12. Ingestion Flow (Final)

```
React uploads file
  │
  └─ POST /api/ingest
          │
       Node saves file record (status: uploading)
          │
       [Phase 1-3: synchronous]
       [Phase 4: Bull job queued]
          │
       POST /parse (Python)
          │
       Modality detector:
          ├─ PDF  → PyMuPDF
          │         ├─ text blocks    → text expert chunks
          │         ├─ tables         → table expert chunks
          │         └─ embedded imgs  → image expert (captioning queue)
          ├─ CSV  → table expert chunks
          └─ IMG  → image expert chunks
          │
       POST /embed per expert (parallel where possible)
          │
       pgvector upsert (chunks table)
          │
       Node updates file status → indexed
          │
       React shows ✓ indexed
```

---

## 13. Cost Profile

| Operation | Cost | How |
|---|---|---|
| Embedding at ingestion | $0 | BGE-M3 runs locally |
| Embedding at query time | $0 | BGE-M3 runs locally |
| Gate inference | $0 | Local PyTorch classifier |
| Gate training data generation | $0 one-time | Single Groq call offline |
| Vector search | $0 | pgvector on Supabase free tier |
| Reranking | $0 | MiniLM runs locally |
| Generation | 1 Groq API call | Per unique query |
| Cached queries | $0 | In-memory cache hit |
| Database hosting | $0 | Supabase free tier |

**Recurring cost: essentially zero.** Only unique queries that miss cache hit Groq.

---

## 14. Build Order & Milestones

```
Phase 0 — Gate classifier
  ✓ generate_data.py runs, produces gate_training.json
  ✓ train.py trains, produces gate_model.pt
  ✓ gate.py routes 20 hand-written queries correctly
  → MILESTONE: gate works in isolation

Phase 1 — Text pipeline
  ✓ text.py: parse + embed PDF
  ✓ pgvector: chunks upserted, retrieved
  ✓ Node: ingest → query → Groq → SSE stream
  ✓ React: upload UI + streaming answer + source citations
  → MILESTONE: upload PDF, ask question, get grounded streamed answer

Phase 2 — Table expert
  ✓ table.py: detect + linearize + embed tables
  ✓ gate routes numerical queries to table expert
  ✓ RRF fusion merges text + table results
  ✓ React: [TABLE] badge on citations
  → MILESTONE: table query uses table expert, not text

Phase 3 — Image expert
  ✓ image.py: extract images + BLIP-2 caption + embed
  ✓ captioning shows in file status progress
  ✓ image queries retrieve caption-based chunks
  → MILESTONE: diagram query returns caption with correct page

Phase 4 — Harden
  ✓ LangGraph replaces ad-hoc query orchestration
  ✓ MiniLM reranker on fused results
  ✓ Bull + Redis for async ingestion
  ✓ org_id isolation enforced everywhere
  ✓ GET/PUT /api/config live config
  ✓ Gate retrained on real query logs
  → MILESTONE: multi-tenant, production-shaped, gate improving on real data
```

**Rule: never start next phase until current milestone is fully hit.**
No horizontal building. Vertical slices only.

---

## 15. Resume Framing

**Wrong framing (too AI-hype, no signal):**
> "Built a multimodal RAG system using LLMs and vector databases"

**Right framing (systems signal, explainable decisions):**
> "Designed and built a multi-service retrieval system with MoE-style expert routing across
> text, table, and image modalities. Trained a local PyTorch gate classifier on synthetic data
> to route queries with zero inference API cost. Used pgvector for per-modality vector search
> with org-level isolation, RRF fusion for cross-modal result merging, and SSE streaming for
> real-time answer delivery. One LLM API call per unique query."

**Questions you must be able to answer cold:**
- Why separate vector indexes per modality instead of one?
- Why RRF over learned fusion for MVP?
- What's the cold start problem for the gate and how did you solve it?
- Why pgvector instead of FAISS or Qdrant?
- Why BLIP-2 over LLaVA?
- What happens when the gate misroutes a query?
- How does multi-tenancy work at the vector search layer?
- Why Node for orchestration and Python for ML?
- What's the latency breakdown per query step?

If you can answer all of these, you own the project. Interviewers will know.
