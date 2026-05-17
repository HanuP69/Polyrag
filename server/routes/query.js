const express = require("express");
const router = express.Router();
const engine = require("../services/engine");
const cache = require("../services/cache");
const { rrfFuse } = require("../services/fuse");

const GATE_THRESHOLD = 0.4;

router.post("/api/query", async (req, res) => {
  const start = Date.now();
  const { query, top_k = 10, system_prompt, model, chat_history = [], file_ids } = req.body;
  const org_id = req.user?.id || "default";

  if (!query) {
    return res.status(400).json({ error: "query is required" });
  }

  const cached = cache.get(query, org_id);
  if (cached) {
    return res.json({ ...cached, cached: true, latency_ms: Date.now() - start });
  }

  try {
    let gateResult = await engine.gate(query);
    let gateWeights = gateResult.raw || gateResult;

    let finalQuery = query;

    const activeExperts = Object.entries(gateWeights).filter(
      ([, w]) => w > GATE_THRESHOLD
    );

    const retrievalPromises = [];
    const resultKeys = [];

    for (const [expertId] of activeExperts) {
      retrievalPromises.push(
        engine.retrieve(finalQuery, expertId, org_id, top_k, file_ids).catch((err) => {
          console.error(`[Query] Vector retrieve ${expertId} failed:`, err.message);
          return { chunks: [] };
        })
      );
      resultKeys.push(expertId);

      retrievalPromises.push(
        engine.retrieveBM25(finalQuery, expertId, org_id, 5).catch((err) => {
          console.error(`[Query] BM25 retrieve ${expertId} failed:`, err.message);
          return { chunks: [], total: 0 };
        })
      );
      resultKeys.push(`${expertId}_bm25`);
    }

    const retrievalResults = await Promise.all(retrievalPromises);

    const expertResults = {};
    resultKeys.forEach((key, i) => {
      const result = retrievalResults[i];
      const chunks = result?.chunks || (Array.isArray(result) ? result : []);
      if (chunks.length > 0) {
        expertResults[key] = { chunks };
      }
    });

    let fused = rrfFuse(expertResults, gateWeights);

    if (fused.length > 0) {
      const avgSim = fused.reduce((sum, c) => sum + (c.rrf_score || 0), 0) / fused.length;
      if (avgSim < 0.001 && activeExperts.length < 3) {
        console.log(`[Query] Fallback cascade triggered (avgRRF=${avgSim.toFixed(4)})`);
        const allExpertIds = ["text", "table", "image"];
        const missingExperts = allExpertIds.filter(
          (id) => !activeExperts.some(([eid]) => eid === id)
        );

        const fallbackPromises = missingExperts.map((expertId) =>
          engine.retrieve(finalQuery, expertId, org_id, top_k, file_ids).catch(() => ({ chunks: [] }))
        );
        const fallbackResults = await Promise.all(fallbackPromises);
        missingExperts.forEach((expertId, i) => {
          const result = fallbackResults[i];
          const chunks = result?.chunks || (Array.isArray(result) ? result : []);
          if (chunks.length > 0) {
            expertResults[`${expertId}_fallback`] = { chunks };
            gateWeights[expertId] = 0.3;
          }
        });

        fused = rrfFuse(expertResults, gateWeights);
      }
    }

    const topChunks = fused.slice(0, 20);

    let reranked = topChunks;
    try {
      reranked = await engine.rerankChunks(finalQuery, topChunks);
      if (Array.isArray(reranked)) {
        reranked = reranked.slice(0, 8);
      } else if (reranked && reranked.chunks) {
        reranked = reranked.chunks.slice(0, 8);
      } else {
        reranked = topChunks.slice(0, 8);
      }
    } catch (err) {
      console.error("[Query] Rerank failed:", err.message);
      reranked = topChunks.slice(0, 8);
    }

    const sources = reranked.map((c) => ({
      chunk_id: c.chunk_id,
      expert_id: c.expert_id,
      content: (c.content || "").slice(0, 1000),
      metadata: c.metadata || {},
    }));

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    res.write(
      `data: ${JSON.stringify({
        type: "meta",
        gate: gateWeights,
        sources,

        active_experts: activeExperts.map(([id]) => id),
      })}\n\n`
    );

    let fullAnswer = "";
    let earlyGuardPromise = null;
    try {
      const streamResp = await engine.streamGenerate(
        buildPrompt(finalQuery, reranked, system_prompt),
        finalQuery,
        model,
        chat_history
      );

      await new Promise((resolve, reject) => {
        streamResp.data.on("data", (chunk) => {
          const lines = chunk.toString().split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const event = JSON.parse(line.slice(6));
                if (event.type === "token") {
                  fullAnswer += event.content;
                  res.write(`data: ${JSON.stringify({ type: "token", content: event.content })}\n\n`);

                  if (fullAnswer.length > 500 && !earlyGuardPromise) {
                    earlyGuardPromise = engine.guard(
                      fullAnswer,
                      sources.map((s) => s.content)
                    ).catch(() => null);
                  }
                }
              } catch {}
            }
          }
        });
        streamResp.data.on("end", resolve);
        streamResp.data.on("error", reject);
      });
    } catch (err) {
      fullAnswer = `[LLM Error] ${err.message}`;
      res.write(`data: ${JSON.stringify({ type: "token", content: fullAnswer })}\n\n`);
    }

    let guardResult = null;
    try {
      const fullGuardPromise = engine.guard(
        fullAnswer,
        sources.map((s) => s.content)
      ).catch(() => null);

      if (earlyGuardPromise) {
        guardResult = await Promise.race([earlyGuardPromise, fullGuardPromise]);
      } else {
        guardResult = await fullGuardPromise;
      }
    } catch (err) {
      console.error("[Query] Guard failed:", err.message);
    }

    const elapsed = Date.now() - start;
    res.write(
      `data: ${JSON.stringify({
        type: "guard",
        verified: guardResult ? guardResult.verified : null,
        score: guardResult ? guardResult.score : null,
        claims: guardResult ? guardResult.claims : [],
      })}\n\n`
    );
    res.write(
      `data: ${JSON.stringify({ type: "done", latency_ms: elapsed })}\n\n`
    );
    res.end();

    cache.set(query, org_id, {
      answer: fullAnswer,
      sources,
      gate_weights: gateWeights,
      guard: guardResult,
      latency_ms: elapsed,
    });
  } catch (err) {
    console.error("[Query] Pipeline error:", err.message);
    if (!res.headersSent) {
      return res.status(500).json({ error: err.message });
    }
    res.write(`data: ${JSON.stringify({ type: "error", message: err.message })}\n\n`);
    res.end();
  }
});

router.post("/api/query/sync", async (req, res) => {
  const { query, org_id = "default", top_k = 10 } = req.body;
  if (!query) return res.status(400).json({ error: "query is required" });

  try {
    const cached = cache.get(query, org_id);
    if (cached) return res.json({ ...cached, cached: true });

    const gateResult = await engine.gate(query);
    const gateWeights = gateResult.raw || gateResult;

    const activeExperts = Object.entries(gateWeights).filter(([, w]) => w > GATE_THRESHOLD);

    const results = await Promise.all(
      activeExperts.map(([expertId]) =>
        engine.retrieve(query, expertId, org_id, top_k).catch(() => ({ chunks: [] }))
      )
    );

    const expertResults = {};
    activeExperts.forEach(([expertId], i) => {
      const chunks = results[i]?.chunks || (Array.isArray(results[i]) ? results[i] : []);
      if (chunks.length > 0) {
        expertResults[expertId] = { chunks };
      }
    });

    const fused = rrfFuse(expertResults, gateWeights).slice(0, 8);

    res.json({
      gate: gateWeights,
      sources: fused,
      total: fused.length,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

function buildPrompt(query, chunks, systemPrompt) {
  const sys =
    systemPrompt ||
    "You are a document (or codebase given the content) Q&A assistant. You MUST answer ONLY using the provided sources below. " +
    "Do NOT use your own knowledge. Every claim you make must come from the sources. " +
    "Cite sources using [Source N] notation. If the sources don't answer the question, say: " +
    "'The uploaded documents do not contain information about this topic.'";

  let context = "";
  (chunks || []).forEach((chunk, i) => {
    const content = (chunk.content || "").slice(0, 1500);
    context += `\n[Source ${i + 1} (${chunk.expert_id || "unknown"})]:\n${content}\n`;
  });

  return `${sys}\n\n--- Sources ---\n${context}\n\n--- Question ---\n${query}`;
}

module.exports = router;