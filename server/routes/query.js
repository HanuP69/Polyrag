const express = require("express");
const router = express.Router();
const engine = require("../services/engine");
const cache = require("../services/cache");

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
    const retrieveResult = await engine.retrieve(query, org_id, top_k, file_ids, model);
    const chunks = retrieveResult?.chunks || [];

    const sources = chunks.slice(0, 8).map((c) => ({
      chunk_id: c.chunk_id,
      modality: c.modality || c.expert_id,
      expert_id: c.expert_id || c.modality,
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
        sources,
      })}\n\n`
    );

    let fullAnswer = "";
    try {
      const streamResp = await engine.streamGenerate(
        buildPrompt(query, chunks.slice(0, 8), system_prompt),
        query,
        model,
        chat_history,
        org_id
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
      guardResult = await engine.guard(
        fullAnswer,
        sources.map((s) => s.content)
      ).catch(() => null);
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

    const retrieveResult = await engine.retrieve(query, org_id, top_k);
    const chunks = retrieveResult?.chunks || [];

    res.json({
      sources: chunks.slice(0, 8),
      total: chunks.length,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

function buildPrompt(query, chunks, systemPrompt) {
  const sys =
    systemPrompt ||
    "You are an elite document and codebase Q&A assistant. You MUST answer ONLY using the provided sources below. " +
    "Every claim you make must come from the sources and be direct and grounded.\n\n" +
    "FORMATTING REQUIREMENTS:\n" +
    "1. Structure your answer beautifully using clean Markdown headers (##, ###).\n" +
    "2. Present key points, facts, or instructions in clear bulleted lists or numbered lists.\n" +
    "3. Wrap all code snippets in complete, syntax-highlighted markdown code blocks (e.g. ```python, ```javascript).\n" +
    "4. Cite sources using [Source N] notation (e.g. [Source 1]).\n" +
    "5. TABLE FORMATTING RULE: If the user query asks for results, performance, metrics, comparisons, or statistics, or if any retrieved source contains tabular data (e.g., chunks starting with '[TABLE]' or containing Markdown table rows), you MUST present this data in a clean Markdown table format with columns and rows. Do not present it as a plain text list or simple bullet points.\n" +
    "6. CRITICAL IMAGE EMBEDDING RULE: If a retrieved source is an image and is directly relevant to answering the user's query, you MUST display/embed the image inline in your response using markdown format: ![Short Description of Image](source_N) where N is the source number. Generate the description dynamically based on the image caption/OCR (e.g. if it is a results table, use ![Results Table](source_N)). Do NOT embed images that are unrelated or irrelevant to what the user specifically asked for. You MUST output the actual markdown tag inline for relevant images. CRITICAL: Do NOT wrap the image markdown tag in code blocks, code tags, or backticks (e.g. do NOT output ```markdown\n![alt](source_1)\n```). Write it directly as plain text in your response so the markdown parser can render it visually.\n" +
    "7. Avoid conversational filler or salutations (e.g. do not say 'Sure, here is...' or 'Hope this helps!'). Be direct, precise, and highly informative.";

  let context = "";
  (chunks || []).forEach((chunk, i) => {
    const content = (chunk.content || "").slice(0, 1500);
    const modality = chunk.modality || chunk.expert_id || "unknown";
    let metaStr = modality;
    if ((modality === "image" || chunk.expert_id === "image") && chunk.metadata?.source) {
      metaStr += `, Image URL: /api/uploads/${chunk.metadata.source}`;
    }
    context += `\n[Source ${i + 1} (${metaStr})]:\n${content}\n`;
  });

  const hasImageSource = (chunks || []).some(c => c.modality === "image" || c.expert_id === "image");
  const reminder = hasImageSource
    ? "\n\nCRITICAL REMINDER: Retrieved sources contain one or more images. If any retrieved image is relevant to answering the user's query, you MUST embed it inline using markdown format: ![Short Description of Image](source_N) (where N is the source number). Do NOT wrap it in code blocks, code tags, or backticks! Write the markdown tag directly in your response text."
    : "";

  return `${sys}\n\n--- Sources ---\n${context}\n\n--- Question ---\n${query}${reminder}`;
}

module.exports = router;