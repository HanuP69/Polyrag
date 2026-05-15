function rrfFuse(expertResults, gateWeights, k = 60) {
  const scores = {};
  const chunkMap = {};

  for (const [key, result] of Object.entries(expertResults)) {
    const expertId = key.replace(/_bm25$/, "").replace(/_fallback$/, "");
    const weight = gateWeights[expertId] || 0.3;
    const isBM25 = key.endsWith("_bm25");
    const bm25Boost = isBM25 ? 0.3 : 1.0;

    const chunks = result.chunks || result;
    if (!Array.isArray(chunks)) continue;

    chunks.forEach((chunk, rank) => {
      const id = chunk.chunk_id;
      if (!scores[id]) scores[id] = 0;
      scores[id] += (weight * bm25Boost) / (k + rank + 1);
      if (!chunkMap[id]) chunkMap[id] = chunk;
    });
  }

  return Object.entries(scores)
    .sort(([, a], [, b]) => b - a)
    .map(([id, score]) => ({ ...chunkMap[id], rrf_score: score }));
}

module.exports = { rrfFuse };
