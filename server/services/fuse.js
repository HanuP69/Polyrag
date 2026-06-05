function rrfFuse(rankedLists, k = 60) {
  const scores = {};
  const chunkMap = {};

  for (const chunks of rankedLists) {
    if (!Array.isArray(chunks)) continue;
    chunks.forEach((chunk, rank) => {
      const id = chunk.chunk_id;
      if (!scores[id]) scores[id] = 0;
      scores[id] += 1 / (k + rank + 1);
      if (!chunkMap[id]) chunkMap[id] = chunk;
    });
  }

  return Object.entries(scores)
    .sort(([, a], [, b]) => b - a)
    .map(([id, score]) => ({ ...chunkMap[id], rrf_score: score }));
}

module.exports = { rrfFuse };
