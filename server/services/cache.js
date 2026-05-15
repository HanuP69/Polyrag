const { LRUCache } = require("lru-cache");
const crypto = require("crypto");

const cache = new LRUCache({
  max: 500,
  ttl: 1000 * 60 * 30,
});

function hashKey(query, orgId) {
  const hash = crypto.createHash("md5").update(`${orgId}:${query}`).digest("hex");
  return hash;
}

function get(query, orgId) {
  return cache.get(hashKey(query, orgId));
}

function set(query, orgId, value) {
  cache.set(hashKey(query, orgId), value);
}

function clear() {
  cache.clear();
}

function stats() {
  return {
    size: cache.size,
    max: cache.max,
  };
}

module.exports = { get, set, clear, stats };
