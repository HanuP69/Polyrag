const axios = require("axios");
const http = require("http");

const ENGINE_URL = process.env.ENGINE_URL || "http://localhost:8000";
const TIMEOUT = 30000;

// Keep-alive pool: reuse TCP connections to Python engine
const keepAliveAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 20,
  maxFreeSockets: 5,
  timeout: 60000,
});

const client = axios.create({
  baseURL: ENGINE_URL,
  timeout: TIMEOUT,
  headers: { "Content-Type": "application/json" },
  httpAgent: keepAliveAgent,
});

async function gate(query) {
  const { data } = await client.post("/gate", { query });
  return data;
}

async function rewrite(query, chatHistory = [], model = null) {
  const { data } = await client.post("/rewrite", { query, chat_history: chatHistory, model });
  return data;
}

async function retrieve(query, expertId, orgId, topK = 10) {
  const { data } = await client.post("/retrieve", {
    query,
    expert_id: expertId,
    org_id: orgId,
    top_k: topK,
  });
  return data;
}

async function retrieveBM25(query, expertId, orgId, topK = 5) {
  const { data } = await client.post("/retrieve/bm25", {
    query,
    expert_id: expertId,
    org_id: orgId,
    top_k: topK,
  });
  return data;
}

async function rerankChunks(query, chunks) {
  const { data } = await client.post("/rerank", { query, chunks });
  return data;
}

async function guard(answer, sources) {
  const { data } = await client.post("/guard", { answer, sources });
  return data;
}

function streamGenerate(prompt, query, model, chatHistory = []) {
  return client.post(
    "/generate/stream",
    { prompt, query, model, chat_history: chatHistory },
    { responseType: "stream", timeout: 120000 }
  );
}

async function ingestFile(filePath, orgId) {
  const FormData = require("form-data");
  const fs = require("fs");
  const form = new FormData();
  form.append("file", fs.createReadStream(filePath));
  form.append("org_id", orgId);
  const { data } = await client.post("/ingest/async", form, {
    headers: form.getHeaders(),
    timeout: 60000,
  });
  return data;
}

async function getIngestStatus(fileId) {
  const { data } = await client.get(`/ingest/status/${fileId}`);
  return data;
}

async function getOrgConfig(orgId) {
  const { data } = await client.get(`/config/${orgId}`);
  return data;
}

async function updateOrgConfig(orgId, name, config) {
  const { data } = await client.put(`/config/${orgId}`, { name, config });
  return data;
}

async function submitFeedback(queryLogId, rating, correctExpert = null) {
  const { data } = await client.post("/feedback", {
    query_log_id: queryLogId,
    rating,
    correct_expert: correctExpert,
  });
  return data;
}

async function getPipelineHealth(orgId = null) {
  const params = orgId ? { org_id: orgId } : {};
  const { data } = await client.get("/health/pipeline", { params });
  return data;
}

async function getModels() {
  const { data } = await client.get("/models");
  return data;
}

async function ingestGithub(repoUrl, orgId) {
  const FormData = require("form-data");
  const form = new FormData();
  form.append("repo_url", repoUrl);
  form.append("org_id", orgId);
  const { data } = await client.post("/ingest/github", form, {
    headers: form.getHeaders(),
    timeout: 300000, // 5 minutes for downloading and parsing a whole repo
  });
  return data;
}

module.exports = {
  gate,
  rewrite,
  retrieve,
  retrieveBM25,
  rerankChunks,
  guard,
  streamGenerate,
  ingestFile,
  ingestGithub,
  getIngestStatus,
  getOrgConfig,
  updateOrgConfig,
  submitFeedback,
  getPipelineHealth,
  getModels,
};
