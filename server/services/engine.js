const axios = require("axios");
const http = require("http");
const https = require("https");
const config = require("./config");

const ENGINE_URL = config.ENGINE_URL;
const TIMEOUT = 0;

// Keep-alive pools: reuse TCP connections to Python engine
const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 20,
  maxFreeSockets: 5,
  timeout: 60000,
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 20,
  maxFreeSockets: 5,
  timeout: 60000,
});

const client = axios.create({
  baseURL: ENGINE_URL,
  timeout: TIMEOUT,
  httpAgent: httpAgent,
  httpsAgent: httpsAgent,
});


async function retrieve(query, orgId, topK = 10, fileIds = null, model = null) {
  const body = {
    query,
    org_id: orgId,
    top_k: topK,
  };
  if (fileIds && fileIds.length > 0) body.file_ids = fileIds;
  if (model) body.model = model;
  const { data } = await client.post("/retrieve", body, { headers: { "x-tenant-id": orgId } });
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

function streamGenerate(prompt, query, model, chatHistory = [], orgId = "default") {
  return client.post(
    "/generate/stream",
    { prompt, query, model, chat_history: chatHistory, org_id: orgId },
    { responseType: "stream", timeout: 120000 }
  );
}

async function generate(prompt, query, model, chatHistory = [], orgId = "default") {
  const { data } = await client.post("/generate", {
    prompt,
    query,
    model,
    chat_history: chatHistory,
    org_id: orgId
  });
  return data;
}

async function ingestFile(filePath, orgId, models = {}) {
  const FormData = require("form-data");
  const fs = require("fs");
  const form = new FormData();
  form.append("file", fs.createReadStream(filePath));
  form.append("org_id", orgId);
  form.append("models", JSON.stringify(models));
  try {
    const { data } = await client.post("/ingest/async", form, {
      headers: { ...form.getHeaders(), "x-tenant-id": orgId },
      timeout: 60000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    return data;
  } catch (err) {
    console.error("[Engine] ingestFile failed:", err.message);
    if (err.response) {
      console.error("[Engine] Response status:", err.response.status);
      console.error("[Engine] Response data:", JSON.stringify(err.response.data));
    }
    throw err;
  }
}

async function getIngestStatus(fileId, orgId) {
  const { data } = await client.get(`/file/${fileId}`, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function getOrgConfig(orgId) {
  const { data } = await client.get(`/config/${orgId}`, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function updateOrgConfig(orgId, name, config) {
  const { data } = await client.put(`/config/${orgId}`, { name, config }, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function submitFeedback(queryLogId, rating, orgId = "default") {
  const { data } = await client.post("/feedback", {
    query_log_id: queryLogId,
    rating,
  }, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function getPipelineHealth(orgId = null) {
  const params = orgId ? { org_id: orgId } : {};
  const headers = orgId ? { "x-tenant-id": orgId } : {};
  const { data } = await client.get("/health/pipeline", { params, headers });
  return data;
}

async function getModels() {
  const { data } = await client.get("/models");
  return data;
}

async function ingestGithub(repoUrl, orgId, models = {}) {
  const FormData = require("form-data");
  const form = new FormData();
  form.append("repo_url", repoUrl);
  form.append("org_id", orgId);
  form.append("models", JSON.stringify(models));
  const { data } = await client.post("/ingest/github", form, {
    headers: { ...form.getHeaders(), "x-tenant-id": orgId },
    timeout: 300000, // 5 minutes for downloading and parsing a whole repo
  });
  return data;
}

async function getOrgFiles(orgId) {
  const { data } = await client.get(`/files/${orgId}`, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function deleteOrgFile(orgId, fileId) {
  const { data } = await client.delete(`/files/${orgId}/${fileId}`, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function getChatSessions(orgId) {
  const { data } = await client.get(`/chat/sessions/${orgId}`, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function createChatSession(sessionId, orgId, title) {
  const { data } = await client.post("/chat/sessions", { session_id: sessionId, org_id: orgId, title }, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function deleteChatSession(orgId, sessionId) {
  const { data } = await client.delete(`/chat/sessions/${orgId}/${sessionId}`, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function getChatMessages(sessionId, orgId) {
  const { data } = await client.get(`/chat/sessions/${sessionId}/messages`, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function addChatMessage(sessionId, messageId, role, content, sources = [], orgId) {
  const { data } = await client.post(`/chat/sessions/${sessionId}/messages`, { message_id: messageId, role, content, sources }, { headers: { "x-tenant-id": orgId } });
  return data;
}

async function getSessionOwner(orgId, sessionId) {
  const { data } = await client.get(`/chat/sessions/${sessionId}/owner`, { headers: { "x-tenant-id": orgId } });
  return data?.org_id || null;
}

async function deleteAllChatSessions(orgId) {
  const { data } = await client.post(`/chat/logout`, {}, { headers: { "x-tenant-id": orgId } });
  return data?.deleted_sessions || 0;
}

module.exports = {
  retrieve,
  rerankChunks,
  guard,
  streamGenerate,
  generate,
  ingestFile,
  ingestGithub,
  getIngestStatus,
  getOrgConfig,
  updateOrgConfig,
  submitFeedback,
  getPipelineHealth,
  getModels,
  getOrgFiles,
  deleteOrgFile,
  getChatSessions,
  createChatSession,
  deleteChatSession,
  getChatMessages,
  addChatMessage,
  getSessionOwner,
  deleteAllChatSessions,
};