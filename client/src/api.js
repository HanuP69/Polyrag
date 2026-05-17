import { supabase } from "./supabaseClient";

const API_BASE = "http://localhost:3001";

async function getAuthHeaders(headers = {}) {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (session?.access_token) {
      headers["Authorization"] = `Bearer ${session.access_token}`;
    } else {
      // Local dev: no Supabase session — send a passthrough token
      headers["Authorization"] = "Bearer local-dev-token";
    }
  } catch {
    headers["Authorization"] = "Bearer local-dev-token";
  }
  return headers;
}

export async function queryStream(query, orgId = "default", model = "llama3.2:3b", chatHistory = [], fileIds = [], onMeta, onToken, onGuard, onDone) {
  const headers = await getAuthHeaders({ "Content-Type": "application/json" });
  const body = { query, org_id: orgId, model, chat_history: chatHistory };
  if (fileIds && fileIds.length > 0) body.file_ids = fileIds;
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Query failed (${res.status}): ${errText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === "meta") onMeta(event);
          else if (event.type === "token") onToken(event.content);
          else if (event.type === "guard") onGuard(event);
          else if (event.type === "done") onDone(event);
        } catch { }
      }
    }
  }
}

export async function uploadFile(file, orgId = "default") {
  const form = new FormData();
  form.append("file", file);
  form.append("org_id", orgId);
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/ingest`, { method: "POST", headers, body: form });
  return res.json();
}

export async function uploadGithub(repoUrl, orgId = "default") {
  const headers = await getAuthHeaders({ "Content-Type": "application/json" });
  const res = await fetch(`${API_BASE}/api/ingest/github`, {
    method: "POST",
    headers,
    body: JSON.stringify({ repo_url: repoUrl, org_id: orgId }),
  });
  return res.json();
}

export async function getIngestStatus(fileId) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/ingest/${fileId}`, { headers });
  return res.json();
}

export async function submitFeedback(queryLogId, rating, correctExpert = null) {
  const headers = await getAuthHeaders({ "Content-Type": "application/json" });
  const res = await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query_log_id: queryLogId, rating, correct_expert: correctExpert }),
  });
  return res.json();
}

export async function getPipelineHealth() {
  const res = await fetch(`${API_BASE}/api/health/pipeline`);
  return res.json();
}

export async function getModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  return res.json();
}
export async function getIngestedFiles(orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/files?org_id=${orgId}`, { headers });
  return res.json();
}

export async function deleteFile(fileId, orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/files/${fileId}?org_id=${orgId}`, {
    method: "DELETE",
    headers,
  });
  return res.json();
}

export async function fetchQueryLog(logId) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/query-logs/${logId}`, { headers });
  return res.json();
}

export async function exportQueryLog(queryLogId, orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/query-logs/${queryLogId}/export?org_id=${orgId}`, {
    headers,
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `query_log_${queryLogId}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export async function fetchDebugInfo(queryLogId, orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/debug/${queryLogId}?org_id=${orgId}`, {
    headers,
  });
  return res.json();
}

export async function forceRetrainGate(orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/admin/retrain-gate?org_id=${orgId}`, {
    method: "POST",
    headers,
  });
  return res.json();
}

export async function resetGate(orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/admin/reset-gate?org_id=${orgId}`, {
    method: "POST",
    headers,
  });
  return res.json();
}

export async function deleteQueryLog(queryLogId, orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/query-logs/${queryLogId}?org_id=${orgId}`, {
    method: "DELETE",
    headers,
  });
  return res.json();
}

export async function getRagDataReport(orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/admin/rag-report?org_id=${orgId}`, {
    headers,
  });
  return res.json();
}

export async function getActiveQueryLog(orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/active-query-log?org_id=${orgId}`, {
    headers,
  });
  return res.json();
}

// --- Missing exports that App.jsx needs ---

export async function getConfig(orgId = "default") {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}/api/config?org_id=${orgId}`, { headers });
  if (!res.ok) return null;
  return res.json();
}

export async function updateConfig(config, orgId = "default") {
  const headers = await getAuthHeaders({ "Content-Type": "application/json" });
  const res = await fetch(`${API_BASE}/api/config`, {
    method: "PUT",
    headers,
    body: JSON.stringify({ org_id: orgId, config }),
  });
  return res.json();
}

// Alias: App.jsx imports `getFiles`, server exposes /api/files
export async function getFiles(orgId = "default") {
  return getIngestedFiles(orgId);
}

export async function getDbHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health/db`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

