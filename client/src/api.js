import { supabase } from "./supabaseClient";

const API_BASE = "http://localhost:3001";

async function getAuthHeaders(headers = {}) {
  const { data: { session } } = await supabase.auth.getSession();
  if (session?.access_token) {
    headers["Authorization"] = `Bearer ${session.access_token}`;
  }
  return headers;
}

export async function queryStream(query, orgId = "default", model = "llama3.2:3b", chatHistory = [], onMeta, onToken, onGuard, onDone) {
  const headers = await getAuthHeaders({ "Content-Type": "application/json" });
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query, org_id: orgId, model, chat_history: chatHistory }),
  });

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
        } catch {}
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
