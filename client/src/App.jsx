import { useState, useRef, useEffect, useCallback } from "react";
import "./index.css";
import {
  queryStream, uploadFile, uploadGithub, getIngestStatus,
  submitFeedback, getPipelineHealth, getModels, getConfig,
  updateConfig, getFiles, deleteFile, getDbHealth,
} from "./api";

function MainApp({ session }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedFileIds, setSelectedFileIds] = useState(new Set());
  const [health, setHealth] = useState(null);
  const [dbHealth, setDbHealth] = useState(null);
  const [githubUrl, setGithubUrl] = useState("");
  const [dragging, setDragging] = useState(false);
  const [model, setModel] = useState("llama3.2:3b");
  const [modelRegistry, setModelRegistry] = useState({});
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState({
    useLlmText: false,
    useLlmCode: false,
    imageModel: "llava:7b",
    tableModel: "llama3.2:3b",
    enablePlanner: false,
    groqApiKey: "",
    geminiApiKey: "",
  });
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const prevFilesRef = useRef([]);

  useEffect(() => {
    getPipelineHealth().then(setHealth).catch(() => {});
    getDbHealth().then(setDbHealth).catch(() => {});
    getModels().then((data) => setModelRegistry(data.models || {})).catch(() => {});
    getConfig().then((data) => {
      if (data && data.config) setConfig(data.config);
    }).catch(() => {});
    getFiles().then((data) => {
      if (Array.isArray(data)) {
        setFiles(data);
        const indexedIds = new Set(data.filter(f => f.status === "indexed" && f.id).map(f => f.id));
        setSelectedFileIds(indexedIds);
      }
    }).catch(() => {});

    const interval = setInterval(() => {
      getPipelineHealth().then(setHealth).catch(() => {});
      getDbHealth().then(setDbHealth).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-select files when they finish indexing
  useEffect(() => {
    const prev = prevFilesRef.current;
    files.forEach(f => {
      const prevF = prev.find(p => p.id === f.id);
      if (f.id && f.status === "indexed" && prevF && prevF.status !== "indexed") {
        setSelectedFileIds(s => new Set([...s, f.id]));
      }
    });
    prevFilesRef.current = files;
  }, [files]);

  const toggleFileSelect = useCallback((fileId) => {
    setSelectedFileIds(prev => {
      const next = new Set(prev);
      if (next.has(fileId)) next.delete(fileId); else next.add(fileId);
      return next;
    });
  }, []);

  const indexedFiles = files.filter(f => f.status === "indexed" && f.id);
  const allSelected = indexedFiles.length > 0 && indexedFiles.every(f => selectedFileIds.has(f.id));

  const selectAllFiles = useCallback(() => {
    setSelectedFileIds(new Set(indexedFiles.map(f => f.id)));
  }, [indexedFiles]);

  const deselectAllFiles = useCallback(() => setSelectedFileIds(new Set()), []);

  const handleConfigChange = async (key, value) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    try { await updateConfig(newConfig); } catch {}
  };

  const handleSubmit = useCallback(async (e) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userQuery = input.trim();
    setInput("");
    setLoading(true);

    setMessages(prev => [...prev, { role: "user", content: userQuery }]);
    setMessages(prev => [...prev, { role: "assistant", content: "", meta: null, guard: null, latency: null, streaming: true }]);

    try {
      const chatHistory = messages
        .filter(m => !m.streaming && m.content)
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content.slice(0, 500) }));

      const fileIdsToQuery = [...selectedFileIds];

      await queryStream(
        userQuery, "default", model, chatHistory, fileIdsToQuery,
        (meta) => setMessages(prev => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], meta };
          return msgs;
        }),
        (token) => setMessages(prev => {
          const msgs = [...prev];
          const last = { ...msgs[msgs.length - 1] };
          last.content += token;
          msgs[msgs.length - 1] = last;
          return msgs;
        }),
        (guard) => setMessages(prev => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], guard };
          return msgs;
        }),
        (done) => setMessages(prev => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], latency: done.latency_ms, streaming: false };
          return msgs;
        })
      );
    } catch (err) {
      setMessages(prev => {
        const msgs = [...prev];
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: `Error: ${err.message}`, streaming: false };
        return msgs;
      });
    }
    setLoading(false);
  }, [input, loading, selectedFileIds, model, messages]);

  const handleFileUpload = useCallback(async (fileList) => {
    for (const file of fileList) {
      setFiles(prev => [...prev, { name: file.name, status: "uploading", progress: 0, id: null }]);
      try {
        const result = await uploadFile(file, "default");
        const fileId = result.file_id;
        setFiles(prev => prev.map(f =>
          f.name === file.name && f.status === "uploading" ? { ...f, id: fileId, status: "queued" } : f
        ));
        const poll = setInterval(async () => {
          try {
            const status = await getIngestStatus(fileId);
            setFiles(prev => prev.map(f => f.id === fileId ? {
              ...f, status: status.status, progress: status.progress || 0,
              experts: status.experts || null, expert_names: status.expert_names || null,
              total_chunks: status.total_chunks || null, experts_used: status.experts_used || null,
            } : f));
            if (status.status === "indexed" || status.status === "failed") clearInterval(poll);
          } catch { clearInterval(poll); }
        }, 800);
      } catch {
        setFiles(prev => prev.map(f =>
          f.name === file.name && f.status === "uploading" ? { ...f, status: "failed" } : f
        ));
      }
    }
  }, []);

  const handleGithubUpload = useCallback(async (repoUrl) => {
    if (!repoUrl) return;
    const repoName = repoUrl.split("/").pop() || "repository";
    setFiles(prev => [...prev, { name: repoName, status: "uploading", progress: 0, id: null }]);
    try {
      const result = await uploadGithub(repoUrl, "default");
      const fileId = result.file_id;
      setFiles(prev => prev.map(f =>
        f.name === repoName && f.status === "uploading" ? { ...f, id: fileId, status: "queued" } : f
      ));
      const poll = setInterval(async () => {
        try {
          const status = await getIngestStatus(fileId);
          setFiles(prev => prev.map(f => f.id === fileId ? {
            ...f, status: status.status, progress: status.progress || 0,
            experts: status.experts || null, expert_names: status.expert_names || null,
            total_chunks: status.total_chunks || null, experts_used: status.experts_used || null,
          } : f));
          if (status.status === "indexed" || status.status === "failed") clearInterval(poll);
        } catch { clearInterval(poll); }
      }, 800);
    } catch {
      setFiles(prev => prev.map(f =>
        f.name === repoName && f.status === "uploading" ? { ...f, status: "failed" } : f
      ));
    }
  }, []);

  const handleFeedback = useCallback(async (msgIndex, rating) => {
    setMessages(prev => {
      const msgs = [...prev];
      msgs[msgIndex] = { ...msgs[msgIndex], userRating: rating };
      return msgs;
    });
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFileUpload(e.dataTransfer.files);
  }, [handleFileUpload]);

  const handleDeleteFile = useCallback(async (fileId) => {
    if (!fileId) return;
    try {
      await deleteFile(fileId);
      setFiles(prev => prev.filter(f => f.id !== fileId));
      setSelectedFileIds(prev => { const next = new Set(prev); next.delete(fileId); return next; });
    } catch {}
  }, []);

  const getFileIcon = (name) => {
    const ext = name.split(".").pop().toLowerCase();
    if (ext === "pdf") return "📄";
    if (ext === "csv") return "📊";
    if (["png", "jpg", "jpeg", "webp"].includes(ext)) return "🖼️";
    return "📝";
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="header-logo">P</div>
          <div>
            <div className="header-title">PolyRAG</div>
            <div className="header-subtitle">Multimodal RAG Engine</div>
          </div>
        </div>
        <div className="header-right">
          {dbHealth && (
            <div style={{ display: "flex", gap: "6px", marginRight: "1rem", alignItems: "center" }}>
              <DbBadge label="Engine" status={dbHealth.engine} />
              <DbBadge label="Postgres" status={dbHealth.postgres_docker} />
            </div>
          )}
          <select
            className="model-select"
            value={model}
            onChange={e => setModel(e.target.value)}
            style={{ marginRight: "1rem", padding: "6px 10px", borderRadius: "6px", border: "1px solid var(--border)", background: "var(--bg-card)", color: "var(--text)", fontSize: "12px", cursor: "pointer" }}
          >
            {Object.keys(modelRegistry).length > 0 ? (
              Object.entries(
                Object.entries(modelRegistry).reduce((groups, [key, m]) => {
                  const g = m.group || "Other";
                  if (!groups[g]) groups[g] = [];
                  groups[g].push({ key, ...m });
                  return groups;
                }, {})
              ).map(([group, models]) => (
                <optgroup key={group} label={group}>
                  {models.filter(m => m.caps?.includes("text")).map(m => (
                    <option key={m.key} value={m.key}>{m.display}</option>
                  ))}
                </optgroup>
              ))
            ) : (
              <>
                <option value="llama3.2:3b">Llama 3.2 (Local)</option>
                <option value="gemma3:4b">Gemma 3 4B (Local)</option>
                <option value="llama-3.1-70b-versatile">Llama 3.1 70B (Groq)</option>
                <option value="gemma2-9b-it">Gemma 2 9B (Groq)</option>
                <option value="mixtral-8x7b-32768">Mixtral 8x7B (Groq)</option>
              </>
            )}
          </select>
          {health && (
            <div className="health-badge" style={{ marginRight: "1rem" }}>
              <div className="health-dot"></div>
              {health.total_queries} queries · {Math.round(health.avg_latency_ms)}ms avg
            </div>
          )}
          <button
            onClick={() => setShowSettings(true)}
            style={{ padding: "6px 12px", borderRadius: "6px", border: "1px solid var(--border)", background: "var(--bg-card)", color: "var(--text)", fontSize: "12px", cursor: "pointer", display: "flex", alignItems: "center", gap: "6px" }}
          >
            ⚙️ Settings
          </button>
        </div>
      </header>

      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}
          style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.7)", zIndex: 1000, display: "flex", justifyContent: "center", alignItems: "center" }}>
          <div className="modal-content" onClick={e => e.stopPropagation()}
            style={{ background: "var(--bg-card)", border: "1px solid var(--border)", padding: "2rem", borderRadius: "12px", width: "500px", maxWidth: "90vw", maxHeight: "90vh", overflowY: "auto" }}>
            <h2 style={{ margin: "0 0 1.5rem 0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              Configuration
              <button onClick={() => setShowSettings(false)} style={{ background: "none", border: "none", color: "var(--text)", cursor: "pointer", fontSize: "1.2rem" }}>×</button>
            </h2>
            <div style={{ marginBottom: "2rem" }}>
              <h3 style={{ fontSize: "1rem", marginBottom: "1rem", borderBottom: "1px solid var(--border)", paddingBottom: "0.5rem" }}>API Keys</h3>
              <div style={{ marginBottom: "1rem" }}>
                <label style={{ display: "block", fontSize: "0.85rem", marginBottom: "0.3rem", color: "var(--text-muted)" }}>Groq API Key</label>
                <input type="password" value={config.groqApiKey || ""} onChange={e => handleConfigChange("groqApiKey", e.target.value)}
                  style={{ width: "100%", padding: "8px", borderRadius: "6px", border: "1px solid var(--border)", background: "var(--bg-main)", color: "var(--text)" }} placeholder="gsk_..." />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.85rem", marginBottom: "0.3rem", color: "var(--text-muted)" }}>Gemini API Key</label>
                <input type="password" value={config.geminiApiKey || ""} onChange={e => handleConfigChange("geminiApiKey", e.target.value)}
                  style={{ width: "100%", padding: "8px", borderRadius: "6px", border: "1px solid var(--border)", background: "var(--bg-main)", color: "var(--text)" }} placeholder="AIza..." />
              </div>
            </div>
            <div style={{ marginBottom: "2rem" }}>
              <h3 style={{ fontSize: "1rem", marginBottom: "1rem", borderBottom: "1px solid var(--border)", paddingBottom: "0.5rem" }}>Ingestion Parsing</h3>
              <label style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "1rem", cursor: "pointer" }}>
                <input type="checkbox" checked={config.useLlmText} onChange={e => handleConfigChange("useLlmText", e.target.checked)} />
                <span style={{ fontSize: "0.9rem" }}>Deep LLM Parsing for Text (Slower)</span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "1rem", cursor: "pointer" }}>
                <input type="checkbox" checked={config.useLlmCode} onChange={e => handleConfigChange("useLlmCode", e.target.checked)} />
                <span style={{ fontSize: "0.9rem" }}>Deep LLM Parsing for Code (Slower)</span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "1rem", cursor: "pointer" }}>
                <input type="checkbox" checked={config.useLlmTable} onChange={e => handleConfigChange("useLlmTable", e.target.checked)} />
                <span style={{ fontSize: "0.9rem" }}>Deep LLM Parsing for Tables (Slower)</span>
              </label>
            </div>
            <div>
              <h3 style={{ fontSize: "1rem", marginBottom: "1rem", borderBottom: "1px solid var(--border)", paddingBottom: "0.5rem" }}>Query Options</h3>
              <label style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer" }}>
                <input type="checkbox" checked={config.enablePlanner} onChange={e => handleConfigChange("enablePlanner", e.target.checked)} />
                <span style={{ fontSize: "0.9rem" }}>Enable Context-Based Planner Expert (Higher Latency, Higher Accuracy)</span>
              </label>
            </div>
          </div>
        </div>
      )}

      <div className="main-content">
        <aside className="sidebar">
          <div className="sidebar-section">
            <div className="sidebar-label">Upload Documents</div>
            <div
              className={`upload-zone ${dragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <div className="upload-icon">📁</div>
              <div className="upload-text">
                <strong>Click to upload</strong> or drag and drop<br />
                PDF, CSV, TXT, PNG, JPG
              </div>
              <input ref={fileInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.png,.jpg,.jpeg,.webp"
                style={{ display: "none" }} onChange={e => handleFileUpload(e.target.files)} />
            </div>

            <div className="github-upload" style={{ marginTop: "1rem" }}>
              <div className="sidebar-label" style={{ marginBottom: "0.5rem" }}>Ingest GitHub Repo</div>
              <div style={{ display: "flex", gap: "8px" }}>
                <input type="text" placeholder="https://github.com/user/repo" value={githubUrl}
                  onChange={e => setGithubUrl(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter") { handleGithubUpload(githubUrl); setGithubUrl(""); } }}
                  style={{ flex: 1, padding: "8px", borderRadius: "6px", border: "1px solid var(--border)", background: "var(--bg-card)", color: "var(--text)" }} />
                <button onClick={() => { handleGithubUpload(githubUrl); setGithubUrl(""); }}
                  style={{ padding: "8px 12px", background: "var(--primary)", color: "white", border: "none", borderRadius: "6px", cursor: "pointer" }}>
                  Ingest
                </button>
              </div>
            </div>
          </div>

          {files.length > 0 && (
            <div className="sidebar-section">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
                <div className="sidebar-label" style={{ margin: 0 }}>Ingested Files</div>
                {indexedFiles.length > 0 && (
                  <button
                    onClick={allSelected ? deselectAllFiles : selectAllFiles}
                    style={{ fontSize: "10px", padding: "2px 7px", borderRadius: "4px", border: "1px solid var(--border)", background: "var(--bg-main)", color: "var(--text-muted)", cursor: "pointer" }}
                  >
                    {allSelected ? "Deselect all" : "Select all"}
                  </button>
                )}
              </div>

              {indexedFiles.length > 0 && (
                <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px", padding: "5px 8px", background: "var(--bg-main)", borderRadius: "6px", border: "1px solid var(--border)" }}>
                  {selectedFileIds.size === 0
                    ? "⚠️ No files selected — queries search all data"
                    : selectedFileIds.size === indexedFiles.length
                    ? `✓ Querying all ${indexedFiles.length} file${indexedFiles.length > 1 ? "s" : ""}`
                    : `✓ Querying ${selectedFileIds.size} of ${indexedFiles.length} file${indexedFiles.length > 1 ? "s" : ""}`}
                </div>
              )}

              <div className="file-list">
                {files.map((file, i) => {
                  const isIndexed = file.status === "indexed" && file.id;
                  const isChecked = isIndexed && selectedFileIds.has(file.id);
                  return (
                    <div className="file-card" key={i}
                      style={{
                        flexDirection: "column", alignItems: "stretch", gap: "6px",
                        opacity: isIndexed ? 1 : 0.7,
                        border: isChecked ? "1px solid var(--primary)" : "1px solid var(--border)",
                        transition: "border-color 0.15s",
                      }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        {isIndexed ? (
                          <input type="checkbox" checked={isChecked}
                            onChange={() => toggleFileSelect(file.id)}
                            style={{ cursor: "pointer", flexShrink: 0 }}
                            title={isChecked ? "Exclude from queries" : "Include in queries"} />
                        ) : (
                          <div style={{ width: "13px", flexShrink: 0 }} />
                        )}
                        <div className="file-icon">{getFileIcon(file.name)}</div>
                        <div className="file-info"
                          style={{ flex: 1, overflow: "hidden", cursor: isIndexed ? "pointer" : "default" }}
                          onClick={isIndexed ? () => toggleFileSelect(file.id) : undefined}>
                          <div className="file-name" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{file.name}</div>
                          <div className={`file-status ${file.status}`}>
                            {file.status}{file.total_chunks ? ` · ${file.total_chunks} chunks` : ""}
                          </div>
                        </div>
                        <div className="delete-file-btn" onClick={() => handleDeleteFile(file.id)}
                          style={{ cursor: "pointer", opacity: 0.6, fontSize: "14px", padding: "4px" }} title="Remove file">
                          ❌
                        </div>
                      </div>

                      {file.status !== "indexed" && file.status !== "failed" && (
                        <div className="progress-bar">
                          <div className="progress-fill" style={{ width: `${file.progress}%` }}></div>
                        </div>
                      )}

                      {file.experts && Object.keys(file.experts).length > 0 && (
                        <div style={{ display: "flex", flexDirection: "column", gap: "4px", marginTop: "2px" }}>
                          {(file.expert_names || Object.keys(file.experts)).map(eid => {
                            const es = file.experts[eid];
                            if (!es) return null;
                            const isRunning = es.state === "running";
                            const isDone = es.state === "done";
                            return (
                              <div key={eid} style={{
                                display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", padding: "3px 6px", borderRadius: "4px",
                                background: isRunning ? "rgba(99,102,241,0.1)" : isDone ? "rgba(16,185,129,0.1)" : "rgba(244,63,94,0.1)",
                                border: `1px solid ${isRunning ? "rgba(99,102,241,0.3)" : isDone ? "rgba(16,185,129,0.3)" : "rgba(244,63,94,0.3)"}`,
                              }}>
                                <span style={{ fontSize: "10px" }}>{isRunning ? "⏳" : isDone ? "✅" : "❌"}</span>
                                <span className={`expert-badge ${eid}`} style={{ fontSize: "10px", padding: "1px 5px" }}>{eid}</span>
                                <span style={{ marginLeft: "auto", color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
                                  {isDone ? `${es.chunks} chunks` : isRunning ? "parsing..." : "failed"}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {file.status === "indexed" && file.experts_used && (
                        <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", marginTop: "2px" }}>
                          {file.experts_used.map(eid => (
                            <span key={eid} className={`expert-badge ${eid}`} style={{ fontSize: "10px", padding: "1px 5px" }}>{eid}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {health && (
            <div className="sidebar-section">
              <div className="sidebar-label">Pipeline Health</div>
              <div className="file-card" style={{ flexDirection: "column", gap: "4px", alignItems: "stretch" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Queries</span><span>{health.total_queries}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Avg Latency</span><span>{Math.round(health.avg_latency_ms)}ms</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Feedback</span>
                  <span>👍 {health.positive_feedback} · 👎 {health.negative_feedback}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Retrain</span>
                  <span style={{ color: health.retrain_recommended ? "var(--accent-rose)" : "var(--accent-emerald)" }}>
                    {health.retrain_recommended ? "Recommended" : "Not needed"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </aside>

        <main className="chat-area">
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">🔮</div>
                <div className="empty-title">Ask anything about your documents</div>
                <div className="empty-desc">
                  Upload files, then check which ones to include on the left. PolyRAG routes to the
                  right expert (text, table, or image) automatically.
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`message message-${msg.role}`}>
                {msg.meta?.rewritten_query && (
                  <div className="rewrite-notice">✨ Query rewritten: "{msg.meta.rewritten_query}"</div>
                )}
                <div className="message-bubble">
                  {msg.content}
                  {msg.streaming && (
                    <span className="typing-indicator">
                      <span className="typing-dot"></span>
                      <span className="typing-dot"></span>
                      <span className="typing-dot"></span>
                    </span>
                  )}
                </div>

                {msg.role === "assistant" && msg.meta && (
                  <div className="message-meta">
                    {msg.meta.active_experts?.map(exp => (
                      <span key={exp} className={`expert-badge ${exp}`}>{exp}</span>
                    ))}
                    {msg.guard && (
                      <span className={`guard-badge ${msg.guard.verified ? "verified" : msg.guard.score > 0.5 ? "partial" : "unverified"}`}>
                        {msg.guard.verified ? "✓" : msg.guard.score > 0.5 ? "~" : "✗"}{" "}
                        {msg.guard.verified ? "Verified" : msg.guard.score > 0.5 ? "Partially verified" : "Unverified"}
                      </span>
                    )}
                    {msg.latency && <span>{(msg.latency / 1000).toFixed(1)}s</span>}
                  </div>
                )}

                {msg.role === "assistant" && msg.meta?.sources?.length > 0 && (
                  <SourceCards sources={msg.meta.sources} />
                )}

                {msg.role === "assistant" && !msg.streaming && (
                  <div className="feedback-bar">
                    <button className={`feedback-btn ${msg.userRating === 5 ? "active-up" : ""}`}
                      onClick={() => handleFeedback(i, 5)} title="Good answer">👍</button>
                    <button className={`feedback-btn ${msg.userRating === 1 ? "active-down" : ""}`}
                      onClick={() => handleFeedback(i, 1)} title="Bad answer">👎</button>
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <form className="input-wrapper" onSubmit={handleSubmit}>
              <input className="input-field" type="text" value={input} onChange={e => setInput(e.target.value)}
                placeholder={
                  selectedFileIds.size > 0
                    ? `Ask about ${selectedFileIds.size} selected file${selectedFileIds.size > 1 ? "s" : ""}...`
                    : "Ask about your documents..."
                }
                disabled={loading} />
              <button className="send-btn" type="submit" disabled={loading || !input.trim()}>→</button>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}

function DbBadge({ label, status }) {
  const up = status === "up";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "4px", fontSize: "11px",
      padding: "3px 8px", borderRadius: "10px",
      background: up ? "rgba(16,185,129,0.12)" : "rgba(244,63,94,0.12)",
      border: `1px solid ${up ? "rgba(16,185,129,0.3)" : "rgba(244,63,94,0.3)"}`,
      color: up ? "var(--accent-emerald)" : "var(--accent-rose)",
      fontWeight: 500,
    }} title={`${label}: ${status}`}>
      <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: up ? "var(--accent-emerald)" : "var(--accent-rose)", display: "inline-block", flexShrink: 0 }} />
      {label}
    </div>
  );
}

function SourceCards({ sources }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="sources-container">
      <button className="sources-toggle" onClick={() => setExpanded(!expanded)}>
        {expanded ? "▾" : "▸"} {sources.length} sources
      </button>
      {expanded && sources.map((s, i) => (
        <div className="source-card" key={i}>
          <div className="source-card-header">
            <span className={`expert-badge ${s.expert_id}`}>{s.expert_id}</span>
            {s.metadata?.page && <span>Page {s.metadata.page}</span>}
            {s.metadata?.similarity && (
              <span style={{ color: "var(--text-muted)" }}>sim: {(s.metadata.similarity * 100).toFixed(1)}%</span>
            )}
          </div>
          {s.content}
        </div>
      ))}
    </div>
  );
}

export default function App() {
  return <MainApp session={{ user: { id: "local", email: "local@polyrag" } }} />;
}
