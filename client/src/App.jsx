import { useState, useRef, useEffect, useCallback } from "react";
import "./index.css";
import {
  queryStream, uploadFile, uploadGithub, getIngestStatus,
  submitFeedback, getPipelineHealth, getModels, getConfig,
  updateConfig, getFiles, deleteFile, getDbHealth, forceRetrainGate,
} from "./api";

const MarkdownRenderer = ({ content }) => {
  if (!content) return null;

  // Split by "```" to handle code blocks elegantly
  const parts = content.split("```");
  return (
    <div className="markdown-content">
      {parts.map((part, index) => {
        const isCode = index % 2 === 1;

        if (isCode) {
          const lines = part.split("\n");
          let language = "text";
          let codeLines = lines;
          if (lines[0] && !lines[0].includes(" ") && lines[0].length < 15) {
            language = lines[0].toLowerCase();
            codeLines = lines.slice(1);
          }
          const codeText = codeLines.join("\n");

          return (
            <div key={index} className="code-block-container" style={{ margin: "12px 0" }}>
              <div className="code-block-header" style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "6px 12px",
                background: "var(--bg-card)",
                borderBottom: "1px solid var(--border)",
                fontSize: "11px",
                fontFamily: "monospace",
                color: "var(--text-muted)",
                borderTopLeftRadius: "6px",
                borderTopRightRadius: "6px"
              }}>
                <span>{language.toUpperCase()}</span>
                <button
                  className="copy-btn"
                  onClick={() => {
                    navigator.clipboard.writeText(codeText);
                  }}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--accent-emerald)",
                    cursor: "pointer",
                    fontSize: "11px",
                    fontWeight: "bold"
                  }}
                >
                  Copy
                </button>
              </div>
              <pre className="code-block" style={{
                margin: 0,
                padding: "12px",
                background: "#19191e",
                color: "#e3e3e6",
                overflowX: "auto",
                borderBottomLeftRadius: "6px",
                borderBottomRightRadius: "6px",
                fontFamily: "var(--font-mono)",
                fontSize: "13px",
                lineHeight: "1.5"
              }}>
                <code>{codeText}</code>
              </pre>
            </div>
          );
        }

        // Regular Text
        const lines = part.split("\n");
        let listItems = [];
        let insideList = false;
        const renderedLines = [];

        const renderInline = (text) => {
          let cleanedText = text;
          if (typeof cleanedText === "string") {
            cleanedText = cleanedText.replace(/\\\((.*?)\\\)/g, "$1");
            cleanedText = cleanedText.replace(/\\\[(.*?)\\\]/g, "$1");
          }

          let segments = [cleanedText];
          // bold
          let newSegments = [];
          for (let seg of segments) {
            if (typeof seg === "string") {
              const subSegs = seg.split(/(\*\*.*?\*\*)/g);
              newSegments.push(...subSegs.map((sub, i) => {
                if (sub.startsWith("**") && sub.endsWith("**")) {
                  return <strong key={i}>{sub.slice(2, -2)}</strong>;
                }
                return sub;
              }));
            } else {
              newSegments.push(seg);
            }
          }
          segments = newSegments;

          // inline code
          newSegments = [];
          for (let seg of segments) {
            if (typeof seg === "string") {
              const subSegs = seg.split(/(\`.*?\`)/g);
              newSegments.push(...subSegs.map((sub, i) => {
                if (sub.startsWith("`") && sub.endsWith("`")) {
                  return <code key={i} className="inline-code" style={{
                    background: "var(--bg-body)",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.9em",
                    border: "1px solid var(--border)"
                  }}>{sub.slice(1, -1)}</code>;
                }
                return sub;
              }));
            } else {
              newSegments.push(seg);
            }
          }
          segments = newSegments;

          // math sub/superscript parser
          const parseSubAndSuper = (txt) => {
            if (typeof txt !== "string") return txt;
            const regex = /(\b[a-zA-Z0-9]+_(?:{[^}]+}|[a-zA-Z0-9+-=]+)|\b[a-zA-Z0-9]+\^(?:{[^}]+}|[a-zA-Z0-9+-=()]+))/g;
            const tokens = txt.split(regex);
            return tokens.map((token, i) => {
              const subMatch = token.match(/^([a-zA-Z0-9]+)_(?:{([^}]+)}|([a-zA-Z0-9+-=]+))$/);
              if (subMatch) {
                const base = subMatch[1];
                const subText = subMatch[2] || subMatch[3];
                return <span key={i} style={{ fontStyle: "italic", fontFamily: "var(--font-serif)", fontSize: "1.1em" }}>{base}<sub>{subText}</sub></span>;
              }
              const superMatch = token.match(/^([a-zA-Z0-9]+)\^(?:{([^}]+)}|([a-zA-Z0-9+-=()]+))$/);
              if (superMatch) {
                const base = superMatch[1];
                const superText = superMatch[2] || superMatch[3];
                return <span key={i} style={{ fontStyle: "italic", fontFamily: "var(--font-serif)", fontSize: "1.1em" }}>{base}<sup>{superText}</sup></span>;
              }
              return token;
            });
          };

          let finalSegments = [];
          for (let seg of segments) {
            if (typeof seg === "string") {
              const parsed = parseSubAndSuper(seg);
              if (Array.isArray(parsed)) {
                finalSegments.push(...parsed);
              } else {
                finalSegments.push(parsed);
              }
            } else {
              finalSegments.push(seg);
            }
          }

          return finalSegments;
        };

        const flushList = (key) => {
          if (listItems.length > 0) {
            renderedLines.push(
              <ul key={`ul-${key}`} className="markdown-list" style={{ paddingLeft: "20px", margin: "8px 0" }}>
                {listItems.map((item, idx) => (
                  <li key={idx} style={{ margin: "4px 0" }}>{renderInline(item)}</li>
                ))}
              </ul>
            );
            listItems = [];
            insideList = false;
          }
        };

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          const trimmed = line.trim();

          if (trimmed.startsWith("# ")) {
            flushList(i);
            renderedLines.push(<h1 key={i} style={{ margin: "16px 0 8px 0", fontSize: "1.8em", fontWeight: "bold" }}>{renderInline(trimmed.slice(2))}</h1>);
          } else if (trimmed.startsWith("## ")) {
            flushList(i);
            renderedLines.push(<h2 key={i} style={{ margin: "14px 0 6px 0", fontSize: "1.4em", fontWeight: "bold" }}>{renderInline(trimmed.slice(3))}</h2>);
          } else if (trimmed.startsWith("### ")) {
            flushList(i);
            renderedLines.push(<h3 key={i} style={{ margin: "12px 0 4px 0", fontSize: "1.2em", fontWeight: "bold" }}>{renderInline(trimmed.slice(4))}</h3>);
          } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
            insideList = true;
            listItems.push(trimmed.slice(2));
          } else if (/^\d+\.\s/.test(trimmed)) {
            flushList(i);
            const content = trimmed.replace(/^\d+\.\s/, "");
            const num = trimmed.match(/^\d+/)?.[0] || "1";
            renderedLines.push(<div key={i} className="list-item-numbered" style={{ margin: "6px 0", display: "flex", gap: "8px" }}>
              <span style={{ fontWeight: "bold", color: "var(--accent-emerald)" }}>{num}.</span>
              <div>{renderInline(content)}</div>
            </div>);
          } else if (trimmed === "") {
            flushList(i);
          } else {
            if (insideList) {
              flushList(i);
            }
            renderedLines.push(<p key={i} style={{ margin: "8px 0", lineHeight: "1.6" }}>{renderInline(line)}</p>);
          }
        }
        flushList(lines.length);

        return <div key={index}>{renderedLines}</div>;
      })}
    </div>
  );
};

function MainApp({ session }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);

  const [wheelOpen, setWheelOpen] = useState(false);
  const [hoveredSlice, setHoveredSlice] = useState(null);
  const [activeModal, setActiveModal] = useState(null); // null, 'files', 'health', 'settings'

  const getArcPath = (cx, cy, r_in, r_out, start_angle, end_angle) => {
    const rad = Math.PI / 180;
    const x1_in = cx + r_in * Math.cos(start_angle * rad);
    const y1_in = cy + r_in * Math.sin(start_angle * rad);
    const x2_in = cx + r_in * Math.cos(end_angle * rad);
    const y2_in = cy + r_in * Math.sin(end_angle * rad);

    const x1_out = cx + r_out * Math.cos(start_angle * rad);
    const y1_out = cy + r_out * Math.sin(start_angle * rad);
    const x2_out = cx + r_out * Math.cos(end_angle * rad);
    const y2_out = cy + r_out * Math.sin(end_angle * rad);

    return `
      M ${x1_in} ${y1_in}
      L ${x1_out} ${y1_out}
      A ${r_out} ${r_out} 0 0 1 ${x2_out} ${y2_out}
      L ${x2_in} ${y2_in}
      A ${r_in} ${r_in} 0 0 0 ${x1_in} ${y1_in}
      Z
    `;
  };

  const getMidpointCoords = (cx, cy, r, angle) => {
    const rad = Math.PI / 180;
    const x = cx + r * Math.cos(angle * rad);
    const y = cy + r * Math.sin(angle * rad);
    return { x, y };
  };
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
    if (ext === "pdf") return "[PDF]";
    if (ext === "csv") return "[CSV]";
    if (["png", "jpg", "jpeg", "webp"].includes(ext)) return "[IMG]";
    return "[TXT]";
  };

  return (
    <div className="app">
      {/* Rich Japanese Bamboo Forest & Misty Dawn background */}
      <div className="bamboo-forest-environment">
        {/* Tall segmented bamboo trunks */}
        <div className="bamboo-trunk trunk-left-1"></div>
        <div className="bamboo-trunk trunk-left-2"></div>
        <div className="bamboo-trunk trunk-mid-left"></div>
        <div className="bamboo-trunk trunk-mid-right"></div>
        <div className="bamboo-trunk trunk-right-1"></div>
        <div className="bamboo-trunk trunk-right-2"></div>
        
        {/* Falling animated Sakura Cherry Blossom Petals */}
        <div className="sakura-petal petal-1"></div>
        <div className="sakura-petal petal-2"></div>
        <div className="sakura-petal petal-3"></div>
        <div className="sakura-petal petal-4"></div>
        <div className="sakura-petal petal-5"></div>
        <div className="sakura-petal petal-6"></div>
      </div>

      {/* Serene Japanese Waterfall Spout & Water stream */}
      <div className="japanese-water-spout">
        <div className="bamboo-spout-element"></div>
        <div className="waterfall-cascade">
          <div className="water-drop-1"></div>
          <div className="water-drop-2"></div>
          <div className="water-drop-3"></div>
        </div>
      </div>

      {/* Ripple Pond at Bottom Left */}
      <div className="japanese-ripple-pond">
        <div className="pond-ripple ripple-1"></div>
        <div className="pond-ripple ripple-2"></div>
        <div className="pond-ripple ripple-3"></div>
      </div>

      <header className="header" style={{ backdropFilter: "blur(4px)", borderBottom: "1px solid var(--border-subtle)", background: "rgba(250, 246, 238, 0.85)" }}>
        <div className="header-left">
          <div className="header-logo" style={{ background: "var(--border-accent)", color: "var(--bg-primary)" }}>P</div>
          <div>
            <div className="header-title">PolyRAG</div>
            <div className="header-subtitle">Multimodal RAG Archival Search</div>
          </div>
        </div>
        <div className="header-right">
          {dbHealth && (
            <div style={{ display: "flex", gap: "6px", marginRight: "1rem", alignItems: "center" }}>
              <DbBadge label="Engine" status={dbHealth.engine} />
              <DbBadge label="Postgres" status={dbHealth.postgres_docker} />
            </div>
          )}
          {health && (
            <div className="health-badge" style={{ marginRight: "1rem" }}>
              <span className="health-dot"></span>
              {health.total_queries} queries · {Math.round(health.avg_latency_ms)}ms avg
            </div>
          )}
          <button
            onClick={() => setActiveModal("settings")}
            className="header-settings-btn"
            style={{ padding: "6px 12px", border: "1px solid var(--border-accent)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", cursor: "pointer", fontFamily: "var(--font-mono)", fontWeight: 700 }}
          >
            SYSTEM SETUP
          </button>
        </div>
      </header>

      <div className="main-content" style={{ gridTemplateColumns: "1fr", maxWidth: "1080px", margin: "0 auto", width: "100%", padding: "24px" }}>
        <div className="japanese-scroll-wrapper">
          <div className="scroll-wood-roller top-roller"></div>
          <div className="scroll-parchment-paper">
            <main className="chat-area" style={{ width: "100%", background: "none" }}>
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <pre style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "11px",
                  lineHeight: "1.2",
                  color: "var(--text-muted)",
                  margin: "0 0 16px 0",
                  textAlign: "left",
                  display: "inline-block",
                  background: "rgba(142, 125, 96, 0.03)",
                  padding: "16px 24px",
                  border: "1px dashed var(--border-subtle)",
                  borderRadius: "4px"
                }}>{`   ._________________________.
   | [RAG] [SYS] [IND] [SEC] |
   |=========================|
   | [ ] Ledger Index A-M    |
   | [ ] Ledger Index N-Z    |
   |=========================|
   | [x] Active Archives     |
   \`-------------------------'`}</pre>
                <div className="empty-title" style={{ fontFamily: "var(--font-header)", fontStyle: "italic", fontSize: "22px" }}>Archival Search Ledger</div>
                <div className="empty-desc" style={{ fontFamily: "var(--font-serif)" }}>
                  Select records from the system index on the left. PolyRAG will cross-reference materials and auto-route your query to the appropriate registry.
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`message message-${msg.role}`}>
                {msg.meta?.rewritten_query && (
                  <div className="rewrite-notice">[ REFOCUSED QUERY ] "{msg.meta.rewritten_query}"</div>
                )}
                <div className="message-bubble">
                  <MarkdownRenderer content={msg.content} />
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
                        {msg.guard.verified ? "PASSED" : msg.guard.score > 0.5 ? "WARN" : "FAIL"}
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
                      onClick={() => handleFeedback(i, 5)} title="Good answer" style={{ fontSize: "10px", fontFamily: "var(--font-mono)", width: "auto", padding: "0 8px", fontWeight: "700" }}>ACCEPT</button>
                    <button className={`feedback-btn ${msg.userRating === 1 ? "active-down" : ""}`}
                      onClick={() => handleFeedback(i, 1)} title="Bad answer" style={{ fontSize: "10px", fontFamily: "var(--font-mono)", width: "auto", padding: "0 8px", fontWeight: "700" }}>REJECT</button>
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
              <button className="send-btn" type="submit" disabled={loading || !input.trim()}>SUBMIT</button>
            </form>
          </div>
            </main>
          </div>
          <div className="scroll-wood-roller bottom-roller"></div>
        </div>
      </div>
      {/* Steampunk Mechanical Cog & Weapon Wheel */}
      <div className={`cog-container ${wheelOpen ? "open" : ""}`} onClick={() => setWheelOpen(!wheelOpen)} title="Rotate Japanese Mizuguruma Water Wheel">
        <svg viewBox="0 0 100 100" className="cog-svg">
          {/* Outer wooden rim */}
          <circle cx="50" cy="50" r="45" fill="none" stroke="#4a3a2c" strokeWidth="5" />
          <circle cx="50" cy="50" r="33" fill="none" stroke="#4a3a2c" strokeWidth="2" />
          {/* Center hub */}
          <circle cx="50" cy="50" r="12" fill="#2d2218" stroke="#4a3a2c" strokeWidth="3" />
          <circle cx="50" cy="50" r="4" fill="#faf6ee" />
          {/* Wooden Spokes */}
          <g stroke="#4a3a2c" strokeWidth="3">
            <line x1="50" y1="5" x2="50" y2="95" />
            <line x1="5" y1="50" x2="95" y2="50" />
            <line x1="18.2" y1="18.2" x2="81.8" y2="81.8" />
            <line x1="18.2" y1="81.8" x2="81.8" y2="18.2" />
          </g>
          {/* Wooden Buckets/Scoops on Rim */}
          <g fill="#4a3a2c">
            <path d="M 46 5 L 54 5 L 51 14 L 47 14 Z" />
            <path d="M 46 95 L 54 95 L 51 86 L 47 86 Z" />
            <path d="M 5 46 L 5 54 L 14 51 L 14 47 Z" />
            <path d="M 95 46 L 95 54 L 86 51 L 86 47 Z" />
            <path d="M 18.2 18.2 L 23.8 23.8 L 20.8 26.8 L 15.2 21.2 Z" />
            <path d="M 81.8 81.8 L 76.2 76.2 L 79.2 73.2 L 84.8 78.8 Z" />
            <path d="M 18.2 81.8 L 23.8 76.2 L 20.8 73.2 L 15.2 78.8 Z" />
            <path d="M 81.8 18.2 L 76.2 23.8 L 79.2 26.8 L 84.8 21.2 Z" />
          </g>
        </svg>
      </div>

      {wheelOpen && (
        <div className="weapon-wheel-overlay" onClick={() => setWheelOpen(false)}>
          <div className="weapon-wheel-menu" style={{ width: "260px", height: "460px" }} onClick={e => e.stopPropagation()}>
            <svg viewBox="0 0 240 480" className="weapon-wheel-svg">
              {[
                {
                  id: "clear",
                  start: -90,
                  end: -60,
                  label: "CLEAR",
                  desc: "Clear Chat Dossier Ledger",
                  action: () => setMessages([]),
                },
                {
                  id: "files",
                  start: -60,
                  end: -30,
                  label: "FILES",
                  desc: "Manage and Upload RAG Documents",
                  action: () => setActiveModal("files"),
                },
                {
                  id: "health",
                  start: -30,
                  end: 0,
                  label: "HEALTH",
                  desc: "Inspect Database & Pipeline Health Metrics",
                  action: () => setActiveModal("health"),
                },
                {
                  id: "planner",
                  start: 0,
                  end: 30,
                  label: "PLANNER",
                  desc: config.enablePlanner ? "Deactivate Context-Based Planner Expert" : "Activate Context-Based Planner Expert",
                  action: () => {
                    const nextPlanner = !config.enablePlanner;
                    setConfig(prev => ({ ...prev, enablePlanner: nextPlanner }));
                    updateConfig({ ...config, enablePlanner: nextPlanner })
                      .then(() => alert(`[ROUTING STATUS] Planner ${nextPlanner ? "Activated" : "Deactivated"}`))
                      .catch(() => {});
                  },
                },
                {
                  id: "settings",
                  start: 30,
                  end: 60,
                  label: "SETTINGS",
                  desc: "Configure RAG Models & System API Keys",
                  action: () => setActiveModal("settings"),
                },
                {
                  id: "toggle-rag",
                  start: 60,
                  end: 90,
                  label: "RAG TOGGLE",
                  desc: selectedFileIds.size > 0 ? "Deselect All Active RAG Sources" : "Select All Active RAG Sources",
                  action: () => {
                    if (selectedFileIds.size > 0) {
                      setSelectedFileIds(new Set());
                    } else {
                      setSelectedFileIds(new Set(files.filter(f => f.status === "indexed" && f.id).map(f => f.id)));
                    }
                  },
                },
              ].map((slice) => {
                const isHovered = hoveredSlice === slice.id;
                const path = getArcPath(0, 240, 65, 220, slice.start, slice.end);
                const textPos = getMidpointCoords(0, 240, 145, (slice.start + slice.end) / 2);

                return (
                  <g
                    key={slice.id}
                    className="weapon-wheel-sector"
                    onMouseEnter={() => setHoveredSlice(slice.id)}
                    onMouseLeave={() => setHoveredSlice(null)}
                    onClick={() => {
                      slice.action();
                      setWheelOpen(false);
                    }}
                  >
                    <path
                      d={path}
                      className={`weapon-wheel-path ${isHovered ? "hovered" : ""}`}
                    />
                    <text
                      x={textPos.x}
                      y={textPos.y}
                      transform={`rotate(${(slice.start + slice.end) / 2}, ${textPos.x}, ${textPos.y})`}
                      className={`weapon-wheel-text ${isHovered ? "hovered" : ""}`}
                      style={{ fontSize: "10px", fontWeight: "700" }}
                    >
                      {slice.label}
                    </text>
                  </g>
                );
              })}
            </svg>
            
            {/* Center Description Display */}
            <div className="weapon-wheel-center-display" style={{ left: "240px" }}>
              <div className="weapon-wheel-center-label">[ MIZUGURUMA WHEEL ]</div>
              <div className="weapon-wheel-center-desc">
                {hoveredSlice
                  ? [
                      { id: "clear", desc: "Clear all records in chat search ledger" },
                      { id: "files", desc: "Open source registry panel to ingest or select documents" },
                      { id: "health", desc: "Inspect current metrics, average latency and gate recommendation" },
                      { id: "planner", desc: config.enablePlanner ? "Disable deep multihop contextual planning expert" : "Enable deep multihop contextual planning expert" },
                      { id: "settings", desc: "Configure local LLM servers, Groq, Gemini keys & parser depth" },
                      { id: "toggle-rag", desc: selectedFileIds.size > 0 ? "Clear active file filter constraints (default all)" : "Engage constraint filtering on all indexed materials" },
                    ].find(d => d.id === hoveredSlice)?.desc
                  : "Rotate Mizuguruma wheel to manage deep RAG operations"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modern Floating Modal Cards */}
      {activeModal && (
        <div className="active-modal-overlay" onClick={() => setActiveModal(null)}>
          <div className="active-modal-card" onClick={e => e.stopPropagation()}>
            <div className="active-modal-header">
              <span className="active-modal-title">
                {activeModal === "files" && "[ REGISTRY SOURCE LEDGER ]"}
                {activeModal === "health" && "[ PIPELINE METRICS & SYSTEM BADGES ]"}
                {activeModal === "settings" && "[ ARCHIVAL SYSTEM SETUP ]"}
              </span>
              <button className="active-modal-close" onClick={() => setActiveModal(null)}>×</button>
            </div>
            <div className="active-modal-body">
              {activeModal === "files" && (
                <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                  <div style={{ background: "rgba(142, 125, 96, 0.02)", border: "1px dashed var(--border-accent)", borderRadius: "6px", padding: "20px", display: "flex", gap: "20px" }}>
                    <div style={{ flex: 1 }}>
                      <div className="sidebar-label" style={{ margin: "0 0 10px 0" }}>Ingest New Materials</div>
                      <div
                        className={`upload-zone ${dragging ? "dragging" : ""}`}
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={e => { e.preventDefault(); setDragging(true); }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={handleDrop}
                        style={{ border: "1px dashed var(--border-subtle)", padding: "24px", borderRadius: "6px", cursor: "pointer", background: "var(--bg-card)", textAlign: "center" }}
                      >
                        <div className="upload-icon" style={{ fontFamily: "var(--font-mono)", fontSize: "11px", letterSpacing: "0.5px", border: "1px dashed var(--border-accent)", padding: "2px 8px", borderRadius: "3px", background: "rgba(142, 125, 96, 0.04)", display: "inline-block", marginBottom: "8px" }}>[ ADD LOCAL FILES ]</div>
                        <div className="upload-text">
                          <strong>Click to search</strong> or drag documents here<br />
                          PDF, CSV, TXT, PNG, JPG, MD
                        </div>
                        <input ref={fileInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.png,.jpg,.jpeg,.webp"
                          style={{ display: "none" }} onChange={e => handleFileUpload(e.target.files)} />
                      </div>
                    </div>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
                      <div className="sidebar-label" style={{ margin: "0 0 8px 0" }}>GitHub Repository Link</div>
                      <div style={{ display: "flex", gap: "8px" }}>
                        <input type="text" placeholder="https://github.com/user/repo" value={githubUrl}
                          onChange={e => setGithubUrl(e.target.value)}
                          onKeyDown={e => { if (e.key === "Enter") { handleGithubUpload(githubUrl); setGithubUrl(""); } }}
                          style={{ flex: 1, padding: "8px", borderRadius: "4px", border: "1px solid var(--border-accent)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "12px", fontFamily: "var(--font-mono)" }} />
                        <button onClick={() => { handleGithubUpload(githubUrl); setGithubUrl(""); }}
                          style={{ padding: "8px 16px", background: "var(--border-accent)", color: "var(--bg-primary)", border: "none", borderRadius: "4px", cursor: "pointer", fontFamily: "var(--font-mono)", fontSize: "11px", fontWeight: "700" }}>
                          INGEST
                        </button>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                      <div className="sidebar-label" style={{ margin: 0 }}>Ingested Archives ({files.length})</div>
                      {indexedFiles.length > 0 && (
                        <button
                          onClick={allSelected ? deselectAllFiles : selectAllFiles}
                          style={{ fontSize: "10px", padding: "3px 8px", borderRadius: "4px", border: "1px solid var(--border-accent)", background: "var(--bg-card)", color: "var(--text-primary)", cursor: "pointer", fontFamily: "var(--font-mono)", fontWeight: "700" }}
                        >
                          {allSelected ? "DESELECT ALL" : "SELECT ALL"}
                        </button>
                      )}
                    </div>

                    {indexedFiles.length > 0 && (
                      <div style={{ fontSize: "11px", color: "var(--border-accent)", marginBottom: "12px", padding: "8px 12px", background: "rgba(142,125,96,0.04)", borderRadius: "4px", border: "1px solid var(--border-subtle)", fontFamily: "var(--font-serif)" }}>
                        {selectedFileIds.size === 0
                          ? "⚠️ All filters cleared — querying will cross-reference all global archives"
                          : selectedFileIds.size === indexedFiles.length
                          ? `✓ cross-referencing all ${indexedFiles.length} cataloged resources`
                          : `✓ filtering search constraints to ${selectedFileIds.size} selected dossiers`}
                      </div>
                    )}

                    <div className="file-list" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", maxHeight: "280px", overflowY: "auto" }}>
                      {files.map((file, i) => {
                        const isIndexed = file.status === "indexed" && file.id;
                        const isChecked = isIndexed && selectedFileIds.has(file.id);
                        return (
                          <div className="file-card" key={i}
                            style={{
                              flexDirection: "column", alignItems: "stretch", gap: "6px",
                              opacity: isIndexed ? 1 : 0.7,
                              border: isChecked ? "1px solid var(--border-accent)" : "1px solid var(--border-subtle)",
                              background: "var(--bg-card)"
                            }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              {isIndexed ? (
                                <input type="checkbox" checked={isChecked}
                                  onChange={() => toggleFileSelect(file.id)}
                                  style={{ cursor: "pointer", flexShrink: 0 }} />
                              ) : (
                                <div style={{ width: "13px", flexShrink: 0 }} />
                              )}
                              <div className="file-icon" style={{ fontSize: "11px", fontFamily: "var(--font-mono)" }}>{getFileIcon(file.name)}</div>
                              <div className="file-info"
                                style={{ flex: 1, overflow: "hidden", cursor: isIndexed ? "pointer" : "default" }}
                                onClick={isIndexed ? () => toggleFileSelect(file.id) : undefined}>
                                <div className="file-name" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontSize: "12px" }}>{file.name}</div>
                                <div className={`file-status ${file.status}`} style={{ fontSize: "10px" }}>
                                  {file.status}{file.total_chunks ? ` · ${file.total_chunks} chunks` : ""}
                                </div>
                              </div>
                              <div className="delete-file-btn" onClick={() => handleDeleteFile(file.id)}
                                style={{ cursor: "pointer", opacity: 0.6, fontSize: "11px", padding: "4px" }} title="Remove dossier">
                                ❌
                              </div>
                            </div>

                            {file.status !== "indexed" && file.status !== "failed" && (
                              <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${file.progress}%` }}></div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {activeModal === "health" && (
                <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                    <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "16px", background: "var(--bg-card)" }}>
                      <div className="sidebar-label" style={{ margin: "0 0 12px 0" }}>Database Node status</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", borderBottom: "1px dashed var(--border-subtle)", paddingBottom: "6px" }}>
                          <span>Engine Server</span>
                          <span style={{ color: dbHealth?.engine === "up" ? "var(--accent-emerald)" : "var(--accent-rose)", fontWeight: "bold" }}>
                            {dbHealth?.engine === "up" ? "ONLINE" : "OFFLINE"}
                          </span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px" }}>
                          <span>PostgreSQL Core</span>
                          <span style={{ color: dbHealth?.postgres_docker === "up" ? "var(--accent-emerald)" : "var(--accent-rose)", fontWeight: "bold" }}>
                            {dbHealth?.postgres_docker === "up" ? "ONLINE" : "OFFLINE"}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "16px", background: "var(--bg-card)" }}>
                      <div className="sidebar-label" style={{ margin: "0 0 12px 0" }}>Pipeline Metrics</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", borderBottom: "1px dashed var(--border-subtle)", paddingBottom: "6px" }}>
                          <span>Cross-Reference Queries</span>
                          <span>{health?.total_queries || 0}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px" }}>
                          <span>Average Response Latency</span>
                          <span>{Math.round(health?.avg_latency_ms || 0)}ms</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "20px", background: "rgba(142, 125, 96, 0.02)", textAlign: "center" }}>
                    <div className="sidebar-label" style={{ margin: "0 0 6px 0" }}>Gate Retrain Recommendation</div>
                    <div style={{ fontSize: "13px", fontFamily: "var(--font-serif)", color: "var(--text-secondary)", marginBottom: "16px" }}>
                      {health?.retrain_recommended
                        ? "⚠️ Multiple unverified feedback patterns detected. System highly recommends forced database gate retraining."
                        : "✓ Embedding space classification vectors are balanced. Manual retraining is optional."}
                    </div>
                    <button
                      onClick={() => {
                        forceRetrainGate()
                          .then(() => alert("[SUCCESS] Embedding gate retraining triggered successfully."))
                          .catch(() => alert("[ERROR] Retraining execution failed."));
                      }}
                      style={{ padding: "10px 24px", background: health?.retrain_recommended ? "var(--accent-rose)" : "var(--border-accent)", color: "var(--bg-primary)", border: "none", borderRadius: "4px", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: "700", cursor: "pointer" }}
                    >
                      FORCE MANUAL GATE RETRAIN
                    </button>
                  </div>
                </div>
              )}

              {activeModal === "settings" && (
                <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                  <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "16px", background: "var(--bg-card)" }}>
                    <div className="sidebar-label" style={{ margin: "0 0 12px 0" }}>Active Search Language Model</div>
                    <select
                      className="model-select"
                      value={model}
                      onChange={e => setModel(e.target.value)}
                      style={{ width: "100%", padding: "8px", borderRadius: "4px", border: "1px solid var(--border-accent)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "12px", fontFamily: "var(--font-mono)", cursor: "pointer" }}
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
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                    <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "16px", background: "var(--bg-card)" }}>
                      <div className="sidebar-label" style={{ margin: "0 0 12px 0" }}>API Credentials</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                        <div>
                          <label style={{ display: "block", fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginBottom: "4px" }}>GROQ API KEY</label>
                          <input type="password" value={config.groqApiKey || ""} onChange={e => handleConfigChange("groqApiKey", e.target.value)}
                            style={{ width: "100%", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border-subtle)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", fontFamily: "var(--font-mono)" }} placeholder="gsk_..." />
                        </div>
                        <div>
                          <label style={{ display: "block", fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginBottom: "4px" }}>GEMINI API KEY</label>
                          <input type="password" value={config.geminiApiKey || ""} onChange={e => handleConfigChange("geminiApiKey", e.target.value)}
                            style={{ width: "100%", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border-subtle)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", fontFamily: "var(--font-mono)" }} placeholder="AIza..." />
                        </div>
                      </div>
                    </div>

                    <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "16px", background: "var(--bg-card)" }}>
                      <div className="sidebar-label" style={{ margin: "0 0 12px 0" }}>Deep LLM Parsing Depth</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                        <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", fontSize: "11px", fontFamily: "var(--font-serif)" }}>
                          <input type="checkbox" checked={config.useLlmText} onChange={e => handleConfigChange("useLlmText", e.target.checked)} />
                          <span>Deep Parsing for Text Content</span>
                        </label>
                        <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", fontSize: "11px", fontFamily: "var(--font-serif)" }}>
                          <input type="checkbox" checked={config.useLlmCode} onChange={e => handleConfigChange("useLlmCode", e.target.checked)} />
                          <span>Deep Parsing for Source Code</span>
                        </label>
                        <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", fontSize: "11px", fontFamily: "var(--font-serif)" }}>
                          <input type="checkbox" checked={config.useLlmTable} onChange={e => handleConfigChange("useLlmTable", e.target.checked)} />
                          <span>Deep Parsing for Tables</span>
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
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
