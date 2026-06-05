import { useState, useRef, useEffect, useCallback } from "react";
import "./index.css";
import { supabase } from "./supabaseClient";
import Login from "./Login";
import {
  queryStream, uploadFile, uploadGithub, getIngestStatus,
  submitFeedback, getPipelineHealth, getModels, getConfig,
  updateConfig, getFiles, deleteFile, getDbHealth,
  getChatSessions, createChatSession, deleteChatSession, getChatMessages, addChatMessage
} from "./api";

const API_BASE = import.meta.env.VITE_API_URL || "";

const MarkdownRenderer = ({ content, sources }) => {
  if (!content) return null;

  let resolvedContent = content;
  if (typeof resolvedContent === "string") {
    resolvedContent = resolvedContent.replace(/\\n/g, "\n");
  }
  if (sources && sources.length > 0) {
    // Replace short references like source_1 or /api/uploads/source_1 or source_N
    resolvedContent = resolvedContent.replace(/(?:\/api\/uploads\/)?source_(\d+)/gi, (match, p1) => {
      const idx = parseInt(p1) - 1;
      if (sources[idx] && sources[idx].metadata?.source) {
        return `/api/uploads/${sources[idx].metadata.source}`;
      }
      return match;
    });
  }

  // Split by "```" to handle code blocks elegantly
  const parts = resolvedContent.split("```");
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

          // markdown images and uploads links: ![alt](url) or [alt](/api/uploads/...)
          let newSegments = [];
          for (let seg of segments) {
            if (typeof seg === "string") {
              const subSegs = seg.split(/(!?\[.*?\]\(.*?\))/g);
              newSegments.push(...subSegs.map((sub, i) => {
                const imgMatch = sub.match(/^(!?)\[(.*?)\]\((.*?)\)$/);
                if (imgMatch) {
                  const hasExclamation = imgMatch[1] === "!";
                  const alt = imgMatch[2];
                  let url = imgMatch[3];
                  
                  // Render as image if it has '!' OR if the URL is an upload image
                  const isUploadImage = url.includes("/api/uploads/") && (url.endsWith(".png") || url.endsWith(".jpg") || url.endsWith(".jpeg") || url.endsWith(".webp") || url.includes("img"));
                  
                  if (hasExclamation || isUploadImage) {
                    if (url.startsWith("/api/uploads/")) {
                      url = `${API_BASE}${url}`;
                    }
                    return (
                      <div key={i} className="chat-image-container" style={{ margin: "16px 0", maxWidth: "100%" }}>
                        <img
                          src={url}
                          alt={alt}
                          style={{
                            maxWidth: "100%",
                            maxHeight: "380px",
                            objectFit: "contain",
                            borderRadius: "3px", // dossier folder sharp photo clip style
                            border: "1px solid var(--border-accent)",
                            padding: "6px",
                            background: "#ffffff",
                            boxShadow: "0 4px 14px rgba(47, 41, 33, 0.08)",
                            display: "block"
                          }}
                        />
                        <div style={{
                          fontSize: "10px",
                          fontFamily: "var(--font-mono)",
                          color: "var(--text-muted)",
                          marginTop: "6px",
                          fontStyle: "italic",
                          letterSpacing: "0.5px"
                        }}>
                          {alt || "Dossier Visual Attachment"}
                        </div>
                      </div>
                    );
                  }
                }
                return sub;
              }));
            } else {
              newSegments.push(seg);
            }
          }
          segments = newSegments;

          // bold
          newSegments = [];
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
                  <li key={idx} style={{ margin: "4px 0", whiteSpace: "pre-wrap" }}>{renderInline(item)}</li>
                ))}
              </ul>
            );
            listItems = [];
            insideList = false;
          }
        };

        let currentPara = [];
        const flushPara = (key) => {
          if (currentPara.length > 0) {
            renderedLines.push(
              <p key={`p-${key}`} style={{ margin: "0 0 12px 0", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
                {renderInline(currentPara.join("\n"))}
              </p>
            );
            currentPara = [];
          }
        };

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          const trimmed = line.trim();

          if (trimmed.startsWith("# ")) {
            flushList(i);
            flushPara(i);
            renderedLines.push(<h1 key={i} style={{ margin: "16px 0 8px 0", fontSize: "1.8em", fontWeight: "bold" }}>{renderInline(trimmed.slice(2))}</h1>);
          } else if (trimmed.startsWith("## ")) {
            flushList(i);
            flushPara(i);
            renderedLines.push(<h2 key={i} style={{ margin: "14px 0 6px 0", fontSize: "1.4em", fontWeight: "bold" }}>{renderInline(trimmed.slice(3))}</h2>);
          } else if (trimmed.startsWith("### ")) {
            flushList(i);
            flushPara(i);
            renderedLines.push(<h3 key={i} style={{ margin: "12px 0 4px 0", fontSize: "1.2em", fontWeight: "bold" }}>{renderInline(trimmed.slice(4))}</h3>);
          } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
            flushPara(i);
            insideList = true;
            listItems.push(trimmed.slice(2));
          } else if (/^\d+\.\s/.test(trimmed)) {
            flushList(i);
            flushPara(i);
            const content = trimmed.replace(/^\d+\.\s/, "");
            const num = trimmed.match(/^\d+/)?.[0] || "1";
            renderedLines.push(<div key={i} className="list-item-numbered" style={{ margin: "6px 0", display: "flex", gap: "8px" }}>
              <span style={{ fontWeight: "bold", color: "var(--accent-emerald)" }}>{num}.</span>
              <div style={{ whiteSpace: "pre-wrap" }}>{renderInline(content)}</div>
            </div>);
          } else if (trimmed === "") {
            flushList(i);
            flushPara(i);
          } else {
            if (insideList) {
              flushList(i);
            }
            currentPara.push(line);
          }
        }
        flushList(lines.length);
        flushPara(lines.length);

        return <div key={index}>{renderedLines}</div>;
      })}
    </div>
  );
};

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
  return `M ${x1_in} ${y1_in} L ${x1_out} ${y1_out} A ${r_out} ${r_out} 0 0 1 ${x2_out} ${y2_out} L ${x2_in} ${y2_in} A ${r_in} ${r_in} 0 0 0 ${x1_in} ${y1_in} Z`;
};

const getMidpointCoords = (cx, cy, r, angle) => {
  const rad = Math.PI / 180;
  return {
    x: cx + r * Math.cos(angle * rad),
    y: cy + r * Math.sin(angle * rad)
  };
};

function MainApp({ session }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatSessions, setChatSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);

  const [wheelOpen, setWheelOpen] = useState(false);
  const [activeWheelIndex, setActiveWheelIndex] = useState(0);
  const [showHints, setShowHints] = useState(false);
  const [activeModal, setActiveModal] = useState(null);

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
    tableModel: "gemini-2.5-flash",
    enablePlanner: false,
    groqApiKey: "",
    geminiApiKey: "",
  });
  const [showGroqKeys, setShowGroqKeys] = useState({});
  const [showGeminiKeys, setShowGeminiKeys] = useState({});

  const [p1State, setP1State] = useState("idle");
  const [p2State, setP2State] = useState("idle");
  const [p3State, setP3State] = useState("idle");

  const handlePandaClick = useCallback((id) => {
    if (id === 1) {
      if (p1State !== "idle") return;
      setP1State("falling");
      setTimeout(() => {
        setP1State("climbing");
        setTimeout(() => {
          setP1State("idle");
        }, 2500);
      }, 700);
    } else if (id === 2) {
      if (p2State !== "idle") return;
      setP2State("falling");
      setTimeout(() => {
        setP2State("climbing");
        setTimeout(() => {
          setP2State("idle");
        }, 2500);
      }, 700);
    } else if (id === 3) {
      if (p3State !== "idle") return;
      setP3State("falling");
      setTimeout(() => {
        setP3State("climbing");
        setTimeout(() => {
          setP3State("idle");
        }, 2500);
      }, 700);
    }
  }, [p1State, p2State, p3State]);

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

    if (session) {
      getChatSessions()
        .then((data) => {
          if (Array.isArray(data)) {
            setChatSessions(data);
          }
        })
        .catch((e) => console.error("Failed to load chat sessions:", e));
    } else {
      setChatSessions([]);
      setCurrentSessionId(null);
    }

    const interval = setInterval(() => {
      getPipelineHealth().then(setHealth).catch(() => {});
      getDbHealth().then(setDbHealth).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, [session]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // High performance tracking effect for interactive Panda head Z-axis neck rotation
  useEffect(() => {
    const handleMouseMove = (e) => {
      const w = window.innerWidth;
      const h = window.innerHeight;

      // Panda 1: Left bamboo (approx neck at 12% width, 65% height)
      const p1x = 0.12 * w;
      const p1y = 0.65 * h;
      const angle1 = Math.atan2(e.clientY - p1y, e.clientX - p1x) * (180 / Math.PI) - 90; 
      const clamp1 = Math.max(-55, Math.min(55, angle1));

      // Panda 2: Right bamboo (approx neck at 92% width, 45% height)
      const p2x = 0.92 * w;
      const p2y = 0.45 * h;
      const angle2 = Math.atan2(e.clientY - p2y, e.clientX - p2x) * (180 / Math.PI) - 90;
      const clamp2 = Math.max(-55, Math.min(55, angle2));

      // Panda 3: Bottom right bamboo (approx neck at 97% width, 78% height)
      const p3x = 0.97 * w;
      const p3y = 0.78 * h;
      const angle3 = Math.atan2(e.clientY - p3y, e.clientX - p3x) * (180 / Math.PI) - 90;
      const clamp3 = Math.max(-55, Math.min(55, angle3));

      document.documentElement.style.setProperty('--panda-rot-1', `rotate(${clamp1}deg)`);
      document.documentElement.style.setProperty('--panda-rot-2', `rotate(${clamp2}deg)`);
      document.documentElement.style.setProperty('--panda-rot-3', `rotate(${clamp3}deg)`);
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

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

  const selectSession = useCallback(async (sessionId) => {
    setCurrentSessionId(sessionId);
    setLoading(true);
    try {
      const msgs = await getChatMessages(sessionId);
      if (Array.isArray(msgs)) {
        const formatted = msgs.map(m => ({
          role: m.role,
          content: m.content,
          meta: m.sources && m.sources.length > 0 ? { sources: m.sources } : null,
          streaming: false,
          message_id: m.message_id
        }));
        setMessages(formatted);
      }
    } catch (err) {
      console.error("Failed to load messages:", err);
    }
    setLoading(false);
  }, []);

  const startNewSession = useCallback(async () => {
    const newSessionId = crypto.randomUUID();
    const title = `Session ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    try {
      await createChatSession(newSessionId, title);
      setChatSessions(prev => [{ session_id: newSessionId, title, created_at: new Date().toISOString() }, ...prev]);
      setCurrentSessionId(newSessionId);
      setMessages([]);
    } catch (e) {
      console.error("Failed to create chat session:", e);
    }
  }, []);

  const handleDeleteSession = useCallback(async (sessionId, e) => {
    e.stopPropagation();
    try {
      await deleteChatSession(sessionId);
      setChatSessions(prev => prev.filter(s => s.session_id !== sessionId));
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
        setMessages([]);
      }
    } catch (e) {
      console.error("Failed to delete chat session:", e);
    }
  }, [currentSessionId]);

  const handleSubmit = useCallback(async (e) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    // API Key availability check
    const modelInfo = modelRegistry[model];
    const modelType = modelInfo?.type || (
      model.includes("gemini") ? "gemini" :
      (model.includes("groq") || ["llama-3.3-70b-specdec", "gemma2-9b-it", "mixtral-8x7b-32768"].includes(model)) ? "groq" :
      "ollama"
    );

    if (modelType === "groq") {
      const hasGroqKey = 
        (config.groqApiKey && config.groqApiKey.trim() !== "") || 
        (Array.isArray(config.groqApiKeys) && config.groqApiKeys.some(k => k && k.trim() !== ""));
      if (!hasGroqKey) {
        alert("Select kiye gaye model ke liye Groq API Key missing hai. Please settings mein jaakar key add karein.");
        return;
      }
    } else if (modelType === "gemini") {
      const hasGeminiKey = 
        (config.geminiApiKey && config.geminiApiKey.trim() !== "") || 
        (Array.isArray(config.geminiApiKeys) && config.geminiApiKeys.some(k => k && k.trim() !== ""));
      if (!hasGeminiKey) {
        alert("Select kiye gaye model ke liye Gemini API Key missing hai. Please settings mein jaakar key add karein.");
        return;
      }
    }

    const userQuery = input.trim();
    setInput("");
    setLoading(true);

    const isSessionSavingEnabled = !!session;
    let activeSessionId = currentSessionId;
    if (isSessionSavingEnabled && !activeSessionId) {
      activeSessionId = crypto.randomUUID();
      const title = userQuery.slice(0, 30) + (userQuery.length > 30 ? "..." : "");
      try {
        await createChatSession(activeSessionId, title);
        setChatSessions(prev => [{ session_id: activeSessionId, title, created_at: new Date().toISOString() }, ...prev]);
        setCurrentSessionId(activeSessionId);
      } catch (err) {
        console.error("Failed to auto-create session:", err);
      }
    }

    setMessages(prev => [...prev, { role: "user", content: userQuery }]);
    setMessages(prev => [...prev, { role: "assistant", content: "", meta: null, guard: null, latency: null, streaming: true }]);

    if (isSessionSavingEnabled && activeSessionId) {
      const userMsgId = crypto.randomUUID();
      try {
        await addChatMessage(activeSessionId, userMsgId, "user", userQuery, []);
      } catch (err) {
        console.error("Failed to save user message:", err);
      }
    }

    try {
      const chatHistory = messages
        .filter(m => !m.streaming && m.content)
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content.slice(0, 500) }));

      const fileIdsToQuery = [...selectedFileIds];
      let metaData = null;
      let accumulatedContent = "";

      await queryStream(
        userQuery, "default", model, chatHistory, fileIdsToQuery,
        (meta) => {
          metaData = meta;
          setMessages(prev => {
            const msgs = [...prev];
            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], meta };
            return msgs;
          });
        },
        (token) => {
          accumulatedContent += token;
          setMessages(prev => {
            const msgs = [...prev];
            const last = { ...msgs[msgs.length - 1] };
            last.content += token;
            msgs[msgs.length - 1] = last;
            return msgs;
          });
        },
        (guard) => setMessages(prev => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], guard };
          return msgs;
        }),
        async (done) => {
          if (isSessionSavingEnabled && activeSessionId) {
            const assistantMsgId = crypto.randomUUID();
            addChatMessage(activeSessionId, assistantMsgId, "assistant", accumulatedContent, metaData?.sources || [])
              .catch((err) => console.error("Failed to save assistant message:", err));
          }
          setMessages(prev => {
            const msgs = [...prev];
            const lastMsg = msgs[msgs.length - 1];
            msgs[msgs.length - 1] = { ...lastMsg, latency: done.latency_ms, streaming: false };
            return msgs;
          });
        }
      );
    } catch (err) {
      if (isSessionSavingEnabled && activeSessionId) {
        const assistantMsgId = crypto.randomUUID();
        addChatMessage(activeSessionId, assistantMsgId, "assistant", `Error: ${err.message}`, [])
          .catch((e) => console.error("Failed to save assistant error response:", e));
      }
      setMessages(prev => {
        const msgs = [...prev];
        const last = { ...msgs[msgs.length - 1] };
        last.content = `Error: ${err.message}`;
        last.streaming = false;
        msgs[msgs.length - 1] = last;
        return msgs;
      });
    }
    setLoading(false);
  }, [input, loading, selectedFileIds, model, messages, currentSessionId, session, config, modelRegistry]);

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
            if (status.status === "indexed" || status.status === "failed" || status.status === "error") clearInterval(poll);
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
          if (status.status === "indexed" || status.status === "failed" || status.status === "error") clearInterval(poll);
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
    if (!name) return "[?]";
    const ext = name.split(".").pop().toLowerCase();
    if (ext === "pdf") return "[PDF]";
    if (ext === "csv") return "[CSV]";
    if (["png", "jpg", "jpeg", "webp"].includes(ext)) return "[IMG]";
    return "[TXT]";
  };

  return (
    <div className="app">
      {/* Sidebar Drawer */}
      <div className={`chat-history-sidebar ${sidebarOpen ? "open" : ""}`} style={{
        position: "fixed",
        top: 0,
        left: sidebarOpen ? 0 : "-320px",
        width: "320px",
        height: "100vh",
        background: "rgba(244, 238, 223, 0.95)",
        borderRight: "2px solid var(--border-accent)",
        boxShadow: "4px 0 24px rgba(0,0,0,0.15)",
        zIndex: 1000,
        transition: "left 0.3s cubic-bezier(0.16, 1, 0.3, 1)",
        display: "flex",
        flexDirection: "column",
        padding: "24px 16px",
        backdropFilter: "blur(8px)"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
          <h2 style={{ fontFamily: "var(--font-header)", fontStyle: "italic", fontSize: "20px", color: "var(--text-primary)" }}>Archival History</h2>
          <button 
            onClick={() => setSidebarOpen(false)}
            style={{ background: "none", border: "none", cursor: "pointer", fontSize: "16px", color: "var(--text-secondary)" }}
          >
            ✕
          </button>
        </div>

        {session ? (
          <>
            <button
              onClick={startNewSession}
              style={{
                width: "100%",
                padding: "10px",
                border: "1px solid var(--border-accent)",
                background: "var(--bg-card)",
                color: "var(--text-primary)",
                fontFamily: "var(--font-mono)",
                fontSize: "12px",
                cursor: "pointer",
                marginBottom: "20px",
                textAlign: "center",
                textTransform: "uppercase"
              }}
            >
              [ NEW LEDGER SESSION ]
            </button>

            <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "8px" }}>
              {chatSessions.length === 0 ? (
                <div style={{ textAlign: "center", padding: "20px", color: "var(--text-muted)", fontStyle: "italic", fontSize: "12px" }}>
                  No previous ledgers.
                </div>
              ) : (
                chatSessions.map((sessionItem) => (
                  <div
                    key={sessionItem.session_id}
                    onClick={() => {
                      selectSession(sessionItem.session_id);
                      setSidebarOpen(false);
                    }}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "10px 12px",
                      background: currentSessionId === sessionItem.session_id ? "rgba(184, 122, 45, 0.1)" : "var(--bg-card)",
                      border: `1px solid ${currentSessionId === sessionItem.session_id ? "var(--accent-amber)" : "var(--border-subtle)"}`,
                      borderRadius: "var(--radius-sm)",
                      cursor: "pointer",
                      transition: "all 0.2s ease"
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0, paddingRight: "8px" }}>
                      <div style={{
                        fontWeight: currentSessionId === sessionItem.session_id ? "bold" : "normal",
                        color: "var(--text-primary)",
                        fontSize: "13px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap"
                      }}>
                        {sessionItem.title}
                      </div>
                      <div style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "2px", fontFamily: "var(--font-mono)" }}>
                        {sessionItem.created_at ? new Date(sessionItem.created_at).toLocaleDateString() : ""}
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteSession(sessionItem.session_id, e)}
                      style={{
                        background: "none",
                        border: "none",
                        color: "var(--accent-rose)",
                        cursor: "pointer",
                        padding: "4px",
                        fontSize: "12px"
                      }}
                      title="Delete Session"
                    >
                      ✕
                    </button>
                  </div>
                ))
              )}
            </div>
          </>
        ) : (
          <div style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            padding: "24px 16px",
            textAlign: "center",
            background: "rgba(142, 125, 96, 0.03)",
            border: "1px dashed var(--border-accent)",
            borderRadius: "4px",
            margin: "20px 0"
          }}>
            <div style={{ fontSize: "12px", color: "var(--text-secondary)", fontFamily: "var(--font-serif)", marginBottom: "20px", lineHeight: "1.6" }}>
              Archival query logging is offline. Sign in with Google to maintain permanent search history and save your query sessions.
            </div>
            <button
              onClick={() => supabase.auth.signInWithOAuth({
                provider: 'google',
                options: { redirectTo: window.location.origin }
              })}
              style={{
                width: "100%",
                padding: "10px 12px",
                border: "1px solid var(--border-accent)",
                background: "var(--bg-card)",
                color: "var(--text-primary)",
                fontFamily: "var(--font-mono)",
                fontSize: "11px",
                cursor: "pointer",
                textTransform: "uppercase"
              }}
            >
              [ CONNECT GOOGLE ]
            </button>
          </div>
        )}
      </div>

      {/* Sidebar Backdrop Overlay */}
      {sidebarOpen && (
        <div 
          onClick={() => setSidebarOpen(false)}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100vw",
            height: "100vh",
            background: "rgba(0, 0, 0, 0.3)",
            backdropFilter: "blur(2px)",
            zIndex: 999
          }}
        ></div>
      )}

      <div className="bamboo-forest-environment">
        <svg style={{ position: "absolute", width: 0, height: 0 }}>
          <defs>
            <linearGradient id="bamboo-grad-dark" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#222e1d" />
              <stop offset="40%" stopColor="#3d4f32" />
              <stop offset="80%" stopColor="#4b613e" />
              <stop offset="100%" stopColor="#1b2518" />
            </linearGradient>
            <linearGradient id="bamboo-grad-mid" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#37482f" />
              <stop offset="35%" stopColor="#556c47" />
              <stop offset="70%" stopColor="#698559" />
              <stop offset="100%" stopColor="#293723" />
            </linearGradient>
            <linearGradient id="bamboo-grad-light" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#4e6541" />
              <stop offset="35%" stopColor="#729260" />
              <stop offset="70%" stopColor="#87aa73" />
              <stop offset="100%" stopColor="#3d5133" />
            </linearGradient>
          </defs>
        </svg>

        {/* Left Bamboo Cluster (Thicker, Denser, 12 layered trunks) */}
        {/* Far Background Trunk 3 */}
        <div className="bamboo-detailed trunk-left-3" style={{ position: "absolute", left: "17%", bottom: "-10px", width: "70px", height: "105%", opacity: 0.1, filter: "blur(3px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-dark)" />
            <path d="M 38 150 L 62 150 M 38 300 L 62 300 M 38 450 L 62 450 M 38 600 L 62 600" stroke="#1b2518" strokeWidth="3" />
          </svg>
        </div>
        {/* Far Background Trunk 4 */}
        <div className="bamboo-detailed trunk-left-4" style={{ position: "absolute", left: "28%", bottom: "-10px", width: "80px", height: "105%", opacity: 0.12, filter: "blur(2.5px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-dark)" />
            <path d="M 38 140 L 62 140 M 38 280 L 62 280 M 38 420 L 62 420 M 38 560 L 62 560 M 38 700 L 62 700" stroke="#1b2518" strokeWidth="3.5" />
          </svg>
        </div>
        {/* Far Background Trunk 5 */}
        <div className="bamboo-detailed trunk-left-5" style={{ position: "absolute", left: "36%", bottom: "-10px", width: "60px", height: "105%", opacity: 0.07, filter: "blur(3.5px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-dark)" />
            <path d="M 38 160 L 62 160 M 38 320 L 62 320 M 38 480 L 62 480 M 38 640 L 62 640" stroke="#1b2518" strokeWidth="3" />
          </svg>
        </div>
        {/* Mid Background Trunk 6 */}
        <div className="bamboo-detailed trunk-left-6" style={{ position: "absolute", left: "22%", bottom: "-10px", width: "70px", height: "105%", opacity: 0.15, filter: "blur(2px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-mid)" />
            <path d="M 38 150 L 62 150 M 38 300 L 62 300 M 38 450 L 62 450 M 38 600 L 62 600" stroke="#23301e" strokeWidth="3" />
          </svg>
        </div>
        {/* Background Trunk 1 */}
        <div className="bamboo-detailed trunk-left-1" style={{ position: "absolute", left: "1%", bottom: "-10px", width: "100px", height: "105%", opacity: 0.2, filter: "blur(1.5px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-mid)" />
            <path d="M 38 150 L 62 150 M 38 300 L 62 300 M 38 450 L 62 450 M 38 600 L 62 600" stroke="#23301e" strokeWidth="4" />
          </svg>
        </div>
        {/* Thick Front Trunk 2 */}
        <div className="bamboo-detailed trunk-left-2" style={{ position: "absolute", left: "8%", bottom: "-10px", width: "155px", height: "105%", opacity: 0.85 }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-light)" />
            <path d="M 35 120 L 65 120 M 35 240 L 65 240 M 35 360 L 65 360 M 35 480 L 65 480 M 35 600 L 65 600 M 35 720 L 65 720" stroke="#37482f" strokeWidth="5.5" />
            <path d="M 55 240 Q 85 210 95 180" fill="none" stroke="#37482f" strokeWidth="4" />
            <path d="M 95 180 Q 110 185 95 195 Q 85 190 95 180" fill="#37482f" />
            <path d="M 45 480 Q 15 450 5 420" fill="none" stroke="#37482f" strokeWidth="4" />
            <path d="M 5 420 Q -10 425 5 435 Q 15 430 5 420" fill="#37482f" />
          </svg>
        </div>

        {/* Right Bamboo Cluster (Thicker, Denser, 12 layered trunks) */}
        {/* Far Background Trunk 3 */}
        <div className="bamboo-detailed trunk-right-3" style={{ position: "absolute", right: "17%", bottom: "-10px", width: "70px", height: "105%", opacity: 0.1, filter: "blur(3px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-dark)" />
            <path d="M 38 150 L 62 150 M 38 300 L 62 300 M 38 450 L 62 450 M 38 600 L 62 600" stroke="#1b2518" strokeWidth="3" />
          </svg>
        </div>
        {/* Far Background Trunk 4 */}
        <div className="bamboo-detailed trunk-right-4" style={{ position: "absolute", right: "27%", bottom: "-10px", width: "80px", height: "105%", opacity: 0.12, filter: "blur(2.5px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-dark)" />
            <path d="M 38 140 L 62 140 M 38 280 L 62 280 M 38 420 L 62 420 M 38 560 L 62 560 M 38 700 L 62 700" stroke="#1b2518" strokeWidth="3.5" />
          </svg>
        </div>
        {/* Far Background Trunk 5 */}
        <div className="bamboo-detailed trunk-right-5" style={{ position: "absolute", right: "35%", bottom: "-10px", width: "60px", height: "105%", opacity: 0.07, filter: "blur(3.5px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-dark)" />
            <path d="M 38 160 L 62 160 M 38 320 L 62 320 M 38 480 L 62 480 M 38 640 L 62 640" stroke="#1b2518" strokeWidth="3" />
          </svg>
        </div>
        {/* Mid Background Trunk 6 */}
        <div className="bamboo-detailed trunk-right-6" style={{ position: "absolute", right: "22%", bottom: "-10px", width: "75px", height: "105%", opacity: 0.15, filter: "blur(2px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-mid)" />
            <path d="M 38 150 L 62 150 M 38 300 L 62 300 M 38 450 L 62 450 M 38 600 L 62 600" stroke="#23301e" strokeWidth="3" />
          </svg>
        </div>
        {/* Thick Front Trunk 1 */}
        <div className="bamboo-detailed trunk-right-1" style={{ position: "absolute", right: "4%", bottom: "-10px", width: "165px", height: "105%", opacity: 0.85 }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-light)" />
            <path d="M 33 140 L 67 140 M 33 280 L 67 280 M 33 420 L 67 420 M 33 560 L 67 560 M 33 700 L 67 700" stroke="#293723" strokeWidth="5.5" />
            <path d="M 45 280 Q 15 250 5 220" fill="none" stroke="#293723" strokeWidth="4" />
            <path d="M 5 220 Q -10 225 5 235 Q 15 230 5 220" fill="#293723" />
            <path d="M 55 560 Q 85 530 95 500" fill="none" stroke="#293723" strokeWidth="4" />
            <path d="M 95 500 Q 110 505 95 515 Q 85 510 95 500" fill="#293723" />
          </svg>
        </div>
        {/* Background Trunk 2 */}
        <div className="bamboo-detailed trunk-right-2" style={{ position: "absolute", right: "0.5%", bottom: "-10px", width: "100px", height: "105%", opacity: 0.18, filter: "blur(1.5px)" }}>
          <svg viewBox="0 0 100 800" preserveAspectRatio="none" style={{ width: "100%", height: "100%" }}>
            <path d="M 45 0 L 55 0 L 55 800 L 45 800 Z" fill="url(#bamboo-grad-mid)" />
            <path d="M 36 130 L 64 130 M 36 260 L 64 260 M 36 390 L 64 390 M 36 520 L 64 520 M 36 650 L 64 650" stroke="#23301e" strokeWidth="4" />
          </svg>
        </div>

        {/* Climbed Panda 1 (Left Front Trunk, side-profile clinging facing left) */}
        <div 
          onClick={() => handlePandaClick(1)}
          style={{ 
            position: "absolute", 
            left: "9.2%", 
            bottom: "28%", 
            width: "130px", 
            height: "130px", 
            cursor: "pointer",
            pointerEvents: "auto", 
            zIndex: 5, 
            filter: "drop-shadow(2px 5px 6px rgba(0,0,0,0.3))",
            transform: p1State === "falling" ? "translateY(350px) rotate(15deg) scale(0.95)" : p1State === "climbing" ? "translateY(0)" : "none",
            transition: p1State === "falling" ? "transform 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)" : p1State === "climbing" ? "transform 2.5s cubic-bezier(0.455, 0.03, 0.515, 0.955)" : "transform 0.4s ease"
          }}
          title="Click to see me slide down and climb back up!"
        >
          <svg viewBox="0 0 80 80" style={{ width: "100%", height: "100%" }}>
            {/* Stubby front arms clutched tightly to Leftward trunk (x=24) */}
            <rect className={`panda-limb-left ${p1State === "climbing" ? "wiggling" : ""}`} x="16" y="38" width="22" height="13" rx="6.5" fill="#222222" style={{ transformOrigin: "27px 44px" }} />
            {/* Stubby hind legs clutched tightly */}
            <rect className={`panda-limb-right ${p1State === "climbing" ? "wiggling" : ""}`} x="14" y="53" width="24" height="14" rx="7" fill="#222222" style={{ transformOrigin: "26px 60px" }} />
            {/* Chubby tail */}
            <circle cx="50" cy="58" r="4.5" fill="#222222" />
            {/* Shoulder saddle */}
            <path d="M 28 38 C 30 32, 46 32, 44 42 C 44 48, 28 46, 28 38 Z" fill="#222222" />
            {/* Chubby white body snug against stalk */}
            <ellipse cx="36" cy="51" rx="13" ry="15" fill="#ffffff" stroke="#1b2518" strokeWidth="1" />
            {/* Interactive Head rotating looking at user cursor */}
            <g style={{ transform: "var(--panda-rot-1)", transformOrigin: "34px 26px", transition: "transform 0.1s ease-out" }}>
              <circle cx="24" cy="16" r="5.5" fill="#222222" />
              <circle cx="42" cy="17" r="5" fill="#222222" />
              <circle cx="34" cy="26" r="13.5" fill="#ffffff" stroke="#1b2518" strokeWidth="1.2" />
              
              {/* Adorable Innocent Sparkly Eyes (Two white glints inside black) */}
              <ellipse cx="29" cy="25" rx="3.5" ry="4.8" fill="#222222" transform="rotate(-10, 29, 25)" />
              <circle cx="28.2" cy="23.8" r="1.1" fill="#ffffff" />
              <circle cx="29.8" cy="26.2" r="0.65" fill="#ffffff" />
              
              <ellipse cx="39" cy="25" rx="3" ry="4" fill="#222222" transform="rotate(10, 39, 25)" />
              <circle cx="38.2" cy="23.8" r="1.1" fill="#ffffff" />
              <circle cx="39.8" cy="26.2" r="0.65" fill="#ffffff" />
              
              <polygon points="32,29 36,29 34,31" fill="#222222" />
              <circle cx="26" cy="29" r="1.5" fill="#ffa8b6" opacity="0.6" />
              <circle cx="41" cy="30" r="1.5" fill="#ffa8b6" opacity="0.6" />
            </g>
          </svg>
        </div>

        {/* Climbed Panda 2 (Right Front Trunk, side-profile clinging facing right) */}
        <div 
          onClick={() => handlePandaClick(2)}
          style={{ 
            position: "absolute", 
            right: "5.2%", 
            bottom: "48%", 
            width: "130px", 
            height: "130px", 
            cursor: "pointer",
            pointerEvents: "auto", 
            zIndex: 5, 
            filter: "drop-shadow(-2px 5px 6px rgba(0,0,0,0.3))",
            transform: p2State === "falling" ? "translateY(450px) rotate(-15deg) scale(0.95)" : p2State === "climbing" ? "translateY(0)" : "none",
            transition: p2State === "falling" ? "transform 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)" : p2State === "climbing" ? "transform 2.5s cubic-bezier(0.455, 0.03, 0.515, 0.955)" : "transform 0.4s ease"
          }}
          title="Click to see me slide down and climb back up!"
        >
          <svg viewBox="0 0 80 80" style={{ width: "100%", height: "100%" }}>
            {/* Stubby front arms clutched tightly to Rightward trunk (x=56) */}
            <rect className={`panda-limb-left ${p2State === "climbing" ? "wiggling" : ""}`} x="42" y="38" width="22" height="13" rx="6.5" fill="#222222" style={{ transformOrigin: "53px 44px" }} />
            {/* Stubby hind legs clutched tightly */}
            <rect className={`panda-limb-right ${p2State === "climbing" ? "wiggling" : ""}`} x="42" y="53" width="24" height="14" rx="7" fill="#222222" style={{ transformOrigin: "54px 60px" }} />
            {/* Chubby tail */}
            <circle cx="30" cy="58" r="4.5" fill="#222222" />
            {/* Shoulder saddle */}
            <path d="M 52 38 C 50 32, 34 32, 36 42 C 36 48, 52 46, 52 38 Z" fill="#222222" />
            {/* Chubby white body snug against stalk */}
            <ellipse cx="44" cy="51" rx="13" ry="15" fill="#ffffff" stroke="#1b2518" strokeWidth="1" />
            {/* Interactive Head rotating looking at user cursor */}
            <g style={{ transform: "var(--panda-rot-2)", transformOrigin: "46px 26px", transition: "transform 0.1s ease-out" }}>
              <circle cx="56" cy="16" r="5.5" fill="#222222" />
              <circle cx="38" cy="17" r="5" fill="#222222" />
              <circle cx="46" cy="26" r="13.5" fill="#ffffff" stroke="#1b2518" strokeWidth="1.2" />
              
              {/* Adorable Innocent Sparkly Eyes (Two white glints inside black) */}
              <ellipse cx="41" cy="25" rx="3" ry="4" fill="#222222" transform="rotate(-10, 41, 25)" />
              <circle cx="40.2" cy="23.8" r="1.1" fill="#ffffff" />
              <circle cx="41.8" cy="26.2" r="0.65" fill="#ffffff" />
              
              <ellipse cx="51" cy="25" rx="3.5" ry="4.8" fill="#222222" transform="rotate(10, 51, 25)" />
              <circle cx="50.2" cy="23.8" r="1.1" fill="#ffffff" />
              <circle cx="51.8" cy="26.2" r="0.65" fill="#ffffff" />
              
              <polygon points="44,29 48,29 46,31" fill="#222222" />
              <circle cx="54" cy="29" r="1.5" fill="#ffa8b6" opacity="0.6" />
              <circle cx="39" cy="30" r="1.5" fill="#ffa8b6" opacity="0.6" />
            </g>
          </svg>
        </div>

        {/* Climbed Panda 3 (Far Right Background Trunk, mirrored facing left) */}
        <div 
          onClick={() => handlePandaClick(3)}
          style={{ 
            position: "absolute", 
            right: "0%", 
            bottom: "16%", 
            width: "105px", 
            height: "105px", 
            cursor: "pointer",
            pointerEvents: "auto", 
            zIndex: 4, 
            opacity: 0.8, 
            filter: "drop-shadow(-1px 3px 4px rgba(0,0,0,0.25))",
            transform: p3State === "falling" ? "translateY(180px) rotate(-15deg) scale(0.95) scaleX(-1)" : p3State === "climbing" ? "translateY(0) scaleX(-1)" : "scaleX(-1)",
            transition: p3State === "falling" ? "transform 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)" : p3State === "climbing" ? "transform 2.5s cubic-bezier(0.455, 0.03, 0.515, 0.955)" : "transform 0.4s ease"
          }}
          title="Click to see me slide down and climb back up!"
        >
          <svg viewBox="0 0 80 80" style={{ width: "100%", height: "100%" }}>
            {/* Stubby front arms clutched tightly */}
            <rect className={`panda-limb-left ${p3State === "climbing" ? "wiggling" : ""}`} x="42" y="38" width="22" height="13" rx="6.5" fill="#222222" style={{ transformOrigin: "53px 44px" }} />
            {/* Stubby hind legs clutched tightly */}
            <rect className={`panda-limb-right ${p3State === "climbing" ? "wiggling" : ""}`} x="42" y="53" width="24" height="14" rx="7" fill="#222222" style={{ transformOrigin: "54px 60px" }} />
            {/* Chubby tail */}
            <circle cx="30" cy="58" r="4.5" fill="#222222" />
            {/* Shoulder saddle */}
            <path d="M 52 38 C 50 32, 34 32, 36 42 C 36 48, 52 46, 52 38 Z" fill="#222222" />
            {/* Chubby white body snug against stalk */}
            <ellipse cx="44" cy="51" rx="13" ry="15" fill="#ffffff" stroke="#1b2518" strokeWidth="1" />
            {/* Interactive Head rotating looking at user cursor */}
            <g style={{ transform: "var(--panda-rot-3)", transformOrigin: "46px 26px", transition: "transform 0.1s ease-out" }}>
              <circle cx="56" cy="16" r="5.5" fill="#222222" />
              <circle cx="38" cy="17" r="5" fill="#222222" />
              <circle cx="46" cy="26" r="13.5" fill="#ffffff" stroke="#1b2518" strokeWidth="1.2" />
              
              {/* Adorable Innocent Sparkly Eyes (Two white glints inside black) */}
              <ellipse cx="41" cy="25" rx="3" ry="4" fill="#222222" transform="rotate(-10, 41, 25)" />
              <circle cx="40.2" cy="23.8" r="1.1" fill="#ffffff" />
              <circle cx="41.8" cy="26.2" r="0.65" fill="#ffffff" />
              
              <ellipse cx="51" cy="25" rx="3.5" ry="4.8" fill="#222222" transform="rotate(10, 51, 25)" />
              <circle cx="50.2" cy="23.8" r="1.1" fill="#ffffff" />
              <circle cx="51.8" cy="26.2" r="0.65" fill="#ffffff" />
              
              <polygon points="44,29 48,29 46,31" fill="#222222" />
              <circle cx="54" cy="29" r="1.5" fill="#ffa8b6" opacity="0.6" />
              <circle cx="39" cy="30" r="1.5" fill="#ffa8b6" opacity="0.6" />
            </g>
          </svg>
        </div>

        <div className="sakura-petal petal-1"></div>
        <div className="sakura-petal petal-2"></div>
        <div className="sakura-petal petal-3"></div>
        <div className="sakura-petal petal-4"></div>
        <div className="sakura-petal petal-5"></div>
        <div className="sakura-petal petal-6"></div>
      </div>

      <div className="leaf-vignette-container">
        <div className="leaf-vignette leaf-vignette-tl">
          <svg viewBox="0 0 120 120" width="100%" height="100%" fill="rgba(74, 93, 62, 0.08)">
            <path d="M10 0 C30 20, 50 10, 80 40 C60 50, 30 30, 10 0 Z" />
            <path d="M30 0 C50 30, 80 20, 100 60 C80 70, 50 40, 30 0 Z" />
            <path d="M0 20 C20 50, 40 40, 60 85 C45 90, 20 65, 0 20 Z" />
          </svg>
        </div>
        <div className="leaf-vignette leaf-vignette-tr">
          <svg viewBox="0 0 120 120" width="100%" height="100%" fill="rgba(74, 93, 62, 0.08)">
            <path d="M110 0 C90 20, 70 10, 40 40 C60 50, 90 30, 110 0 Z" />
            <path d="M90 0 C70 30, 40 20, 20 60 C40 70, 70 40, 90 0 Z" />
            <path d="M120 20 C100 50, 80 40, 60 85 C75 90, 100 65, 120 20 Z" />
          </svg>
        </div>
        <div className="leaf-vignette leaf-vignette-bl">
          <svg viewBox="0 0 120 120" width="100%" height="100%" fill="rgba(74, 93, 62, 0.06)">
            <path d="M0 100 C30 80, 50 90, 80 60 C65 50, 40 70, 0 100 Z" />
            <path d="M20 120 C50 90, 80 100, 100 60 C80 50, 50 80, 20 120 Z" />
          </svg>
        </div>
        <div className="leaf-vignette leaf-vignette-br">
          <svg viewBox="0 0 120 120" width="100%" height="100%" fill="rgba(74, 93, 62, 0.06)">
            <path d="M120 100 C90 80, 70 90, 40 60 C55 50, 80 70, 120 100 Z" />
            <path d="M100 120 C70 90, 40 100, 20 60 C40 50, 70 80, 100 120 Z" />
          </svg>
        </div>
      </div>

      <div className="japanese-stony-mountain" style={{ zIndex: 2 }}>
        <svg viewBox="0 0 160 400" preserveAspectRatio="none" style={{ width: "100%", height: "100%", overflow: "visible" }}>
          <defs>
            <linearGradient id="stone-light" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#a3b1bc" />
              <stop offset="50%" stopColor="#768490" />
              <stop offset="100%" stopColor="#55606a" />
            </linearGradient>
            <linearGradient id="stone-dark" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#5d666e" />
              <stop offset="100%" stopColor="#2c3035" />
            </linearGradient>
            <linearGradient id="stone-accent" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8794a0" />
              <stop offset="100%" stopColor="#3d444a" />
            </linearGradient>
          </defs>
          
          {/* Base crevice depths */}
          <path d="M 0 -30 Q 40 -50 70 -15 T 105 0 C 130 150, 85 280, 140 360 C 120 380, 125 390, 115 400 L 0 400 Z" fill="#151719" />
          <path d="M 0 -30 Q 40 -50 70 -15 T 105 0 C 125 140, 80 270, 135 350 C 115 375, 120 385, 110 400 L 0 400 Z" fill="url(#stone-dark)" />

          {/* Densely Stacked Natural River Cobblestones & Boulders */}
          
          {/* Row 1: Cap Stones */}
          <path d="M 0 -30 C 25 -35, 55 -25, 60 5 C 40 15, 15 10, 0 5 Z" fill="url(#stone-light)" stroke="#272a2e" strokeWidth="1" />
          <path d="M 3 -27 Q 25 -30 48 -10" fill="none" stroke="#dbe7f2" strokeWidth="1.2" opacity="0.7" strokeLinecap="round" /> {/* Highlight */}
          
          <path d="M 50 -20 C 70 -35, 95 -15, 105 0 C 105 30, 80 40, 55 35 C 45 25, 45 -5, 50 -20 Z" fill="url(#stone-accent)" stroke="#272a2e" strokeWidth="1" />
          <path d="M 58 -22 Q 80 -25 98 -5" fill="none" stroke="#dbe7f2" strokeWidth="1.2" opacity="0.7" strokeLinecap="round" /> {/* Highlight */}
          
          {/* Row 2: Upper Mid Stones */}
          <path d="M 0 5 C 20 10, 40 20, 35 50 C 15 60, 0 45, 0 35 Z" fill="url(#stone-dark)" stroke="#1c1e21" strokeWidth="1" />
          <path d="M 35 30 C 55 20, 85 25, 80 60 C 60 75, 35 70, 35 35 Z" fill="url(#stone-light)" stroke="#272a2e" strokeWidth="1" />
          <path d="M 42 32 Q 62 25 78 45" fill="none" stroke="#dbe7f2" strokeWidth="1.2" opacity="0.7" strokeLinecap="round" />
          
          <path d="M 0 45 C 18 50, 28 80, 18 110 C 5 120, 0 105, 0 80 Z" fill="url(#stone-accent)" stroke="#1c1e21" strokeWidth="1" />
          
          {/* Row 3: Center Stones */}
          <path d="M 18 100 C 40 90, 70 100, 65 140 C 45 160, 20 140, 18 100 Z" fill="url(#stone-dark)" stroke="#1c1e21" strokeWidth="1" />
          <path d="M 23 103 Q 45 95 62 120" fill="none" stroke="#dbe7f2" strokeWidth="1.2" opacity="0.7" strokeLinecap="round" />
          
          <path d="M 0 105 C 15 115, 25 150, 15 190 C 0 200, 0 170, 0 140 Z" fill="url(#stone-light)" stroke="#272a2e" strokeWidth="1" />
          <path d="M 2 110 Q 14 125 12 165" fill="none" stroke="#dbe7f2" strokeWidth="1.2" opacity="0.7" strokeLinecap="round" />

          {/* Row 4: Lower Mid Stones */}
          <path d="M 15 180 C 45 165, 80 180, 75 230 C 45 250, 20 230, 15 180 Z" fill="url(#stone-accent)" stroke="#272a2e" strokeWidth="1" />
          <path d="M 22 182 Q 52 170 72 205" fill="none" stroke="#dbe7f2" strokeWidth="1.2" opacity="0.7" strokeLinecap="round" />
          
          <path d="M 0 185 C 10 200, 15 240, 5 290 C 0 300, 0 270, 0 220 Z" fill="url(#stone-dark)" stroke="#1c1e21" strokeWidth="1" />
          
          <path d="M 5 285 C 35 270, 65 290, 60 340 C 35 360, 15 340, 5 285 Z" fill="url(#stone-light)" stroke="#272a2e" strokeWidth="1" />
          <path d="M 10 287 Q 40 275 58 315" fill="none" stroke="#dbe7f2" strokeWidth="1.2" opacity="0.7" strokeLinecap="round" />

          {/* Row 5: Bottom Base Stones */}
          <path d="M 60 330 C 90 315, 125 325, 115 400 L 50 400 Z" fill="url(#stone-dark)" stroke="#1c1e21" strokeWidth="1" />
          <path d="M 0 285 C 25 300, 35 365, 0 400 Z" fill="url(#stone-accent)" stroke="#272a2e" strokeWidth="1" />

          {/* Detailed Hanging Ivy Vines & Leaves */}
          <path d="M 55 35 Q 50 60 55 85 Q 58 95 53 110" fill="none" stroke="#485c3b" strokeWidth="1.8" strokeLinecap="round" opacity="0.85" />
          <circle cx="53" cy="50" r="3" fill="#5c7a4b" stroke="#374b2a" strokeWidth="0.5" />
          <circle cx="49" cy="70" r="2.5" fill="#4d663e" stroke="#374b2a" strokeWidth="0.5" />
          <circle cx="56" cy="80" r="3.2" fill="#5c7a4b" stroke="#374b2a" strokeWidth="0.5" />
          <circle cx="54" cy="98" r="2" fill="#4d663e" stroke="#374b2a" strokeWidth="0.5" />

          <path d="M 25 120 Q 30 145 24 175" fill="none" stroke="#3d4f32" strokeWidth="1.5" strokeLinecap="round" opacity="0.8" />
          <circle cx="28" cy="135" r="2.2" fill="#4d663e" stroke="#293d20" strokeWidth="0.5" />
          <circle cx="23" cy="155" r="2.8" fill="#3d5231" stroke="#293d20" strokeWidth="0.5" />

          {/* Crevice Moss Details */}
          <path d="M 45 10 Q 55 25 55 35" fill="none" stroke="#5c7a4b" strokeWidth="4" strokeLinecap="round" opacity="0.85" />
          
          {/* Ledge Peak Spillover Edge */}
          <path d="M 0 -30 Q 40 -50 70 -15 T 105 0" fill="none" stroke="#688059" strokeWidth="5" strokeLinecap="round" opacity="0.9" />
        </svg>
      </div>

      <div className="japanese-water-spout">
        <svg width="100%" height="100%" viewBox="0 0 60 2000" preserveAspectRatio="none" style={{ overflow: "visible" }}>
          <defs>
            <filter id="water-turbulence">
              <feTurbulence type="fractalNoise" baseFrequency="0.03 0.18" numOctaves="3" result="noise" />
              <feDisplacementMap in="SourceGraphic" in2="noise" scale="8" xChannelSelector="R" yChannelSelector="G" />
            </filter>
            <linearGradient id="water-fall-bg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(255, 255, 255, 0)" />
              <stop offset="15%" stopColor="rgba(165, 205, 227, 0.22)" />
              <stop offset="55%" stopColor="rgba(226, 241, 247, 0.16)" />
              <stop offset="85%" stopColor="rgba(165, 205, 227, 0.08)" />
              <stop offset="100%" stopColor="rgba(255, 255, 255, 0)" />
            </linearGradient>
            <linearGradient id="water-fall-foam" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(255, 255, 255, 0.95)" />
              <stop offset="25%" stopColor="rgba(230, 242, 247, 0.45)" />
              <stop offset="75%" stopColor="rgba(205, 225, 235, 0.35)" />
              <stop offset="100%" stopColor="rgba(255, 255, 255, 0.1)" />
            </linearGradient>
          </defs>
          
          {/* Curved Background Water Deep Volume */}
          <path d="M 5 0 C 12 300, 2 600, 15 1000 C 28 1400, 8 1700, 10 2000 L 50 2000 C 48 1700, 68 1400, 55 1000 C 42 600, 52 300, 45 0 Z" fill="url(#water-fall-bg)" opacity="0.65" />
          
          {/* Curved foaming rapids under turbulence */}
          <g filter="url(#water-turbulence)">
            <path d="M 5 0 C 12 300, 2 600, 15 1000 C 28 1400, 8 1700, 10 2000 L 50 2000 C 48 1700, 68 1400, 55 1000 C 42 600, 52 300, 45 0 Z" fill="url(#water-fall-foam)" opacity="0.5" />
            
            {/* Curved flowing highlight streams */}
            <path d="M 12 0 C 19 300, 9 600, 22 1000 C 35 1400, 15 1700, 17 2000" fill="none" stroke="#ffffff" strokeWidth="3" className="water-streak-fast" strokeDasharray="30 100" strokeLinecap="round" opacity="0.85" />
            <path d="M 23 0 C 30 300, 20 600, 33 1000 C 46 1400, 26 1700, 28 2000" fill="none" stroke="#e6f2f7" strokeWidth="4" className="water-streak-slow" strokeDasharray="50 150" strokeLinecap="round" opacity="0.65" />
            <path d="M 34 0 C 41 300, 31 600, 44 1000 C 57 1400, 37 1700, 39 2000" fill="none" stroke="#ffffff" strokeWidth="2.5" className="water-streak-fastest" strokeDasharray="20 80" strokeLinecap="round" opacity="0.9" />
            <path d="M 44 0 C 51 300, 41 600, 54 1000 C 67 1400, 47 1700, 49 2000" fill="none" stroke="#e6f2f7" strokeWidth="3.5" className="water-streak-slow" strokeDasharray="40 120" strokeLinecap="round" opacity="0.7" />
          </g>
        </svg>
      </div>

      <div className="japanese-ripple-pond">
        <div style={{ position: "absolute", bottom: 0, left: 0, width: "100%", height: "100%", background: "linear-gradient(to top, rgba(74, 134, 166, 0.65) 0%, rgba(110, 168, 196, 0.35) 40%, transparent 100%)" }}></div>
        <svg viewBox="0 0 1000 100" preserveAspectRatio="none" style={{ position: "absolute", bottom: 0, left: 0, width: "200%", height: "100%", opacity: 0.8 }} className="flowing-water-wave">
          <path d="M 0 50 Q 125 70 250 50 T 500 50 T 750 50 T 1000 50" fill="none" stroke="rgba(74, 134, 166, 0.6)" strokeWidth="2.5" />
          <path d="M 0 65 Q 125 45 250 65 T 500 65 T 750 65 T 1000 65" fill="none" stroke="rgba(74, 134, 166, 0.4)" strokeWidth="3" />
          <path d="M 0 80 Q 125 100 250 80 T 500 80 T 750 80 T 1000 80" fill="none" stroke="rgba(74, 134, 166, 0.8)" strokeWidth="2" />
        </svg>
        <svg width="100%" height="100%" style={{ position: "absolute", top: 0, left: 0 }}>
          <ellipse cx="10%" cy="85%" rx="60" ry="8" fill="none" stroke="rgba(165,205,227,0.4)" strokeWidth="1.5" className="pond-ripple-svg ripple-delay-1" />
          <ellipse cx="10%" cy="85%" rx="100" ry="14" fill="none" stroke="rgba(165,205,227,0.2)" strokeWidth="1" className="pond-ripple-svg ripple-delay-2" />
          <ellipse cx="45%" cy="75%" rx="80" ry="12" fill="none" stroke="rgba(165,205,227,0.3)" strokeWidth="1.5" className="pond-ripple-svg ripple-delay-3" />
          <ellipse cx="45%" cy="75%" rx="130" ry="18" fill="none" stroke="rgba(165,205,227,0.15)" strokeWidth="1" className="pond-ripple-svg ripple-delay-4" />
          <ellipse cx="85%" cy="80%" rx="70" ry="10" fill="none" stroke="rgba(165,205,227,0.35)" strokeWidth="1.5" className="pond-ripple-svg ripple-delay-5" />
          <ellipse cx="85%" cy="80%" rx="115" ry="16" fill="none" stroke="rgba(165,205,227,0.15)" strokeWidth="1" className="pond-ripple-svg ripple-delay-6" />
        </svg>
        <div className="pond-koi-fish koi-1" style={{ position: "absolute", left: "10%", bottom: "25px", transform: "scale(0.85) rotate(70deg)", opacity: 0.85, filter: "drop-shadow(2px 4px 4px rgba(0,0,0,0.3))" }}>
          <svg viewBox="0 0 100 100" width="45" height="45">
            <path d="M 50 80 Q 25 105 50 95 Q 75 105 50 80" fill="#cc4e28" opacity="0.6" />
            <path d="M 35 45 Q 10 65 35 55" fill="#cc4e28" opacity="0.7" />
            <path d="M 65 45 Q 90 65 65 55" fill="#cc4e28" opacity="0.7" />
            <path d="M 50 10 C 20 30, 30 75, 50 85 C 70 75, 80 30, 50 10 Z" fill="#e85d31" />
            <path d="M 50 15 C 35 25, 60 35, 50 45 C 40 35, 65 25, 50 15 Z" fill="#fcf9f2" opacity="0.9" />
          </svg>
        </div>
        <div className="pond-koi-fish koi-2" style={{ position: "absolute", left: "86%", bottom: "20px", transform: "scale(0.95) rotate(-25deg)", opacity: 0.8, filter: "drop-shadow(2px 4px 4px rgba(0,0,0,0.3))" }}>
          <svg viewBox="0 0 100 100" width="45" height="45">
            <path d="M 50 80 Q 25 105 50 95 Q 75 105 50 80" fill="#e8e1d5" opacity="0.6" />
            <path d="M 35 45 Q 10 65 35 55" fill="#e8e1d5" opacity="0.7" />
            <path d="M 65 45 Q 90 65 65 55" fill="#e8e1d5" opacity="0.7" />
            <path d="M 50 10 C 20 30, 30 75, 50 85 C 70 75, 80 30, 50 10 Z" fill="#fcf9f2" />
            <path d="M 50 20 C 35 30, 60 45, 50 55 C 40 45, 65 30, 50 20 Z" fill="#cc4e28" opacity="0.9" />
          </svg>
        </div>
        <div className="pond-koi-fish koi-3" style={{ position: "absolute", left: "93%", bottom: "50px", transform: "scale(0.7) rotate(110deg)", opacity: 0.75, filter: "drop-shadow(2px 4px 4px rgba(0,0,0,0.3))" }}>
          <svg viewBox="0 0 100 100" width="45" height="45">
            <path d="M 50 80 Q 25 105 50 95 Q 75 105 50 80" fill="#bfa145" opacity="0.6" />
            <path d="M 35 45 Q 10 65 35 55" fill="#bfa145" opacity="0.7" />
            <path d="M 65 45 Q 90 65 65 55" fill="#bfa145" opacity="0.7" />
            <path d="M 50 10 C 20 30, 30 75, 50 85 C 70 75, 80 30, 50 10 Z" fill="#dfbc54" />
            <path d="M 45 45 C 35 55, 55 60, 48 70 C 40 60, 55 55, 45 45 Z" fill="#fcf9f2" opacity="0.6" />
          </svg>
        </div>
        <div className="advanced-lily-pad" style={{ left: "12%", bottom: "40px", transform: "scale(0.85) rotate(15deg)" }}>
          <svg viewBox="0 0 100 100">
            <path d="M50 5 C75 5 95 25 95 50 C95 75 75 95 50 95 C25 95 5 75 5 50 C5 35 15 20 30 10 L50 50 Z" fill="#4d6641" stroke="#384f2e" strokeWidth="2" />
            <path d="M50 50 L35 15 M50 50 L15 35 M50 50 L15 65 M50 50 L35 85 M50 50 L65 85 M50 50 L85 65 M50 50 L85 35 M50 50 L65 15" stroke="#384f2e" strokeWidth="1" opacity="0.6" />
          </svg>
        </div>
        <div className="advanced-lily-pad" style={{ left: "85%", bottom: "10px", transform: "scale(1.1) rotate(-25deg)" }}>
          <svg viewBox="0 0 100 100">
            <path d="M50 5 C75 5 95 25 95 50 C95 75 75 95 50 95 C25 95 5 75 5 50 C5 35 15 20 30 10 L50 50 Z" fill="#58754b" stroke="#384f2e" strokeWidth="2" />
            <path d="M50 50 L35 15 M50 50 L15 35 M50 50 L15 65 M50 50 L35 85 M50 50 L65 85 M50 50 L85 65 M50 50 L85 35 M50 50 L65 15" stroke="#384f2e" strokeWidth="1" opacity="0.6" />
          </svg>
        </div>
        <div className="advanced-lily-pad" style={{ left: "92%", bottom: "30px", transform: "scale(0.6) rotate(60deg)" }}>
          <svg viewBox="0 0 100 100">
            <path d="M50 5 C75 5 95 25 95 50 C95 75 75 95 50 95 C25 95 5 75 5 50 C5 35 15 20 30 10 L50 50 Z" fill="#435c38" stroke="#293d20" strokeWidth="2" />
            <path d="M50 50 L35 15 M50 50 L15 35 M50 50 L15 65 M50 50 L35 85 M50 50 L65 85 M50 50 L85 65 M50 50 L85 35 M50 50 L65 15" stroke="#293d20" strokeWidth="1" opacity="0.6" />
          </svg>
        </div>
        
        {/* Waterfall Splash Spray */}
        <div className="waterfall-splash" style={{ position: "absolute", left: "62px", bottom: "95px", zIndex: 3, pointerEvents: "none" }}>
          <svg width="45" height="50" viewBox="0 0 45 50">
            <ellipse cx="22.5" cy="45" rx="18" ry="4" fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth="1.2" className="pond-ripple-svg ripple-delay-1" />
            <circle cx="12" cy="38" r="3.5" fill="rgba(255,255,255,0.9)" className="splash-droplet droplet-1" />
            <circle cx="22.5" cy="30" r="2.5" fill="rgba(255,255,255,0.95)" className="splash-droplet droplet-2" />
            <circle cx="32" cy="35" r="3" fill="rgba(255,255,255,0.9)" className="splash-droplet droplet-3" />
            <circle cx="17" cy="22" r="1.8" fill="rgba(255,255,255,1)" className="splash-droplet droplet-4" />
            <circle cx="28" cy="26" r="2" fill="rgba(255,255,255,1)" className="splash-droplet droplet-5" />
          </svg>
        </div>
      </div>

      <div
        className={`japanese-water-wheel ${wheelOpen ? "menu-open" : ""}`}
        style={{
          transform: wheelOpen ? `rotate(${activeWheelIndex * -25}deg)` : undefined
        }}
        onClick={() => setWheelOpen(!wheelOpen)}
        title="Japanese Mizuguruma Water Wheel"
      >
        <svg viewBox="0 0 100 100" style={{ width: "100%", height: "100%", filter: "drop-shadow(4px 8px 12px rgba(0,0,0,0.5))" }}>
          <defs>
            <linearGradient id="wood-grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#664c38" />
              <stop offset="30%" stopColor="#4d3525" />
              <stop offset="70%" stopColor="#3d2a1c" />
              <stop offset="100%" stopColor="#24180f" />
            </linearGradient>
            <linearGradient id="metal-grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#d6d6d6" />
              <stop offset="30%" stopColor="#9e9e9e" />
              <stop offset="70%" stopColor="#5c5c5c" />
              <stop offset="100%" stopColor="#383838" />
            </linearGradient>
            <linearGradient id="gold-grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#ffe891" />
              <stop offset="50%" stopColor="#dfb743" />
              <stop offset="100%" stopColor="#876513" />
            </linearGradient>
          </defs>
          {/* Outer Wood Rim base */}
          <circle cx="50" cy="50" r="46" fill="none" stroke="url(#wood-grad)" strokeWidth="6" />
          <circle cx="50" cy="50" r="43" fill="none" stroke="#1c130b" strokeWidth="1" />
          
          {/* Outer Gold Band Accent */}
          <circle cx="50" cy="50" r="40" fill="none" stroke="url(#gold-grad)" strokeWidth="1.5" />
          
          {/* Spoke brackets and spokes */}
          <g stroke="url(#wood-grad)" strokeWidth="4.5" strokeLinecap="round">
            <line x1="50" y1="12" x2="50" y2="88" />
            <line x1="12" y1="50" x2="88" y2="50" />
            <line x1="23.2" y1="23.2" x2="76.8" y2="76.8" />
            <line x1="23.2" y1="76.8" x2="76.8" y2="23.2" />
          </g>
          
          {/* Metal Spoke Reinforcements */}
          <g stroke="url(#metal-grad)" strokeWidth="1.5" opacity="0.85">
            <line x1="50" y1="18" x2="50" y2="82" />
            <line x1="18" y1="50" x2="82" y2="50" />
            <line x1="27.4" y1="27.4" x2="72.6" y2="72.6" />
            <line x1="27.4" y1="72.6" x2="72.6" y2="27.4" />
          </g>
          
          {/* Inner Ring */}
          <circle cx="50" cy="50" r="28" fill="none" stroke="url(#wood-grad)" strokeWidth="3" />
          <circle cx="50" cy="50" r="29.5" fill="none" stroke="#1c130b" strokeWidth="0.5" />
          <circle cx="50" cy="50" r="26.5" fill="none" stroke="#1c130b" strokeWidth="0.5" />
          
          {/* Core Rivet and Gold Accent Ring */}
          <circle cx="50" cy="50" r="14" fill="url(#wood-grad)" stroke="#1c130b" strokeWidth="2.5" />
          <circle cx="50" cy="50" r="9" fill="url(#metal-grad)" stroke="#111111" strokeWidth="1" />
          <circle cx="50" cy="50" r="4" fill="url(#gold-grad)" stroke="#4d3525" strokeWidth="0.75" />
          
          {/* Core details */}
          <circle cx="50" cy="42.5" r="1" fill="#111" />
          <circle cx="50" cy="57.5" r="1" fill="#111" />
          <circle cx="42.5" cy="50" r="1" fill="#111" />
          <circle cx="57.5" cy="50" r="1" fill="#111" />
          
          {/* Perimeter Water Buckets (Detailed wood & metal scoops) */}
          {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, idx) => (
            <g key={idx} transform={`rotate(${angle}, 50, 50)`}>
              <path d="M 42 2 L 58 2 L 55 13 L 45 13 Z" fill="url(#wood-grad)" stroke="#1c130b" strokeWidth="1" />
              <path d="M 42 2 L 58 2 L 58 4 L 42 4 Z" fill="url(#metal-grad)" stroke="#111" strokeWidth="0.5" />
              <path d="M 45 13 L 55 13 L 50 20 Z" fill="#111" opacity="0.45" />
            </g>
          ))}
        </svg>
      </div>

      <header className="header" style={{ backdropFilter: "blur(4px)", borderBottom: "1px double rgba(184, 122, 45, 0.25)", background: "rgba(250, 246, 238, 0.9)", zIndex: 999, padding: "10px 24px" }}>
        <div className="header-left">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="hamburger-btn"
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              width: "20px",
              height: "14px",
              padding: 0,
              marginRight: "14px",
              zIndex: 1000
            }}
            title="Toggle Chat History"
          >
            <span style={{ width: "100%", height: "2px", backgroundColor: "var(--text-primary)", transition: "all 0.3s" }}></span>
            <span style={{ width: "100%", height: "2px", backgroundColor: "var(--text-primary)", transition: "all 0.3s" }}></span>
            <span style={{ width: "100%", height: "2px", backgroundColor: "var(--text-primary)", transition: "all 0.3s" }}></span>
          </button>
          <div className="header-logo" style={{ background: "#b53e3e", color: "#ffffff", borderRadius: "2px", width: "24px", height: "24px", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "Georgia, serif", fontSize: "14px", fontWeight: "bold", boxShadow: "0 0 0 1.5px #b53e3e, 0 1px 3px rgba(0,0,0,0.15)", marginRight: "12px" }} title="理 (Ri) - Reason/Logic Seal">理</div>
          <div>
            <div className="header-title" style={{ fontFamily: "Georgia, serif", fontWeight: 600, letterSpacing: "0.5px", color: "var(--text-primary)" }}>PolyRAG</div>
            <div className="header-subtitle" style={{ fontFamily: "Georgia, serif", fontStyle: "italic", color: "var(--text-secondary)", fontSize: "10px", letterSpacing: "0.2px" }}>Multimodal RAG Archival Search</div>
          </div>
        </div>
        <div className="header-right">
          {dbHealth && (
            <div style={{ display: "flex", gap: "6px", marginRight: "1rem", alignItems: "center" }}>
              <DbBadge label="Engine" status={dbHealth.engine} />
              <DbBadge label="Postgres" status={dbHealth.postgres_docker} />
            </div>
          )}
          <button
            onClick={() => setShowHints(!showHints)}
            className={`hints-bulb-btn ${showHints ? "active" : ""}`}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "4px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginRight: "0.5rem",
              transition: "all 0.25s cubic-bezier(0.16, 1, 0.3, 1)"
            }}
            title="Click for Ledger Hints & Navigation Map"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill={showHints ? "rgba(184, 122, 45, 0.25)" : "none"} xmlns="http://www.w3.org/2000/svg">
              <path d="M9 21H15" stroke={showHints ? "var(--accent-amber)" : "var(--text-secondary)"} strokeWidth="2" strokeLinecap="round"/>
              <path d="M10 18H14" stroke={showHints ? "var(--accent-amber)" : "var(--text-secondary)"} strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 2C7.58 2 4 5.58 4 10C4 12.78 5.42 15.22 7.58 16.59C8.47 17.15 9 18.06 9 19V19.5" stroke={showHints ? "var(--accent-amber)" : "var(--text-secondary)"} strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 2C16.42 2 20 5.58 20 10C20 12.78 18.58 15.22 16.42 16.59C15.53 17.15 15 18.06 15 19V19.5" stroke={showHints ? "var(--accent-amber)" : "var(--text-secondary)"} strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 6V12" stroke={showHints ? "var(--accent-amber)" : "var(--text-secondary)"} strokeWidth="2" strokeLinecap="round"/>
              <path d="M9.5 9.5H14.5" stroke={showHints ? "var(--accent-amber)" : "var(--text-secondary)"} strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
          <button
            onClick={() => setActiveModal("settings")}
            className="header-settings-btn"
            style={{ padding: "5px 12px", border: "1px solid var(--border-accent)", background: "none", color: "var(--text-primary)", fontSize: "10px", cursor: "pointer", fontFamily: "Georgia, serif", fontWeight: 600, letterSpacing: "0.5px", textTransform: "uppercase", transition: "all 0.2s ease" }}
          >
            [ SETUP SYSTEM ]
          </button>
          {session ? (
            <button
              onClick={() => supabase.auth.signOut()}
              className="header-settings-btn"
              style={{
                padding: "5px 12px",
                border: "1px solid var(--accent-rose)",
                background: "none",
                color: "var(--accent-rose)",
                fontSize: "10px",
                cursor: "pointer",
                fontFamily: "Georgia, serif",
                fontWeight: 600,
                letterSpacing: "0.5px",
                textTransform: "uppercase",
                transition: "all 0.2s ease",
                marginLeft: "8px"
              }}
            >
              [ SIGN OUT ]
            </button>
          ) : (
            <button
              onClick={() => supabase.auth.signInWithOAuth({
                provider: 'google',
                options: { redirectTo: window.location.origin }
              })}
              className="header-settings-btn"
              style={{
                padding: "5px 12px",
                border: "1px solid var(--accent-indigo)",
                background: "none",
                color: "var(--accent-indigo)",
                fontSize: "10px",
                cursor: "pointer",
                fontFamily: "Georgia, serif",
                fontWeight: 600,
                letterSpacing: "0.5px",
                textTransform: "uppercase",
                transition: "all 0.2s ease",
                marginLeft: "8px"
              }}
            >
              [ SIGN IN ]
            </button>
          )}
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
                  <MarkdownRenderer content={msg.content} sources={msg.meta?.sources} />
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
      {wheelOpen && (
        <div className="weapon-wheel-overlay" onClick={() => setWheelOpen(false)}>
          <div
            style={{
              position: "fixed",
              bottom: "-20px",
              left: "-112px",
              width: "225px",
              height: "225px",
              zIndex: 1001,
              pointerEvents: "none"
            }}
            onClick={e => e.stopPropagation()}
          >
            <svg
              viewBox="0 0 360 360"
              style={{
                width: "540px",
                height: "540px",
                position: "absolute",
                left: "-157.5px",
                bottom: "-157.5px",
                overflow: "visible",
                pointerEvents: "none"
              }}
            >
              {/* Stationary Pointer Arrow pointing directly at active slot (0 degrees / horizontal right) */}
              <g transform="translate(180, 180)">
                <polygon points="175,-8 163,0 175,8" fill="#dfb743" stroke="#5e5041" strokeWidth="1.5" style={{ filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.35))" }} />
              </g>

              {/* Rotating slices group - centers active slice at 0 degrees horizontally */}
              <g
                style={{
                  transform: `rotate(${-(-62.5 + activeWheelIndex * 25)}deg)`,
                  transformOrigin: "180px 180px",
                  transition: "transform 0.6s cubic-bezier(0.16, 1, 0.3, 1)"
                }}
              >
                {[
                  { id: "clear", label: "CLEAR", start: -75, end: -50, action: () => setMessages([]) },
                  { id: "files", label: "FILES", start: -50, end: -25, action: () => setActiveModal("files") },
                  { id: "health", label: "HEALTH", start: -25, end: 0, action: () => setActiveModal("health") },
                  { id: "planner", label: "PLANNER", start: 0, end: 25,
                    action: () => {
                      const nextPlanner = !config.enablePlanner;
                      setConfig(prev => ({ ...prev, enablePlanner: nextPlanner }));
                      updateConfig({ ...config, enablePlanner: nextPlanner })
                        .then(() => alert(`[ROUTING STATUS] Planner ${nextPlanner ? "Activated" : "Deactivated"}`))
                        .catch(() => {});
                    }
                  },
                  { id: "settings", label: "SETTINGS", start: 25, end: 50, action: () => setActiveModal("settings") },
                  { id: "toggle-rag", label: "RAG TOGGLE", start: 50, end: 75,
                    action: () => {
                      if (selectedFileIds.size > 0) {
                        setSelectedFileIds(new Set());
                      } else {
                        setSelectedFileIds(new Set(files.filter(f => f.status === "indexed" && f.id).map(f => f.id)));
                      }
                    }
                  }
                ].map((slice, idx) => {
                  const isActive = activeWheelIndex === idx;
                  const arcD = getArcPath(180, 180, 78, 172, slice.start, slice.end);
                  const textCoords = getMidpointCoords(180, 180, 125, (slice.start + slice.end) / 2);
                  return (
                    <g
                      key={slice.id}
                      onClick={() => setActiveWheelIndex(idx)}
                      style={{ cursor: "pointer", pointerEvents: "auto" }}
                    >
                      <path
                        d={arcD}
                        fill={isActive ? "rgba(140, 59, 48, 0.9)" : "rgba(251, 248, 240, 0.95)"}
                        stroke="#5e5041"
                        strokeWidth="2"
                        style={{ transition: "all 0.25s ease" }}
                      />
                      <text
                        x={textCoords.x}
                        y={textCoords.y}
                        transform={`rotate(${(slice.start + slice.end) / 2}, ${textCoords.x}, ${textCoords.y})`}
                        fill={isActive ? "#ffffff" : "var(--text-primary)"}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontWeight: "700",
                          fontSize: "11px",
                          pointerEvents: "none",
                          letterSpacing: "0.5px"
                        }}
                      >
                        {slice.label}
                      </text>
                    </g>
                  );
                })}
              </g>

              <g
                onClick={() => setActiveWheelIndex(prev => (prev === 0 ? 5 : prev - 1))}
                style={{ cursor: "pointer", pointerEvents: "auto" }}
              >
                <path
                  d={getArcPath(180, 180, 78, 172, -90, -77)}
                  fill="#f4eedf"
                  stroke="#5e5041"
                  strokeWidth="2"
                />
                <text
                  x={getMidpointCoords(180, 180, 125, -83.5).x}
                  y={getMidpointCoords(180, 180, 125, -83.5).y}
                  fill="var(--text-primary)"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ fontFamily: "var(--font-mono)", fontWeight: "bold", fontSize: "14px" }}
                >
                  ▲
                </text>
              </g>

              <g
                onClick={() => setActiveWheelIndex(prev => (prev === 5 ? 0 : prev + 1))}
                style={{ cursor: "pointer", pointerEvents: "auto" }}
              >
                <path
                  d={getArcPath(180, 180, 78, 172, 77, 90)}
                  fill="#f4eedf"
                  stroke="#5e5041"
                  strokeWidth="2"
                />
                <text
                  x={getMidpointCoords(180, 180, 125, 83.5).x}
                  y={getMidpointCoords(180, 180, 125, 83.5).y}
                  fill="var(--text-primary)"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ fontFamily: "var(--font-mono)", fontWeight: "bold", fontSize: "14px" }}
                >
                  ▼
                </text>
              </g>
            </svg>

            <div
              className="mizuguruma-rack-menu"
              style={{
                position: "absolute",
                left: "400px",
                bottom: "0px",
                width: "295px",
                background: "#fbf8f0",
                border: "2px solid #5e5041",
                borderLeft: "6px solid #8c3b30",
                borderRadius: "4px",
                padding: "16px",
                boxShadow: "var(--shadow-dossier)",
                display: "flex",
                flexDirection: "column",
                gap: "10px",
                pointerEvents: "auto"
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px dashed var(--border-subtle)", paddingBottom: "6px" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", fontWeight: "700", color: "#8c3b30", letterSpacing: "1px" }}>
                  [ MIZUGURUMA DIAL ]
                </span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                  {activeWheelIndex + 1} / 6
                </span>
              </div>

              <p style={{ fontFamily: "var(--font-serif)", fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.5", minHeight: "44px", margin: 0 }}>
                {[
                  "Clear all records in chat search ledger",
                  "Open source registry panel to ingest or select documents",
                  "Inspect current metrics and average latency",
                  config.enablePlanner ? "Disable deep multihop contextual planning expert" : "Enable deep multihop contextual planning expert",
                  "Configure local LLM servers, Groq, Gemini keys & parser depth",
                  selectedFileIds.size > 0 ? "Clear active file filter constraints (default all)" : "Engage constraint filtering on all indexed materials"
                ][activeWheelIndex]}
              </p>

              <button
                onClick={() => {
                  const items = [
                    { action: () => setMessages([]) },
                    { action: () => setActiveModal("files") },
                    { action: () => setActiveModal("health") },
                    {
                      action: () => {
                        const nextPlanner = !config.enablePlanner;
                        setConfig(prev => ({ ...prev, enablePlanner: nextPlanner }));
                        updateConfig({ ...config, enablePlanner: nextPlanner })
                          .then(() => alert(`[ROUTING STATUS] Planner ${nextPlanner ? "Activated" : "Deactivated"}`))
                          .catch(() => {});
                      }
                    },
                    { action: () => setActiveModal("settings") },
                    {
                      action: () => {
                        if (selectedFileIds.size > 0) {
                          setSelectedFileIds(new Set());
                        } else {
                          setSelectedFileIds(new Set(files.filter(f => f.status === "indexed" && f.id).map(f => f.id)));
                        }
                      }
                    }
                  ];
                  items[activeWheelIndex].action();
                  setWheelOpen(false);
                }}
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "var(--accent-indigo)",
                  color: "#ffffff",
                  border: "none",
                  borderRadius: "3px",
                  fontFamily: "var(--font-sans)",
                  fontWeight: "700",
                  fontSize: "11px",
                  cursor: "pointer",
                  letterSpacing: "0.5px"
                }}
              >
                EXECUTE SELECTION
              </button>
            </div>
          </div>
        </div>
      )}

      {showHints && (
        <div className="hints-scroll-overlay" onClick={() => setShowHints(false)}>
          <div className="hints-scroll-card" onClick={e => e.stopPropagation()}>
            <div className="hints-scroll-header">
              <span className="hints-scroll-title">[ ANCIENT ARCHIVAL USER MAP ]</span>
              <button className="hints-scroll-close" onClick={() => setShowHints(false)}>×</button>
            </div>
            <div className="hints-scroll-body">
              <div className="hint-item">
                <span className="hint-icon">⚙</span>
                <div className="hint-content">
                  <h4>Mizuguruma Water Wheel</h4>
                  <p>Located submerged at the bottom left of the garden pond. Click to access archival controls.</p>
                </div>
              </div>
              <div className="hint-item">
                <span className="hint-icon">⚖</span>
                <div className="hint-content">
                  <h4>Search Constraints</h4>
                  <p>Toggle active files under the <strong>RAG TOGGLE</strong> menu action to index query scope.</p>
                </div>
              </div>
              <div className="hint-item">
                <span className="hint-icon">✦</span>
                <div className="hint-content">
                  <h4>Expert Abstraction</h4>
                  <p>PolyRAG routes tasks automatically to code, ledger, or image experts depending on query type.</p>
                </div>
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

                            {file.status !== "indexed" && file.status !== "failed" && file.status !== "uploading" && (
                              <div style={{ marginTop: "4px" }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "3px" }}>
                                  <span style={{
                                    fontFamily: "var(--font-mono)", fontSize: "9px", fontWeight: 700,
                                    textTransform: "uppercase", letterSpacing: "0.5px",
                                    color: file.status === "embedding" ? "var(--accent-indigo)"
                                         : file.status === "storing" || file.status === "indexing" ? "var(--accent-emerald)"
                                         : "var(--accent-amber)"
                                  }}>
                                    {file.status === "parsing" ? "⟳ Parsing document"
                                     : file.status === "captioning" ? "⟳ Captioning images"
                                     : file.status === "preparing" ? "⟳ Preparing pipeline"
                                     : file.status === "embedding" ? "⟳ Embedding chunks"
                                     : file.status === "storing" ? "⟳ Storing to DB"
                                     : file.status === "indexing" ? "⟳ Building indexes"
                                     : `⟳ ${file.status || "processing"}`}
                                  </span>
                                  <span style={{
                                    fontFamily: "var(--font-mono)", fontSize: "10px", fontWeight: 700,
                                    color: "var(--text-secondary)"
                                  }}>
                                    {file.progress || 0}%
                                  </span>
                                </div>
                                <div className="progress-bar" style={{ height: "5px" }}>
                                  <div className="progress-fill" style={{
                                    width: `${file.progress || 0}%`,
                                    background: (file.progress || 0) >= 80 ? "var(--accent-emerald)" : "var(--accent-cyan)"
                                  }}></div>
                                </div>
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
                          <option value="gemini-2.5-flash">Gemini 2.5 Flash (Gemini)</option>
                          <option value="llama3.2:3b">Llama 3.2 (Local)</option>
                          <option value="gemma3:4b">Gemma 3 4B (Local)</option>
                          <option value="llama-3.3-70b-specdec">Llama 3.3 70B (Groq)</option>
                          <option value="gemma2-9b-it">Gemma 2 9B (Groq)</option>
                          <option value="mixtral-8x7b-32768">Mixtral 8x7B (Groq)</option>
                        </>
                      )}
                    </select>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                    <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "16px", background: "var(--bg-card)" }}>
                      <div className="sidebar-label" style={{ margin: "0 0 12px 0" }}>API Credentials</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: "14px", maxHeight: "300px", overflowY: "auto" }}>
                        <div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                            <label style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>GROQ API KEYS</label>
                            <button onClick={() => {
                              const keys = Array.isArray(config.groqApiKeys) ? [...config.groqApiKeys] : (config.groqApiKey ? [config.groqApiKey] : []);
                              keys.push("");
                              handleConfigChange("groqApiKeys", keys);
                            }} style={{ background: "none", border: "none", color: "var(--accent-indigo)", fontSize: "11px", cursor: "pointer", padding: 0, fontWeight: "bold" }}>+ Add Key</button>
                          </div>
                          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                            {(() => {
                              const keys = Array.isArray(config.groqApiKeys) ? config.groqApiKeys : (config.groqApiKey ? [config.groqApiKey] : [""]);
                              if (keys.length === 0) return <span style={{ fontSize: "10px", color: "var(--text-muted)", fontStyle: "italic" }}>No keys added. Click "+ Add Key" to add one.</span>;
                              return keys.map((key, index) => {
                                const isVisible = !!showGroqKeys[index];
                                return (
                                  <div key={index} style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                                    <div style={{ position: "relative", flex: 1, display: "flex", alignItems: "center" }}>
                                      <input type={isVisible ? "text" : "password"} value={key || ""} onChange={e => {
                                        const updated = [...keys];
                                        updated[index] = e.target.value;
                                        handleConfigChange("groqApiKeys", updated);
                                      }} style={{ flex: 1, padding: "6px 28px 6px 10px", borderRadius: "4px", border: "1px solid var(--border-subtle)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", fontFamily: "var(--font-mono)" }} placeholder="gsk_..." />
                                      <button onClick={() => {
                                        setShowGroqKeys(prev => ({ ...prev, [index]: !prev[index] }));
                                      }} style={{ position: "absolute", right: "8px", background: "none", border: "none", color: "var(--text-muted)", fontSize: "12px", cursor: "pointer", padding: 0, display: "flex", alignItems: "center" }}>
                                        {isVisible ? "👁️" : "🙈"}
                                      </button>
                                    </div>
                                    <button onClick={() => {
                                      const updated = keys.filter((_, i) => i !== index);
                                      handleConfigChange("groqApiKeys", updated);
                                    }} style={{ background: "none", border: "none", color: "var(--accent-rose)", fontSize: "12px", cursor: "pointer", padding: "0 4px" }}>✕</button>
                                  </div>
                                );
                              });
                            })()}
                          </div>
                        </div>

                        <div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                            <label style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>GEMINI API KEYS</label>
                            <button onClick={() => {
                              const keys = Array.isArray(config.geminiApiKeys) ? [...config.geminiApiKeys] : (config.geminiApiKey ? [config.geminiApiKey] : []);
                              keys.push("");
                              handleConfigChange("geminiApiKeys", keys);
                            }} style={{ background: "none", border: "none", color: "var(--accent-indigo)", fontSize: "11px", cursor: "pointer", padding: 0, fontWeight: "bold" }}>+ Add Key</button>
                          </div>
                          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                            {(() => {
                              const keys = Array.isArray(config.geminiApiKeys) ? config.geminiApiKeys : (config.geminiApiKey ? [config.geminiApiKey] : [""]);
                              if (keys.length === 0) return <span style={{ fontSize: "10px", color: "var(--text-muted)", fontStyle: "italic" }}>No keys added. Click "+ Add Key" to add one.</span>;
                              return keys.map((key, index) => {
                                const isVisible = !!showGeminiKeys[index];
                                return (
                                  <div key={index} style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                                    <div style={{ position: "relative", flex: 1, display: "flex", alignItems: "center" }}>
                                      <input type={isVisible ? "text" : "password"} value={key || ""} onChange={e => {
                                        const updated = [...keys];
                                        updated[index] = e.target.value;
                                        handleConfigChange("geminiApiKeys", updated);
                                      }} style={{ flex: 1, padding: "6px 28px 6px 10px", borderRadius: "4px", border: "1px solid var(--border-subtle)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", fontFamily: "var(--font-mono)" }} placeholder="AIza..." />
                                      <button onClick={() => {
                                        setShowGeminiKeys(prev => ({ ...prev, [index]: !prev[index] }));
                                      }} style={{ position: "absolute", right: "8px", background: "none", border: "none", color: "var(--text-muted)", fontSize: "12px", cursor: "pointer", padding: 0, display: "flex", alignItems: "center" }}>
                                        {isVisible ? "👁️" : "🙈"}
                                      </button>
                                    </div>
                                    <button onClick={() => {
                                      const updated = keys.filter((_, i) => i !== index);
                                      handleConfigChange("geminiApiKeys", updated);
                                    }} style={{ background: "none", border: "none", color: "var(--accent-rose)", fontSize: "12px", cursor: "pointer", padding: "0 4px" }}>✕</button>
                                  </div>
                                );
                              });
                            })()}
                          </div>
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

                  <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "6px", padding: "16px", background: "var(--bg-card)", marginTop: "16px" }}>
                    <div className="sidebar-label" style={{ margin: "0 0 12px 0" }}>Pipeline Model Configuration</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
                      <div>
                        <label style={{ display: "block", fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginBottom: "4px" }}>EMBEDDING MODEL</label>
                        <select value={config.embedderProvider || "local"} onChange={e => handleConfigChange("embedderProvider", e.target.value)}
                          style={{ width: "100%", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border-subtle)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", fontFamily: "var(--font-mono)", cursor: "pointer" }}>
                          <option value="local">Local BGE-M3 (1024-dim)</option>
                          <option value="gemini">Gemini text-embedding-004 (768-dim, Zero-Padded)</option>
                        </select>
                        <span style={{ fontSize: "9px", color: "var(--accent-amber)", marginTop: "4px", display: "block" }}>
                          ⚠️ Changing embedder requires clearing and re-ingesting your files.
                        </span>
                      </div>
                      <div>
                        <label style={{ display: "block", fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginBottom: "4px" }}>RERANKER MODE</label>
                        <select value={config.rerankerProvider || "local"} onChange={e => handleConfigChange("rerankerProvider", e.target.value)}
                          style={{ width: "100%", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border-subtle)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", fontFamily: "var(--font-mono)", cursor: "pointer" }}>
                          <option value="local">Local BGE-Reranker (Enabled)</option>
                          <option value="none">Disabled (Bypassed — Saves ~1.5GB VRAM/RAM)</option>
                        </select>
                      </div>
                      <div>
                        <label style={{ display: "block", fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginBottom: "4px" }}>VISION / CAPTIONING MODEL</label>
                        <select value={config.visionModel || "gemini"} onChange={e => handleConfigChange("visionModel", e.target.value)}
                          style={{ width: "100%", padding: "6px 10px", borderRadius: "4px", border: "1px solid var(--border-subtle)", background: "var(--bg-card)", color: "var(--text-primary)", fontSize: "11px", fontFamily: "var(--font-mono)", cursor: "pointer" }}>
                          <option value="gemini">Gemini (Cloud) - Fast & Accurate</option>
                          <option value="local">LLaVA (Local via Ollama)</option>
                        </select>
                        <span style={{ fontSize: "9px", color: "var(--accent-amber)", marginTop: "4px", display: "block" }}>
                          Used during document ingestion.
                        </span>
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
  const themeColor = up ? "var(--accent-emerald)" : "var(--accent-rose)";
  const bgColor = up ? "rgba(28, 82, 55, 0.05)" : "rgba(138, 46, 36, 0.05)";
  const borderColor = up ? "var(--accent-emerald)" : "var(--accent-rose)";
  const borderStyle = up ? "dashed" : "solid";
  const statusText = up ? "ACTIVE" : "OFFLINE";

  return (
    <div style={{
      display: "inline-flex",
      alignItems: "center",
      gap: "6px",
      fontSize: "9px",
      padding: "4px 8px",
      borderRadius: "2px",
      background: bgColor,
      border: `1px ${borderStyle} ${borderColor}`,
      color: themeColor,
      fontFamily: "var(--font-mono)",
      fontWeight: 700,
      letterSpacing: "0.5px",
      textTransform: "uppercase",
      transition: "all 0.2s ease"
    }} title={`${label}: ${status.toUpperCase()}`}>
      <span style={{ opacity: 0.8 }}>{label}:</span>
      <span style={{ fontWeight: 800 }}>{statusText}</span>
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
            <span className={`expert-badge ${s.modality || s.expert_id}`}>{s.modality || s.expert_id}</span>
            {s.metadata?.page && <span>Page {s.metadata.page}</span>}
            {s.metadata?.similarity && (
              <span style={{ color: "var(--text-muted)" }}>sim: {(s.metadata.similarity * 100).toFixed(1)}%</span>
            )}
          </div>
          <div className="source-card-content" style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {(s.expert_id === "image" || s.modality === "image") && s.metadata?.source && (
              <div className="source-image-wrapper" style={{
                marginTop: "4px",
                borderRadius: "8px",
                overflow: "hidden",
                border: "1px solid var(--border)",
                background: "rgba(0, 0, 0, 0.2)",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                padding: "8px"
              }}>
                <img 
                  src={`${API_BASE}/api/uploads/${s.metadata.source}`} 
                  alt={s.metadata.source || "source image"} 
                  style={{
                    maxWidth: "100%",
                    maxHeight: "220px",
                    objectFit: "contain",
                    borderRadius: "6px",
                    boxShadow: "0 4px 10px rgba(0,0,0,0.3)"
                  }} 
                  onError={(e) => { e.target.parentElement.style.display = 'none'; }}
                />
              </div>
            )}
            <span style={{ whiteSpace: "pre-wrap" }}>{s.content}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg-primary)' }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "12px", color: "var(--text-muted)" }}>
          LOADING POLYRAG LEDGER CORE...
        </div>
      </div>
    );
  }

  return <MainApp session={session} setSession={setSession} />;
}