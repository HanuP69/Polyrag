const express = require("express");
const router = express.Router();
const engine = require("../services/engine");

// Get all sessions for the user's org
router.get("/api/chat/sessions", async (req, res) => {
  const orgId = req.user?.id || "default";
  try {
    const sessions = await engine.getChatSessions(orgId);
    res.json(sessions);
  } catch (err) {
    console.error("[Chat Routes] Get sessions failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Create a new chat session
router.post("/api/chat/sessions", async (req, res) => {
  const orgId = req.user?.id || "default";
  const { session_id, title } = req.body;
  if (!session_id || !title) {
    return res.status(400).json({ error: "session_id and title are required" });
  }
  try {
    const result = await engine.createChatSession(session_id, orgId, title);
    res.json(result);
  } catch (err) {
    console.error("[Chat Routes] Create session failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Delete a chat session
router.delete("/api/chat/sessions/:id", async (req, res) => {
  const orgId = req.user?.id || "default";
  const sessionId = req.params.id;
  try {
    // Ensure session belongs to this org before deleting
    const owner = await engine.getSessionOwner(orgId, sessionId);
    if (!owner || owner !== orgId) {
      return res.status(404).json({ error: "Session not found" });
    }
    const result = await engine.deleteChatSession(orgId, sessionId);
    res.json(result);
  } catch (err) {
    console.error("[Chat Routes] Delete session failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Get all messages in a session
router.get("/api/chat/sessions/:id/messages", async (req, res) => {
  const orgId = req.user?.id || "default";
  const sessionId = req.params.id;
  try {
    // Verify session ownership
    const owner = await engine.getSessionOwner(orgId, sessionId);
    if (!owner || owner !== orgId) {
      return res.status(404).json({ error: "Session not found" });
    }
    const messages = await engine.getChatMessages(sessionId, orgId);
    res.json(messages);
  } catch (err) {
    console.error("[Chat Routes] Get messages failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Add a message to a session
router.post("/api/chat/sessions/:id/messages", async (req, res) => {
  const orgId = req.user?.id || "default";
  const sessionId = req.params.id;
  const { message_id, role, content, sources = [] } = req.body;
  if (!message_id || !role || !content) {
    return res.status(400).json({ error: "message_id, role, and content are required" });
  }
  try {
    // Verify session ownership before adding a message
    const owner = await engine.getSessionOwner(orgId, sessionId);
    if (!owner || owner !== orgId) {
      return res.status(404).json({ error: "Session not found" });
    }
    const result = await engine.addChatMessage(sessionId, message_id, role, content, sources, orgId);
    res.json(result);
  } catch (err) {
    console.error("[Chat Routes] Add message failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Logout: clear all chat sessions for the current org
router.post("/api/chat/logout", async (req, res) => {
  const orgId = req.user?.id || "default";
  try {
    const deleted = await engine.deleteAllChatSessions(orgId);
    res.json({ status: "ok", deleted_sessions: deleted });
  } catch (err) {
    console.error("[Chat Routes] Logout failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
