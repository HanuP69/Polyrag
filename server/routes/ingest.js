const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const engine = require("../services/engine");

const UPLOAD_DIR = path.join(__dirname, "..", "..", "data", "uploads");

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = [".pdf", ".csv", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${ext}`));
    }
  },
});

router.post("/api/ingest", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const orgId = req.user?.id || "default";
  let models = {};
  if (req.body.models) {
    try { models = JSON.parse(req.body.models); } catch (e) {}
  }

  try {
    const result = await engine.ingestFile(req.file.path, orgId, models);
    res.json({
      status: result.status,
      file_id: result.file_id,
      filename: req.file.originalname,
    });
  } catch (err) {
    console.error("[Ingest] Failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

router.get("/api/ingest/:fileId", async (req, res) => {
  try {
    const status = await engine.getIngestStatus(req.params.fileId);
    res.json(status);
  } catch (err) {
    if (err.response && err.response.status === 404) {
      return res.status(404).json({ error: "File ID not found" });
    }
    res.status(500).json({ error: err.message });
  }
});

router.post("/api/ingest/github", async (req, res) => {
  const repoUrl = req.body.repo_url;
  const orgId = req.user?.id || "default";
  const models = req.body.models || {};

  if (!repoUrl) {
    return res.status(400).json({ error: "No repo_url provided" });
  }

  try {
    const result = await engine.ingestGithub(repoUrl, orgId, models);
    res.json({
      status: result.status,
      file_id: result.file_id,
      repo: result.repo,
    });
  } catch (err) {
    console.error("[Ingest] GitHub ingest failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

router.get("/api/files", async (req, res) => {
  const orgId = req.user?.id || "default";
  try {
    const files = await engine.getOrgFiles(orgId);
    res.json(files);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Support both old (/api/file/:fileId) and new (/api/files/:fileId) client paths
async function handleDeleteFile(req, res) {
  const orgId = req.user?.id || "default";
  try {
    const data = await engine.deleteOrgFile(orgId, req.params.fileId);
    res.json(data);
  } catch (err) {
    if (err.response && err.response.status === 404) {
      return res.status(404).json({ error: "File not found" });
    }
    res.status(500).json({ error: err.message });
  }
}
router.delete("/api/file/:fileId", handleDeleteFile);
router.delete("/api/files/:fileId", handleDeleteFile);

// Proxy pipeline health and models to Python engine
router.get("/api/health/pipeline", async (req, res) => {
  try {
    const data = await engine.getPipelineHealth(req.query.org_id || null);
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

router.get("/api/models", async (req, res) => {
  try {
    const data = await engine.getModels();
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

// DB / Docker health check
router.get("/api/health/db", async (req, res) => {
  const { exec } = require("child_process");
  const ENGINE_URL = process.env.ENGINE_URL || "http://localhost:8000";

  // Check Python engine
  let engineUp = false;
  try {
    const axios = require("axios");
    const resp = await axios.get(`${ENGINE_URL}/health`, { timeout: 3000 });
    engineUp = resp.status === 200;
  } catch {}

  // Check Docker postgres container
  const checkDocker = () => new Promise((resolve) => {
    exec("docker ps --filter name=postgres --filter name=polyrag --filter status=running --format \"{{.Names}}\"", (err, stdout) => {
      if (err) return resolve({ running: false, error: err.message });
      const containers = stdout.trim().split("\n").filter(Boolean);
      resolve({ running: containers.length > 0, containers });
    });
  });

  const dockerResult = await checkDocker();

  res.json({
    engine: engineUp ? "up" : "down",
    postgres_docker: dockerResult.running ? "up" : "down",
    postgres_containers: dockerResult.containers || [],
    docker_error: dockerResult.error || null,
  });
});

module.exports = router;
