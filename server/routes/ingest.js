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

  const orgId = req.body.org_id || "default";

  try {
    const result = await engine.ingestFile(req.file.path, orgId);
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

module.exports = router;
