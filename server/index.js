require("dotenv").config();

const cluster = require("cluster");
const os = require("os");
const config = require("./services/config");

// Resolve worker count from config
const NUM_WORKERS = config.NUM_WORKERS;

if (cluster.isPrimary && NUM_WORKERS > 1) {
  console.log("=".repeat(60));
  console.log("  PolyRAG Node.js Orchestration Server (Cluster)");
  console.log(`  Workers: ${NUM_WORKERS}`);
  console.log(`  Engine: ${config.ENGINE_URL}`);
  console.log("=".repeat(60));

  for (let i = 0; i < NUM_WORKERS; i++) {
    cluster.fork();
  }

  cluster.on("exit", (worker, code) => {
    console.log(`[Cluster] Worker ${worker.process.pid} died (code=${code}). Restarting...`);
    cluster.fork();
  });
} else {
  // If we only need 1 worker, run directly without clustering
  if (cluster.isPrimary) {
    console.log("=".repeat(60));
    console.log("  PolyRAG Node.js Orchestration Server (Single Process)");
    console.log(`  Engine: ${config.ENGINE_URL}`);
    console.log("=".repeat(60));
  }
  const express = require("express");
  const cors = require("cors");
  const morgan = require("morgan");
  const rateLimit = require("express-rate-limit");

  const queryRoutes = require("./routes/query");
  const ingestRoutes = require("./routes/ingest");
  const configRoutes = require("./routes/config");
  const feedbackRoutes = require("./routes/feedback");
  const chatRoutes = require("./routes/chat");
  const cache = require("./services/cache");

  const PORT = config.PORT;
  const app = express();
  const path = require("path");

  app.use(cors({ origin: "*", credentials: true }));
  app.use(express.json({ limit: "10mb" }));
  app.use("/api/uploads", (req, res, next) => {
    // Correct URL if the separator underscore is missing between UUID and page/img markers
    // e.g. /1a9229f2-c7c8-412e-8825-89ce749acf7bp1_img0.png -> /1a9229f2-c7c8-412e-8825-89ce749acf7b_p1_img0.png
    const filename = req.path.slice(1); // remove leading slash
    const uuidPatternWithoutUnderscore = /^([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})_?p(\d+)_?img(\d+)\.png$/i;
    const match = filename.match(uuidPatternWithoutUnderscore);
    if (match) {
      const correctedFilename = `${match[1]}_p${match[2]}_img${match[3]}.png`;
      if (correctedFilename !== filename) {
        req.url = "/" + correctedFilename;
      }
    }
    next();
  });
  app.use("/api/uploads", express.static(path.join(__dirname, "../data/uploads")));
  app.use(morgan("short"));

  const limiter = rateLimit({
    windowMs: 60 * 1000,
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Too many requests, slow down" },
  });
  app.use("/api/query", limiter);

  const authMiddleware = require("./middleware/auth");

  app.get("/api/health", (req, res) => {
    res.json({
      status: "ok",
      worker: process.pid,
      uptime: process.uptime(),
      cache: cache.stats(),
      engine_url: config.ENGINE_URL,
    });
  });

  // Public routes (no auth needed) — specific routers mounted below

  // Protected
  app.use("/api/query", authMiddleware);
  app.use("/api/ingest", authMiddleware);
  app.use("/api/files", authMiddleware);
  app.use("/api/feedback", authMiddleware);
  app.use("/api/config", authMiddleware);
  app.use("/api/chat", authMiddleware);
  
  app.use(queryRoutes);
  app.use(ingestRoutes);
  app.use(feedbackRoutes);
  app.use(configRoutes);
  app.use(chatRoutes);

  app.use((err, req, res, next) => {
    console.error(`[Server] Unhandled error on ${req.method} ${req.url}:`, err.stack || err.message);
    res.status(500).json({ error: "Internal server error" });
  });

  app.listen(PORT, () => {
    console.log(`  [Worker ${process.pid}] listening on port ${PORT}`);
  });
}
