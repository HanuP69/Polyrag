require("dotenv").config();

const cluster = require("cluster");
const os = require("os");

const NUM_WORKERS = Math.min(os.cpus().length, 4);

if (cluster.isPrimary) {
  console.log("=".repeat(60));
  console.log("  PolyRAG Node.js Orchestration Server (Cluster)");
  console.log(`  Workers: ${NUM_WORKERS}`);
  console.log(`  Engine: ${process.env.ENGINE_URL || "http://localhost:8000"}`);
  console.log("=".repeat(60));

  for (let i = 0; i < NUM_WORKERS; i++) {
    cluster.fork();
  }

  cluster.on("exit", (worker, code) => {
    console.log(`[Cluster] Worker ${worker.process.pid} died (code=${code}). Restarting...`);
    cluster.fork();
  });
} else {
  const express = require("express");
  const cors = require("cors");
  const morgan = require("morgan");
  const rateLimit = require("express-rate-limit");

  const queryRoutes = require("./routes/query");
  const ingestRoutes = require("./routes/ingest");
  const configRoutes = require("./routes/config");
  const feedbackRoutes = require("./routes/feedback");
  const cache = require("./services/cache");

  const PORT = process.env.PORT || 3001;
  const app = express();

  app.use(cors({ origin: "*", credentials: true }));
  app.use(express.json({ limit: "10mb" }));
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
      engine_url: process.env.ENGINE_URL || "http://localhost:8000",
    });
  });

  const authMiddleware = require("./middleware/auth");

  // Public
  app.use(configRoutes);
  
  // Protected
  app.use("/api/query", authMiddleware);
  app.use("/api/ingest", authMiddleware);
  app.use("/api/feedback", authMiddleware);
  
  app.use(queryRoutes);
  app.use(ingestRoutes);
  app.use(feedbackRoutes);

  app.use((err, req, res, next) => {
    console.error("[Server] Unhandled error:", err.message);
    res.status(500).json({ error: "Internal server error" });
  });

  app.listen(PORT, () => {
    console.log(`  [Worker ${process.pid}] listening on port ${PORT}`);
  });
}
