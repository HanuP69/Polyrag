const express = require("express");
const router = express.Router();
const engine = require("../services/engine");

router.post("/api/feedback", async (req, res) => {
  const { query_log_id, rating } = req.body;
  const orgId = req.user?.id || "default";

  if (!query_log_id || rating === undefined) {
    return res.status(400).json({ error: "query_log_id and rating are required" });
  }

  try {
    const result = await engine.submitFeedback(query_log_id, rating, orgId);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get("/api/health/pipeline", async (req, res) => {
  try {
    const health = await engine.getPipelineHealth(req.query.org_id);
    res.json(health);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
