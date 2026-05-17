const express = require("express");
const router = express.Router();
const engine = require("../services/engine");

router.get("/api/config", async (req, res) => {
  const orgId = req.user?.id || "default";
  try {
    const config = await engine.getOrgConfig(orgId);
    res.json(config);
  } catch (err) {
    if (err.response && err.response.status === 404) {
      return res.status(404).json({ error: "Org not found" });
    }
    res.status(500).json({ error: err.message });
  }
});

router.put("/api/config", async (req, res) => {
  const orgId = req.user?.id || "default";
  const { name, config } = req.body;
  try {
    const result = await engine.updateOrgConfig(orgId, name, config);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get("/api/models", async (req, res) => {
  try {
    const result = await engine.getModels();
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
