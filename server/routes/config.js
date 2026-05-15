const express = require("express");
const router = express.Router();
const engine = require("../services/engine");

router.get("/api/config/:orgId", async (req, res) => {
  try {
    const config = await engine.getOrgConfig(req.params.orgId);
    res.json(config);
  } catch (err) {
    if (err.response && err.response.status === 404) {
      return res.status(404).json({ error: "Org not found" });
    }
    res.status(500).json({ error: err.message });
  }
});

router.put("/api/config/:orgId", async (req, res) => {
  const { name, config } = req.body;
  try {
    const result = await engine.updateOrgConfig(req.params.orgId, name, config);
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
