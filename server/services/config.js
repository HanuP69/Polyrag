const fs = require("fs");
const path = require("path");

let config = {};
try {
  const configPath = path.resolve(__dirname, "../../polyrag.config.json");
  if (fs.existsSync(configPath)) {
    config = JSON.parse(fs.readFileSync(configPath, "utf8"));
  }
} catch (err) {
  console.error("[Config] Error reading polyrag.config.json:", err.message);
}

const getSetting = (pathStr, defaultValue) => {
  const parts = pathStr.split(".");
  let current = config;
  for (const part of parts) {
    if (current && typeof current === "object" && part in current) {
      current = current[part];
    } else {
      return defaultValue;
    }
  }
  return current !== undefined ? current : defaultValue;
};

// Orchestrator Port
const PORT = process.env.PORT || getSetting("server.node_port", 3001);

// Python Engine Config
let engineHost = getSetting("server.engine_host", "localhost");
if (engineHost === "0.0.0.0") {
  engineHost = "127.0.0.1";
}
const enginePort = getSetting("server.engine_port", 8000);
const ENGINE_URL = process.env.ENGINE_URL || `http://${engineHost}:${enginePort}`;

// Authentication Config
const LOCAL_DEV = process.env.LOCAL_DEV !== undefined
  ? process.env.LOCAL_DEV !== "false"
  : getSetting("auth.local_dev", true);

const SUPABASE_URL = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL || getSetting("auth.supabase_url", "");
const SUPABASE_KEY = process.env.VITE_SUPABASE_ANON_KEY || process.env.SUPABASE_KEY || getSetting("auth.supabase_anon_key", "");

const NUM_WORKERS = process.env.NUM_WORKERS !== undefined
  ? parseInt(process.env.NUM_WORKERS)
  : getSetting("server.node_workers", 1);

module.exports = {
  PORT,
  ENGINE_URL,
  LOCAL_DEV,
  SUPABASE_URL,
  SUPABASE_KEY,
  NUM_WORKERS,
  rawConfig: config
};
