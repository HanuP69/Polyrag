const path = require('path');
const config = require('../services/config');

// LOCAL_DEV mode: skip Supabase auth and use a default local user.
// Set LOCAL_DEV=false and configure Supabase env vars for production.
const LOCAL_DEV = config.LOCAL_DEV;

let supabase = null;
if (!LOCAL_DEV) {
  try {
    const { createClient } = require('@supabase/supabase-js');
    const supabaseUrl = config.SUPABASE_URL;
    const supabaseKey = config.SUPABASE_KEY;
    if (supabaseUrl && supabaseKey) {
      supabase = createClient(supabaseUrl, supabaseKey);
      console.log('[Auth] Supabase client initialized');
    } else {
      console.warn('[Auth] Supabase URL or key missing — auth will reject non-dev tokens');
    }
  } catch (err) {
    console.error('[Auth] Failed to initialize Supabase client:', err.message);
  }
}

// Track which orgs have been ensured this session to avoid repeated calls
const _ensuredOrgs = new Set();

function authMiddleware(req, res, next) {
  // --- Local dev bypass ---
  if (LOCAL_DEV) {
    req.user = { id: 'default', email: 'local@polyrag' };
    if (req.body) req.body.org_id = 'default';
    return next();
  }

  // --- Production: verify Supabase JWT ---
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid Authorization header' });
  }

  const token = authHeader.split(' ')[1];

  // Allow local dev token passthrough
  if (token === 'local-dev-token') {
    req.user = { id: 'default', email: 'local@polyrag' };
    if (req.body) req.body.org_id = 'default';
    return next();
  }

  if (!supabase) {
    return res.status(500).json({ error: 'Supabase not configured' });
  }

  // Verify token and ensure org exists
  supabase.auth.getUser(token)
    .then(({ data, error }) => {
      if (error || !data?.user) {
        return res.status(401).json({ error: 'Unauthorized: Invalid token' });
      }
      const user = data.user;
      req.user = user;
      if (req.body) req.body.org_id = user.id;

      // Ensure org exists in engine (fire-and-forget, cached per session)
      if (!_ensuredOrgs.has(user.id)) {
        const engine = require('../services/engine');
        engine.getOrgConfig(user.id).then(() => {
          _ensuredOrgs.add(user.id);
        }).catch(() => {
          // Org doesn't exist yet — that's OK, endpoints will create as needed
          _ensuredOrgs.add(user.id);
        });
      }

      next();
    })
    .catch((err) => {
      console.error('[Auth] Token verification failed:', err.message);
      return res.status(401).json({ error: 'Unauthorized: Token verification failed' });
    });
}

module.exports = authMiddleware;
