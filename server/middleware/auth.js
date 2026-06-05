const path = require('path');
const config = require('../services/config');

// LOCAL_DEV mode: skip Supabase auth and use a default local user.
// Set LOCAL_DEV=false and configure Supabase env vars for production.
const LOCAL_DEV = config.LOCAL_DEV;

let supabase = null;
if (!LOCAL_DEV) {
  const { createClient } = require('@supabase/supabase-js');
  const supabaseUrl = config.SUPABASE_URL;
  const supabaseKey = config.SUPABASE_KEY;
  if (supabaseUrl && supabaseKey) {
    supabase = createClient(supabaseUrl, supabaseKey);
  }
}

async function authMiddleware(req, res, next) {
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

  try {
    if (!supabase) {
      return res.status(500).json({ error: 'Supabase not configured' });
    }
    const { data: { user }, error } = await supabase.auth.getUser(token);
    if (error || !user) {
      return res.status(401).json({ error: 'Unauthorized: Invalid token' });
    }
    req.user = user;
    if (req.body) req.body.org_id = user.id;
    next();
  } catch (err) {
    console.error('[Auth] Token verification failed:', err.message);
    return res.status(401).json({ error: 'Unauthorized: Token verification failed' });
  }
}

module.exports = authMiddleware;
