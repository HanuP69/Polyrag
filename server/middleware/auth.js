const { createClient } = require('@supabase/supabase-js');
const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '../../client/.env') });

const supabaseUrl = process.env.VITE_SUPABASE_URL || 'https://mock-supabase-url.supabase.co';
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY || 'mock-anon-key';

const supabase = createClient(supabaseUrl, supabaseKey);

async function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid Authorization header' });
  }

  const token = authHeader.split(' ')[1];
  
  try {
    const { data: { user }, error } = await supabase.auth.getUser(token);
    
    if (error || !user) {
      return res.status(401).json({ error: 'Unauthorized: Invalid token' });
    }

    // Inject user info into the request
    req.user = user;
    
    // Enforce org_id to be the user's ID
    if (req.body) {
      req.body.org_id = user.id;
    }
    
    next();
  } catch (err) {
    console.error('[Auth] Token verification failed:', err.message);
    return res.status(401).json({ error: 'Unauthorized: Token verification failed' });
  }
}

module.exports = authMiddleware;
