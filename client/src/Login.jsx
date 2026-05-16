import React, { useState, useEffect } from 'react';
import { supabase } from './supabaseClient';

export default function Login({ setSession }) {
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    return () => subscription.unsubscribe();
  }, [setSession]);

  const handleGoogleLogin = async () => {
    setLoading(true);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: window.location.origin
      }
    });
    
    if (error) {
      alert(error.message);
    }
    setLoading(false);
  };

  return (
    <div className="login-container" style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', 
      justifyContent: 'center', height: '100vh', backgroundColor: '#0f1115', color: '#fff'
    }}>
      <div style={{
        background: '#1a1d24', padding: '40px', borderRadius: '12px',
        boxShadow: '0 8px 24px rgba(0,0,0,0.5)', textAlign: 'center', maxWidth: '400px', width: '100%'
      }}>
        <h1 style={{ marginBottom: '10px', fontSize: '28px', background: 'linear-gradient(90deg, #60a5fa, #3b82f6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Welcome to PolyRAG
        </h1>
        <p style={{ color: '#94a3b8', marginBottom: '30px' }}>Sign in to access your personal knowledge base.</p>
        
        <button 
          onClick={handleGoogleLogin} 
          disabled={loading}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
            width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid #334155',
            background: '#1e293b', color: '#fff', fontSize: '16px', cursor: 'pointer',
            transition: 'background 0.2s', fontWeight: '500'
          }}
          onMouseOver={(e) => e.currentTarget.style.background = '#334155'}
          onMouseOut={(e) => e.currentTarget.style.background = '#1e293b'}
        >
          <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google" style={{ width: '20px', height: '20px' }} />
          {loading ? 'Signing in...' : 'Continue with Google'}
        </button>
      </div>
    </div>
  );
}
