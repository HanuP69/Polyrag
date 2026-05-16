"""
gate.py — Gate inference.

Loads BGE-M3 + gate_model.pt once at startup.
Per query: embed → forward pass → sigmoid → threshold → return expert weights.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.config import EMBEDDING_MODEL, GATE_MODEL_PATH, GATE_THRESHOLD, EMBEDDING_DIM
from engine.gate.train import GateClassifier


class Gate:
    """
    MoE-style gate that routes queries to the right expert(s).
    
    Loads embedding model + classifier once.
    Returns per-expert weights for each query.
    """
    
    EXPERT_NAMES = ["text", "table", "image", "code"]
    
    def __init__(self):
        self.embed_model = None
        self.classifier = None
        self._loaded = False
    
    def load(self):
        """Load models. Called once at startup."""
        if self._loaded:
            return
        
        from sentence_transformers import SentenceTransformer
        
        print("[Gate] Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        
        print(f"[Gate] Loading classifier from {GATE_MODEL_PATH}")
        if not os.path.exists(GATE_MODEL_PATH):
            raise FileNotFoundError(
                f"Gate model not found at {GATE_MODEL_PATH}. "
                "Run generate_data.py then train.py first."
            )
        
        checkpoint = torch.load(GATE_MODEL_PATH, map_location="cpu", weights_only=True)
        input_dim = checkpoint.get("input_dim", EMBEDDING_DIM)
        num_classes = checkpoint.get("num_classes", 4)
        
        self.classifier = GateClassifier(input_dim=input_dim, num_classes=num_classes)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()
        
        self._loaded = True
        print("[Gate] [OK] Gate loaded and ready")
    
    def route(self, query: str, threshold: float = GATE_THRESHOLD) -> dict[str, float]:
        """
        Route a query to expert(s).
        
        Returns:
            dict like {"text": 0.82, "table": 0.61, "image": 0.09, "code": 0.01}
            Only experts above threshold are included.
        """
        if not self._loaded:
            self.load()
        
        # Embed query
        embedding = self.embed_model.encode(
            query, normalize_embeddings=True
        )
        
        # Forward pass
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            logits = self.classifier(x)
            probs = torch.sigmoid(logits).squeeze(0).numpy()
        
        # Build weights dict
        weights = {}
        for i, name in enumerate(self.EXPERT_NAMES):
            score = float(probs[i])
            weights[name] = score
        
        # Filter by threshold
        active = {k: v for k, v in weights.items() if v > threshold}
        
        # Fallback: if nothing above threshold, take the highest
        if not active:
            best_idx = np.argmax(probs)
            active = {self.EXPERT_NAMES[best_idx]: float(probs[best_idx])}
        
        return active
    
    def route_raw(self, query: str) -> dict[str, float]:
        """Return all expert weights without thresholding (for debugging)."""
        if not self._loaded:
            self.load()
        
        embedding = self.embed_model.encode(query, normalize_embeddings=True)
        
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            logits = self.classifier(x)
            probs = torch.sigmoid(logits).squeeze(0).numpy()
        
        return {name: float(probs[i]) for i, name in enumerate(self.EXPERT_NAMES)}


# Singleton instance — loaded once, reused
_gate_instance = None

def get_gate() -> Gate:
    """Get or create the global Gate instance."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = Gate()
        _gate_instance.load()
    return _gate_instance


if __name__ == "__main__":
    # Quick test
    gate = get_gate()
    
    test_queries = [
        "summarize the force majeure clause",
        "how many rows have revenue > 100k",
        "describe the architecture diagram on page 3",
        "compare Q1 and Q2 sales figures from the chart",
        "what does the contract say about termination",
        "show me the top 5 products by unit sales",
        "what information is in the pie chart on page 7",
        "explain the methodology described in section 3",
        "where is the handle_request function defined?",
    ]
    
    print("\n" + "=" * 70)
    print("GATE ROUTING TEST")
    print("=" * 70)
    
    for q in test_queries:
        raw = gate.route_raw(q)
        active = gate.route(q)
        raw_str = " | ".join(f"{k}: {v:.2f}" for k, v in raw.items())
        active_str = ", ".join(active.keys())
        print(f"\n  Q: \"{q}\"")
        print(f"  Raw:    {raw_str}")
        print(f"  Active: [{active_str}]")
