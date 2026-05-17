import os
import sys
import threading
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.config import EMBEDDING_MODEL, GATE_MODEL_PATH, GATE_THRESHOLD, EMBEDDING_DIM
from engine.gate.train import GateClassifier


class Gate:
    EXPERT_NAMES = ["text", "table", "image", "code"]

    def __init__(self):
        self.embed_model = None
        self.classifier = None
        self._loaded = False

    def load(self):
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
        if not self._loaded:
            self.load()

        embedding = self.embed_model.encode(query, normalize_embeddings=True)

        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            logits = self.classifier(x)
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        weights = {name: float(probs[i]) for i, name in enumerate(self.EXPERT_NAMES)}
        active = {k: v for k, v in weights.items() if v > threshold}

        if not active:
            best_idx = np.argmax(probs)
            active = {self.EXPERT_NAMES[best_idx]: float(probs[best_idx])}

        return active

    def route_raw(self, query: str) -> dict[str, float]:
        if not self._loaded:
            self.load()

        embedding = self.embed_model.encode(query, normalize_embeddings=True)

        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            logits = self.classifier(x)
            probs = torch.sigmoid(logits).squeeze(0).numpy()

        return {name: float(probs[i]) for i, name in enumerate(self.EXPERT_NAMES)}


_gate_instance = None
_gate_lock = threading.Lock()


def get_gate() -> Gate:
    global _gate_instance
    with _gate_lock:
        if _gate_instance is None:
            _gate_instance = Gate()
            _gate_instance.load()
    return _gate_instance


if __name__ == "__main__":
    gate = get_gate()

    test_queries = [
        "summarize the force majeure clause",
        "how many rows have revenue > 100k",
        "describe the architecture diagram on page 3",
        "compare Q1 and Q2 sales figures from the chart",
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