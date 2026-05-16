"""
train.py — Train the gate classifier.

Pipeline:
1. Load training data (synthetic JSON or Postgres query_logs)
2. Embed all queries with BGE-M3
3. Train a 3-layer classifier: Linear(dim→128) → ReLU → Dropout → Linear(128→3) → Sigmoid
4. Multi-label BCEWithLogitsLoss (a query can route to multiple experts)
5. Save best model weights to gate_model.pt
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from engine.config import (
    EMBEDDING_MODEL, EMBEDDING_DIM,
    GATE_MODEL_PATH, GATE_TRAINING_DATA_PATH, DATA_DIR
)


# ──────────────────────────── Gate Classifier Model ──────────────────
class GateClassifier(nn.Module):
    """
    4-class multi-label classifier.
    Input: 768-dim or 1024-dim BGE-M3 embedding
    Output: 4-dim logits (text, table, image, code)
    """
    def __init__(self, input_dim: int = EMBEDDING_DIM, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


# ──────────────────────────── Data Loading ───────────────────────────
def load_synthetic_data(path: str) -> tuple[list[str], np.ndarray]:
    """
    Load synthetic training data from JSON.
    Returns (queries, labels) where labels is a multi-hot array [N, 4].
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    queries = []
    labels = []
    
    expert_to_idx = {"text": 0, "table": 1, "image": 2, "code": 3}
    
    for expert_id, expert_queries in data.items():
        if expert_id not in expert_to_idx:
            continue
        idx = expert_to_idx[expert_id]
        for q in expert_queries:
            queries.append(q)
            label = [0.0, 0.0, 0.0, 0.0]
            label[idx] = 1.0
            labels.append(label)
    
    return queries, np.array(labels, dtype=np.float32)


# ──────────────────────────── Embedding ──────────────────────────────
def embed_queries(queries: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed queries using BGE-M3 via sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    
    print(f"[Gate] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"[Gate] Embedding {len(queries)} queries (batch_size={batch_size})...")
    embeddings = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    print(f"[Gate] Embedding shape: {embeddings.shape}")
    return embeddings


# ──────────────────────────── Training ───────────────────────────────
def train_gate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32
):
    """Train the gate classifier."""
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels.argmax(axis=1)
    )
    
    print(f"[Gate] Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    input_dim = embeddings.shape[1]
    model = GateClassifier(input_dim=input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                
                # Accuracy: check if argmax matches
                preds = torch.sigmoid(logits)
                pred_labels = (preds > 0.4).float()
                correct += (pred_labels == y_batch).all(dim=1).sum().item()
                total += len(y_batch)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Gate] Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {accuracy:.2%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
    
    # Save best model
    os.makedirs(os.path.dirname(GATE_MODEL_PATH), exist_ok=True)
    
    # Save model with input_dim metadata
    save_dict = {
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "num_classes": 4,
        "best_val_loss": best_val_loss,
    }
    torch.save(save_dict, GATE_MODEL_PATH)
    print(f"[Gate] Best model saved to {GATE_MODEL_PATH} (val_loss={best_val_loss:.4f})")
    
    return model


# ──────────────────────────── Main ───────────────────────────────────
def main():
    if not os.path.exists(GATE_TRAINING_DATA_PATH):
        print(f"[Gate] ERROR: No training data at {GATE_TRAINING_DATA_PATH}")
        print("[Gate] Run generate_data.py first.")
        sys.exit(1)
    
    # Load data
    print(f"[Gate] Loading training data from {GATE_TRAINING_DATA_PATH}")
    queries, labels = load_synthetic_data(GATE_TRAINING_DATA_PATH)
    print(f"[Gate] Loaded {len(queries)} queries, {labels.shape[1]} classes")
    
    # Embed
    embeddings = embed_queries(queries)
    
    # Train
    train_gate(embeddings, labels)
    
    print("[Gate] Training complete!")


if __name__ == "__main__":
    main()
