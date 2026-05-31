"""
Embedder (BGE-M3) and Reranker (BGE-reranker-v2-m3).
Direct port from v4 notebook with VRAM management methods.
"""
import numpy as np
import torch
import gc
from typing import List
from engine_v4.config import CFG


class Embedder:
    """BGE-M3 embedder — lazy-loaded, 1024-dim, normalized."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedder] Loading BGE-M3 on {CFG.embed_device}...")
            self._model = SentenceTransformer(CFG.embedder_model, device=CFG.embed_device)
            self._model.max_seq_length = 512  # VRAM safety: limits 8192→512
            print(f"[Embedder] Loaded. max_seq_length={self._model.max_seq_length}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        self._load()
        vecs = self._model.encode(
            texts,
            batch_size=min(CFG.embed_batch, 8),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def to_cpu(self):
        """Move model to CPU to free VRAM."""
        if self._model is not None:
            self._model = self._model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            print("[Embedder] Moved to CPU")

    def to_gpu(self):
        """Move model back to CUDA if configured."""
        if CFG.embed_device == "cpu":
            print("[Embedder] Staying on CPU as configured.")
            self._load()
            return
        if self._model is not None:
            self._model = self._model.to("cuda")
            print("[Embedder] Moved to CUDA")
        else:
            self._load()

    def unload(self):
        """Fully unload model from memory."""
        self._model = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[Embedder] Unloaded")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:
        return 1024


class Reranker:
    """BGE-reranker-v2-m3 — lazy-loaded, batched cross-encoder."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] Loading on {CFG.rerank_device}...")
            self._model = CrossEncoder(
                CFG.reranker_model,
                device=CFG.rerank_device,
                max_length=1024,
            )
            print("[Reranker] Loaded.")

    def rerank(self, query: str, texts: List[str], top_n: int) -> List[int]:
        """Returns indices sorted by descending score, limited to top_n."""
        if not texts:
            return []
        self._load()
        top_n = min(top_n, len(texts))
        pairs = [(query, t) for t in texts]
        all_scores = []
        for i in range(0, len(pairs), 16):
            scores = self._model.predict(pairs[i : i + 16], show_progress_bar=False)
            all_scores.extend(
                scores.tolist() if hasattr(scores, "tolist") else list(scores)
            )
        return np.argsort(all_scores)[::-1].tolist()[:top_n]

    def to_cpu(self):
        if self._model is not None:
            self._model.model = self._model.model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            print("[Reranker] Moved to CPU")

    def to_gpu(self):
        """Move model back to CUDA if configured."""
        if CFG.rerank_device == "cpu":
            print("[Reranker] Staying on CPU as configured.")
            self._load()
            return
        if self._model is not None:
            self._model.model = self._model.model.to("cuda")
            print("[Reranker] Moved to CUDA")
        else:
            self._load()

    def unload(self):
        self._model = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[Reranker] Unloaded")

    @property
    def loaded(self) -> bool:
        return self._model is not None


# ── Singletons ───────────────────────────────────────────────────────────────
embedder = Embedder()
reranker = Reranker()
