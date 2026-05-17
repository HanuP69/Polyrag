"""
base.py — Abstract expert base class.

Every expert (text, table, image) must implement:
  - parse(file_path) → list[Chunk]
  - embed(chunks) → list[np.ndarray]

Retrieve is implemented once in the base class — same pgvector query
for every expert, just filtered by expert_id.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import uuid


@dataclass
class Chunk:
    """A single chunk of content extracted from a document."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str = ""
    file_id: str = ""
    expert_id: str = ""       # 'text' | 'table' | 'image'
    content: str = ""         # raw text or caption
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class BaseExpert(ABC):
    """
    Abstract base class for all PolyRAG experts.
    
    Subclasses must implement:
      - parse: extract chunks from a file
      - embed: generate embeddings for chunks
    
    Retrieve is shared — all experts use the same pgvector cosine search,
    just filtered by expert_id.
    """
    
    expert_id: str          # 'text' | 'table' | 'image'
    embed_dim: int = 1024   # BGE-M3 dimension
    
    @abstractmethod
    def parse(self, file_path: str, file_id: str = "", org_id: str = "", config: dict = None) -> list[Chunk]:
        """
        Extract chunks from a file.
        
        Args:
            file_path: Path to the file on disk
            file_id: UUID of the file record
            org_id: Organization ID for multi-tenancy
        
        Returns:
            List of Chunk objects with content and metadata filled.
        """
        ...
    
    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed chunks. Fills the `embedding` field on each Chunk.
        
        Args:
            chunks: List of Chunk objects with content filled
        
        Returns:
            Same chunks with embedding field populated (768-dim or 1024-dim vectors)
        """
        ...
    
    def retrieve(self, query_vec: np.ndarray, org_id: str, top_k: int = 10) -> list[Chunk]:
        """
        pgvector cosine search filtered by org_id + expert_id.
        Shared by all experts — implemented once here.
        
        This is a synchronous wrapper; the actual async version lives in db.py.
        """
        from engine.db import search_chunks
        return search_chunks(
            query_vec=query_vec,
            org_id=org_id,
            expert_id=self.expert_id,
            top_k=top_k
        )
