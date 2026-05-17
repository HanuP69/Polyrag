"""
text.py — Text Expert.

Parse: PyMuPDF extracts text blocks from PDF. Chunks at 512 tokens, 64 overlap.
Embed: BGE-M3 via sentence-transformers. Batch embed for speed.
"""

import os
import re
from typing import Optional
import numpy as np

from engine.experts.base import BaseExpert, Chunk
from engine.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


class TextExpert(BaseExpert):
    """Expert for text/prose retrieval from documents."""
    
    expert_id = "text"
    
    def __init__(self):
        self._embed_model = None
    
    @property
    def embed_model(self):
        """Lazy-load embedding model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[TextExpert] Loading embedding model: {EMBEDDING_MODEL}")
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embed_model
    
    def parse(self, file_path: str, file_id: str = "", org_id: str = "", config: dict = None) -> list[Chunk]:
        """
        Extract text chunks from a PDF file.
        
        Uses PyMuPDF for text extraction, chunks at sentence boundaries.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            return self._parse_pdf(file_path, file_id, org_id)
        elif ext == ".txt":
            return self._parse_txt(file_path, file_id, org_id)
        elif ext in (".md", ".markdown"):
            return self._parse_txt(file_path, file_id, org_id)
        else:
            print(f"[TextExpert] Unsupported file type: {ext}")
            return []
    
    def _parse_pdf(self, file_path: str, file_id: str, org_id: str) -> list[Chunk]:
        """Extract text from PDF using PyMuPDF."""
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        chunks = []
        total_pages = len(doc)
        print(f"[TextExpert] Parsing {total_pages} pages from {os.path.basename(file_path)}...")
        
        for page_num in range(total_pages):
            if page_num > 0 and page_num % 100 == 0:
                print(f"[TextExpert] Progress: {page_num}/{total_pages} pages parsed ({len(chunks)} chunks so far)")
            
            page = doc[page_num]
            
            # Get text blocks with position info
            blocks = page.get_text("blocks")
            
            # Concatenate text blocks for this page
            page_text = ""
            for block in blocks:
                if block[6] == 0:  # text block (not image)
                    page_text += block[4] + "\n"
            
            if not page_text.strip():
                continue
            
            # Detect section headings (heuristic: short lines, possibly uppercase or larger font)
            section = self._detect_section(page, blocks)
            
            # Chunk the page text
            page_chunks = self._chunk_text(
                text=page_text.strip(),
                page_num=page_num + 1,
                section=section,
                file_id=file_id,
                org_id=org_id
            )
            chunks.extend(page_chunks)
        
        doc.close()
        print(f"[TextExpert] Extracted {len(chunks)} chunks from {os.path.basename(file_path)}")
        return chunks
    
    def _parse_txt(self, file_path: str, file_id: str, org_id: str) -> list[Chunk]:
        """Extract text from a plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        chunks = self._chunk_text(
            text=text.strip(),
            page_num=1,
            section="",
            file_id=file_id,
            org_id=org_id
        )
        print(f"[TextExpert] Extracted {len(chunks)} chunks from {os.path.basename(file_path)}")
        return chunks
    
    def _detect_section(self, page, blocks) -> str:
        """
        Heuristic section heading detection.
        Look for short text blocks that might be headings.
        """
        for block in blocks:
            if block[6] != 0:
                continue
            text = block[4].strip()
            # Short lines that look like headers
            if len(text) < 100 and text and not text.endswith("."):
                # Check if it looks like a heading (uppercase or numbered)
                if (text.isupper() or 
                    re.match(r'^\d+[\.\)]\s', text) or
                    re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$', text)):
                    return text[:100]
        return ""
    
    def _chunk_text(
        self,
        text: str,
        page_num: int,
        section: str,
        file_id: str,
        org_id: str
    ) -> list[Chunk]:
        """
        Chunk text at ~CHUNK_SIZE tokens with CHUNK_OVERLAP overlap.
        Respects sentence boundaries.
        """
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_tokens = []
        current_text_parts = []
        char_offset = 0
        
        for sentence in sentences:
            # Rough token count (words ≈ tokens * 0.75)
            sentence_tokens = sentence.split()
            
            if len(current_tokens) + len(sentence_tokens) > CHUNK_SIZE and current_text_parts:
                # Create chunk
                chunk_text = " ".join(current_text_parts)
                chunks.append(Chunk(
                    org_id=org_id,
                    file_id=file_id,
                    expert_id=self.expert_id,
                    content=chunk_text,
                    metadata={
                        "page": page_num,
                        "section": section,
                        "char_offset": char_offset
                    }
                ))
                
                # Overlap: keep last CHUNK_OVERLAP tokens worth of sentences
                overlap_tokens = 0
                overlap_start = len(current_text_parts)
                for i in range(len(current_text_parts) - 1, -1, -1):
                    overlap_tokens += len(current_text_parts[i].split())
                    if overlap_tokens >= CHUNK_OVERLAP:
                        overlap_start = i
                        break
                
                current_text_parts = current_text_parts[overlap_start:]
                current_tokens = []
                for part in current_text_parts:
                    current_tokens.extend(part.split())
                char_offset += len(chunk_text) - len(" ".join(current_text_parts))
            
            current_text_parts.append(sentence)
            current_tokens.extend(sentence_tokens)
        
        # Last chunk
        if current_text_parts:
            chunk_text = " ".join(current_text_parts)
            if chunk_text.strip():
                chunks.append(Chunk(
                    org_id=org_id,
                    file_id=file_id,
                    expert_id=self.expert_id,
                    content=chunk_text,
                    metadata={
                        "page": page_num,
                        "section": section,
                        "char_offset": char_offset
                    }
                ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, respecting abbreviations."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed chunks using BGE-M3.
        Batch embed for speed. Returns chunks with embedding field filled.
        """
        if not chunks:
            return chunks
        
        texts = [c.content for c in chunks]
        
        print(f"[TextExpert] Embedding {len(texts)} chunks...")
        embeddings = self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True
        )
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        
        print(f"[TextExpert] [OK] Embedded {len(chunks)} chunks (dim={embeddings.shape[1]})")
        return chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_model.encode(
            query,
            normalize_embeddings=True
        )
