import os
import re
import uuid
from typing import List, Dict

from engine.experts.base import BaseExpert, Chunk

CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css"
}

class CodeExpert(BaseExpert):
    """
    Expert for parsing and embedding raw code files.
    """
    
    @property
    def expert_id(self) -> str:
        return "code"

    def parse(self, file_path: str, file_id: str, org_id: str) -> List[Chunk]:
        """
        Extract code chunks from the file.
        Uses a line-based chunking strategy to preserve logic blocks.
        """
        ext = os.path.splitext(file_path)[1].lower()
        language = CODE_EXTENSIONS.get(ext, "unknown")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Skip binary or improperly encoded files
            return []

        lines = content.split('\n')
        chunks = []
        
        # Simple chunking strategy: block of max ~500 lines, with 50 lines overlap
        # Ideal for capturing full functions while keeping within token limits.
        MAX_LINES = 100
        OVERLAP = 20
        
        i = 0
        while i < len(lines):
            end_idx = min(i + MAX_LINES, len(lines))
            chunk_lines = lines[i:end_idx]
            
            # Reconstruct text
            chunk_text = "\n".join(chunk_lines).strip()
            
            if chunk_text:
                # Add file context to the content itself to help the embedding model
                # understand where this code comes from.
                # Use just the basename or relative path if possible, but we might only have abs path.
                rel_path = os.path.basename(file_path)
                
                enriched_content = f"File: {rel_path}\nLanguage: {language}\nLines: {i+1}-{end_idx}\n\n```\n{chunk_text}\n```"
                
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        file_id=file_id,
                        org_id=org_id,
                        expert_id=self.expert_id,
                        content=enriched_content,
                        metadata={
                            "file_path": rel_path,
                            "language": language,
                            "start_line": i + 1,
                            "end_line": end_idx
                        }
                    )
                )
            
            if end_idx == len(lines):
                break
                
            i += (MAX_LINES - OVERLAP)

        return chunks

    def embed(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Use the shared BGE-M3 model for code embedding.
        It handles code syntax well enough without needing a specialized code model.
        """
        from engine.main import _cached_embed_query
        
        # Note: the main.py pipeline handles embedding directly using the shared model,
        # but if this is called individually, we route to the global embedding.
        from engine.config import _embed_model
        
        texts = [c.content for c in chunks]
        embs = _embed_model.encode(texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
        return embs.tolist()
