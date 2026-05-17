import os
import re
import uuid
import numpy as np
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

    def parse(self, file_path: str, file_id: str, org_id: str, config: dict = None) -> List[Chunk]:
        """
        Extract code chunks from the file.
        Uses a line-based chunking strategy to preserve logic blocks.
        If config['useLlmCode'] is True, generates a propositional summary for each chunk.
        """
        config = config or {}
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
                rel_path = os.path.basename(file_path)
                enriched_content = f"File: {rel_path}\nLanguage: {language}\nLines: {i+1}-{end_idx}\n\n```\n{chunk_text}\n```"

                # Deep LLM Parsing
                if config.get("useLlmCode", False):
                    try:
                        from engine.main import _resolve_model, _generate_ollama
                        prompt = f"Summarize the purpose and functionality of the following {language} code snippet from '{rel_path}'. Keep it concise (2-3 sentences):\n\n{chunk_text}"
                        
                        code_model = config.get("models", {}).get("codeModel")
                        provider, api_name = _resolve_model(code_model)
                        summary = ""
                        
                        if provider == "groq" and config.get("groqApiKey"):
                            import requests
                            resp = requests.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers={"Authorization": f"Bearer {config.get('groqApiKey')}", "Content-Type": "application/json"},
                                json={"model": api_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 150},
                                timeout=10
                            )
                            if resp.status_code == 200:
                                summary = resp.json()["choices"][0]["message"]["content"]
                        elif provider == "gemini" and config.get("geminiApiKey"):
                            import requests
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_name}:generateContent?key={config.get('geminiApiKey')}"
                            resp = requests.post(
                                url,
                                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 150}},
                                timeout=10
                            )
                            if resp.status_code == 200:
                                summary = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            # Fallback to Ollama if no keys or specifically requested
                            summary = _generate_ollama(prompt, api_name)
                            
                        if summary:
                            enriched_content = f"{enriched_content}\n\nLLM Summary:\n{summary}"
                    except Exception as e:
                        print(f"[CodeExpert] LLM parsing failed for chunk: {e}")
                
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
        # Note: the main.py pipeline handles embedding directly using the shared model,
        # but if this is called individually, we route to the global embedding.
        from engine.main import _embed_model
        
        if _embed_model is None:
            return [[0.0] * 1024 for _ in chunks]
            
        texts = [c.content for c in chunks]
        embs = _embed_model.encode(texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
        return embs.tolist()

    def embed_query(self, query: str) -> 'np.ndarray':
        from engine.main import _embed_model
        if _embed_model is None:
            import numpy as np
            return np.zeros(1024)
        emb = _embed_model.encode([query], normalize_embeddings=True)[0]
        return emb
