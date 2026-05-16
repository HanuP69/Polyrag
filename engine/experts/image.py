import os
import base64
import requests
import numpy as np

from engine.experts.base import BaseExpert, Chunk
from engine.config import (
    EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
)


class ImageExpert(BaseExpert):

    expert_id = "image"

    def __init__(self):
        self._embed_model = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[ImageExpert] Loading embedding model: {EMBEDDING_MODEL}")
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embed_model

    def parse(self, file_path: str, file_id: str = "", org_id: str = "") -> list[Chunk]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
            return self._parse_standalone_image(file_path, file_id, org_id)
        elif ext == ".pdf":
            return self._parse_pdf_images(file_path, file_id, org_id)
        return []

    def _parse_standalone_image(self, file_path: str, file_id: str, org_id: str) -> list[Chunk]:
        b64 = self._encode_image(file_path)
        caption = self._caption_image(b64)
        if not caption:
            return []

        chunk = Chunk(
            org_id=org_id,
            file_id=file_id,
            expert_id=self.expert_id,
            content=caption,
            metadata={
                "source": os.path.basename(file_path),
                "type": "standalone_image",
            },
        )
        print(f"[ImageExpert] Captioned standalone image: {os.path.basename(file_path)}")
        return [chunk]

    def _parse_pdf_images(self, file_path: str, file_id: str, org_id: str) -> list[Chunk]:
        import fitz
        from concurrent.futures import ThreadPoolExecutor, as_completed

        MAX_IMAGES = 30
        MIN_SIZE = 100

        doc = fitz.open(file_path)
        candidates = []
        seen_xrefs = set()

        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)

            for img_info in images:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    pix = fitz.Pixmap(doc, xref)

                    if pix.alpha:
                        pix = fitz.Pixmap(pix, 0)

                    if pix.colorspace and pix.colorspace.n != 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    if pix.width < MIN_SIZE or pix.height < MIN_SIZE:
                        continue

                    # Resize if too large to prevent Ollama 500 OOM errors
                    MAX_DIM = 1024
                    if pix.width > MAX_DIM or pix.height > MAX_DIM:
                        scale = MAX_DIM / max(pix.width, pix.height)
                        mat = fitz.Matrix(scale, scale)
                        pix = fitz.Pixmap(pix, mat)

                    img_bytes = pix.tobytes("png")
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    candidates.append((page_num, pix.width, pix.height, b64))
                except Exception as e:
                    pass

            if len(candidates) >= MAX_IMAGES:
                break

        doc.close()

        candidates = candidates[:MAX_IMAGES]
        total = len(candidates)

        if not total:
            print("[ImageExpert] No valid images found")
            return []

        try:
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        except Exception:
            print(f"[ImageExpert] Ollama unavailable at {OLLAMA_BASE_URL}, skipping {total} images")
            return []

        print(f"[ImageExpert] Found {total} images, captioning concurrently...")

        chunks: list[Chunk] = []

        def caption_one(idx, page_num, w, h, b64):
            cap = self._caption_image(b64)
            print(f"[ImageExpert] Captioned {idx+1}/{total} (page {page_num+1})")
            return page_num, w, h, cap

        # Use max_workers=1 because local Ollama llava crashes with 500 on concurrent vision inference
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {
                pool.submit(caption_one, i, pn, w, h, b64): i
                for i, (pn, w, h, b64) in enumerate(candidates)
            }
            for future in as_completed(futures):
                try:
                    page_num, w, h, caption = future.result()
                    if not caption:
                        continue
                    chunks.append(Chunk(
                        org_id=org_id,
                        file_id=file_id,
                        expert_id=self.expert_id,
                        content=caption,
                        metadata={
                            "page": page_num + 1,
                            "type": "pdf_image",
                            "width": w,
                            "height": h,
                        },
                    ))
                except Exception as e:
                    print(f"[ImageExpert] Caption task failed: {e}")

        print(f"[ImageExpert] PDF -> {len(chunks)} image chunks from {os.path.basename(file_path)}")
        return chunks

    def _encode_image(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _caption_image(self, b64_image: str) -> str:
        prompt = (
            "Describe this image in detail. Include all visible text, labels, "
            "data values, diagram elements, and any structural information. "
            "Be thorough and factual."
        )
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_VISION_MODEL,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 1024,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[ImageExpert] Caption failed: {e}")
            return ""

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        print(f"[ImageExpert] Embedding {len(texts)} image captions...")
        embeddings = self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 5,
            normalize_embeddings=True,
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        print(f"[ImageExpert] [OK] Embedded {len(chunks)} chunks (dim={embeddings.shape[1]})")
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode(query, normalize_embeddings=True)
