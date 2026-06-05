"""
Chunking functions — direct port from v4 notebook.
Chunk dataclass + chunk_text, chunk_table, chunk_image.
"""
import re
import io
import base64
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from engine_v4.config import CFG
from engine_v4.ollama import ollama_generate


@dataclass
class Chunk:
    chunk_id:   str
    doc_id:     str
    section_id: int
    modality:   str               # "text" | "table" | "image"
    content:    str
    metadata:   dict = field(default_factory=dict)
    embedding:  Optional[np.ndarray] = None

    # Extra fields used by the server (compatible with old engine)
    org_id:     str = "default"
    file_id:    str = ""
    expert_id:  str = ""          # alias for modality in old API

    def to_dict(self) -> dict:
        return {
            "chunk_id":   self.chunk_id,
            "doc_id":     self.doc_id,
            "section_id": self.section_id,
            "modality":   self.modality,
            "expert_id":  self.modality,   # old API compat
            "content":    self.content,
            "metadata":   self.metadata,
            "org_id":     self.org_id,
            "file_id":    self.file_id,
        }


def clean_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip()


# ── CAPTION PROMPT (from v4 notebook) ────────────────────────────────────────

CAPTION_PROMPT = (
    "Describe this image in detail for search and retrieval purposes. "
    "Include: (1) what type of image this is (chart, diagram, photo, screenshot, "
    "flowchart, table, map, etc.), (2) all visible text, labels, numbers, legends, "
    "axes, or annotations, (3) the main subject or topic, (4) key information, "
    "relationships, or trends shown. Be specific and factual. Max 150 words."
)


def should_skip_image(ocr_text: str, caption: str) -> bool:
    if not CFG.skip_decorative_images:
        return False
    if len(ocr_text.strip()) < 10:
        skip_words = ["logo", "icon", "banner", "decorative", "border", "watermark", "seal", "badge"]
        if any(w in caption.lower() for w in skip_words):
            return True
    return False


def caption_image_gemini(b64_str: str, org_id: str = "default") -> str:
    """Caption via Gemini model. b64_str is base64 representation of image."""
    from engine_v4 import db
    import requests
    
    org_data = db.get_org_config(org_id) or {}
    db_cfg = org_data.get("config", {})
    
    # Check custom keys first, then fallback
    from engine_v4.ollama import get_next_key
    db_keys = db_cfg.get("geminiApiKeys") or []
    if not isinstance(db_keys, list):
        db_keys = [db_keys]
    if db_cfg.get("geminiApiKey"):
        db_keys.insert(0, db_cfg.get("geminiApiKey"))
        
    api_key = get_next_key(org_id, "gemini", db_keys, CFG.gemini_api_key)
    if not api_key:
        print("[Caption Gemini] Warning: Gemini API key not set. Skipping captioning.")
        return ""

    # Strip prefix if any
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]

    # Call Gemini REST API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CFG.gemini_model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": CAPTION_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": b64_str
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        caption = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return clean_text(caption)
    except Exception as e:
        print(f"[Caption Gemini] API call failed: {e}")
        return ""


def caption_image_pil(img, org_id: str = "default") -> str:
    """Caption via Gemini. img is a PIL Image."""
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return caption_image_gemini(b64, org_id)


def caption_image_b64(b64_str: str, org_id: str = "default") -> str:
    """Caption from base64 string (used during server ingestion)."""
    return caption_image_gemini(b64_str, org_id)


# ── CHUNK TEXT (from v4 notebook) ────────────────────────────────────────────

def chunk_text(doc_id: str, sec_idx: int, text: str, org_id="default", file_id="") -> List[Chunk]:
    MAX_CHARS, OVERLAP = 1800, 200
    text = clean_text(text)
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) < MAX_CHARS:
            cur += " " + s
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = cur[-OVERLAP:] + " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return [
        Chunk(
            chunk_id=f"{doc_id}_s{sec_idx}_t{i}",
            doc_id=doc_id, section_id=sec_idx,
            modality="text", content=c,
            metadata={"chunk_idx": i},
            org_id=org_id, file_id=file_id, expert_id="text",
        )
        for i, c in enumerate(chunks) if c
    ]


# ── CHUNK TABLE (from v4 notebook) ──────────────────────────────────────────

def chunk_table(doc_id: str, sec_idx: int, table_id: str, md_table: str,
                org_id="default", file_id="") -> List[Chunk]:
    lines = [l.strip() for l in md_table.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return [Chunk(
            chunk_id=f"{doc_id}_s{sec_idx}_{table_id}_r0",
            doc_id=doc_id, section_id=sec_idx,
            modality="table", content=f"[TABLE] {clean_text(md_table)}",
            metadata={"table_id": table_id},
            org_id=org_id, file_id=file_id, expert_id="table",
        )]
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    data_lines = [l for l in lines[1:] if not re.match(r'^[|\s\-]+$', l)]
    chunks = []
    for gi, start in enumerate(range(0, max(1, len(data_lines)), 5)):
        rg = data_lines[start : start + 5]
        rows_nat = []
        for row in rg:
            cells = [c.strip() for c in row.split("|") if c.strip()]
            rows_nat.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, cells)))
        content = (
            f"[TABLE] Columns: {', '.join(headers)}.\n"
            + "\n".join(rows_nat)
            + f"\n\nMarkdown:\n{lines[0]}\n" + "\n".join(rg)
        )
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_s{sec_idx}_{table_id}_r{gi}",
            doc_id=doc_id, section_id=sec_idx,
            modality="table", content=content,
            metadata={"table_id": table_id, "row_group": gi, "headers": headers},
            org_id=org_id, file_id=file_id, expert_id="table",
        ))
    return chunks


# ── CHUNK IMAGE (from v4 notebook) ──────────────────────────────────────────

def chunk_image(
    doc_id: str, sec_idx: int, image_id: str, b64_str: str,
    nearby_text: str = "", fig_caption: str = "",
    org_id: str = "default", file_id: str = "",
) -> Optional[Chunk]:
    from PIL import Image

    try:
        # Strip data URI prefix
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    except Exception:
        return None

    ocr_text = ""
    try:
        import pytesseract
        ocr_text = clean_text(pytesseract.image_to_string(img))
    except Exception:
        pass

    caption = ""
    try:
        caption = caption_image_pil(img, org_id)
    except Exception:
        pass

    if should_skip_image(ocr_text, caption):
        return None

    parts = []
    if caption:      parts.append(f"[Caption]: {caption}")
    if ocr_text:     parts.append(f"[OCR]: {ocr_text[:500]}")
    if fig_caption:  parts.append(f"[Figure caption]: {fig_caption}")
    if nearby_text:  parts.append(f"[Context]: {nearby_text[:300]}")
    if not parts:
        return None

    return Chunk(
        chunk_id=f"{doc_id}_s{sec_idx}_{image_id}",
        doc_id=doc_id, section_id=sec_idx,
        modality="image", content=" ".join(parts),
        metadata={"image_id": image_id, "has_ocr": bool(ocr_text), "has_caption": bool(caption)},
        org_id=org_id, file_id=file_id, expert_id="image",
    )
