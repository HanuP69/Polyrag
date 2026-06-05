"""
PDF ingestion pipeline.
Reuses extract_pdf from the old engine + v4 notebook chunking.
"""
import os
import io
import re
import uuid
import base64
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from engine_v4.config import CFG
from engine_v4.chunker import Chunk, chunk_text, chunk_table, chunk_image, clean_text, caption_image_b64
from engine_v4.models import embedder
from engine_v4.ollama import ollama_unload_all
from engine_v4 import db


# ── PDF Parser (from old engine, proven to work) ────────────────────────────

def extract_pdf_chunks(
    pdf_path: str, file_id: str, org_id: str = "default"
) -> List[Chunk]:
    """
    Parse a PDF into chunks using PyMuPDF.
    Extracts text, tables, and images — creates Chunk objects matching v4 schema.
    """
    import fitz

    doc = fitz.open(pdf_path)
    all_chunks: List[Chunk] = []
    chunk_counts = {"text": 0, "table": 0, "image": 0}

    for page_no, page in enumerate(doc):
        # ── Text blocks ──────────────────────────────────────────────────
        text_content = page.get_text("text") or ""
        text_content = re.sub(r'\s+', ' ', unicodedata.normalize("NFKD", text_content)).strip()
        if text_content:
            for ch in chunk_text(file_id, page_no, text_content, org_id, file_id):
                all_chunks.append(ch)
                chunk_counts["text"] += 1

        # ── Tables ───────────────────────────────────────────────────────
        try:
            tables = page.find_tables()
            for t_idx, tbl in enumerate(tables.tables):
                rows = tbl.extract()
                if not rows:
                    continue
                header = rows[0]
                sep = ["---"] * len(header)
                body = rows[1:]
                def fmt_row(r):
                    return "| " + " | ".join(str(c or "").replace('\n', ' ') for c in r) + " |"
                md = "\n".join([fmt_row(header), fmt_row(sep)] + [fmt_row(r) for r in body])
                if md.strip():
                    for ch in chunk_table(file_id, page_no, f"tbl{t_idx}", md, org_id, file_id):
                        all_chunks.append(ch)
                        chunk_counts["table"] += 1
        except Exception as e:
            print(f"[Ingest] Table extraction failed page {page_no}: {e}")

        # ── Images ───────────────────────────────────────────────────────
        img_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(img_list[:CFG.max_images_per_section]):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 32 or pix.height < 32:
                    continue

                # Resize large images
                img_bytes = pix.tobytes(output="png")
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes))
                if img.width > 800 or img.height > 800:
                    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()

                b64 = base64.b64encode(img_bytes).decode()

                # Save extracted image to CFG.upload_dir (data/uploads) so the frontend can serve it
                img_filename = f"{file_id}_p{page_no}_img{img_idx}.png"
                img_save_path = os.path.join(CFG.upload_dir, img_filename)
                try:
                    os.makedirs(CFG.upload_dir, exist_ok=True)
                    with open(img_save_path, "wb") as img_file:
                        img_file.write(img_bytes)
                except Exception as e:
                    print(f"[Ingest] Warning: Could not write image file to disk: {e}")

                # Get nearby text for context
                nearby = text_content[:400] if text_content else ""

                # Caption via Ollama llava
                caption = ""
                try:
                    caption = caption_image_b64(b64)
                except Exception:
                    pass

                # OCR
                ocr_text = ""
                try:
                    import pytesseract
                    ocr_text = clean_text(pytesseract.image_to_string(img))
                except Exception:
                    pass

                # Build chunk content (same format as v4 notebook)
                parts = []
                if caption:    parts.append(f"[Caption]: {caption}")
                if ocr_text:   parts.append(f"[OCR]: {ocr_text[:500]}")
                if nearby:     parts.append(f"[Context]: {nearby[:300]}")
                if not parts:
                    continue

                ch = Chunk(
                    chunk_id=f"{file_id}_p{page_no}_img{img_idx}",
                    doc_id=file_id, section_id=page_no,
                    modality="image", content=" ".join(parts),
                    metadata={"image_id": f"img{img_idx}", "has_ocr": bool(ocr_text),
                              "has_caption": bool(caption), "page": page_no + 1,
                              "source": img_filename},
                    org_id=org_id, file_id=file_id, expert_id="image",
                )
                all_chunks.append(ch)
                chunk_counts["image"] += 1

            except Exception as e:
                print(f"[Ingest] Image extraction failed page {page_no}: {e}")
                continue

    doc.close()
    print(f"[Ingest] PDF parsed: {sum(chunk_counts.values())} chunks "
          f"(text={chunk_counts['text']}, table={chunk_counts['table']}, image={chunk_counts['image']})")
    return all_chunks


# ── Text/CSV/MD file parsing ────────────────────────────────────────────────

def extract_text_file(file_path: str, file_id: str, org_id: str = "default") -> List[Chunk]:
    """Parse plain text / markdown / CSV files."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    return chunk_text(file_id, 0, text, org_id, file_id)


# ── Full ingestion pipeline ─────────────────────────────────────────────────

def ingest_file(file_path: str, file_id: str, org_id: str = "default", progress_callback=None) -> dict:
    """
    Full ingestion pipeline (v4 architecture):
    1. Parse file → chunks
    2. Unload Ollama (captioning done) → free VRAM
    3. Embed all chunks with BGE-M3
    4. Store chunks + embeddings + BM25 to PostgreSQL
    5. Reload in-memory indexes
    """
    import torch, gc

    def _progress(pct: int, stage: str):
        if progress_callback:
            progress_callback(pct, stage)

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Parse
    _progress(5, "parsing")
    print(f"[Ingest] Parsing {file_path} (ext={ext})...")
    if ext == ".pdf":
        all_chunks = extract_pdf_chunks(file_path, file_id, org_id)
    elif ext in (".txt", ".md", ".csv"):
        all_chunks = extract_text_file(file_path, file_id, org_id)
    else:
        return {"status": "error", "error": f"Unsupported file type: {ext}"}

    if not all_chunks:
        return {"status": "error", "error": "No chunks extracted"}

    # Check if any image chunks needed captioning
    has_images = any(ch.modality == "image" for ch in all_chunks)
    if has_images:
        _progress(25, "captioning")
    else:
        _progress(20, "parsing")

    # 2. Unload Ollama (llava was used for captioning) → free VRAM for embedder
    _progress(40, "preparing")
    print("[Ingest] Unloading Ollama models...")
    ollama_unload_all()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Embed
    _progress(45, "embedding")
    print(f"[Ingest] Embedding {len(all_chunks)} chunks...")
    embedder.to_gpu()
    texts = [ch.content for ch in all_chunks]
    BATCH = 64
    all_vecs = []
    total_batches = max(1, (len(texts) + BATCH - 1) // BATCH)
    for batch_idx, start in enumerate(range(0, len(texts), BATCH)):
        batch = texts[start : start + BATCH]
        vecs = embedder.embed(batch)
        all_vecs.append(vecs)
        # Progress from 45% to 70% during embedding
        embed_progress = 45 + int((batch_idx + 1) / total_batches * 25)
        _progress(min(embed_progress, 70), "embedding")
        if (start // BATCH) % 5 == 0:
            print(f"  {start + len(batch)}/{len(texts)} embedded")

    import numpy as np
    vecs = np.vstack(all_vecs)
    for ch, v in zip(all_chunks, vecs):
        ch.embedding = v

    # 4. Store to PostgreSQL
    _progress(80, "storing")
    print("[Ingest] Storing to PostgreSQL...")
    conn = db.get_conn()
    db.store_chunks(all_chunks, conn)
    db.store_embeddings(all_chunks, conn)
    conn.close()

    # Rebuild BM25 from ALL org chunks (not just this file)
    db.rebuild_bm25(org_id)

    # 5. Reload in-memory indexes
    _progress(90, "indexing")
    from engine_v4.retrieval import reload_indexes
    reload_indexes(org_id)

    _progress(100, "indexed")
    print(f"[Ingest] Done. {len(all_chunks)} chunks stored.")
    return {"status": "completed", "chunk_count": len(all_chunks)}
