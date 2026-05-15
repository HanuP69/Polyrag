"""
table.py -- Table Expert.

Parse: PyMuPDF detects table regions in PDFs. CSV files handled directly.
       Each table is linearized row-by-row: "Col1: val | Col2: val | Col3: val"
       Whole table = one chunk (context must be preserved).
Embed: BGE-M3 on linearized text (same model as text expert).
"""

import os
import csv
import io
import numpy as np

from engine.experts.base import BaseExpert, Chunk
from engine.config import EMBEDDING_MODEL


class TableExpert(BaseExpert):

    expert_id = "table"

    def __init__(self):
        self._embed_model = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[TableExpert] Loading embedding model: {EMBEDDING_MODEL}")
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embed_model

    # ── parsing ──────────────────────────────────────────────────────

    def parse(self, file_path: str, file_id: str = "", org_id: str = "") -> list[Chunk]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return self._parse_csv(file_path, file_id, org_id)
        elif ext == ".pdf":
            return self._parse_pdf_tables(file_path, file_id, org_id)
        else:
            return []

    def _parse_csv(self, file_path: str, file_id: str, org_id: str) -> list[Chunk]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            return []

        headers = rows[0]
        data_rows = rows[1:]
        linearized = self._linearize_rows(headers, data_rows)

        chunk = Chunk(
            org_id=org_id,
            file_id=file_id,
            expert_id=self.expert_id,
            content=linearized,
            metadata={
                "row_count": len(data_rows),
                "col_names": headers,
                "source": os.path.basename(file_path),
            },
        )
        print(f"[TableExpert] CSV -> 1 chunk, {len(data_rows)} rows, {len(headers)} cols")
        return [chunk]

    def _parse_pdf_tables(self, file_path: str, file_id: str, org_id: str) -> list[Chunk]:
        import fitz

        doc = fitz.open(file_path)
        chunks: list[Chunk] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()

            for table in tables:
                cells = table.extract()  # list[list[str | None]]
                if not cells or len(cells) < 2:
                    continue

                headers = [c or "" for c in cells[0]]
                data_rows = [[c or "" for c in row] for row in cells[1:]]
                linearized = self._linearize_rows(headers, data_rows)

                chunks.append(Chunk(
                    org_id=org_id,
                    file_id=file_id,
                    expert_id=self.expert_id,
                    content=linearized,
                    metadata={
                        "page": page_num + 1,
                        "row_count": len(data_rows),
                        "col_names": headers,
                    },
                ))

        doc.close()
        print(f"[TableExpert] PDF -> {len(chunks)} table chunks from {os.path.basename(file_path)}")
        return chunks

    # ── linearization ────────────────────────────────────────────────

    @staticmethod
    def _linearize_rows(headers: list[str], data_rows: list[list[str]]) -> str:
        """
        Convert tabular data into a text format the embedding model can reason about.
        Format: "Col1: val | Col2: val | Col3: val"
        """
        lines = []
        header_line = " | ".join(headers)
        lines.append(f"Columns: {header_line}")

        for row in data_rows:
            parts = []
            for h, v in zip(headers, row):
                v = v.strip() if v else ""
                if v:
                    parts.append(f"{h}: {v}")
            if parts:
                lines.append(" | ".join(parts))

        return "\n".join(lines)

    # ── embedding ────────────────────────────────────────────────────

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        print(f"[TableExpert] Embedding {len(texts)} table chunks...")
        embeddings = self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 5,
            normalize_embeddings=True,
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        print(f"[TableExpert] [OK] Embedded {len(chunks)} chunks (dim={embeddings.shape[1]})")
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode(query, normalize_embeddings=True)
