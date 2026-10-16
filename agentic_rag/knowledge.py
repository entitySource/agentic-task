"""The knowledge base the agent retrieves from.

Two kinds of source, both split into chunks and embedded into a LangChain
`InMemoryVectorStore`:

* Column cards, generated from every loaded dataset: one card per column
  (type, range, example values) plus a dataset card. They let retrieval point
  the model at the right columns for a question.
* Documents the user uploads (txt, md, csv data dictionaries, pdf): business
  glossaries, metric definitions, data dictionaries. Split with LangChain's
  `RecursiveCharacterTextSplitter`.

Chunk text and metadata live in SQLite so the knowledge base survives a
restart; vectors are rebuilt from it at startup.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .data import Dataset
from .embeddings import words
from .memory import utcnow

DOC_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".csv", ".json"}
MAX_DOC_BYTES = 5 * 1024 * 1024


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    kind: str  # column | dataset | document
    text: str
    score: float
    location: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def extract_text(filename: str, payload: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in DOC_SUFFIXES:
        raise ValueError(f"Unsupported document type '{suffix}'. Use .txt, .md, .pdf, .csv or .json.")
    if len(payload) > MAX_DOC_BYTES:
        raise ValueError("Document is larger than 5 MB.")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(payload))
        pages = [f"[page {i + 1}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)]
        text = "\n\n".join(pages)
    else:
        text = payload.decode("utf-8", errors="replace")
    if not text.strip():
        raise ValueError("No readable text found in this document.")
    return text


class KnowledgeBase:
    def __init__(self, path: str | Path, embeddings: Embeddings, chunk_size: int = 700, chunk_overlap: int = 100):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""])
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")], strip_headers=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, kind TEXT NOT NULL, dataset_fp TEXT,
                chars INTEGER NOT NULL, chunks INTEGER NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, position INTEGER NOT NULL,
                text TEXT NOT NULL, meta_json TEXT NOT NULL DEFAULT '{}');
            """)
        self.conn.commit()
        self.store = InMemoryVectorStore(embeddings)
        self._rebuild()

    # ------------------------------------------------------------------ index
    def _rebuild(self):
        self.store = InMemoryVectorStore(self.embeddings)
        rows = self.conn.execute(
            "SELECT c.id, c.doc_id, c.text, c.meta_json, d.title, d.kind, d.dataset_fp FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id").fetchall()
        if rows:
            self.store.add_documents([self._doc(r) for r in rows], ids=[r["id"] for r in rows])

    @staticmethod
    def _doc(r) -> Document:
        meta = json.loads(r["meta_json"] or "{}")
        meta.update({"doc_id": r["doc_id"], "title": r["title"], "kind": r["kind"],
                     "dataset_fp": r["dataset_fp"], "chunk_id": r["id"]})
        return Document(page_content=r["text"], metadata=meta)

    def _add(self, title: str, kind: str, dataset_fp: str | None, pieces: list[tuple[str, dict]],
             doc_id: str | None = None, chars: int | None = None) -> str:
        doc_id = doc_id or uuid.uuid4().hex[:10]
        with self.lock:
            self.conn.execute("INSERT INTO documents(id, title, kind, dataset_fp, chars, chunks, created_at) "
                              "VALUES (?,?,?,?,?,?,?)",
                              (doc_id, title, kind, dataset_fp, chars or sum(len(t) for t, _ in pieces),
                               len(pieces), utcnow()))
            rows = []
            for i, (text, meta) in enumerate(pieces):
                cid = f"{doc_id}-{i}"
                self.conn.execute("INSERT INTO chunks(id, doc_id, position, text, meta_json) VALUES (?,?,?,?,?)",
                                  (cid, doc_id, i, text, json.dumps(meta)))
                rows.append({"id": cid, "doc_id": doc_id, "text": text, "meta_json": json.dumps(meta),
                             "title": title, "kind": kind, "dataset_fp": dataset_fp})
            self.conn.commit()
            if rows:
                self.store.add_documents([self._doc(r) for r in rows], ids=[r["id"] for r in rows])
        return doc_id

    # ------------------------------------------------------------------ sources
    def split(self, filename: str, text: str) -> list[tuple[str, dict]]:
        """Markdown is split on its headings first (one glossary entry per chunk), then by size."""
        pieces: list[tuple[str, dict]] = []
        if Path(filename).suffix.lower() in (".md", ".markdown"):
            sections = self.md_splitter.split_text(text)
            for sec in sections:
                trail = [sec.metadata[h] for h in ("h1", "h2", "h3", "h4") if sec.metadata.get(h)]
                section = " › ".join(trail[-2:])
                for chunk in self.splitter.split_text(sec.page_content):
                    pieces.append((chunk.strip(), {"section": section} if section else {}))
            return [p for p in pieces if p[0]]
        page = None
        for chunk in self.splitter.split_text(text):
            found = re.findall(r"\[page (\d+)\]", chunk)
            page = int(found[-1]) if found else page
            meta = {"page": page} if page else {}
            pieces.append((re.sub(r"\[page \d+\]\s*", "", chunk).strip(), meta))
        return [p for p in pieces if p[0]]

    def add_document(self, filename: str, payload: bytes, title: str | None = None) -> dict:
        text = extract_text(filename, payload)
        pieces = self.split(filename, text)
        doc_id = self._add(title or Path(filename).name, "document", None, pieces, chars=len(text))
        return self.get_document(doc_id)

    def index_dataset(self, ds: Dataset) -> str:
        """(Re)build the column cards for a dataset."""
        doc_id = f"ds-{ds.fingerprint}"
        self.delete_document(doc_id)
        kinds = ds.kinds()
        overview = (f"Dataset {ds.name}: {len(ds.df):,} rows and {len(ds.columns)} columns. Columns: "
                    + ", ".join(f"{c} ({k})" for c, k in kinds.items()) + ".")
        pieces = [(overview, {"column": None})]
        for col in ds.columns:
            readable = col.replace("_", " ")
            pieces.append((f"Column '{col}' ({readable}) in dataset {ds.name}: {ds.describe_column(col)}.",
                           {"column": col}))
        self._add(f"{ds.name} columns", "dataset", ds.fingerprint, pieces, doc_id=doc_id)
        return doc_id

    # ------------------------------------------------------------------ admin
    def list_documents(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC")]

    def get_document(self, doc_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def document_chunks(self, doc_id: str) -> list[dict]:
        return [{"id": r["id"], "position": r["position"], "text": r["text"], **json.loads(r["meta_json"])}
                for r in self.conn.execute("SELECT * FROM chunks WHERE doc_id=? ORDER BY position", (doc_id,))]

    def delete_document(self, doc_id: str) -> bool:
        with self.lock:
            ids = [r["id"] for r in self.conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            cur = self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.commit()
            if ids:
                self.store.delete(ids)
            return cur.rowcount > 0

    # ------------------------------------------------------------------ search
    def search(self, query: str, dataset_fp: str | None = None, k: int = 5,
               kinds: tuple[str, ...] = ("document", "dataset")) -> list[Hit]:
        """Hybrid search: vector similarity, re-ranked with a small keyword-overlap bonus.

        Column cards are only searched for the active dataset; documents always.
        """
        def allowed(doc: Document) -> bool:
            md = doc.metadata
            if md.get("kind") not in kinds:
                return False
            return md.get("kind") != "dataset" or md.get("dataset_fp") == dataset_fp

        with self.lock:
            candidates = self.store.similarity_search_with_score(query, k=max(k * 4, 12), filter=allowed)
        q_words = set(words(query))
        hits = []
        for doc, score in candidates:
            md = doc.metadata
            overlap = len(q_words & set(words(doc.page_content)))
            col = md.get("column")
            bonus = 0.04 * overlap + (0.15 if col and col.lower().replace("_", " ") in query.lower() else 0)
            location = ", ".join(p for p in [f"p. {md['page']}" if md.get("page") else "",
                                             md.get("section") or "", f"column {col}" if col else ""] if p)
            hits.append(Hit(md["chunk_id"], md["doc_id"], md["title"], md["kind"], doc.page_content,
                            round(float(score) + bonus, 3), location))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

