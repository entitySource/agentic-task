"""Curated long-term memory: SQLite for the facts, embeddings for retrieval.

Each memory is one small typed record (a definition, a named filter, a
preference, a correction or a column note), never a transcript.

* One active value per key. Saving a new value for a key supersedes the old
  one; the old one stays in history with an audit trail.
* A fact the user stated cannot be overwritten by something the agent
  inferred: the store reports a conflict and the agent must ask.
* Scope: preferences are global; other memories belong to the dataset they
  were learned on, but carry over to another file that has every column
  they reference. Memories whose columns are missing are reported as stale.
* Retrieval is semantic (embeddings) plus a boost when the question uses the
  remembered key itself, so "live accounts" can find "active customers".
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .embeddings import singular, words

KINDS = {
    "definition": "What a business term means, ideally with a filter that implements it "
                  "(e.g. 'active customers' = status is not cancelled).",
    "named_filter": "A reusable, named set of row conditions (e.g. 'big north deals').",
    "preference": "How the user likes answers (time_granularity=weekly, decimal_places=1).",
    "correction": "A corrected fact about the data or domain (e.g. 'revenue is in thousands of INR').",
    "column_note": "What a column means or how to read it.",
}

GRANULARITIES = {"daily": "D", "weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y",
                 "d": "D", "w": "W", "m": "M", "q": "Q", "y": "Y"}
STRUCTURED_PREFERENCES = ("time_granularity", "decimal_places")
MAX_CONTENT_CHARS = 600


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def normalise_key(key: str) -> str:
    return " ".join(singular(t) for t in re.sub(r"[^a-z0-9]+", " ", key.lower()).split())


def granularity_code(value: str | None) -> str | None:
    return GRANULARITIES.get(value.strip().lower()) if value else None


@dataclass
class Memory:
    id: int
    kind: str
    key: str
    content: str
    filter: list | None
    scope: str
    dataset_fp: str | None
    dataset_name: str | None
    columns: list
    source: str
    status: str
    created_at: str
    updated_at: str
    use_count: int
    last_used_at: str | None
    supersedes_id: int | None
    note: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def render(self) -> str:
        line = f"{self.kind} '{self.key}': {self.content}"
        if self.filter:
            line += f" | filter={json.dumps(self.filter)}"
        if self.source == "user":
            line += " (stated by user)"
        if self.note:
            line += f" ({self.note})"
        return line


@dataclass
class SaveResult:
    status: str  # created | updated | unchanged | conflict | rejected
    memory: Memory | None = None
    previous: Memory | None = None
    message: str = ""

    def as_dict(self) -> dict:
        out = {"status": self.status, "message": self.message}
        if self.memory:
            out["memory"] = self.memory.render()
            out["memory_id"] = self.memory.id
        if self.previous:
            out["previous"] = self.previous.render()
        return out


@dataclass
class MemoryHit:
    memory: Memory
    score: float


@dataclass
class MemoryRetrieval:
    hits: list[MemoryHit] = field(default_factory=list)
    stale: list[tuple[Memory, str]] = field(default_factory=list)


class MemoryStore:
    def __init__(self, path: str | Path, embeddings: Embeddings):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.embeddings = embeddings
        self.index = InMemoryVectorStore(embeddings)
        self._init()
        self._reindex()

    def _init(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, key TEXT NOT NULL,
                norm_key TEXT NOT NULL, content TEXT NOT NULL, filter_json TEXT,
                scope TEXT NOT NULL DEFAULT 'dataset', dataset_fp TEXT, dataset_name TEXT,
                columns_json TEXT NOT NULL DEFAULT '[]', source TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 0, last_used_at TEXT, supersedes_id INTEGER);
            CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(norm_key, status);
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER NOT NULL,
                action TEXT NOT NULL, detail TEXT, at TEXT NOT NULL);
            """)
        self.conn.commit()

    # ------------------------------------------------------------------ index
    def _doc(self, m: Memory) -> Document:
        return Document(page_content=f"{m.key}: {m.content}", metadata={"memory_id": m.id})

    def _reindex(self):
        self.index = InMemoryVectorStore(self.embeddings)
        mems = self.list()
        if mems:
            self.index.add_documents([self._doc(m) for m in mems], ids=[str(m.id) for m in mems])

    def _index_add(self, m: Memory):
        self.index.add_documents([self._doc(m)], ids=[str(m.id)])

    def _index_remove(self, memory_id: int):
        self.index.delete([str(memory_id)])

    # ------------------------------------------------------------------ rows
    def _row(self, r: sqlite3.Row) -> Memory:
        return Memory(
            id=r["id"], kind=r["kind"], key=r["key"], content=r["content"],
            filter=json.loads(r["filter_json"]) if r["filter_json"] else None,
            scope=r["scope"], dataset_fp=r["dataset_fp"], dataset_name=r["dataset_name"],
            columns=json.loads(r["columns_json"] or "[]"), source=r["source"], status=r["status"],
            created_at=r["created_at"], updated_at=r["updated_at"], use_count=r["use_count"],
            last_used_at=r["last_used_at"], supersedes_id=r["supersedes_id"])

    def _log(self, memory_id: int, action: str, detail: str = ""):
        self.conn.execute("INSERT INTO memory_events(memory_id, action, detail, at) VALUES (?,?,?,?)",
                          (memory_id, action, detail, utcnow()))

    def get(self, memory_id: int) -> Memory | None:
        r = self.conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, include_inactive: bool = False) -> list[Memory]:
        sql = "SELECT * FROM memories" + ("" if include_inactive else " WHERE status='active'")
        return [self._row(r) for r in self.conn.execute(sql + " ORDER BY kind, key")]

    def history(self, memory_id: int) -> list[dict]:
        m = self.get(memory_id)
        while m:  # walk forward to the newest version
            nxt = self.conn.execute("SELECT * FROM memories WHERE supersedes_id=?", (m.id,)).fetchone()
            if not nxt:
                break
            m = self._row(nxt)
        chain, seen = [], set()
        while m and m.id not in seen:
            seen.add(m.id)
            events = [dict(e) for e in self.conn.execute(
                "SELECT action, detail, at FROM memory_events WHERE memory_id=? ORDER BY id", (m.id,))]
            chain.append({"memory": m.as_dict(), "events": events})
            m = self.get(m.supersedes_id) if m.supersedes_id else None
        return chain

    def _find_active(self, norm_key: str, scope: str, dataset_fp: str | None) -> Memory | None:
        if scope == "global":
            r = self.conn.execute("SELECT * FROM memories WHERE norm_key=? AND status='active' "
                                  "AND scope='global'", (norm_key,)).fetchone()
        else:
            r = self.conn.execute("SELECT * FROM memories WHERE norm_key=? AND status='active' AND "
                                  "(scope='global' OR dataset_fp=?) ORDER BY scope='dataset' DESC",
                                  (norm_key, dataset_fp)).fetchone()
        return self._row(r) if r else None

    # ------------------------------------------------------------------ writes
    def remember(self, kind: str, key: str, content: str, *, filter: list | None = None,
                 scope: str = "dataset", dataset_fp: str | None = None, dataset_name: str | None = None,
                 columns: list | None = None, source: str = "user",
                 confirm_overwrite: bool = False) -> SaveResult:
        kind = (kind or "").strip().lower()
        key, content = (key or "").strip(), (content or "").strip()
        if kind not in KINDS:
            return SaveResult("rejected", message=f"Unknown kind '{kind}'. Use one of {sorted(KINDS)}.")
        if not key or not content:
            return SaveResult("rejected", message="Both key and content are required.")
        if len(content) > MAX_CONTENT_CHARS:
            return SaveResult("rejected", message=(
                f"Content is {len(content)} characters (max {MAX_CONTENT_CHARS}). Store a short fact, "
                "not raw data or a transcript."))
        if kind == "named_filter" and not filter:
            return SaveResult("rejected", message="A named_filter needs a 'filter' list of conditions.")
        if kind == "preference":
            norm = normalise_key(key).replace(" ", "_")
            if norm in STRUCTURED_PREFERENCES:
                key, scope = norm, "global"
                err = _validate_preference(norm, content)
                if err:
                    return SaveResult("rejected", message=err)
                content = content.strip().lower()
        if scope not in ("global", "dataset") or (scope == "dataset" and not dataset_fp):
            scope = "global"
        source = source if source in ("user", "agent") else "agent"
        cols = sorted(set(columns or []) | {c["column"] for c in (filter or []) if "column" in c})
        norm_key = normalise_key(key)

        with self.lock:
            existing = self._find_active(norm_key, scope, dataset_fp)
            now = utcnow()
            if existing:
                if existing.content == content and (existing.filter or None) == (filter or None) \
                        and existing.kind == kind:
                    self._log(existing.id, "reconfirmed", f"source={source}")
                    self.conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (now, existing.id))
                    self.conn.commit()
                    return SaveResult("unchanged", memory=self.get(existing.id),
                                      message="Already remembered exactly this.")
                if existing.source == "user" and source != "user" and not confirm_overwrite:
                    return SaveResult("conflict", memory=existing, message=(
                        "The user stated a different value for this earlier. Do not overwrite it on your "
                        "own: ask the user which is right, then save again with source='user'."))
            cur = self.conn.execute(
                """INSERT INTO memories(kind, key, norm_key, content, filter_json, scope, dataset_fp,
                   dataset_name, columns_json, source, status, created_at, updated_at, supersedes_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,'active',?,?,?)""",
                (kind, key, norm_key, content, json.dumps(filter) if filter else None, scope,
                 dataset_fp if scope == "dataset" else None, dataset_name, json.dumps(cols), source,
                 now, now, existing.id if existing else None))
            new_id = cur.lastrowid
            if existing:
                self.conn.execute("UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                                  (now, existing.id))
                self._log(existing.id, "superseded", f"by #{new_id}")
                self._log(new_id, "updated", f"replaced #{existing.id}: {existing.content}")
                self._index_remove(existing.id)
            else:
                self._log(new_id, "created", f"source={source}")
            self.conn.commit()
            new = self.get(new_id)
            self._index_add(new)
        if existing:
            return SaveResult("updated", memory=new, previous=existing,
                              message="Replaced the older value; the old one is kept in history.")
        return SaveResult("created", memory=new, message="Saved.")

    def edit(self, memory_id: int, content: str | None = None, filter: list | None = None,
             key: str | None = None) -> SaveResult:
        m = self.get(memory_id)
        if not m or m.status != "active":
            return SaveResult("rejected", message=f"No active memory #{memory_id}.")
        if key and normalise_key(key) != normalise_key(m.key):
            self.delete(memory_id, reason=f"renamed to '{key}'")
            return self.remember(m.kind, key, content or m.content, filter=filter or m.filter, scope=m.scope,
                                 dataset_fp=m.dataset_fp, dataset_name=m.dataset_name, columns=m.columns)
        return self.remember(m.kind, m.key, content or m.content,
                             filter=filter if filter is not None else m.filter, scope=m.scope,
                             dataset_fp=m.dataset_fp, dataset_name=m.dataset_name, columns=m.columns,
                             source="user", confirm_overwrite=True)

    def delete(self, memory_id: int, reason: str = "deleted by user") -> bool:
        with self.lock:
            m = self.get(memory_id)
            if not m or m.status != "active":
                return False
            self.conn.execute("UPDATE memories SET status='deleted', updated_at=? WHERE id=?",
                              (utcnow(), memory_id))
            self._log(memory_id, "deleted", reason)
            self.conn.commit()
            self._index_remove(memory_id)
            return True

    def forget_key(self, key: str, dataset_fp: str | None = None) -> list[int]:
        ids = [r["id"] for r in self.conn.execute(
            "SELECT id FROM memories WHERE norm_key=? AND status='active' AND "
            "(scope='global' OR dataset_fp=? OR ? IS NULL)", (normalise_key(key), dataset_fp, dataset_fp))]
        for i in ids:
            self.delete(i, reason="forgotten on request")
        return ids

    def clear(self):
        for m in self.list():
            self.delete(m.id, reason="cleared all")

    def mark_used(self, memory_ids):
        now = utcnow()
        with self.lock:
            for i in set(memory_ids):
                self.conn.execute("UPDATE memories SET use_count=use_count+1, last_used_at=? WHERE id=?",
                                  (now, i))
            self.conn.commit()

    def export(self) -> list[dict]:
        return [m.as_dict() for m in self.list(include_inactive=True)]

    # ------------------------------------------------------------------ reads
    def _applies(self, m: Memory, dataset_fp: str | None, columns: list[str] | None) -> tuple[bool, str]:
        """(usable, reason-if-stale). Unrelated-dataset memories are simply not usable."""
        if m.scope == "global" or m.dataset_fp == dataset_fp:
            missing = [c for c in m.columns if columns is not None and c not in columns]
            return (not missing, f"columns {missing} are not in this dataset" if missing else "")
        if columns is not None and m.columns and set(m.columns) <= set(columns):
            m.note = f"learned on '{m.dataset_name}', which has the same columns"
            return True, ""
        if m.columns and columns is not None:
            missing = [c for c in m.columns if c not in columns]
            return False, f"columns {missing} are not in this dataset"
        return False, ""

    def lookup(self, key: str, dataset_fp: str | None, columns: list[str] | None = None) -> Memory | None:
        rows = [self._row(r) for r in self.conn.execute(
            "SELECT * FROM memories WHERE norm_key=? AND status='active'", (normalise_key(key),))]
        rows.sort(key=lambda m: (m.dataset_fp != dataset_fp, m.scope != "dataset", -m.id))
        for m in rows:
            ok, _ = self._applies(m, dataset_fp, columns)
            if ok:
                return m
        return None

    def preference(self, name: str) -> str | None:
        r = self.conn.execute("SELECT content FROM memories WHERE kind='preference' AND norm_key=? "
                              "AND status='active' ORDER BY id DESC LIMIT 1", (normalise_key(name),)).fetchone()
        return r["content"] if r else None

    def retrieve(self, question: str, dataset_fp: str | None, columns: list[str] | None,
                 k: int = 6, min_score: float = 0.0) -> MemoryRetrieval:
        out = MemoryRetrieval()
        active = {m.id: m for m in self.list()}
        if not active:
            return out
        q_norm = f" {normalise_key(question)} "
        q_words = set(words(question))
        with self.lock:
            scored = {int(d.metadata["memory_id"]): s
                      for d, s in self.index.similarity_search_with_score(question, k=len(active))}
        ranked = []
        for mid, m in active.items():
            ok, why = self._applies(m, dataset_fp, columns)
            if not ok:
                if why:
                    out.stale.append((m, why))
                continue
            score = scored.get(mid, 0.0)
            if f" {normalise_key(m.key)} " in q_norm:
                score += 1.0  # the question uses the remembered term itself
            score += 0.05 * len(q_words & set(words(m.key + " " + m.content + " " + " ".join(m.columns))))
            if m.kind == "preference":
                score += 2.0  # cheap to include, and tools depend on them
            ranked.append(MemoryHit(m, round(score, 3)))
        ranked.sort(key=lambda h: (-h.score, -h.memory.id))
        out.hits = [h for h in ranked if h.score >= min_score][:k]
        return out


def _validate_preference(name: str, value: str) -> str | None:
    if name == "time_granularity" and value.strip().lower() not in GRANULARITIES:
        return "time_granularity must be one of daily, weekly, monthly, quarterly, yearly."
    if name == "decimal_places" and (not value.strip().isdigit() or int(value) > 10):
        return "decimal_places must be a whole number between 0 and 10."
    return None
