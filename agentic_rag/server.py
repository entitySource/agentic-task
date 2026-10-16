"""FastAPI backend for the web app (single user, runs on localhost).

Run with:  python -m agentic_rag   (serves http://localhost:8501)
"""

# No `from __future__ import annotations` here: FastAPI must see the request
# models defined inside create_app() as real classes, not strings.
import json
import logging
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import RagAgent
from .config import DEFAULT_MODELS, KEY_ENV, ROOT, Settings
from .data import TABLE_SUFFIXES, Dataset, column_kind, load_table
from .embeddings import make_embeddings
from .knowledge import DOC_SUFFIXES, KnowledgeBase
from .llm import make_chat_model
from .memory import KINDS, MemoryStore

log = logging.getLogger("agentic_rag")
WEB = ROOT / "web"


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.RLock()
        self.ask_lock = threading.Lock()  # a plain Lock: the stream may resume on another thread
        settings.store_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = make_embeddings(settings.embeddings, cache_dir=str(settings.store_dir / "models"))
        self.memory = MemoryStore(settings.store_dir / "memory.db", self.embeddings)
        self.knowledge = KnowledgeBase(settings.store_dir / "knowledge.db", self.embeddings)
        self.use_memory = True
        self.datasets: dict[str, Dataset] = {}
        self.agents: dict[str, RagAgent] = {}
        self.active: str | None = None
        self._seed_knowledge()
        for path in sorted(settings.data_dir.glob("*")):
            if path.suffix.lower() in TABLE_SUFFIXES:
                try:
                    self.add_dataset(load_table(path))
                except Exception as e:
                    log.warning("Skipped %s: %s", path.name, e)

    def _seed_knowledge(self):
        titles = {d["title"] for d in self.knowledge.list_documents()}
        for path in sorted((ROOT / "knowledge").glob("*")):
            if path.suffix.lower() in DOC_SUFFIXES and path.name not in titles:
                self.knowledge.add_document(path.name, path.read_bytes())

    def add_dataset(self, ds: Dataset):
        with self.lock:
            self.datasets[ds.name] = ds
            self.agents.pop(ds.name, None)
            self.knowledge.index_dataset(ds)
            self.active = self.active or ds.name

    def agent(self, name: str | None = None) -> RagAgent:
        name = name or self.active
        if not name or name not in self.datasets:
            raise HTTPException(400, "Load a dataset first.")
        with self.lock:
            if name not in self.agents:
                plain = None
                if self.settings.provider == "anthropic":
                    plain = make_chat_model(self.settings, fallbacks=False)
                self.agents[name] = RagAgent(self.datasets[name], make_chat_model(self.settings), self.memory,
                                             self.knowledge, use_memory=self.use_memory, plain_model=plain)
            return self.agents[name]


def _dataset_info(ds: Dataset, active: bool) -> dict:
    return {"name": ds.name, "rows": len(ds.df), "columns": len(ds.columns), "fingerprint": ds.fingerprint,
            "active": active, "source": Path(ds.source).name}


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Agentic RAG")
    state = AppState(settings or Settings.from_env())
    app.state.agentic = state

    # ------------------------------------------------------------------ app state
    @app.get("/api/state")
    def get_state():
        s = state.settings
        active = state.datasets.get(state.active) if state.active else None
        return {
            "provider": s.provider, "model": s.model_name, "use_memory": state.use_memory,
            "providers": [{"id": p, "default_model": m, "needs_key": p in KEY_ENV, "key_set": bool(s.key_for(p))}
                          for p, m in DEFAULT_MODELS.items()],
            "embeddings": getattr(state.embeddings, "name", "custom"),
            "datasets": [_dataset_info(d, d.name == state.active) for d in state.datasets.values()],
            "schema": [{"column": c, "kind": k} for c, k in active.kinds().items()] if active else [],
            "counts": {"documents": len([d for d in state.knowledge.list_documents() if d["kind"] == "document"]),
                       "memories": len(state.memory.list())},
        }

    class SettingsIn(BaseModel):
        provider: str | None = None
        model: str | None = None
        api_key: str | None = None
        use_memory: bool | None = None

    @app.post("/api/settings")
    def update_settings(body: SettingsIn):
        with state.lock:
            s = state.settings
            if body.provider:
                if body.provider not in DEFAULT_MODELS:
                    raise HTTPException(400, f"Unknown provider '{body.provider}'.")
                if body.provider != s.provider:
                    s.model = ""
                s.provider = body.provider
            if body.model is not None:
                s.model = body.model.strip()
            if body.api_key:
                s.api_keys[s.provider] = body.api_key.strip()
            if body.use_memory is not None:
                state.use_memory = body.use_memory
            try:
                make_chat_model(s)
            except Exception as e:
                raise HTTPException(400, f"Couldn't set up {s.provider}: {e}")
            state.agents.clear()
        return get_state()

    # ------------------------------------------------------------------ datasets
    @app.post("/api/datasets/upload")
    async def upload_dataset(file: UploadFile = File(...)):
        payload = await file.read()
        try:
            ds = load_table((file.filename, payload))
        except Exception as e:
            raise HTTPException(400, f"Couldn't read {file.filename}: {e}")
        safe = re.sub(r"[^\w.\- ]", "_", Path(file.filename).name)
        (state.settings.data_dir / safe).write_bytes(payload)
        state.add_dataset(ds)
        state.active = ds.name
        return get_state()

    @app.post("/api/datasets/{name}/activate")
    def activate(name: str):
        if name not in state.datasets:
            raise HTTPException(404, f"No dataset called '{name}'.")
        state.active = name
        return get_state()

    @app.get("/api/datasets/{name}/preview")
    def preview(name: str, rows: int = 12):
        ds = state.datasets.get(name)
        if not ds:
            raise HTTPException(404, f"No dataset called '{name}'.")
        return {"name": ds.name, "columns": [{"column": c, "kind": column_kind(ds.df[c]),
                                              "description": ds.describe_column(c)} for c in ds.columns],
                "rows": ds.preview(min(rows, 50)), "total_rows": len(ds.df)}

    # ------------------------------------------------------------------ asking
    class AskIn(BaseModel):
        question: str

    @app.post("/api/ask")
    def ask(body: AskIn):
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "Type a question first.")
        agent = state.agent()

        def stream():
            with state.ask_lock:
                for event in agent.ask(question):
                    yield json.dumps(event, default=str) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/api/reset")
    def reset():
        if state.active and state.active in state.agents:
            state.agents[state.active].reset()
        return {"ok": True}

    # ------------------------------------------------------------------ knowledge
    @app.get("/api/knowledge")
    def list_knowledge():
        return {"documents": state.knowledge.list_documents()}

    @app.post("/api/knowledge/upload")
    async def upload_knowledge(file: UploadFile = File(...)):
        try:
            doc = state.knowledge.add_document(file.filename, await file.read())
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"document": doc, "documents": state.knowledge.list_documents()}

    @app.get("/api/knowledge/search")
    def search_knowledge(q: str, k: int = 5):
        fp = state.datasets[state.active].fingerprint if state.active else None
        return {"hits": [h.as_dict() for h in state.knowledge.search(q, fp, k=k)],
                "floor": getattr(state.embeddings, "relevance_floor", 0)}

    @app.get("/api/knowledge/{doc_id}")
    def document(doc_id: str):
        doc = state.knowledge.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "No such document.")
        return {"document": doc, "chunks": state.knowledge.document_chunks(doc_id)}

    @app.delete("/api/knowledge/{doc_id}")
    def delete_document(doc_id: str):
        if doc_id.startswith("ds-"):
            raise HTTPException(400, "Column cards are rebuilt from the dataset; remove the dataset instead.")
        if not state.knowledge.delete_document(doc_id):
            raise HTTPException(404, "No such document.")
        return {"documents": state.knowledge.list_documents()}

    # ------------------------------------------------------------------ memory
    def _mem_list():
        return {"memories": [m.as_dict() for m in state.memory.list()], "kinds": KINDS}

    @app.get("/api/memory")
    def list_memory():
        return _mem_list()

    class MemoryIn(BaseModel):
        kind: str
        key: str
        content: str
        scope: str = "global"

    @app.post("/api/memory")
    def add_memory(body: MemoryIn):
        ds = state.datasets.get(state.active) if state.active else None
        res = state.memory.remember(body.kind, body.key, body.content, scope=body.scope,
                                    dataset_fp=ds.fingerprint if ds else None,
                                    dataset_name=ds.name if ds else None, source="user", confirm_overwrite=True)
        if res.status == "rejected":
            raise HTTPException(400, res.message)
        return {**_mem_list(), "result": res.as_dict()}

    class MemoryEdit(BaseModel):
        content: str | None = None
        key: str | None = None

    @app.patch("/api/memory/{memory_id}")
    def edit_memory(memory_id: int, body: MemoryEdit):
        res = state.memory.edit(memory_id, content=body.content, key=body.key)
        if res.status == "rejected":
            raise HTTPException(400, res.message)
        return {**_mem_list(), "result": res.as_dict()}

    @app.delete("/api/memory/{memory_id}")
    def delete_memory(memory_id: int):
        if not state.memory.delete(memory_id):
            raise HTTPException(404, "No such memory.")
        return _mem_list()

    @app.get("/api/memory/{memory_id}/history")
    def memory_history(memory_id: int):
        return {"history": state.memory.history(memory_id)}

    @app.delete("/api/memory")
    def clear_memory():
        state.memory.clear()
        return _mem_list()

    @app.get("/api/memory-export")
    def export_memory():
        return JSONResponse(state.memory.export(),
                            headers={"Content-Disposition": "attachment; filename=agentic-rag-memory.json"})

    # ------------------------------------------------------------------ web
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/_stcore/health")
    @app.get("/health")
    def health():
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app
