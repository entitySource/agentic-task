"""Settings, read from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | Path = ROOT / ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b",
    "openai": "gpt-4o-mini",
    "offline": "rule-based",
}

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}


@dataclass
class Settings:
    provider: str = "offline"
    model: str = ""
    api_keys: dict = field(default_factory=dict)
    embeddings: str = "auto"  # auto | fastembed | hash
    data_dir: Path = ROOT / "data"
    store_dir: Path = ROOT / "store"

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        return cls(
            provider=os.getenv("AGENTIC_RAG_PROVIDER", "offline").lower(),
            model=os.getenv("AGENTIC_RAG_MODEL", ""),
            api_keys={p: os.getenv(env, "") for p, env in KEY_ENV.items()},
            embeddings=os.getenv("AGENTIC_RAG_EMBEDDINGS", "auto").lower(),
            data_dir=Path(os.getenv("AGENTIC_RAG_DATA_DIR", ROOT / "data")),
            store_dir=Path(os.getenv("AGENTIC_RAG_STORE_DIR", ROOT / "store")),
        )

    @property
    def model_name(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    def key_for(self, provider: str | None = None) -> str:
        return self.api_keys.get(provider or self.provider, "") or ""
