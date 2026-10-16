from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agentic_rag.data import load_table
from agentic_rag.embeddings import HashingEmbeddings
from agentic_rag.knowledge import KnowledgeBase
from agentic_rag.memory import MemoryStore

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def embeddings():
    return HashingEmbeddings()


@pytest.fixture
def subs():
    return load_table(ROOT / "data" / "subscriptions.csv")


@pytest.fixture
def sales():
    return load_table(ROOT / "data" / "retail_sales.xlsx")


@pytest.fixture
def memory(embeddings):
    return MemoryStore(":memory:", embeddings)


@pytest.fixture
def knowledge(embeddings):
    kb = KnowledgeBase(":memory:", embeddings)
    glossary = ROOT / "knowledge" / "business-glossary.md"
    kb.add_document(glossary.name, glossary.read_bytes())
    return kb


class ScriptedChatModel(BaseChatModel):
    """A fake LangChain chat model that replays a fixed list of AIMessages."""

    script: list = []
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        msg = self.script.pop(0) if self.script else AIMessage(content="(script finished)")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}])


@pytest.fixture
def scripted():
    def make(*steps):
        return ScriptedChatModel(script=list(steps), seen=[])
    return make
