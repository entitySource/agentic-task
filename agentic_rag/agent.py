"""The retrieval-augmented agent.

For each question:
  1. Retrieve: search the knowledge base (uploaded documents + this dataset's
     column cards) and memory, keep what clears the relevance floor.
  2. Prompt: schema + retrieved passages (labelled [K1], [K2] ...) + memories
     ([M1] ...) go into the system prompt.
  3. Loop: the LangChain chat model (tools bound) calls analysis tools; tool
     results go back as ToolMessages until it answers. It can call
     search_knowledge to retrieve more mid-answer.
  4. Answer: text with inline citations, plus charts, the tool trace and the
     sources, streamed to the caller as events.
"""

from __future__ import annotations

import json
import re
import time
from typing import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .analysis import AnalysisContext
from .data import Dataset
from .knowledge import KnowledgeBase
from .llm import message_text
from .memory import MemoryStore
from .tools import build_tools

SYSTEM_TEMPLATE = """You are Agentic RAG, a careful data analyst. You answer questions about the user's table by \
calling tools that run real pandas and scipy analysis on it, and you use the retrieved knowledge below to \
understand the business terms behind the question.

How to work
1. Every number, ranking or claim about the data must come from a tool result in this conversation. Never \
estimate or invent figures. If a tool returns an error, read its hint, fix the arguments and try again.
2. Pick the tool that matches the question (profile, rows, aggregate, time trend, correlation, outliers, group \
comparison, derived column, chart). Use several when the question needs it.
3. Retrieved knowledge passages are labelled [K1], [K2] and so on. Use them to interpret terms, metrics and \
columns. When a sentence of your answer relies on a passage, cite it inline, e.g. "Churned means cancelled \
within the period [K2]." Cite only labels you were given or got from search_knowledge. If a term is unclear \
and the passages don't cover it, call search_knowledge before assuming.
4. Memories are labelled [M1], [M2] and so on. Apply them when the question uses their terms: pass remembered \
filters through `named_filters`, honour preferences, and trust corrections over your own assumptions. Say \
briefly which memory you applied.
5. Memory is for durable, reusable facts only. Call save_memory when the user defines a term or metric, \
corrects a fact, states how they like answers, or names a filter to reuse. Never save one-off questions, \
results or rows. If save_memory reports a conflict, ask the user which value is right.
6. If the question is ambiguous in a way that changes the answer, state the assumption you made; ask only \
when no reasonable default exists.
7. Answer concisely in markdown: the direct answer first, then the key figures (a small table is fine), then \
any caveat. Don't paste raw JSON.

Dataset
{schema}

Retrieved knowledge
{passages}

What you remember
{memories}
"""


class RagAgent:
    def __init__(self, dataset: Dataset, model: BaseChatModel, memory: MemoryStore | None = None,
                 knowledge: KnowledgeBase | None = None, use_memory: bool = True, max_steps: int = 10,
                 history_turns: int = 6, plain_model: BaseChatModel | None = None):
        self.dataset = dataset
        self.model = model
        self.memory = memory if use_memory else None
        self.knowledge = knowledge
        self.ctx = AnalysisContext(dataset=dataset, memory=self.memory, knowledge=knowledge)
        self.tools = build_tools(self.ctx, use_memory=use_memory)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.bound = model.bind_tools(self.tools)
        self.plain_model = plain_model  # same model without optional beta features, used if those are rejected
        self.max_steps = max_steps
        self.history_turns = history_turns
        self.history: list[BaseMessage] = []  # earlier turns: question + final answer text only

    def reset(self):
        self.history = []

    def _floor(self) -> float:
        emb = getattr(self.knowledge or self.memory, "embeddings", None)
        return getattr(emb, "relevance_floor", 0.0)

    def retrieve(self, question: str) -> tuple[list[dict], list[dict], list[dict]]:
        self.ctx.sources = {}
        passages, memories, stale = [], [], []
        floor = self._floor()
        if self.knowledge:
            for h in self.knowledge.search(question, self.dataset.fingerprint, k=5):
                if h.score >= floor:
                    label = self.ctx.register_source(h)
                    passages.append({"label": label, **h.as_dict()})
        if self.memory:
            r = self.memory.retrieve(question, self.dataset.fingerprint, self.dataset.columns, k=6, min_score=floor)
            for i, hit in enumerate(r.hits, 1):
                memories.append({"label": f"M{i}", "id": hit.memory.id, "kind": hit.memory.kind,
                                 "key": hit.memory.key, "text": hit.memory.render(), "score": hit.score})
            stale = [{"key": m.key, "why": why} for m, why in r.stale]
            if r.hits:
                self.memory.mark_used(h.memory.id for h in r.hits)
        return passages, memories, stale

    def system_prompt(self, passages: list[dict], memories: list[dict], stale: list[dict]) -> str:
        p_lines = [f"[{p['label']}] ({p['title']}{', ' + p['location'] if p['location'] else ''})\n{p['text']}"
                   for p in passages] or ["(nothing retrieved; call search_knowledge if a term is unclear)"]
        m_lines = [f"[{m['label']}] {m['text']}" for m in memories] or ["(nothing relevant yet)"]
        if stale:
            m_lines.append("Not applicable to this dataset, do not use: "
                           + "; ".join(f"{s['key']} ({s['why']})" for s in stale))
        return SYSTEM_TEMPLATE.format(schema=self.dataset.schema_summary(), passages="\n\n".join(p_lines),
                                      memories="\n".join(m_lines))

    def ask(self, question: str) -> Iterator[dict]:
        """Run one question, yielding events: retrieval, tool_start, tool_end, answer (or error)."""
        t_start = time.perf_counter()
        passages, memories, stale = self.retrieve(question)
        yield {"type": "retrieval", "passages": passages, "memories": memories, "stale": stale}

        system = SystemMessage(self.system_prompt(passages, memories, stale))
        turn: list[BaseMessage] = [HumanMessage(question)]
        self.ctx.applied_memory_ids = []
        charts_before = len(self.ctx.charts)
        trace, answer = [], ""

        for _ in range(self.max_steps):
            try:
                ai: AIMessage = self.bound.invoke([system, *self.history, *turn])
            except Exception as e:
                if "fallback" in str(e).lower() and self.plain_model is not None:
                    # The endpoint rejected the refusal-fallback option; carry on without it.
                    self.bound, self.plain_model = self.plain_model.bind_tools(self.tools), None
                    ai = None
                else:
                    yield {"type": "error", "message": _friendly_error(e)}
                    return
            if ai is None:
                try:
                    ai = self.bound.invoke([system, *self.history, *turn])
                except Exception as e:
                    yield {"type": "error", "message": _friendly_error(e)}
                    return
            turn.append(ai)
            if getattr(ai, "response_metadata", {}).get("stop_reason") == "refusal":
                answer = "The model declined to answer this request."
                break
            if not ai.tool_calls:
                answer = message_text(ai) or "(The model returned no text.)"
                break
            if message_text(ai):
                yield {"type": "thought", "text": message_text(ai)}
            for call in ai.tool_calls:
                yield {"type": "tool_start", "name": call["name"], "args": call["args"]}
                t0 = time.perf_counter()
                tool = self.tools_by_name.get(call["name"])
                if tool is None:
                    content = json.dumps({"error": f"Unknown tool '{call['name']}'. "
                                                   f"Available: {list(self.tools_by_name)}"})
                else:
                    try:
                        content = tool.invoke(call["args"])
                    except Exception as e:  # schema validation errors land here
                        content = json.dumps({"error": f"Invalid arguments for {call['name']}: {e}"[:1500]})
                result = json.loads(content) if content.startswith("{") and not content.endswith("[truncated]\"}") \
                    else {"raw": content}
                step = {"name": call["name"], "args": call["args"], "result": result,
                        "seconds": round(time.perf_counter() - t0, 3)}
                trace.append(step)
                yield {"type": "tool_end", **step}
                turn.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        else:
            answer = "I stopped after reaching the tool-call limit without a final answer. Try a narrower question."

        self.history += [HumanMessage(question), AIMessage(answer)]
        self.history = self.history[-2 * self.history_turns:]
        sources = [{"label": label, **hit.as_dict()} for label, hit in self.ctx.sources.items()]
        cited = sorted(set(re.findall(r"\[(K\d+)\]", answer)) & set(self.ctx.sources), key=lambda s: int(s[1:]))
        yield {"type": "answer", "text": answer, "charts": self.ctx.charts[charts_before:], "trace": trace,
               "sources": sources, "cited": cited, "memories": memories,
               "applied_memory_ids": sorted(set(self.ctx.applied_memory_ids)),
               "seconds": round(time.perf_counter() - t_start, 2)}

    def ask_sync(self, question: str) -> dict:
        final = {}
        for event in self.ask(question):
            if event["type"] in ("answer", "error"):
                final = event
        return final


def _friendly_error(e: Exception) -> str:
    text = str(e)
    name = type(e).__name__
    if "authentication" in text.lower() or "api key" in text.lower() or name in ("AuthenticationError",):
        return "The model provider rejected the API key. Paste a valid key under Model settings, or switch to Offline."
    if "connect" in text.lower() or name in ("APIConnectionError", "ConnectError"):
        return "Couldn't reach the model provider. Check your internet connection (or that Ollama is running)."
    if "rate" in text.lower() and "limit" in text.lower():
        return "The model provider is rate-limiting requests. Wait a moment and ask again."
    return f"The model call failed ({name}): {text[:400]}"
