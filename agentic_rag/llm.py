"""Chat models, all behind LangChain's `BaseChatModel` interface.

    anthropic  Claude via langchain-anthropic (default model claude-opus-5-5)
    groq       open models on Groq's OpenAI-compatible endpoint
    ollama     local models through Ollama's OpenAI-compatible endpoint
    openai     OpenAI, or any compatible server (AGENTIC_RAG_BASE_URL)
    offline    a rule-based stand-in so the app works with no key or internet
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import DEFAULT_MODELS, Settings

BASE_URLS = {"groq": "https://api.groq.com/openai/v1", "ollama": "http://localhost:11434/v1"}


def make_chat_model(settings: Settings, fallbacks: bool = True) -> BaseChatModel:
    provider, model = settings.provider, settings.model_name
    if provider in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=16000, thinking={"type": "adaptive"},
            output_config={"effort": os.getenv("AGENTIC_RAG_EFFORT", "medium")}, max_retries=2)
        if settings.key_for("anthropic"):
            kwargs["api_key"] = settings.key_for("anthropic")
        if fallbacks and model in ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"):
            # Server-side fallback: if a safety classifier declines, the API retries on another model.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["model_kwargs"] = {"fallbacks": "default"}
        return ChatAnthropic(**kwargs)
    if provider in ("groq", "ollama", "openai"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, temperature=0.1, max_retries=2,
            api_key=settings.key_for(provider) or ("ollama" if provider == "ollama" else None),
            base_url=os.getenv("AGENTIC_RAG_BASE_URL") or BASE_URLS.get(provider))
    if provider in ("offline", "demo"):
        return OfflineChatModel()
    raise ValueError(f"Unknown provider '{provider}'. Use anthropic, groq, ollama, openai or offline.")


def message_text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Offline model: keyword routing over the schema and context in the system prompt.

_COL_LINE = re.compile(r"^- (.+?) \[(numeric|categorical|datetime|text|boolean)\]", re.M)
_MEM_FILTER = re.compile(r"^\[M\d+\] (?:definition|named_filter) '(.+?)':.*\| filter=", re.M)
_PASSAGE = re.compile(r"^\[(K\d+)\] \((.+?)\)\n(.+?)(?=\n\[K\d+\] |\n\n|\Z)", re.M | re.S)
_DEFINE = re.compile(r"(?:define|remember)\s+[\"']?(?P<name>[\w ]+?)[\"']?\s+(?:as|means|=)\s+(?P<col>[\w ]+?)\s*"
                     r"(?P<op>==|!=|>=|<=|>|<|=| is not | is )\s*(?P<val>.+?)[.]?$", re.I)
_PREF = re.compile(r"\b(daily|weekly|monthly|quarterly|yearly)\b", re.I)
_EXPLAIN = re.compile(r"^\s*(what|how)\s+(does|do|is|are)\b.*\b(mean|means|defined|definition|calculated|measured)\b"
                      r"|^\s*(define|explain|what is meant by|what counts as|what do you mean by)\b"
                      r"|^\s*what(?:'s| is| are)\s+(?:an?\s+|the\s+)?[\w\s-]{2,40}\??\s*$", re.I)
_DATA_WORDS = re.compile(r"\b(how many|count of|number of|total|average|mean|median|sum|trend|over time|chart|plot|"
                         r"by|per|top|highest|lowest|most|least|outlier|correlat|compare)\b", re.I)


class OfflineChatModel(BaseChatModel):
    """Deterministic stand-in for an LLM. Labelled as offline in every answer."""

    @property
    def _llm_type(self) -> str:
        return "offline-rules"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        question = str(messages[last_human].content)
        tool_msgs = [m for m in messages[last_human + 1:] if isinstance(m, ToolMessage)]
        calls = {c["id"]: c for m in messages[last_human + 1:] if isinstance(m, AIMessage) for c in m.tool_calls}
        if tool_msgs:
            msg = AIMessage(content=_summarise(tool_msgs[-1], calls.get(tool_msgs[-1].tool_call_id), system))
        else:
            msg = _route(question, system)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _schema(system: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _COL_LINE.finditer(system)}


def _passages(system: str) -> list[tuple[str, str, str]]:
    return [(m.group(1), m.group(2), m.group(3).strip()) for m in _PASSAGE.finditer(system)]


def _mentioned(question: str, cols: dict[str, str], kinds: tuple) -> list[str]:
    q = question.lower()
    found = []
    for c, k in cols.items():
        for form in (c.lower(), c.lower().replace("_", " ")):
            if k in kinds and form in q:
                found.append((q.find(form), c))
                break
    return [c for _, c in sorted(found)]


def _value(v: str):
    v = v.strip().strip("'\"")
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def _call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_" + uuid.uuid4().hex[:8],
                                              "type": "tool_call"}])


def _route(question: str, system: str) -> AIMessage:
    cols = _schema(system)
    q = question.lower()
    for c in sorted(cols, key=len, reverse=True):  # so "monthly_revenue" doesn't read as "monthly"
        q = q.replace(c.lower(), " _col_ ").replace(c.lower().replace("_", " "), " _col_ ")
    nums = _mentioned(question, cols, ("numeric",))
    cats = _mentioned(question, cols, ("categorical", "boolean", "text"))
    dates = [c for c, k in cols.items() if k == "datetime"]
    named = [n for n in _MEM_FILTER.findall(system) if n.lower() in question.lower()]
    common = {"named_filters": named} if named else {}

    if m := _DEFINE.search(question):
        op = {"=": "==", " is ": "==", " is not ": "!="}.get(m["op"], m["op"].strip())
        c = m["col"].strip()
        return _call("save_memory", kind="definition", key=m["name"].strip(),
                     content=f"{m['name'].strip()} means rows where {c} {op} {m['val'].strip()}",
                     filter=[{"column": c, "op": op, "value": _value(m["val"])}], source="user")
    if (m := _PREF.search(question)) and any(w in q for w in ("prefer", "from now on", "always use")):
        return _call("save_memory", kind="preference", key="time_granularity", content=m[1].lower(),
                     scope="global", source="user")
    if "forget" in q:
        key = re.sub(r".*forget\s+(?:about\s+)?(?:the\s+)?(?:definition\s+of\s+)?", "",
                     question.lower()).strip(" ?.'\"")
        return _call("forget_memory", key=key)
    if (_EXPLAIN.search(question) and not _DATA_WORDS.search(re.sub(r"\bmeant by|mean by\b", "", q))) or q.startswith(("search ", "look up", "find docs")):
        return _call("search_knowledge", query=question)
    if any(w in q for w in ("outlier", "anomal", "unusual")) and nums:
        return _call("detect_outliers", column=nums[0], **common)
    if any(w in q for w in ("correlat", "relationship", "related", "drive")):
        return _call("correlation", **({"target": nums[0]} if nums else {}), **common)
    if any(w in q for w in ("chart", "plot", "graph", "visual")):
        if dates and any(w in q for w in ("over time", "trend", "month", "week", "quarter")):
            return _call("create_chart", kind="line", x=dates[0], **({"y": nums[0]} if nums else {}), **common)
        if cats:
            return _call("create_chart", kind="bar", x=cats[0], **({"y": nums[0]} if nums else {}), **common)
        if nums:
            return _call("create_chart", kind="hist", x=nums[0], **common)
    if dates and any(w in q for w in ("trend", "over time", "per month", "monthly", "weekly", "by month",
                                      "by week", "per week", "growth", "quarter")):
        freq = next((f for w, f in [("quarter", "quarterly"), ("week", "weekly"), ("month", "monthly"),
                                    ("year", "yearly"), ("daily", "daily")] if w in q), None)
        return _call("time_series", date_column=dates[0], value_column=nums[0] if nums else "*",
                     agg="sum" if nums else "count", **({"freq": freq} if freq else {}), **common)
    if any(w in q for w in ("compare", "differ", " vs ", "versus", "significant")) and cats and (nums or len(cats) > 1):
        return _call("compare_groups", group_column=cats[0], value_column=nums[0] if nums else cats[1], **common)
    agg = next((a for w, a in [("average", "mean"), ("mean", "mean"), ("median", "median"), ("total", "sum"),
                               ("sum", "sum"), ("highest", "max"), ("max", "max"), ("lowest", "min"),
                               ("min", "min")] if w in q), None)
    by = cats if re.search(r"\b(by|per|each|across|which)\b", q) else []
    if agg and nums:
        return _call("group_aggregate", metrics=[{"column": nums[0], "agg": agg}],
                     **({"group_by": by[:1]} if by else {}), **common)
    if any(w in q for w in ("how many", "count", "number of")):
        return _call("group_aggregate", metrics=[{"column": "*", "agg": "count"}],
                     **({"group_by": by[:1], "include_share": True} if by else {}), **common)
    if any(w in q for w in ("show", "list", "which rows", "top", "rows")):
        return _call("filter_rows", limit=10, **common)
    if _passages(system) and not (nums or cats):
        return _call("search_knowledge", query=question)
    return _call("dataset_overview")


def _summarise(tool_msg: ToolMessage, call: dict | None, system: str) -> str:
    name = (call or {}).get("name", "tool")
    try:
        r = json.loads(tool_msg.content)
    except (json.JSONDecodeError, TypeError):
        r = {"raw": str(tool_msg.content)}
    lines = [f"_Offline mode: rule-based routing, no language model. Tool used: `{name}`._", ""]
    if "error" in r:
        return "\n".join(lines + [f"The tool reported a problem: {r['error']}"])
    if name == "search_knowledge":
        passages = r.get("passages", [])
        docs = [p for p in passages if not p["source"].endswith(" columns")] or passages
        if docs:
            top = max(p.get("score", 0) for p in docs)
            docs = [p for p in docs if p.get("score", 0) >= top - 0.03]
        mems = [m for m in r.get("memories", []) if not m["memory"].startswith("preference")]
        if not docs and not mems:
            return "\n".join(lines + ["I found nothing in the knowledge base about that. Upload a glossary or "
                                      "data dictionary under Knowledge, or ask about the data itself."])
        for p in docs[:2]:
            lines.append(f"- From **{p['source']}**{' (' + p['where'] + ')' if p.get('where') else ''}: "
                         f"{_first_sentences(p['text'], 3)} [{p['label']}]")
        for m in mems[:2]:
            lines.append(f"- You told me: {m['memory']}")
        return "\n".join(lines)
    if name == "create_chart":
        lines.append("The chart is below.")
    if r.get("applied_memory"):
        lines.append("Applied memory: " + "; ".join(r["applied_memory"]))
    if "rows_after_filters" in r and r["rows_after_filters"] != r.get("rows_total"):
        lines.append(f"Rows after filters: {r['rows_after_filters']:,} of {r['rows_total']:,}")
    if r.get("frequency"):
        lines.append(f"Frequency: {r['frequency']} ({r.get('frequency_source')})")
    for key in ("result", "series", "rows", "most_extreme", "top_pairs", "correlations_with_target"):
        if isinstance(r.get(key), list) and r[key]:
            lines += ["", _table(r[key])]
    if name == "dataset_overview":
        lines.append(f"{r['rows']:,} rows, {r['columns']} columns.")
        lines += ["", _table([{k: v for k, v in p.items() if k in ("column", "kind", "missing", "unique", "mean")}
                              for p in r["profile"]])]
    for key in ("groups", "row_share_%"):
        if isinstance(r.get(key), dict) and all(isinstance(v, dict) for v in r[key].values()):
            lines += ["", _table([{"group": k, **v} for k, v in r[key].items()])]
    for key in ("trend", "t_test_p_value", "mann_whitney_p_value", "p_value", "cohens_d", "significant_at_5%",
                "interpretation", "outlier_count", "bounds", "status", "message", "memory", "previous", "deleted",
                "caution"):
        if key in r:
            value = r[key]
            if isinstance(value, dict):
                value = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in value.items())
            lines.append(f"- {key.replace('_', ' ').replace('%', ' %').strip()}: {value}")
    return "\n".join(lines)


def _first_sentences(text: str, n: int = 2) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"^#+\s*", "", text, flags=re.M)).strip()
    parts = re.split(r"(?<=[.!?])\s", text)
    return " ".join(parts[:n])[:400]


def _table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(row.get(c, "")) for c in cols) + " |" for row in rows[:15]]
    return "\n".join(out)


__all__ = ["make_chat_model", "message_text", "OfflineChatModel", "DEFAULT_MODELS"]
