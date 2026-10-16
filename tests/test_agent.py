"""The RAG agent loop, driven by a scripted LangChain chat model, plus the offline model and the API."""

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agentic_rag.agent import RagAgent
from agentic_rag.llm import OfflineChatModel

from .conftest import call


def test_retrieved_context_goes_into_the_prompt_and_citations_are_kept(subs, memory, knowledge, scripted):
    knowledge.index_dataset(subs)
    model = scripted(
        call("group_aggregate", metrics=[{"column": "*", "agg": "count"}], group_by=["status"]),
        AIMessage(content="Churn means a cancelled status [K1]. Basic churns most. [K99]"))
    agent = RagAgent(subs, model, memory, knowledge)
    events = list(agent.ask("What does churn rate mean and which plan churns most?"))
    retrieval, answer = events[0], events[-1]
    assert retrieval["type"] == "retrieval"
    assert any("Churn and churn rate" in p["location"] for p in retrieval["passages"])
    system = model.seen[0][0]
    assert isinstance(system, SystemMessage) and "[K1]" in system.content and "Retrieved knowledge" in system.content
    assert answer["type"] == "answer" and answer["cited"] == ["K1"]  # K99 was never given, so it isn't a source
    assert [t["name"] for t in answer["trace"]] == ["group_aggregate"]


def test_tool_errors_are_fed_back_so_the_model_can_retry(subs, memory, knowledge, scripted):
    model = scripted(
        call("detect_outliers", column="revenue"),
        call("detect_outliers", column="monthly_revenue"),
        AIMessage(content="Found the outliers."))
    agent = RagAgent(subs, model, memory, knowledge)
    answer = agent.ask_sync("Any odd revenue?")
    assert "error" in answer["trace"][0]["result"] and "monthly_revenue" in answer["trace"][0]["result"]["error"]
    assert answer["trace"][1]["result"]["outlier_count"] >= 4
    tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert json.loads(tool_msgs[0].content)["error"]


def test_memory_from_one_session_changes_the_next(subs, memory, knowledge, scripted):
    first = RagAgent(subs, scripted(
        call("save_memory", kind="definition", key="active customers", content="status is not cancelled",
             filter=[{"column": "status", "op": "!=", "value": "cancelled"}]),
        AIMessage(content="Saved.")), memory, knowledge)
    first.ask_sync("Active customers excludes cancelled accounts. Remember that.")

    model = scripted(call("group_aggregate", metrics=[{"column": "*", "agg": "count"}],
                          named_filters=["active customers"]), AIMessage(content="925 active customers."))
    second = RagAgent(subs, model, memory, knowledge)
    events = list(second.ask("How many active customers are there?"))
    assert events[0]["memories"][0]["key"] == "active customers"
    assert "[M1] definition 'active customers'" in model.seen[0][0].content
    answer = events[-1]
    assert answer["applied_memory_ids"] and answer["trace"][0]["result"]["result"][0]["row_count"] == 925


def test_conversation_history_is_kept_between_turns(subs, memory, knowledge, scripted):
    model = scripted(AIMessage(content="First."), AIMessage(content="Second."))
    agent = RagAgent(subs, model, memory, knowledge)
    agent.ask_sync("one")
    agent.ask_sync("two")
    assert [m.content for m in model.seen[1][1:]] == ["one", "First.", "two"]
    agent.reset()
    assert agent.history == []


def test_memory_off_hides_memory_tools(subs, memory, knowledge, scripted):
    agent = RagAgent(subs, scripted(AIMessage(content="ok")), memory, knowledge, use_memory=False)
    assert "save_memory" not in agent.tools_by_name and agent.memory is None


def test_model_errors_become_an_error_event(subs, memory, knowledge):
    class Broken(OfflineChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("authentication_error: invalid x-api-key")

    events = list(RagAgent(subs, Broken(), memory, knowledge).ask("hi"))
    assert events[-1]["type"] == "error" and "API key" in events[-1]["message"]


def test_offline_model_answers_definitions_from_the_knowledge_base(subs, memory, knowledge):
    knowledge.index_dataset(subs)
    answer = RagAgent(subs, OfflineChatModel(), memory, knowledge).ask_sync("What counts as an active customer?")
    assert answer["trace"][0]["name"] == "search_knowledge"
    assert "Active customer" in answer["text"] and answer["cited"]


def test_offline_model_routes_data_questions(subs, memory, knowledge):
    agent = RagAgent(subs, OfflineChatModel(), memory, knowledge)
    assert agent.ask_sync("What drives nps?")["trace"][0]["name"] == "correlation"
    chart = agent.ask_sync("Chart monthly_revenue by plan")
    assert chart["trace"][0]["args"]["kind"] == "bar" and chart["charts"]


def test_api_end_to_end(tmp_path):
    from pathlib import Path

    from fastapi.testclient import TestClient

    from agentic_rag.config import Settings
    from agentic_rag.server import create_app

    root = Path(__file__).resolve().parent.parent
    client = TestClient(create_app(Settings(provider="offline", embeddings="hash", data_dir=root / "data",
                                            store_dir=tmp_path)))
    state = client.get("/api/state").json()
    assert {d["name"] for d in state["datasets"]} == {"subscriptions", "retail_sales"}
    assert state["counts"]["documents"] == 1
    client.post("/api/datasets/subscriptions/activate")
    lines = client.post("/api/ask", json={"question": "What drives nps?"}).text.splitlines()
    events = [json.loads(line) for line in lines]
    assert events[0]["type"] == "retrieval" and events[-1]["type"] == "answer"
    mem = client.post("/api/memory", json={"kind": "column_note", "key": "nps", "content": "0-10 survey"}).json()
    assert mem["result"]["status"] == "created"
    up = client.post("/api/knowledge/upload", files={"file": ("terms.md", b"# Terms\n## Whale\nA customer paying over 1M INR.")})
    assert up.status_code == 200
    hits = client.get("/api/knowledge/search", params={"q": "what is a whale customer"}).json()["hits"]
    assert hits[0]["title"] == "terms.md"
    assert client.post("/api/settings", json={"provider": "nope"}).status_code == 400
