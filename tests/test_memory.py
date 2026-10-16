"""Memory: write policy, versioning, conflicts, scope and retrieval."""

from agentic_rag.analysis import AnalysisContext, group_aggregate, save_memory, time_series


def test_update_supersedes_and_keeps_history(memory):
    memory.remember("definition", "Active customers", "status is not cancelled", dataset_fp="fp")
    res = memory.remember("definition", "active customer", "status is active or trial", dataset_fp="fp")
    assert res.status == "updated"
    assert len([m for m in memory.list() if m.key.lower().startswith("active")]) == 1
    history = memory.history(res.memory.id)
    assert [h["memory"]["status"] for h in history] == ["active", "superseded"]


def test_agent_cannot_silently_override_the_user(memory):
    memory.remember("correction", "revenue unit", "revenue is in thousands of INR", dataset_fp="fp", source="user")
    res = memory.remember("correction", "revenue unit", "revenue is in INR", dataset_fp="fp", source="agent")
    assert res.status == "conflict"
    assert memory.lookup("revenue unit", "fp").content == "revenue is in thousands of INR"


def test_rejects_dumps_and_bad_preferences(memory):
    assert memory.remember("definition", "x", "y" * 700).status == "rejected"
    assert memory.remember("preference", "time granularity", "fortnightly").status == "rejected"
    assert memory.remember("feeling", "x", "y").status == "rejected"
    assert memory.remember("preference", "Time Granularity", "Weekly").memory.content == "weekly"


def test_retrieval_uses_the_term_and_flags_stale(memory, subs, sales):
    ctx = AnalysisContext(subs, memory)
    save_memory(ctx, "definition", "active customers", "status is not cancelled",
                filter=[{"column": "status", "op": "!=", "value": "Cancelled"}])
    r = memory.retrieve("how many active customers per region", subs.fingerprint, subs.columns)
    assert r.hits[0].memory.key == "active customers"
    assert r.hits[0].memory.filter[0]["value"] == "cancelled"  # normalised to the real label
    other = memory.retrieve("active customers", sales.fingerprint, sales.columns)
    assert not other.hits and other.stale and "status" in other.stale[0][1]


def test_definition_changes_tool_results(memory, subs):
    ctx = AnalysisContext(subs, memory)
    save_memory(ctx, "definition", "active customers", "status is not cancelled",
                filter=[{"column": "status", "op": "!=", "value": "cancelled"}])
    out = group_aggregate(ctx, [{"column": "*", "agg": "count"}], named_filters=["Active Customers"])
    assert out["result"][0]["row_count"] == (subs.df["status"] != "cancelled").sum()
    assert ctx.applied_memory_ids and out["applied_memory"]


def test_granularity_preference_changes_time_series(memory, subs):
    ctx = AnalysisContext(subs, memory)
    assert time_series(ctx, "signup_date")["frequency"] == "monthly"
    memory.remember("preference", "time_granularity", "weekly")
    out = time_series(ctx, "signup_date")
    assert out["frequency"] == "weekly" and "remembered" in out["frequency_source"]


def test_edit_delete_and_persistence(tmp_path, embeddings):
    from agentic_rag.memory import MemoryStore

    store = MemoryStore(tmp_path / "m.db", embeddings)
    m = store.remember("column_note", "nps", "nps is a 0 to 10 score").memory
    edited = store.edit(m.id, content="nps is a 0-10 survey score")
    assert edited.status == "updated"
    reopened = MemoryStore(tmp_path / "m.db", embeddings)
    assert reopened.retrieve("what is nps", None, None).hits[0].memory.content.startswith("nps is a 0-10")
    assert reopened.delete(edited.memory.id) and not reopened.list()
