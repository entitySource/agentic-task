"""Analysis tools on both sample datasets, called through the LangChain tool wrappers."""

import json

import pandas as pd
import pytest

from agentic_rag.analysis import AnalysisContext
from agentic_rag.tools import ANALYSIS_TOOLS, build_tools


def tools_for(ds, memory=None, knowledge=None):
    ctx = AnalysisContext(ds, memory, knowledge)
    return ctx, {t.name: t for t in build_tools(ctx)}


def run(tools, tool_name, **args):
    return json.loads(tools[tool_name].invoke(args))


def test_at_least_eight_distinct_analysis_tools(subs):
    _, tools = tools_for(subs)
    assert len(ANALYSIS_TOOLS) >= 8
    assert {"search_knowledge", "save_memory", "forget_memory"} <= set(tools)


def test_type_inference_dates_and_excel(subs, sales):
    assert subs.kinds()["signup_date"] == "datetime"
    assert sales.kinds()["Order Date"] == "datetime"
    assert sales.df["Order Date"].dt.month.max() == 12  # day-first dates parsed correctly


def test_group_aggregate_matches_pandas(subs):
    _, tools = tools_for(subs)
    out = run(tools, "group_aggregate", metrics=[{"column": "monthly_revenue", "agg": "sum"}], group_by=["plan"])
    expected = subs.df.groupby("plan")["monthly_revenue"].sum().round(2).to_dict()
    assert {r["plan"]: r["sum_monthly_revenue"] for r in out["result"]} == pytest.approx(expected)


def test_filters_are_case_insensitive_and_give_hints(subs):
    _, tools = tools_for(subs)
    out = run(tools, "filter_rows", filters=[{"column": "region", "op": "==", "value": "north"}], limit=3)
    assert out["match_count"] == (subs.df["region"] == "North").sum()
    bad = run(tools, "filter_rows", filters=[{"column": "regoin", "op": "==", "value": "North"}])
    assert "region" in bad["error"]
    bad = run(tools, "filter_rows", filters=[{"column": "plan", "op": "==", "value": "Platinum"}])
    assert "Valid values" in bad["error"]


def test_schema_validation_errors_come_back_as_errors(subs):
    _, tools = tools_for(subs)
    with pytest.raises(Exception):  # pydantic rejects an unknown op before the tool runs
        tools["filter_rows"].invoke({"filters": [{"column": "region", "op": "approx", "value": 1}]})


def test_time_series_and_partial_period(subs):
    _, tools = tools_for(subs)
    out = run(tools, "time_series", date_column="signup_date", freq="monthly")
    assert out["frequency"] == "monthly" and "trend" in out and out["periods"] > 12


def test_correlation_finds_planted_relationship(subs):
    _, tools = tools_for(subs)
    out = run(tools, "correlation", target="nps")
    top = out["correlations_with_target"][0]
    assert top["column"] == "support_tickets" and top["r"] < -0.4


def test_outliers_find_planted_contracts(subs):
    _, tools = tools_for(subs)
    out = run(tools, "detect_outliers", column="monthly_revenue")
    assert out["outlier_count"] >= 4 and out["most_extreme"][0]["plan"] == "Enterprise"


def test_compare_groups_numeric_and_categorical(subs):
    _, tools = tools_for(subs)
    num = run(tools, "compare_groups", group_column="plan", value_column="nps", group_a="Basic", group_b="Pro")
    assert "cohens_d" in num and num["significant_at_5%"]
    cat = run(tools, "compare_groups", group_column="plan", value_column="status", group_a="basic", group_b="pro")
    assert cat["test"].startswith("chi-square") and cat["significant_at_5%"]


def test_derive_column_is_safe(subs):
    _, tools = tools_for(subs)
    ok = run(tools, "derive_column", name="revenue_per_seat", expression="monthly_revenue / seats")
    assert ok["created_column"] == "revenue_per_seat"
    for bad in ["seats.__class__", "__import__('os')", "abs(seats)", "open('x')"]:
        assert "error" in run(tools, "derive_column", name="x", expression=bad)


def test_chart_returns_drawable_data(subs, sales):
    ctx, tools = tools_for(subs)
    run(tools, "create_chart", kind="bar", x="plan", y="monthly_revenue")
    run(tools, "create_chart", kind="hist", x="nps")
    ctx2, tools2 = tools_for(sales)
    run(tools2, "create_chart", kind="line", x="Order Date", y="Revenue", freq="quarterly")
    assert [c["kind"] for c in ctx.charts] == ["bar", "hist"]
    assert [p["label"] for p in ctx2.charts[0]["points"]] == ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]


def test_same_tools_on_a_different_dataset(sales):
    _, tools = tools_for(sales)
    out = run(tools, "group_aggregate", metrics=[{"column": "Revenue", "agg": "mean"}], group_by=["Store"], top_n=1)
    best = sales.df.groupby("Store")["Revenue"].mean().idxmax()
    assert out["result"][0]["Store"] == best
    q = run(tools, "time_series", date_column="Order Date", value_column="Revenue", freq="quarterly")
    assert q["trend"]["peak_period"] == "2025Q4"  # planted festive-season bump


def test_search_knowledge_registers_citable_sources(subs, knowledge):
    knowledge.index_dataset(subs)
    ctx, tools = tools_for(subs, knowledge=knowledge)
    out = run(tools, "search_knowledge", query="how is NPS scored")
    assert out["passages"][0]["label"] == "K1"
    assert "promoters" in " ".join(p["text"] for p in out["passages"])
    assert set(ctx.sources) == {p["label"] for p in out["passages"]}


def test_flag_column_gives_rates(subs):
    _, tools = tools_for(subs)
    run(tools, "derive_column", name="churned", expression="status == 'cancelled'")
    out = run(tools, "group_aggregate", metrics=[{"column": "churned", "agg": "mean"}], group_by=["plan"])
    assert out["result"][0]["plan"] == "Basic"
    assert isinstance(pd.Series([r["mean_churned"] for r in out["result"]]).max(), float)
