"""The agent's tools as LangChain `StructuredTool`s with pydantic schemas.

Tools are built per session so they close over that session's
`AnalysisContext` (its dataset, memory, knowledge base, charts and sources).
Every tool returns a JSON string; failures come back as {"error": ...} so the
model can read the hint and retry instead of the loop crashing.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from . import analysis as A
from .analysis import AnalysisContext, ToolError
from .memory import KINDS

Freq = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]


class Condition(BaseModel):
    column: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "not_contains", "between",
                "is_null", "not_null"]
    value: Any = Field(None, description="Value to compare with; a list for in/not_in/between.")


class Metric(BaseModel):
    column: str = Field(description="Column to aggregate, or '*' to count rows.")
    agg: Literal["sum", "mean", "median", "min", "max", "count", "nunique", "std"]
    alias: Optional[str] = None


class Filtered(BaseModel):
    filters: Optional[list[Condition]] = Field(None, description="Row conditions, ANDed, applied first.")
    named_filters: Optional[list[str]] = Field(
        None, description="Names of remembered definitions or named filters to apply (e.g. 'active customers'). "
                          "Use whenever the question uses a remembered term.")


class OverviewArgs(BaseModel):
    columns: Optional[list[str]] = Field(None, description="Optional subset of columns to profile.")


class FilterRowsArgs(Filtered):
    columns: Optional[list[str]] = None
    sort_by: Optional[str] = None
    ascending: bool = False
    limit: int = Field(10, description="Rows to return (max 25).")


class AggregateArgs(Filtered):
    metrics: list[Metric]
    group_by: Optional[list[str]] = None
    sort_by: Optional[str] = None
    ascending: bool = False
    top_n: Optional[int] = None
    include_share: bool = Field(False, description="Add each group's % share of the first metric.")


class TimeSeriesArgs(Filtered):
    date_column: str
    value_column: str = Field("*", description="Numeric column, or '*' to count rows.")
    agg: Literal["sum", "mean", "median", "min", "max", "count", "nunique"] = "sum"
    freq: Optional[Freq] = Field(None, description="Leave empty to use the user's remembered preference.")
    group_by: Optional[str] = Field(None, description="Optional category to split the series by.")
    last_n_periods: Optional[int] = None


class CorrelationArgs(Filtered):
    columns: Optional[list[str]] = None
    target: Optional[str] = Field(None, description="Rank every other numeric column against this one.")
    method: Literal["pearson", "spearman", "kendall"] = "pearson"


class OutlierArgs(Filtered):
    column: str
    method: Literal["iqr", "zscore"] = "iqr"
    threshold: Optional[float] = Field(None, description="IQR multiplier (default 1.5) or z cutoff (default 3).")
    show_columns: Optional[list[str]] = None


class CompareArgs(Filtered):
    group_column: str
    value_column: str
    group_a: Optional[str] = None
    group_b: Optional[str] = None


class DeriveArgs(BaseModel):
    name: str
    expression: str = Field(description="Arithmetic over columns, e.g. 'revenue / units'. "
                                        "Wrap names containing spaces in backticks.")


class ChartArgs(Filtered):
    kind: Literal["bar", "line", "hist", "scatter", "pie"]
    x: str
    y: Optional[str] = None
    agg: Literal["sum", "mean", "median", "min", "max", "count", "nunique"] = "sum"
    title: Optional[str] = None
    top_n: int = 12
    freq: Optional[Freq] = None


class SearchArgs(BaseModel):
    query: str = Field(description="What to look up, in plain words.")
    k: int = 4


class SaveMemoryArgs(BaseModel):
    kind: Literal["definition", "named_filter", "preference", "correction", "column_note"]
    key: str = Field(description="Short name, e.g. 'active customers' or 'time_granularity'.")
    content: str = Field(description="The fact in one or two sentences, or the preference value.")
    filter: Optional[list[Condition]] = Field(None, description="Conditions that implement a definition.")
    scope: Literal["dataset", "global"] = "dataset"
    source: Literal["user", "agent"] = Field("user", description="'user' if the user said it, 'agent' if inferred.")


class ForgetArgs(BaseModel):
    key: Optional[str] = None
    memory_id: Optional[int] = None


SPECS: list[tuple[str, str, type[BaseModel], Callable]] = [
    ("dataset_overview", "Profile the table: row count, column types, missing values, unique counts, summary "
     "statistics and the most common category values. Use first when unsure what the data holds.",
     OverviewArgs, A.dataset_overview),
    ("filter_rows", "Look up individual rows that match conditions, optionally sorted. Use for 'which, list, "
     "show me the rows' questions.", FilterRowsArgs, A.filter_rows),
    ("group_aggregate", "Aggregate metrics (sum, mean, median, min, max, count, nunique, std), optionally per "
     "group, with sorting, top-N and % share. Column '*' with agg 'count' counts rows.",
     AggregateArgs, A.group_aggregate),
    ("time_series", "Aggregate a value over time (daily to yearly) with period-over-period % change, trend, "
     "peak and lowest period. Omit freq to use the user's remembered time_granularity.",
     TimeSeriesArgs, A.time_series),
    ("correlation", "Measure relationships between numeric columns, or rank the drivers of one target column.",
     CorrelationArgs, A.correlation),
    ("detect_outliers", "Find unusual values in a numeric column (IQR rule or z-score) and return the most "
     "extreme rows.", OutlierArgs, A.detect_outliers),
    ("compare_groups", "Test whether two groups differ: Welch t-test, Mann-Whitney U and Cohen's d for a "
     "numeric value; chi-square for a categorical one.", CompareArgs, A.compare_groups),
    ("derive_column", "Add a column for this session from an arithmetic expression over existing columns. A "
     "comparison such as \"status == 'cancelled'\" gives a 1/0 flag whose mean is a rate.",
     DeriveArgs, A.derive_column),
    ("create_chart", "Draw a chart (bar, line over a date column, hist, scatter, pie) shown under the answer.",
     ChartArgs, A.create_chart),
    ("search_knowledge", "Search the knowledge base (uploaded documents such as glossaries and data "
     "dictionaries, plus this dataset's column cards) and remembered facts. Use when a term, metric or column "
     "meaning is unclear, then cite passages by label.", SearchArgs, A.search_knowledge),
    ("save_memory", "Remember a durable fact for future sessions. Kinds: "
     + "; ".join(f"{k}: {v}" for k, v in KINDS.items())
     + " Give a definition or named_filter a 'filter' so tools can apply it exactly. Structured preferences: "
       "key 'time_granularity' (daily/weekly/monthly/quarterly/yearly) and key 'decimal_places'.",
     SaveMemoryArgs, A.save_memory),
    ("forget_memory", "Delete a remembered item when the user asks to forget it or says it's wrong with no "
     "replacement.", ForgetArgs, A.forget_memory),
]

MEMORY_TOOLS = {"save_memory", "forget_memory"}
ANALYSIS_TOOLS = [name for name, *_ in SPECS if name not in MEMORY_TOOLS | {"search_knowledge"}]


def to_json(result: dict, limit: int = 7000) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + ' ... [truncated]"}'


def run(fn: Callable, ctx: AnalysisContext, args: dict) -> dict:
    try:
        return fn(ctx, **args)
    except ToolError as e:
        return {"error": str(e)}
    except ValidationError as e:
        return {"error": f"Invalid arguments: {e.errors()[:3]}"}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
    except Exception as e:  # keep the loop alive and let the model see what failed
        return {"error": f"{type(e).__name__}: {e}"}


def build_tools(ctx: AnalysisContext, use_memory: bool = True) -> list[StructuredTool]:
    tools = []
    for name, description, schema, fn in SPECS:
        if not use_memory and name in MEMORY_TOOLS:
            continue

        def call(_fn=fn, **kwargs):
            args = {k: v for k, v in kwargs.items() if v is not None}
            return to_json(run(_fn, ctx, args))

        tools.append(StructuredTool.from_function(func=call, name=name, description=description,
                                                  args_schema=schema))
    return tools
