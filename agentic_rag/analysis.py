"""Analysis functions. Every number the agent reports comes from here.

Each function takes an `AnalysisContext` plus plain JSON-style arguments and
returns a JSON-serialisable dict. Problems raise `ToolError` with a hint
(closest column names, valid category values) so the model can fix its call.
Filters are structured conditions, never generated code.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Dataset, column_kind
from .knowledge import Hit, KnowledgeBase
from .memory import MemoryStore, granularity_code

MAX_ROWS = 25
OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "not_contains", "between",
       "is_null", "not_null")
AGGS = ("sum", "mean", "median", "min", "max", "count", "nunique", "std")
FREQ_RULES = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}
_AGG_ALIASES = {"avg": "mean", "average": "mean", "total": "sum", "distinct": "nunique"}
_OP_ALIASES = {"=": "==", "eq": "==", "ne": "!=", "neq": "!=", "notin": "not_in", "not in": "not_in",
               "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


class ToolError(Exception):
    pass


@dataclass
class AnalysisContext:
    dataset: Dataset
    memory: MemoryStore | None = None
    knowledge: KnowledgeBase | None = None
    applied_memory_ids: list[int] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    sources: dict[str, Hit] = field(default_factory=dict)  # label -> hit, e.g. "K3"

    @property
    def df(self) -> pd.DataFrame:
        return self.dataset.df

    def decimals(self) -> int:
        v = self.memory.preference("decimal_places") if self.memory else None
        return int(v) if v and v.isdigit() else 2

    def register_source(self, hit: Hit) -> str:
        for label, existing in self.sources.items():
            if existing.chunk_id == hit.chunk_id:
                return label
        label = f"K{len(self.sources) + 1}"
        self.sources[label] = hit
        return label


# ---------------------------------------------------------------- helpers

def clean(v, decimals: int = 2):
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, pd.Period):
        return str(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), decimals)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def records(df: pd.DataFrame, ctx: AnalysisContext, limit: int = MAX_ROWS) -> list[dict]:
    d = ctx.decimals()
    return [{str(k): clean(v, d) for k, v in row.items()} for row in df.head(limit).to_dict(orient="records")]


def col(ctx: AnalysisContext, name: str) -> str:
    if name in ctx.df.columns:
        return name
    lower = {c.lower(): c for c in ctx.df.columns}
    for candidate in (str(name).lower(), str(name).lower().replace(" ", "_"), str(name).lower().replace("_", " ")):
        if candidate in lower:
            return lower[candidate]
    close = difflib.get_close_matches(str(name), ctx.dataset.columns, n=3, cutoff=0.5)
    raise ToolError(f"Unknown column '{name}'. Closest: {close or ctx.dataset.columns[:15]}")


def is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def numeric_col(ctx: AnalysisContext, name: str) -> str:
    c = col(ctx, name)
    if not is_number(ctx.df[c]):
        raise ToolError(f"Column '{c}' is {column_kind(ctx.df[c])}, not numeric.")
    return c


def _coerce(series: pd.Series, value):
    if isinstance(value, list):
        return [_coerce(series, v) for v in value]
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value)
    if pd.api.types.is_bool_dtype(series) and isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if is_number(series) and isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            raise ToolError(f"Column '{series.name}' is numeric; '{value}' is not a number.")
    return value


def match_categories(series: pd.Series, values: list) -> list:
    """Case-insensitive matching against the labels that exist in the column."""
    if is_number(series) or pd.api.types.is_datetime64_any_dtype(series):
        return values
    existing = [str(e) for e in series.dropna().unique()]
    lookup = {e.lower(): e for e in existing}
    out = []
    for v in values:
        sv = str(v)
        if sv in existing or sv.lower() in lookup:
            out.append(sv if sv in existing else lookup[sv.lower()])
            continue
        if column_kind(series) == "categorical":
            close = difflib.get_close_matches(sv, existing, n=3, cutoff=0.6)
            raise ToolError(f"Value '{v}' not found in '{series.name}'. Valid values: {sorted(existing)[:20]}"
                            + (f"; closest: {close}" if close else ""))
        out.append(sv)
    return out


def apply_conditions(df: pd.DataFrame, conditions: list[dict] | None, ctx: AnalysisContext) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for cond in conditions or []:
        if hasattr(cond, "model_dump"):
            cond = cond.model_dump()
        if not isinstance(cond, dict) or "column" not in cond or "op" not in cond:
            raise ToolError(f"Each filter needs 'column' and 'op': got {cond}")
        c = col(ctx, cond["column"])
        op = str(cond["op"]).strip().lower().replace(" ", "_")
        op = _OP_ALIASES.get(op, op)
        if op not in OPS:
            raise ToolError(f"Unknown op '{cond['op']}'. Use one of {list(OPS)}")
        s, val = df[c], cond.get("value")
        if op in ("is_null", "not_null"):
            m = s.isna() if op == "is_null" else s.notna()
        elif op in ("contains", "not_contains"):
            m = s.astype(str).str.contains(str(val), case=False, na=False, regex=False)
            m = m if op == "contains" else ~m
        elif op in ("in", "not_in"):
            vals = _coerce(s, match_categories(s, val if isinstance(val, list) else [val]))
            m = s.isin(vals) if (is_number(s) or pd.api.types.is_datetime64_any_dtype(s)) \
                else s.astype(str).isin([str(v) for v in vals])
            m = m if op == "in" else ~m
        elif op == "between":
            if not isinstance(val, list) or len(val) != 2:
                raise ToolError("'between' needs value=[low, high]")
            lo, hi = _coerce(s, val)
            m = s.between(lo, hi)
        else:
            if op in ("==", "!=") and not isinstance(val, list):
                val = match_categories(s, [val])[0]
            v = _coerce(s, val)
            if op in ("==", "!=") and not (is_number(s) or pd.api.types.is_datetime64_any_dtype(s)):
                m = s.astype(str) == str(v)
                m = m if op == "==" else ~m & s.notna()
            else:
                m = {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
        mask &= m.fillna(False).astype(bool)
    return df[mask]


def resolve_named(ctx: AnalysisContext, names: list[str] | None) -> tuple[list[dict], list[str]]:
    conditions, used = [], []
    for name in names or []:
        if not ctx.memory:
            raise ToolError("Memory is turned off, so named filters are unavailable.")
        m = ctx.memory.lookup(name, ctx.dataset.fingerprint, ctx.dataset.columns)
        if not m or not m.filter:
            known = [x.key for x in ctx.memory.list() if x.filter]
            raise ToolError(f"No remembered filter called '{name}'. Known: {known or 'none'}. "
                            "Use explicit 'filters' or ask the user to define it.")
        conditions.extend(m.filter)
        ctx.applied_memory_ids.append(m.id)
        used.append(f"{m.key} = {m.content}")
    return conditions, used


def filtered(ctx: AnalysisContext, filters=None, named_filters=None) -> tuple[pd.DataFrame, dict]:
    named_conditions, used = resolve_named(ctx, named_filters)
    df = apply_conditions(ctx.df, named_conditions + list(filters or []), ctx)
    meta = {"rows_total": len(ctx.df), "rows_after_filters": len(df)}
    if used:
        meta["applied_memory"] = used
    return df, meta


def _agg(name: str) -> str:
    a = _AGG_ALIASES.get(str(name).lower(), str(name).lower())
    if a not in AGGS:
        raise ToolError(f"Unknown agg '{name}'. Use one of {list(AGGS)}")
    return a


def _freq(ctx: AnalysisContext, freq: str | None) -> tuple[str, str]:
    code = granularity_code(freq) if freq else None
    if code:
        return code, "argument"
    pref = ctx.memory.preference("time_granularity") if ctx.memory else None
    if granularity_code(pref):
        m = ctx.memory.lookup("time_granularity", None)
        if m:
            ctx.applied_memory_ids.append(m.id)
        return granularity_code(pref), f"remembered preference time_granularity={pref}"
    return "M", "default (monthly)"


# ---------------------------------------------------------------- 1. profile

def dataset_overview(ctx: AnalysisContext, columns: list[str] | None = None) -> dict:
    df, d = ctx.df, ctx.decimals()
    profile = []
    for c in ([col(ctx, x) for x in columns] if columns else list(df.columns)):
        s = df[c]
        kind = column_kind(s)
        entry = {"column": c, "kind": kind, "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if kind == "numeric":
            desc = s.describe()
            entry.update({k: clean(desc[k], d) for k in ("mean", "std", "min", "50%", "max")})
        elif kind == "datetime":
            entry.update({"min": clean(s.min()), "max": clean(s.max())})
        else:
            entry["top_values"] = {str(k): int(v) for k, v in s.dropna().astype(str).value_counts().head(6).items()}
        profile.append(entry)
    return {"name": ctx.dataset.name, "rows": len(df), "columns": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()), "profile": profile}


# ---------------------------------------------------------------- 2. rows

def filter_rows(ctx: AnalysisContext, filters=None, named_filters=None, columns: list[str] | None = None,
                sort_by: str | None = None, ascending: bool = False, limit: int = 10) -> dict:
    df, meta = filtered(ctx, filters, named_filters)
    if sort_by:
        df = df.sort_values(col(ctx, sort_by), ascending=ascending)
    if columns:
        df = df[[col(ctx, c) for c in columns]]
    limit = max(1, min(int(limit or 10), MAX_ROWS))
    return {**meta, "match_count": len(df), "rows": records(df, ctx, limit),
            "note": f"showing {min(limit, len(df))} of {len(df)} matching rows"}


# ---------------------------------------------------------------- 3. aggregate

def group_aggregate(ctx: AnalysisContext, metrics: list[dict], group_by: list[str] | None = None,
                    filters=None, named_filters=None, sort_by: str | None = None, ascending: bool = False,
                    top_n: int | None = None, include_share: bool = False) -> dict:
    df, meta = filtered(ctx, filters, named_filters)
    if not metrics:
        raise ToolError("Give at least one metric, e.g. {'column': '*', 'agg': 'count'}")
    named = {}
    for m in metrics:
        m = m.model_dump() if hasattr(m, "model_dump") else m
        agg = _agg(m.get("agg", ""))
        column = m.get("column") or "*"
        if column == "*":
            named[m.get("alias") or "row_count"] = (None, "size")
            continue
        c = col(ctx, column)
        if agg not in ("count", "nunique"):
            numeric_col(ctx, c)
        named[m.get("alias") or f"{agg}_{c}"] = (c, agg)

    if group_by:
        keys = [col(ctx, g) for g in group_by]
        grouped = df.groupby(keys, dropna=False)
        parts = {alias: (grouped.size() if how == "size" else grouped[c].agg(how)) for alias, (c, how) in named.items()}
        result = pd.DataFrame(parts).reset_index()
    else:
        result = pd.DataFrame([{alias: (len(df) if how == "size" else df[c].agg(how))
                                for alias, (c, how) in named.items()}])
    first = list(named)[0]
    if include_share and group_by:
        total = result[first].sum()
        result[f"share_of_{first}_%"] = result[first] / total * 100 if total else np.nan
    if group_by:
        result = result.sort_values(sort_by if sort_by in result.columns else first, ascending=ascending)
    groups = len(result)
    if top_n:
        result = result.head(int(top_n))
    return {**meta, "groups": groups, "result": records(result, ctx)}


# ---------------------------------------------------------------- 4. time

def time_series(ctx: AnalysisContext, date_column: str, value_column: str = "*", agg: str = "sum",
                freq: str | None = None, group_by: str | None = None, filters=None, named_filters=None,
                last_n_periods: int | None = None) -> dict:
    dc = col(ctx, date_column)
    if not pd.api.types.is_datetime64_any_dtype(ctx.df[dc]):
        dates = [c for c, k in ctx.dataset.kinds().items() if k == "datetime"]
        raise ToolError(f"'{dc}' is not a date column. Date columns: {dates}")
    df, meta = filtered(ctx, filters, named_filters)
    code, source = _freq(ctx, freq)
    rule = FREQ_RULES[code]
    df = df.dropna(subset=[dc])
    if df.empty:
        raise ToolError("No rows left after filtering.")
    count_rows = value_column in ("*", None, "")
    agg = "count" if count_rows else _agg(agg)
    vc = None if count_rows else (numeric_col(ctx, value_column) if agg not in ("count", "nunique")
                                  else col(ctx, value_column))
    grouper = pd.Grouper(key=dc, freq=rule)
    if group_by:
        g = col(ctx, group_by)
        grouped = df.groupby([grouper, g])
        table = (grouped.size() if vc is None else grouped[vc].agg(agg)).unstack(g).fillna(0)
    else:
        grouped = df.groupby(grouper)
        table = (grouped.size() if vc is None else grouped[vc].agg(agg)).to_frame("value")
        table["change_%"] = table["value"].pct_change().replace([np.inf, -np.inf], np.nan) * 100
    last_date = df[dc].max()
    period_end = table.index[-1] if code == "D" else table.index[-1].to_period(code).end_time.normalize()
    partial = bool(len(table) and last_date < period_end)
    if last_n_periods:
        table = table.tail(int(last_n_periods))
    table.index = table.index.to_period(code).astype(str)
    table.index.name = "period"
    table = table.reset_index()
    table.columns = [str(c) for c in table.columns]
    out = {**meta, "frequency": {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "yearly"}[code],
           "frequency_source": source, "periods": len(table), "series": records(table.tail(MAX_ROWS), ctx)}
    if not group_by and len(table) >= 2:
        vals = table["value"].astype(float)
        slope = np.polyfit(np.arange(len(vals)), vals, 1)[0] if len(vals) > 2 else vals.iloc[-1] - vals.iloc[0]
        out["trend"] = {"first": clean(vals.iloc[0]), "last": clean(vals.iloc[-1]),
                        "overall_change_%": clean((vals.iloc[-1] / vals.iloc[0] - 1) * 100) if vals.iloc[0] else None,
                        "avg_change_per_period": clean(slope),
                        "peak_period": str(table.loc[vals.idxmax(), "period"]),
                        "lowest_period": str(table.loc[vals.idxmin(), "period"])}
    if partial:
        out["caution"] = (f"The last period is incomplete (data ends {last_date:%Y-%m-%d}); "
                          "don't read its drop as a real decline.")
    if len(table) > MAX_ROWS:
        out["note"] = f"showing the last {MAX_ROWS} of {len(table)} periods"
    return out


# ---------------------------------------------------------------- 5. correlation

def _strength(r: float) -> str:
    a = abs(r)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "negligible"


def correlation(ctx: AnalysisContext, columns: list[str] | None = None, target: str | None = None,
                method: str = "pearson", filters=None, named_filters=None) -> dict:
    df, meta = filtered(ctx, filters, named_filters)
    method = method if method in ("pearson", "spearman", "kendall") else "pearson"
    cols = [numeric_col(ctx, c) for c in columns] if columns else [c for c in df.columns if is_number(df[c])]
    if target:
        target = numeric_col(ctx, target)
        cols = [c for c in cols if c != target] + [target]
    cols = [c for c in cols if df[c].nunique() > 1]
    if len(cols) < 2:
        raise ToolError("Need at least two numeric columns that vary.")
    corr = df[cols].corr(method=method)
    d = ctx.decimals()
    if target:
        ranked = corr[target].drop(target).dropna().sort_values(key=abs, ascending=False)
        return {**meta, "method": method, "target": target,
                "correlations_with_target": [{"column": k, "r": clean(v, d), "strength": _strength(v)}
                                             for k, v in ranked.items()],
                "caution": "Correlation is not causation; check sample size and outliers."}
    pairs = [{"a": a, "b": b, "r": clean(corr.loc[a, b], d), "strength": _strength(corr.loc[a, b])}
             for i, a in enumerate(cols) for b in cols[i + 1:] if not pd.isna(corr.loc[a, b])]
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {**meta, "method": method, "top_pairs": pairs[:15]}


# ---------------------------------------------------------------- 6. outliers

def detect_outliers(ctx: AnalysisContext, column: str, method: str = "iqr", threshold: float | None = None,
                    filters=None, named_filters=None, show_columns: list[str] | None = None) -> dict:
    c = numeric_col(ctx, column)
    df, meta = filtered(ctx, filters, named_filters)
    s = df[c].dropna()
    if s.empty:
        raise ToolError("No values to analyse after filtering.")
    if method == "zscore":
        k = threshold or 3.0
        std = s.std(ddof=0)
        mask = ((s - s.mean()) / std).abs() > k if std else s * 0 > 0
        bounds = {"mean": clean(s.mean()), "std": clean(std), "z_threshold": k}
    else:
        method, k = "iqr", threshold or 1.5
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        mask = (s < lo) | (s > hi)
        bounds = {"q1": clean(q1), "q3": clean(q3), "lower_bound": clean(lo), "upper_bound": clean(hi)}
    rows = df.loc[s[mask].index]
    if show_columns:
        keep = [col(ctx, x) for x in show_columns]
        rows = rows[keep + ([c] if c not in keep else [])]
    rows = rows.reindex(rows[c].sub(s.median()).abs().sort_values(ascending=False).index)
    return {**meta, "column": c, "method": method, "bounds": bounds, "outlier_count": int(mask.sum()),
            "outlier_share_%": clean(mask.mean() * 100), "high_outliers": int((s[mask] > s.median()).sum()),
            "low_outliers": int((s[mask] < s.median()).sum()), "most_extreme": records(rows, ctx, 10)}


# ---------------------------------------------------------------- 7. compare

def _p(value: float):
    return "<0.0001" if value < 1e-4 else round(float(value), 4)


def compare_groups(ctx: AnalysisContext, group_column: str, value_column: str, group_a: str | None = None,
                   group_b: str | None = None, filters=None, named_filters=None) -> dict:
    from scipy import stats

    gc, vc = col(ctx, group_column), col(ctx, value_column)
    df, meta = filtered(ctx, filters, named_filters)
    labels = df[gc].astype(str).where(df[gc].notna())
    if not group_a or not group_b:
        top = labels.value_counts().index.tolist()
        if len(top) < 2:
            raise ToolError(f"'{gc}' has fewer than two groups after filtering.")
        group_a = group_a or next(t for t in top if t != group_b)
        group_b = group_b or next(t for t in top if t.lower() != str(group_a).lower())
    group_a, group_b = match_categories(df[gc], [group_a, group_b])
    a_rows, b_rows = df[labels == str(group_a)], df[labels == str(group_b)]
    d = ctx.decimals()
    if is_number(df[vc]):
        a, b = a_rows[vc].dropna(), b_rows[vc].dropna()
        if len(a) < 2 or len(b) < 2:
            raise ToolError("Each group needs at least two values.")
        t = stats.ttest_ind(a, b, equal_var=False)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        cohen = (a.mean() - b.mean()) / pooled if pooled else float("nan")
        size = "undefined" if cohen != cohen else ("large" if abs(cohen) >= 0.8 else "medium" if abs(cohen) >= 0.5
                                                   else "small" if abs(cohen) >= 0.2 else "negligible") + " effect"
        return {**meta, "test": "Welch t-test + Mann-Whitney U", "value_column": vc,
                "groups": {str(group_a): {"n": len(a), "mean": clean(a.mean(), d), "median": clean(a.median(), d)},
                           str(group_b): {"n": len(b), "mean": clean(b.mean(), d), "median": clean(b.median(), d)}},
                "difference_in_means": clean(a.mean() - b.mean(), d), "t_test_p_value": _p(t.pvalue),
                "mann_whitney_p_value": _p(u.pvalue), "cohens_d": clean(cohen, 3),
                "significant_at_5%": bool(t.pvalue < 0.05), "interpretation": size}
    sub = pd.concat([a_rows, b_rows])
    table = pd.crosstab(sub[gc].astype(str), sub[vc].astype(str))
    if table.shape[1] < 2:
        raise ToolError(f"'{vc}' has a single value in these groups; nothing to compare.")
    chi2, p, _, _ = stats.chi2_contingency(table)
    shares = (table.div(table.sum(axis=1), axis=0) * 100).round(d)
    return {**meta, "test": "chi-square test of independence", "value_column": vc,
            "counts": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in table.iterrows()},
            "row_share_%": {str(k): {str(c): float(v) for c, v in row.items()} for k, row in shares.iterrows()},
            "chi2": clean(chi2, 3), "p_value": _p(p), "significant_at_5%": bool(p < 0.05)}


# ---------------------------------------------------------------- 8. derive

_SAFE = re.compile(r"^[\w\s\.\+\-\*/\(\)%<>=!&|~,`'\"]+$")


def derive_column(ctx: AnalysisContext, name: str, expression: str) -> dict:
    if "__" in expression or not _SAFE.match(expression) or \
            re.search(r"\b(import|lambda|exec|eval|open)\b", expression):
        raise ToolError("Only arithmetic and comparisons on column names are allowed.")
    if re.search(r"[A-Za-z_]\w*\s*\(", expression):
        raise ToolError("Function calls aren't allowed in expressions; use arithmetic only.")
    if not re.fullmatch(r"[A-Za-z_][\w ]{0,40}", name):
        raise ToolError("Column name must be letters, digits, spaces or underscores.")
    result = ctx.df.eval(expression, engine="python")
    if not isinstance(result, pd.Series) or len(result) != len(ctx.df):
        raise ToolError("The expression must produce one value per row.")
    if pd.api.types.is_bool_dtype(result):
        result = result.astype(int)  # 1/0 so a mean is a rate
    elif is_number(result):
        result = result.replace([np.inf, -np.inf], np.nan)
    ctx.dataset.df[name] = result
    ctx.dataset.derived_columns[name] = expression
    s = ctx.dataset.df[name]
    return {"created_column": name, "expression": expression, "missing": int(s.isna().sum()),
            "summary": {"min": clean(s.min()), "mean": clean(s.mean()), "max": clean(s.max())},
            "note": "The column exists for the rest of this session and other tools can use it."}


# ---------------------------------------------------------------- 9. chart

def create_chart(ctx: AnalysisContext, kind: str, x: str, y: str | None = None, agg: str = "sum",
                 title: str | None = None, filters=None, named_filters=None, top_n: int = 12,
                 freq: str | None = None) -> dict:
    """Return chart data (not an image); the web app draws it."""
    df, meta = filtered(ctx, filters, named_filters)
    xc = col(ctx, x)
    yc = col(ctx, y) if y else None
    kind = kind.lower()
    agg = _agg(agg) if yc else "count"
    d = ctx.decimals()
    spec = {"kind": kind, "x_label": xc, "y_label": f"{agg} of {yc}" if yc else "rows"}
    if kind == "hist":
        numeric_col(ctx, xc)
        counts, edges = np.histogram(df[xc].dropna(), bins=20)
        spec["points"] = [{"label": f"{edges[i]:,.4g} to {edges[i + 1]:,.4g}", "value": int(n)}
                          for i, n in enumerate(counts)]
        spec["y_label"] = "rows"
    elif kind == "scatter":
        if not yc:
            raise ToolError("A scatter chart needs y.")
        numeric_col(ctx, xc), numeric_col(ctx, yc)
        sample = df[[xc, yc]].dropna()
        sample = sample.sample(min(len(sample), 800), random_state=0)
        spec["points"] = [{"x": clean(a, d), "y": clean(b, d)} for a, b in sample.itertuples(index=False)]
        spec["y_label"] = yc
    elif kind == "line":
        if not pd.api.types.is_datetime64_any_dtype(df[xc]):
            raise ToolError("Line charts need a date column as x.")
        code, _ = _freq(ctx, freq)
        g = df.dropna(subset=[xc]).groupby(pd.Grouper(key=xc, freq=FREQ_RULES[code]))
        series = g.size() if not yc else g[yc].agg(agg)
        spec["points"] = [{"label": str(p.to_period(code)), "value": clean(v, d)} for p, v in series.items()]
    elif kind in ("bar", "pie"):
        g = df.groupby(xc)
        series = (g.size() if not yc else g[yc].agg(agg)).sort_values(ascending=False).head(int(top_n))
        spec["points"] = [{"label": str(k), "value": clean(v, d)} for k, v in series.items()]
    else:
        raise ToolError("kind must be one of bar, line, hist, scatter, pie")
    spec["title"] = title or f"{spec['y_label']} by {xc}"
    ctx.charts.append(spec)
    return {**meta, "chart": len(ctx.charts), "kind": kind, "title": spec["title"],
            "points_drawn": len(spec["points"]), "note": "The chart is shown to the user under your answer."}


# ---------------------------------------------------------------- knowledge + memory

def search_knowledge(ctx: AnalysisContext, query: str, k: int = 4) -> dict:
    hits = []
    if ctx.knowledge:
        for h in ctx.knowledge.search(query, ctx.dataset.fingerprint, k=k):
            hits.append({"label": ctx.register_source(h), "source": h.title, "where": h.location,
                         "score": h.score, "text": h.text})
    mems = []
    if ctx.memory:
        r = ctx.memory.retrieve(query, ctx.dataset.fingerprint, ctx.dataset.columns, k=k)
        mems = [{"memory_id": h.memory.id, "memory": h.memory.render()} for h in r.hits]
    return {"passages": hits, "memories": mems,
            "note": "Cite passages by their label, e.g. [K2]." if hits else "Nothing relevant found."}


def save_memory(ctx: AnalysisContext, kind: str, key: str, content: str, filter: list | None = None,
                scope: str = "dataset", source: str = "user") -> dict:
    if not ctx.memory:
        return {"error": "Memory is turned off for this session."}
    if filter:
        filter = [f.model_dump() if hasattr(f, "model_dump") else f for f in filter]
        apply_conditions(ctx.df, filter, ctx)  # validates columns and values first
        filter = [_normalise_condition(ctx, c) for c in filter]
    res = ctx.memory.remember(kind, key, content, filter=filter, scope=scope, dataset_fp=ctx.dataset.fingerprint,
                              dataset_name=ctx.dataset.name, source=source)
    out = res.as_dict()
    if res.status in ("created", "updated") and filter:
        out["rows_matching_now"] = len(apply_conditions(ctx.df, filter, ctx))
    return out


def _normalise_condition(ctx: AnalysisContext, cond: dict) -> dict:
    c = col(ctx, cond["column"])
    out = {k: v for k, v in {**cond, "column": c}.items() if v is not None}
    val = cond.get("value")
    if val is not None and column_kind(ctx.df[c]) == "categorical":
        fixed = match_categories(ctx.df[c], val if isinstance(val, list) else [val])
        out["value"] = fixed if isinstance(val, list) else fixed[0]
    return out


def forget_memory(ctx: AnalysisContext, key: str | None = None, memory_id: int | None = None) -> dict:
    if not ctx.memory:
        return {"error": "Memory is turned off for this session."}
    if memory_id is not None:
        return {"deleted": [int(memory_id)] if ctx.memory.delete(int(memory_id), "forgotten on request") else []}
    if key:
        return {"deleted": ctx.memory.forget_key(key, ctx.dataset.fingerprint)}
    raise ToolError("Give a key or a memory_id.")
