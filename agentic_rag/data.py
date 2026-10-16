"""Loading tables and describing them.

Dataset-agnostic: column types are inferred, text that is really dates or
numbers is converted, and each dataset gets a fingerprint (a hash of its
column names) so knowledge and memory can tell datasets apart.
"""

from __future__ import annotations

import hashlib
import io
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}


def columns_fingerprint(columns) -> str:
    key = "|".join(sorted(str(c).strip().lower() for c in columns))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _looks_like_dates(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
        return False
    if not sample.str.contains(r"\d").all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def _parse_dates(series: pd.Series) -> pd.Series:
    """Day-first only when the data proves it (a first part above 12)."""
    parts = series.astype(str).str.extract(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}")
    dayfirst = False
    if parts.notna().all(axis=1).mean() > 0.5:
        first, second = parts[0].astype(float), parts[1].astype(float)
        dayfirst = bool((first > 12).any() and not (second > 12).any())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=dayfirst)


_NUMBER_NOISE = r"[,\s₹$€£%]"


def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        if _looks_like_dates(s):
            df[col] = _parse_dates(s)
            continue
        cleaned = s.dropna().astype(str).str.replace(_NUMBER_NOISE, "", regex=True)
        if not cleaned.empty and pd.to_numeric(cleaned, errors="coerce").notna().mean() >= 0.95:
            df[col] = pd.to_numeric(s.astype(str).str.replace(_NUMBER_NOISE, "", regex=True), errors="coerce")
    return df


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.nunique(dropna=True) <= max(20, int(len(series) * 0.05)):
        return "categorical"
    return "text"


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:,.4g}"
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}"
    return str(v)


@dataclass
class Dataset:
    name: str
    df: pd.DataFrame
    source: str = ""
    derived_columns: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return columns_fingerprint([c for c in self.df.columns if c not in self.derived_columns])

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.df.columns]

    def kinds(self) -> dict[str, str]:
        return {str(c): column_kind(self.df[c]) for c in self.df.columns}

    def describe_column(self, col: str) -> str:
        s = self.df[col]
        kind = column_kind(s)
        info = f"{col} [{kind}]"
        if kind == "numeric":
            info += f" range {_fmt(s.min())} to {_fmt(s.max())}, mean {_fmt(s.mean())}"
        elif kind == "datetime" and s.notna().any():
            info += f" from {_fmt(s.min())} to {_fmt(s.max())}"
        elif kind in ("categorical", "boolean"):
            info += f" values {s.dropna().astype(str).value_counts().head(8).index.tolist()}"
        else:
            info += f" e.g. {s.dropna().astype(str).head(3).tolist()}"
        missing = int(s.isna().sum())
        if missing:
            info += f"; {missing} missing"
        if col in self.derived_columns:
            info += f"; derived as `{self.derived_columns[col]}`"
        return info

    def schema_summary(self) -> str:
        lines = [f"Dataset '{self.name}': {len(self.df):,} rows x {self.df.shape[1]} columns."]
        lines += [f"- {self.describe_column(c)}" for c in self.columns]
        return "\n".join(lines)

    def preview(self, n: int = 8) -> list[dict]:
        out = []
        for row in self.df.head(n).to_dict(orient="records"):
            out.append({str(k): (_fmt(v) if isinstance(v, (pd.Timestamp, float, np.floating)) else
                                 (None if pd.isna(v) else v if not isinstance(v, np.generic) else v.item()))
                        for k, v in row.items()})
        return out


def load_table(source, name: str | None = None, sheet: str | int | None = 0) -> Dataset:
    """Load CSV/TSV/Excel from a path, or from (filename, bytes)."""
    if isinstance(source, tuple):
        fname, payload = source
        suffix, name, raw = Path(fname).suffix.lower(), name or Path(fname).stem, io.BytesIO(payload)
        label = fname
    else:
        path = Path(source)
        suffix, name, raw, label = path.suffix.lower(), name or path.stem, path, str(path)
    if suffix not in TABLE_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'. Use CSV, TSV or Excel.")
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(raw, sheet_name=sheet)
    else:
        df = pd.read_csv(raw, sep="\t" if suffix == ".tsv" else None, engine="python")
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return Dataset(name=name, df=infer_types(df), source=label)
