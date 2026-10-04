"""
data_loader.py
----------------
Step 1 of the Data Analyst AI Agent.

Responsibilities:
1. Load a dataset (CSV / Excel / JSON) into a pandas DataFrame.
2. Profile it: shape, dtypes, null counts, sample rows, basic stats.
3. Produce a compact "data card" (a text summary) that will later be
   fed to the LLM as context so it "understands" the dataset without
   us dumping the entire file into the prompt.
"""

import pandas as pd
import json
from pathlib import Path


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV, Excel, or JSON file into a DataFrame based on extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df = auto_parse_dates(df)
    return df


def auto_parse_dates(df: pd.DataFrame, min_success_rate: float = 0.8) -> pd.DataFrame:
    """
    Detect object/string columns that look like dates and convert them to
    real datetime64 columns, so both the data card and any code run against
    this DataFrame see a proper datetime dtype instead of raw strings.

    Why this exists: LLM-generated analysis code often calls a bare
    pd.to_datetime(df[col]) on a date column. Pandas' default parser
    guesses month-first (MM/DD/YYYY), which throws
    "ValueError: day is out of range for month" the moment it hits a
    day-first date like 13/02/2024. Parsing once here, with dayfirst=True
    and errors="coerce", avoids that crash entirely - bad/mixed values
    just become NaT instead of blowing up the whole run.
    """
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            parsed = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
            success_rate = parsed.notna().mean()
            if success_rate >= min_success_rate:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df


def profile_dataset(df: pd.DataFrame, sample_rows: int = 5) -> dict:
    """
    Build a structured profile of the dataset:
    - shape
    - column names + dtypes
    - null counts per column
    - basic numeric stats (mean/min/max) for numeric columns
    - unique value counts for low-cardinality categorical columns
    - a small sample of rows
    """
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
        "sample_rows": df.head(sample_rows).to_dict(orient="records"),
    }

    for col in df.columns:
        col_data = df[col]
        col_info = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "null_pct": round(float(col_data.isnull().mean() * 100), 2),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_info["min"] = _safe_stat(col_data.min())
            col_info["max"] = _safe_stat(col_data.max())
            col_info["mean"] = _safe_stat(col_data.mean())
        else:
            n_unique = col_data.nunique(dropna=True)
            col_info["n_unique"] = int(n_unique)
            # Only show top values if cardinality is manageable
            if n_unique <= 20:
                col_info["top_values"] = (
                    col_data.value_counts(dropna=True).head(10).to_dict()
                )

        profile["columns"][col] = col_info

    return profile


def _safe_stat(value):
    """Round floats, pass through other types, handle NaN."""
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 4)
    return value


def profile_to_text(profile: dict) -> str:
    """
    Convert the structured profile into a compact text 'data card'
    suitable for feeding into an LLM prompt as context.
    """
    lines = []
    lines.append(f"Dataset: {profile['n_rows']} rows, {profile['n_columns']} columns")
    lines.append("\nColumns:")

    for col, info in profile["columns"].items():
        line = f"- {col} ({info['dtype']}), nulls: {info['null_count']} ({info['null_pct']}%)"
        if "mean" in info:
            line += f", range: [{info['min']}, {info['max']}], mean: {info['mean']}"
        elif "n_unique" in info:
            line += f", unique values: {info['n_unique']}"
            if "top_values" in info:
                top = ", ".join(f"{k}: {v}" for k, v in info["top_values"].items())
                line += f" (top: {top})"
        lines.append(line)

    lines.append("\nSample rows:")
    lines.append(json.dumps(profile["sample_rows"], indent=2, default=str))

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual test with a tiny inline dataset
    import io

    sample_csv = """name,age,city,salary
Alice,29,Delhi,50000
Bob,34,Mumbai,62000
Carol,,Delhi,58000
Dave,41,Bangalore,
Eve,25,Mumbai,45000
"""
    df = pd.read_csv(io.StringIO(sample_csv))
    profile = profile_dataset(df)
    print(profile_to_text(profile))