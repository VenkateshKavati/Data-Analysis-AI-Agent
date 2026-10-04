"""
executor.py
----------------
Step 2b of the Data Analyst AI Agent.

Runs LLM-generated pandas code in an isolated subprocess rather than
exec()-ing it in the main process. This means:
- A crash or infinite loop in generated code can't take down the agent.
- We can enforce a hard timeout.
- The generated code only gets the dataset file path, not our full
  process memory / env vars / other users' data.

It is NOT a full security sandbox (a determined attacker could still
escape a plain subprocess) — for a public-facing deployment you'd
upgrade this to Docker/gVisor/E2B. For a project / prototype, subprocess
isolation is a solid, honest middle ground.
"""

import subprocess
import sys
import json
import tempfile
import os
from pathlib import Path

TIMEOUT_SECONDS = 30

# Template that wraps the LLM-generated snippet with data loading +
# result capture. {dataset_path} and {generated_code} are filled in.
RUNNER_TEMPLATE = '''
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Safety net: LLM-generated code sometimes plots a PeriodIndex (e.g. from
# .dt.to_period("M")) directly, which matplotlib can't handle and raises
# "TypeError: float() argument must be ... not 'Period'". Auto-convert
# Period-like arguments to timestamps so this fails gracefully instead of
# crashing the whole analysis.
_original_plot = plt.plot

def _safe_plot(*args, **kwargs):
    def _convert(a):
        if isinstance(a, pd.PeriodIndex):
            return a.to_timestamp()
        if isinstance(a, pd.Series) and str(a.dtype).startswith("period"):
            return a.dt.to_timestamp()
        return a
    return _original_plot(*[_convert(a) for a in args], **kwargs)

plt.plot = _safe_plot

_dataset_path = r"{dataset_path}"
_suffix = Path(_dataset_path).suffix.lower()

if _suffix == ".csv":
    df = pd.read_csv(_dataset_path)
elif _suffix in (".xlsx", ".xls"):
    df = pd.read_excel(_dataset_path)
elif _suffix == ".json":
    df = pd.read_json(_dataset_path)
elif _suffix == ".parquet":
    df = pd.read_parquet(_dataset_path)
else:
    raise ValueError(f"Unsupported file type: {{_suffix}}")

# Auto-parse date-like text columns into real datetime64 columns, matching
# what data_loader.py does when building the data card. dayfirst=True +
# errors="coerce" avoids "day is out of range for month" crashes on
# day-first or mixed-format date strings.
for _col in df.columns:
    if pd.api.types.is_object_dtype(df[_col]) or pd.api.types.is_string_dtype(df[_col]):
        _non_null = df[_col].dropna()
        if not _non_null.empty:
            _parsed = pd.to_datetime(_non_null, errors="coerce", dayfirst=True)
            if _parsed.notna().mean() >= 0.8:
                df[_col] = pd.to_datetime(df[_col], errors="coerce", dayfirst=True)

result = None

{generated_code}

# Serialize whatever the generated code put in `result`
def _to_jsonable(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

with open(r"{output_path}", "w") as f:
    json.dump({{"result": _to_jsonable(result)}}, f, default=str)
'''


import re

# Deprecated pandas frequency aliases (pre-2.2) -> their modern replacements.
# pandas 2.2+ / 3.x raises ValueError on the old single-letter forms, but
# LLMs are trained on lots of old code and reach for "M"/"Q"/"Y" by habit.
# We patch these directly in the generated code as a safety net, in
# addition to instructing the model to use the new aliases.
_DEPRECATED_FREQ_MAP = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}


def _fix_deprecated_freq_aliases(code: str) -> str:
    """Rewrite freq="M" / freq='Q' / .resample("M") etc. to modern equivalents."""
    def _replace_kwarg(match):
        quote, old_alias = match.group(1), match.group(2)
        return f"freq={quote}{_DEPRECATED_FREQ_MAP[old_alias]}{quote}"

    def _replace_resample(match):
        quote, old_alias = match.group(1), match.group(2)
        return f".resample({quote}{_DEPRECATED_FREQ_MAP[old_alias]}{quote})"

    code = re.sub(r"freq=(['\"])(M|Q|Y|A)\1", _replace_kwarg, code)
    code = re.sub(r"\.resample\((['\"])(M|Q|Y|A)\1\)", _replace_resample, code)
    return code


def run_generated_code(dataset_path: str, generated_code: str) -> dict:
    """
    Write the generated code + a runner harness to a temp file, execute it
    in a subprocess with a timeout, and return the captured result.

    Returns a dict: {"success": bool, "result": ..., "error": str|None,
                      "chart_path": str|None}
    """
    generated_code = _fix_deprecated_freq_aliases(generated_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run_analysis.py"
        output_path = Path(tmpdir) / "output.json"
        chart_path = Path(tmpdir) / "output_chart.png"

        script = RUNNER_TEMPLATE.format(
            dataset_path=dataset_path,
            generated_code=generated_code,
            output_path=str(output_path),
        )
        script_path.write_text(script)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,               # chart saves land here, not our real cwd
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": None,
                "error": f"Execution timed out after {TIMEOUT_SECONDS}s",
                "chart_path": None,
            }

        if proc.returncode != 0:
            return {
                "success": False,
                "result": None,
                "error": proc.stderr.strip()[-2000:],  # last 2000 chars is usually enough
                "chart_path": None,
            }

        if not output_path.exists():
            return {
                "success": False,
                "result": None,
                "error": "Code ran but did not set `result`.",
                "chart_path": None,
            }

        with open(output_path) as f:
            data = json.load(f)

        # Copy chart out of the temp dir if one was generated, so it
        # survives after the TemporaryDirectory is cleaned up.
        final_chart_path = None
        if chart_path.exists():
            persist_dir = Path("generated_charts")
            persist_dir.mkdir(exist_ok=True)
            final_chart_path = persist_dir / f"chart_{os.getpid()}_{len(list(persist_dir.iterdir()))}.png"
            final_chart_path.write_bytes(chart_path.read_bytes())

        return {
            "success": True,
            "result": data.get("result"),
            "error": None,
            "chart_path": str(final_chart_path) if final_chart_path else None,
        }


if __name__ == "__main__":
    # Manual test: fake "generated code" answering a simple question
    import pandas as pd

    sample_csv_path = os.path.abspath("test_sample.csv")
    pd.DataFrame({
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age": [29, 34, None, 41, 25],
        "city": ["Delhi", "Mumbai", "Delhi", "Bangalore", "Mumbai"],
        "salary": [50000, 62000, 58000, None, 45000],
    }).to_csv(sample_csv_path, index=False)

    fake_generated_code = """
avg_salary_by_city = df.groupby("city")["salary"].mean().to_dict()
result = avg_salary_by_city
"""

    output = run_generated_code(sample_csv_path, fake_generated_code)
    print(json.dumps(output, indent=2))

    os.remove(sample_csv_path)
