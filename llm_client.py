"""
llm_client.py
----------------
Step 2a of the Data Analyst AI Agent.

Wraps the Hugging Face Inference API so the rest of the agent doesn't
care which specific model is behind it. Swap MODEL_ID to try different
code-capable models.

Setup:
    pip install huggingface_hub
    export HF_API_TOKEN="hf_xxx..."   # get one free at huggingface.co/settings/tokens

Good model choices for this task (instruction-tuned, code-capable):
    - "Qwen/Qwen2.5-Coder-32B-Instruct"   (strong at pandas/python code)
    - "mistralai/Mistral-7B-Instruct-v0.3" (general purpose, decent code)
    - "HuggingFaceH4/zephyr-7b-beta"       (general purpose)
"""

import os
import re
from huggingface_hub import InferenceClient

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

SYSTEM_PROMPT = """You are a data analysis assistant. You are given:
1. A description of a pandas DataFrame called `df` (columns, dtypes, stats, sample rows).
2. A user question about the data.

Your job: write a short Python snippet using pandas (and matplotlib if a chart
is requested) that answers the question, using the DataFrame `df` which is
already loaded in memory.

Rules:
- Only use columns that actually exist in the data description given.
- Any column shown with dtype "datetime64[ns]" in the data description is
  already a proper datetime column - do NOT call pd.to_datetime() on it
  again. Only call pd.to_datetime() on a column if its dtype is "object"
  and it clearly contains date strings, and even then always pass
  errors="coerce", dayfirst=True so mixed/invalid formats don't crash.
- When grouping a datetime column by month/quarter/year for a chart
  (e.g. df["date"].dt.to_period("M")), matplotlib CANNOT plot Period
  objects directly - it raises TypeError. Always convert the period back
  to a timestamp before plotting, e.g.:
      grouped = df.groupby(df["date"].dt.to_period("M"))["sales"].sum()
      grouped.index = grouped.index.to_timestamp()
      plt.plot(grouped.index, grouped.values, marker="o")
  Alternatively, group by a Grouper directly:
      grouped = df.groupby(pd.Grouper(key="date", freq="ME"))["sales"].sum()
  which produces a normal DatetimeIndex that plots without conversion.
- Modern pandas (2.2+) renamed several frequency aliases used in
  pd.Grouper(freq=...) or .resample(). Always use the NEW names:
  "ME" for month-end (NOT "M"), "QE" for quarter-end (NOT "Q"),
  "YE" for year-end (NOT "Y" or "A"). The old aliases raise a ValueError.
- Store the final answer in a variable called `result`.
- If a chart is requested, save it to a file called "output_chart.png"
  using plt.savefig("output_chart.png") and set result to a short string
  describing the chart.
- Do not read files, access the network, or import anything beyond
  pandas, matplotlib.pyplot, and numpy.
- Return ONLY the Python code, no explanations, no markdown fences.
"""


def _get_client() -> InferenceClient:
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Set the HF_API_TOKEN environment variable with your Hugging Face token."
        )
    return InferenceClient(model=MODEL_ID, token=token)


def generate_analysis_code(data_card: str, question: str) -> str:
    """
    Ask the Hugging Face model to write pandas code that answers `question`
    given the dataset description in `data_card`. Returns clean Python code.
    """
    client = _get_client()

    user_prompt = f"""DATASET DESCRIPTION:
{data_card}

USER QUESTION:
{question}

PYTHON CODE:"""

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.1,   # low temperature: we want deterministic, correct code
    )

    raw = response.choices[0].message.content
    return _clean_code(raw)


def _clean_code(raw: str) -> str:
    """Strip markdown fences and leading/trailing explanation text if the model adds them."""
    code = raw.strip()
    # Remove ```python ... ``` fences if present
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def summarize_result(question: str, result_repr: str) -> str:
    """
    Second LLM call: turn a raw computed result into a plain-English insight.
    Kept separate from code generation so each call has one clear job.
    """
    client = _get_client()

    user_prompt = f"""You are a data analyst. A user asked a question and an
analysis was run. Explain the result in 2-3 plain English sentences,
suitable for a business audience. Do not invent numbers not shown below.

QUESTION: {question}
RAW RESULT: {result_repr}

SUMMARY:"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
