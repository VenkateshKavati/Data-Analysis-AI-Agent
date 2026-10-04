"""
agent.py
----------------
Step 3: The Agent Orchestrator.

Ties together data_loader.py, llm_client.py, and executor.py into a
single loop:

    dataset -> profile -> data card
    (data card + user question) -> LLM generates pandas code
    generated code -> subprocess executor -> result / chart
    (question + result) -> LLM summarizes -> plain-English insight

This is the piece Streamlit (or any other UI) will call.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

from data_loader import load_dataset, profile_dataset, profile_to_text
from llm_client import generate_analysis_code, summarize_result
from executor import run_generated_code


@dataclass
class AgentResponse:
    success: bool
    question: str
    generated_code: Optional[str] = None
    raw_result: Optional[object] = None
    chart_path: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None


class DataAnalystAgent:
    """
    Holds one loaded dataset + its data card, and answers questions
    against it. One instance per uploaded dataset / user session.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = os.path.abspath(dataset_path)
        self.df = load_dataset(self.dataset_path)
        self.profile = profile_dataset(self.df)
        self.data_card = profile_to_text(self.profile)

        # Cache the already-parsed DataFrame (dates parsed, dtypes settled)
        # as Parquet. Each question spins up a fresh subprocess in
        # executor.py - without this, that subprocess would have to
        # re-parse the full original file (e.g. openpyxl on a large .xlsx)
        # from scratch every single time, which is what causes execution
        # timeouts on bigger datasets. Parquet read-back is dramatically
        # faster and preserves dtypes exactly.
        cache_fd, self._cache_path = tempfile.mkstemp(suffix=".parquet")
        os.close(cache_fd)
        self.df.to_parquet(self._cache_path, index=False)

    def __del__(self):
        cache_path = getattr(self, "_cache_path", None)
        if cache_path and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass

    def ask(self, question: str) -> AgentResponse:
        """
        Full loop for a single user question:
        generate code -> execute -> summarize.
        """
        # 1. Generate pandas code from the question
        try:
            code = generate_analysis_code(self.data_card, question)
        except Exception as e:
            return AgentResponse(
                success=False,
                question=question,
                error=f"Code generation failed: {e}",
            )

        # 2. Execute it safely in a subprocess (against the cached parquet
        #    copy, not the original file - see __init__)
        exec_result = run_generated_code(self._cache_path, code)

        if not exec_result["success"]:
            return AgentResponse(
                success=False,
                question=question,
                generated_code=code,
                error=exec_result["error"],
            )

        # 3. Summarize the raw result in plain English
        try:
            summary = summarize_result(question, str(exec_result["result"]))
        except Exception as e:
            # Analysis succeeded even if summarization failed - don't
            # throw the whole response away, just note it.
            summary = f"(Summary unavailable: {e})"

        return AgentResponse(
            success=True,
            question=question,
            generated_code=code,
            raw_result=exec_result["result"],
            chart_path=exec_result["chart_path"],
            summary=summary,
        )


if __name__ == "__main__":
    # Manual smoke test using a tiny inline dataset.
    # Note: this will actually call the Hugging Face API, so it
    # requires HF_API_TOKEN to be set - this is NOT run automatically,
    # just here for you to try locally.
    import pandas as pd

    sample_path = "smoke_test.csv"
    pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Bangalore", "Delhi", "Mumbai"],
        "sales": [1200, 950, 700, 1100, 980],
    }).to_csv(sample_path, index=False)

    agent = DataAnalystAgent(sample_path)
    response = agent.ask("What is the average sales per city?")

    print("SUCCESS:", response.success)
    print("CODE:\n", response.generated_code)
    print("RESULT:", response.raw_result)
    print("SUMMARY:", response.summary)
    print("ERROR:", response.error)

    os.remove(sample_path)
