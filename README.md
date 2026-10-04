# 📊 Data Analyst AI Agent

A conversational data-analysis agent — upload a CSV, Excel, or JSON dataset, ask questions about it in plain English, and get back computed results, auto-generated charts, and a plain-English summary of the findings.

Built with **Streamlit**, **Pandas**, and the **Hugging Face Inference API**, with LLM-generated analysis code executed safely in an isolated subprocess rather than `exec()`-ed directly.

---

## ✨ Features

- **Upload any tabular dataset** — CSV, Excel (`.xlsx`/`.xls`), or JSON
- **Automatic data profiling** — shape, dtypes, null counts, numeric stats, top categorical values, and sample rows, compiled into a compact "data card" for LLM context
- **Ask questions in plain English** — the agent translates your question into pandas/matplotlib code using an LLM
- **Safe code execution** — generated code runs in an isolated subprocess with an enforced timeout, not in the main app process
- **Auto-generated charts** — with sensible defaults (readable labels, gridlines, rotated ticks, top-N category limiting for high-cardinality columns)
- **Plain-English summaries** — a second LLM call turns the raw result into a short business-friendly explanation
- **Chat-style history** — question/answer pairs persist across the session, each with its chart, raw result, and generated code viewable in expanders

---

## 🏗️ Architecture

```
dataset upload
      │
      ▼
data_loader.py   → loads file, auto-parses ambiguous date columns, builds a profile + "data card"
      │
      ▼
llm_client.py    → sends data card + question to the LLM, gets back pandas/matplotlib code
      │
      ▼
executor.py      → runs the generated code in an isolated subprocess (Parquet-cached dataset, timeout-guarded)
      │
      ▼
llm_client.py    → second LLM call summarizes the raw result in plain English
      │
      ▼
agent.py         → orchestrates the full loop, returns an AgentResponse
      │
      ▼
app.py           → Streamlit UI: upload, ask, view chat history, charts, code
```

**Why a subprocess instead of `exec()`?** LLM-generated code is untrusted. Running it in a separate process means a crash, infinite loop, or bad import can't take down the main app, memory/env vars aren't exposed, and a hard timeout is enforced. (Note: this is process isolation, not a full security sandbox — for a public-facing deployment, upgrade to Docker/gVisor/E2B.)

**Why cache to Parquet?** Each question spins up a fresh subprocess. Without caching, a large Excel file would get re-parsed by `openpyxl` from scratch on every single question. The dataset is parsed and date-normalized once at upload time, then cached as Parquet for fast, consistent reads on every subsequent question.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Data handling | Pandas, NumPy, OpenPyXL, PyArrow (Parquet) |
| Charts | Matplotlib |
| LLM | Hugging Face Inference API (`InferenceClient`) |
| Code isolation | Python `subprocess` |

---

## 🚀 Getting Started

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd data-analyst-ai-agent
pip install -r requirements.txt
```

### 2. Set your Hugging Face API token
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
export HF_API_TOKEN="hf_your_token_here"
```
Or create a `.env` file in the project root:
```
HF_API_TOKEN=hf_your_token_here
```

### 3. Run the app
```bash
streamlit run app.py
```

Then open `http://localhost:8501`, upload a dataset in the sidebar, and start asking questions.

---

## 📁 Project Structure

```
.
├── app.py             # Streamlit UI
├── agent.py           # Orchestrates data_loader → llm_client → executor
├── data_loader.py     # Load, profile, and build a data card for a dataset
├── llm_client.py      # Hugging Face Inference API wrapper (code gen + summarization)
├── executor.py         # Sandboxed subprocess execution of generated code
├── requirements.txt
└── README.md
```

---

## 💡 Example Questions

- "What is the average revenue by product category?"
- "Show me a bar chart of the top 10 customers by total spend"
- "Which month had the highest sales?"
- "Are there any columns with a lot of missing data?"

---

## ⚠️ Limitations

- Code execution is process-isolated, not fully sandboxed — not recommended for public, multi-tenant deployment without further hardening (e.g. Docker/gVisor).
- Chart and code quality depend on the underlying LLM; very ambiguous questions may need to be rephrased.
- Designed for exploratory analysis on small-to-medium datasets (tested up to ~60K rows); very large datasets may need chunked loading.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
