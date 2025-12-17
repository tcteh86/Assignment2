# Loan Assistant (Assn2)

Modern Streamlit console for an internal loan-assistant workflow.  
The UI talks to a single CrewAI agent that fuses:

- Customer records (`data/*.csv`) fetched through the `CustomerDataLookup` tool.
- FAISS-backed retrieval over the PDFs in `policies/` so every answer cites policy text.
- Guardrails that only surface policy-grounded decisions, plus a loan-officer override UI.

## Repo Tour

- `app.py` – Streamlit front end with decision cards, policy evidence, and officer workflow.
- `agents.py` – CrewAI agent, RAG setup, CSV utilities, and normalization/guardrails.
- `prompts/unified_agent.md` – strict JSON contract the agent must follow.
- `data/` – sample customer/account/PR CSV slices plus canned decision history.
- `policies/` – drop PDF policy manuals here (required at runtime).
- `doc/` – design notes, workflow diagrams, and UI guidance.
- `tests/` – pytest coverage for parser/guardrail helpers.

## Prerequisites

- Python 3.10+ (CrewAI and FAISS wheels expect a modern CPython build).
- `pip` plus build tools able to install `faiss-cpu` and `sentence-transformers`.
- OpenAI API key with GPT‑4o or compatible model access.

## Setup

The combination below is tested end-to-end with Python 3.10.12. Pinning these versions keeps CrewAI's LangChain bindings and FAISS' wheels aligned so the Streamlit app, RAG stack, and pytest suite all initialize cleanly.

| Package | Version | Notes |
| ------- | ------- | ----- |
| streamlit | 1.36.0 | Stable UI build with `st.set_page_config` + themed cards used by the console. |
| crewai | 0.36.21 | Exposes `Agent`, `Crew`, and `Task` APIs used in `agents.py`, plus the Windows signal patch still matches this release. |
| langchain | 0.2.11 | Core abstractions that CrewAI expects; pairs with split community/text-splitter packages. |
| langchain-community | 0.2.10 | Provides `PyPDFLoader`, `SentenceTransformerEmbeddings`, and FAISS wrappers. |
| langchain-openai | 0.1.21 | Supplies the `ChatOpenAI` client used by the agent. |
| langchain-text-splitters | 0.2.2 | Required for `RecursiveCharacterTextSplitter` import in `agents.py`. |
| sentence-transformers | 2.7.0 | Matches LangChain's embedding wrapper; will pull in `torch 2.2.x` automatically (install CPU builds on M1/Windows if prompted). |
| faiss-cpu | 1.8.0.post1 | Compatible with Python 3.10 wheels on macOS/Linux; powers the policy vector store. |
| python-dotenv | 1.0.1 | Loads `.env` files before the agent boots. |
| pandas | 2.2.2 | Used for CSV ingestion and guardrails. |
| pytest | 8.3.2 | Runs the regression suite in `tests/`. |

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install \
    streamlit==1.36.0 \
    crewai==0.36.21 \
    langchain==0.2.11 \
    langchain-community==0.2.10 \
    langchain-openai==0.1.21 \
    langchain-text-splitters==0.2.2 \
    sentence-transformers==2.7.0 \
    faiss-cpu==1.8.0.post1 \
    python-dotenv==1.0.1 \
    pandas==2.2.2 \
    pytest==8.3.2
```

1. Create an `.env` in the repo root with `OPENAI_API_KEY=sk-...`.
2. (Optional) Set `LOAN_AGENT_LOG_PATH=/path/to/loan_agents.log` to move the log file.
3. Copy the latest PDF policy manuals into `policies/`. The FAISS cache is rebuilt on the first run or when you call `warm_policy_cache(force_rebuild=True)`.

## Running the App

```bash
streamlit run app.py
```

What happens:

1. `warm_policy_cache()` builds an embedding index over every PDF in `policies/`.
2. Submissions call `handle_user_input()`, which:
   - Validates the text,
   - Pulls customer data + policy snippets via CrewAI tools,
   - Returns one of `qa`, `loan_application`, or `error`.
3. The UI renders policy evidence, AI memos, and lets the human officer record the final decision (tracked in the sidebar stats).

## Data & Policy Expectations

- Customer/account/PR CSVs live in `data/` and are loaded lazily per request.
- Non-Singaporean customers must have a PR record; missing PR data emits an error.
- Policy PDFs are mandatory—without them the FAISS build fails and the UI shows a fatal banner.

## Testing

```bash
pytest
```

`tests/test_agents.py` covers JSON parsing, guardrails, FAISS warming, and tool wiring. Run the suite after editing `agents.py` or prompts to ensure the contract with the UI remains intact.

## Troubleshooting

- **Policy database failed to load** – confirm PDFs exist in `policies/` and rerun `streamlit`.
- **Interest/risk guardrail errors** – the agent could not cite policy text; re-issue the request or inspect `loan_agents.log`.
- **CrewAI on Windows** – `agents.py` ships a defensive telemetry/signal patch that activates automatically on `win32`.
