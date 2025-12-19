# GenAI-Assisted Loan Risk Assessment

Streamlit prototype that normalizes loan applications, grounds responses in policy rules, and uses OpenAI for policy-aware risk reasoning. The UI highlights risk level, recommendation, rationale, citations, follow-ups, and human-review toggles.

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your environment (copy `.env.example`):
   ```bash
   cp .env.example .env
   # set OPENAI_API_KEY inside .env or your shell
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL shown in the terminal, submit sample applications, and review the AI rationale and policy citations.

## Project structure
- `app.py` – Streamlit UI and user flow.
- `services/` – Modular services for RAG, LLM calls, and risk logic.
- `data/policy_rules.md` – Local knowledge base used by the retriever.
- `doc/deliverable.md` – Assignment write-up (Parts 1–3).
- `tests/` – Unit tests for RAG, normalization, and orchestration.

## Running tests
```bash
pytest
```
