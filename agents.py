"""Core loan assistant logic backed by CrewAI and FAISS-powered policy RAG."""
# windows_patch.py
"""
Windows compatibility patch for CrewAI.

This safely:
- Disables CrewAI telemetry (which causes signal issues on Windows)
- Adds missing Unix-only signals (SIGHUP, SIGTSTP, SIGQUIT, SIGCONT)
- Prevents CrewAI from registering signals in non-main threads
- Does nothing on Linux/macOS (safe for cross-platform use)

Usage:
    import windows_patch
"""

import os
import sys
import signal

# ---------------------------------------------------------
# 1. Only apply patches on Windows
# ---------------------------------------------------------
if sys.platform != "win32":
    # On Linux/MacOS, do nothing — signals work normally
    pass
else:
    # ---------------------------------------------------------
    # Disable CrewAI telemetry (root cause of most signal issues)
    # ---------------------------------------------------------
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_TRACKING"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    # ---------------------------------------------------------
    # Add missing Unix signals so imports don't fail
    # ---------------------------------------------------------
    fallback = signal.SIGTERM  # safe substitute

    for sig_name in ("SIGHUP", "SIGTSTP", "SIGQUIT", "SIGCONT"):
        if not hasattr(signal, sig_name):
            setattr(signal, sig_name, fallback)

    # ---------------------------------------------------------
    # Patch CrewAI telemetry to avoid registering signals
    # in non-main threads (Streamlit / FastAPI)
    # ---------------------------------------------------------
    try:
        from crewai.telemetry.telemetry import Telemetry
        import threading

        original = Telemetry._register_signal_handler

        def safe_register(self, sig, handler):
            # Only allow signal registration in the main thread
            if threading.current_thread() is not threading.main_thread():
                return
            try:
                return original(self, sig, handler)
            except ValueError:
                # Ignore "signal only works in main thread" errors
                return

        Telemetry._register_signal_handler = safe_register

    except Exception:
        # If CrewAI internals change, fail silently
        pass

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import logging
import os
import json
from datetime import datetime

import pandas as pd

from crewai import Agent, Crew, Task
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===========================================================
# Environment & logging setup
# ===========================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")

LOG_FILE = os.getenv("LOAN_AGENT_LOG_PATH", "loan_agents.log")
log_handlers: List[logging.Handler] = [logging.StreamHandler()]

try:
    log_handlers.append(logging.FileHandler(LOG_FILE))
except OSError as err:
    print(f"[loan_agents] Unable to open log file '{LOG_FILE}': {err}", file=sys.stderr)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("loan_agents")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
POLICY_DIR = BASE_DIR / "policies"

# Cache for the FAISS policy database so Streamlit sessions do not rebuild it
_POLICY_DB: Optional[FAISS] = None

# ===========================================================
# Prompt loading helper
# ===========================================================


def error_response(error: str, message: str) -> Dict[str, str]:
    """Standardized error payload for the UI."""
    return {
        "type": "error",
        "error": error,
        "message": message,
    }

@lru_cache(maxsize=8)
def load_prompt(filename: str) -> str:
    """Load a prompt template from prompts/<filename>."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, "prompts", filename)
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        logger.exception("Failed to load prompt file '%s'", filename)
        return ""


# ===========================================================
# Data loading utilities
# ===========================================================

@lru_cache(maxsize=1)
def _load_customer_tables():
    """Read customer CSV slices once per session."""
    credit_df = pd.read_csv(DATA_DIR / "credit_scores.csv", dtype={"ID": str})
    account_df = pd.read_csv(DATA_DIR / "account_status.csv", dtype={"ID": str})
    pr_df = pd.read_csv(DATA_DIR / "pr_status.csv", dtype={"ID": str})
    account_df["__name_lower"] = (
        account_df["Name"].fillna("").str.strip().str.lower()
    )
    return credit_df, account_df, pr_df


def clean_str(value: Any, default: str = "Unknown") -> str:
    """Safely convert arbitrary values to a trimmed string."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text or default


def load_customer_data(customer_input: str) -> Optional[Dict[str, Any]]:
    """Return a unified customer object from CSV slices without merging tables."""
    query = str(customer_input or "").strip()
    if not query:
        return {"error": "Customer identifier cannot be empty."}

    try:
        credit_df, account_df, pr_df = _load_customer_tables()
    except FileNotFoundError as err:
        logger.exception("CSV file not found")
        return {"error": f"Required data file missing: {err}"}
    except Exception as err:
        logger.exception("Customer load error")
        return {"error": str(err)}

    is_id = query.isdigit()
    account_rows = (
        account_df[account_df["ID"] == query]
        if is_id
        else account_df[account_df["__name_lower"] == query.lower()]
    )

    if account_rows.empty:
        return None
    if not is_id and len(account_rows) > 1:
        return {
            "error": "Multiple customers share that name. Please provide a unique ID."
        }

    account_row = account_rows.iloc[0]
    customer_id = clean_str(account_row.get("ID"))
    name = clean_str(account_row.get("Name"))
    email = clean_str(account_row.get("Email"))
    nationality = clean_str(account_row.get("Nationality"))
    account_status = clean_str(account_row.get("AccountStatus"))

    credit_row = credit_df[credit_df["ID"] == customer_id]
    if credit_row.empty:
        return {"error": f"Credit score missing for ID {customer_id}."}
    credit_value = credit_row.iloc[0].get("CreditScore")
    try:
        credit_score = int(float(credit_value))
    except (TypeError, ValueError):
        credit_score = clean_str(credit_value, default="Unknown")

    pr_status = "Not Required"
    if nationality.lower() != "singaporean":
        pr_row = pr_df[pr_df["ID"] == customer_id]
        if pr_row.empty:
            return {"error": f"PR status missing for ID {customer_id}."}
        pr_status = clean_str(pr_row.iloc[0].get("PRStatus"))

    return {
        "ID": customer_id,
        "Name": name,
        "Email": email,
        "Nationality": nationality,
        "AccountStatus": account_status,
        "CreditScore": credit_score,
        "PRStatus": pr_status,
    }


# ===========================================================
# Policy RAG utilities
# ===========================================================

def setup_rag() -> Optional[FAISS]:
    """Build a FAISS index over policy PDFs so the LLM can cite official rules."""
    try:
        docs = []

        if not POLICY_DIR.is_dir():
            raise FileNotFoundError(
                f"Policy directory '{POLICY_DIR}' not found. "
                "Ensure your policy PDFs are placed there."
            )

        pdf_paths = sorted(POLICY_DIR.glob("*.pdf"))
        if not pdf_paths:
            raise RuntimeError("No policy PDF documents found in 'policies/'.")

        for path in pdf_paths:
            loader = PyPDFLoader(str(path))
            # Each PDF page becomes a LangChain Document used downstream for RAG.
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding)
        logger.info("Policy RAG index built successfully.")
        return vectorstore

    except Exception:
        logger.exception("RAG setup failed")
        return None


def get_policy_db() -> Optional[FAISS]:
    """Return (and lazily cache) the FAISS store so every Streamlit run shares it."""
    global _POLICY_DB
    if _POLICY_DB is None:
        logger.info("Building policy database (FAISS cache miss)...")
        _POLICY_DB = setup_rag()
        if _POLICY_DB is None:
            logger.error("Policy database build failed.")
        else:
            logger.info("Policy database ready and cached.")
    return _POLICY_DB


def warm_policy_cache(force_rebuild: bool = False) -> bool:
    """
    Eagerly build the FAISS policy database so the first user request never waits.

    Returns True if the cache exists (after an optional rebuild), otherwise False.
    """
    global _POLICY_DB
    if force_rebuild:
        _POLICY_DB = None
    db = get_policy_db()
    return db is not None


def build_policy_tool(policy_db):
    """Wrap the FAISS policy index into a CrewAI tool so agents can call RAG."""
    if policy_db is None:
        return None

    @tool("PolicyRetriever")
    def search_policies(query: Any) -> str:
        """Return policy passages relevant to the given free-text query."""
        try:
            text = str(query or "").strip()
            if not text:
                return "Empty policy query."

            # Similarity search gives the agent up to 5 dense-retrieved passages.
            docs = policy_db.similarity_search(text, k=5)
            if not docs:
                return "No relevant policy sections found."

            joined = []
            for i, doc in enumerate(docs, start=1):
                joined.append(f"[Match {i}]\n{doc.page_content.strip()}")
            return "\n\n".join(joined)
        except Exception as err:
            logger.exception("PolicyRetriever tool failed")
            return f"Policy search failed: {err}"

    return search_policies


def build_customer_and_policy_tools(policy_db) -> List[Any]:
    """Return the tools available to the unified agent."""
    policy_tool = build_policy_tool(policy_db)

    @tool("CustomerDataLookup")
    def customer_data_lookup(query: Any) -> str:
        """Lookup a customer profile by ID or exact name and return JSON."""
        try:
            # The tool is intentionally thin: just bridge CrewAI to Python I/O.
            data = load_customer_data(str(query))
            if data is None:
                return "Customer not found."
            if isinstance(data, dict) and "error" in data:
                return json.dumps({"error": data["error"]}, default=str)
            return json.dumps(data, default=str)
        except Exception as err:
            logger.exception("CustomerDataLookup tool failed")
            return f"CustomerDataLookup failed: {err}"

    tools: List[Any] = [customer_data_lookup]
    if policy_tool is not None:
        tools.append(policy_tool)
    return tools


# ===========================================================
# Unified LLM pipeline
# ===========================================================

def safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    """Try multiple strategies to parse LLM output into JSON."""
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        trimmed = raw.strip().strip("`")
        if "\n" in trimmed:
            first_line, rest = trimmed.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                raw = rest.strip()
            else:
                raw = rest.strip() or first_line.strip()
        else:
            raw = trimmed
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception:
            return None


def normalize_loan_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the response dict matches what the UI expects."""
    FALLBACK = "I don't have the necessary information to answer that."
    logger.debug("HEREEEE Raw agent payload: %s", payload) 
    if not isinstance(payload, dict):
        return error_response(
            "llm_payload_invalid",
            "Loan assistant returned an invalid response.",
        )

    resp_type = payload.get("type", "qa")

    if resp_type == "loan_application":
        customer = payload.get("customer") or {}
        assessment = payload.get("ai_assessment") or {}
        customer_summary = {
            "id": customer.get("id") or customer.get("ID"),
            "name": customer.get("name") or customer.get("Name"),
            "nationality": customer.get("nationality") or customer.get("Nationality"),
            "pr_status": customer.get("pr_status") or customer.get("PRStatus"),
            "account_status": customer.get("account_status") or customer.get("AccountStatus"),
            "credit_score": customer.get("credit_score") or customer.get("CreditScore"),
        }
        assessment_summary = {
            "risk": assessment.get("risk", "Unknown"),
            "ai_recommendation": assessment.get("ai_recommendation", "Pending"),
            "interest_rate": assessment.get("interest_rate", "Unknown"),
            "policy_notes": assessment.get("policy_notes", ""),
            "pr_status_used": assessment.get("pr_status_used")
            or assessment.get("pr_status")
            or customer_summary.get("pr_status"),
        }
        letter = payload.get("letter", "")
        logger.debug("HEREEEE Raw agent payload: %s", payload) 
        logger.debug("HEREEEE Raw agent payload: %s", payload) 
        # Hard guardrails so the UI never displays policy-light loan decisions.
        if not str(assessment_summary["policy_notes"]).strip():
            return error_response(
                "policy_evidence_missing",
                "Policy-backed risk and interest details are required for loan applications.",
            )

        if str(assessment_summary["interest_rate"]).strip().lower() in {"", "unknown", "n/a"}:
            return error_response(
                "interest_rate_missing",
                "Interest rate derived from policy guidance is missing. Please rerun the request.",
            )

        if str(assessment_summary["risk"]).strip().lower() in {"", "unknown"}:
            return error_response(
                "risk_missing",
                "Risk classification based on policy guidance is missing. Please rerun the request.",
            )
        return {
            "type": "loan_application",
            "customer": customer_summary,
            "ai_assessment": assessment_summary,
            "letter": letter,
        }
    
    if resp_type == "error":
        return error_response(
            payload.get("error", "llm_error"),
            payload.get("message", "An unknown error occurred."),
        )

    # Default to a general QA style response
    answer = payload.get("answer") or FALLBACK
    return {
        "type": "qa",
        "answer": answer.strip() or FALLBACK,
    }


def run_unified_pipeline(user_text: str, tools: List[Any]) -> Dict[str, Any]:
    """Execute the single-agent pipeline that blends RAG + customer data tools."""
    prompt = load_prompt("unified_agent.md").strip()
    if not prompt:
        prompt = "You are a cautious loan assistant. Always return valid JSON as instructed."

    # Single agent keeps reasoning consistent and avoids sync issues.
    agent = Agent(
        role="Unified Loan Assistant",
        goal=prompt,
        backstory=prompt,
        tools=tools,
        allow_delegation=False,
    )

    task = Task(
        name="loan_helper",
        description=(
            f"{prompt}\n\n# USER MESSAGE\n{user_text}\n\n"
            "Return ONLY the JSON object specified in the prompt."
        ),
        agent=agent,
        expected_output=(
            '{"type": "qa" | "loan_application" | "error", ... }'
        ),
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    results = crew.kickoff()
    raw_output = str(results.tasks_output[0].raw)

    parsed = safe_json_loads(raw_output)
    if parsed is None:
        logger.warning("Failed to parse unified agent output: %s", raw_output)
        return error_response(
            "llm_output_parse_error",
            "Unable to parse the model response. Please try again.",
        )

    return normalize_loan_response(parsed)


# ===========================================================
# Public entry point for Streamlit
# ===========================================================

def handle_user_input(user_text: str) -> Dict[str, Any]:
    """Streamlit entry point: validate, ensure tooling, and run the LLM."""
    try:
        text = (user_text or "").strip()
        if not text:
            return error_response(
                "empty_input",
                "Please provide a non-empty loan question or request.",
            )

        policy_db = get_policy_db()
        if policy_db is None:
            return error_response(
                "policy_unavailable",
                "Policy database is unavailable. Ensure the policy PDFs exist and try again.",
            )

        tools = build_customer_and_policy_tools(policy_db)
        response = run_unified_pipeline(text, tools)

        if not isinstance(response, dict) or "type" not in response:
            return error_response(
                "unexpected_response",
                "Loan assistant returned an unexpected response.",
            )

        return response

    except Exception as err:
        logger.exception("handle_user_input failed")
        return error_response(
            "internal_error",
            f"An unexpected error occurred: {err}",
        )
