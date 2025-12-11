import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from crewai import Agent, Crew, Task
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===========================================================
# Environment & logging setup
# ===========================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("loan_agents")

# Cache for the FAISS policy database so Streamlit sessions do not rebuild it
_POLICY_DB: Optional[FAISS] = None


# ===========================================================
# Prompt loading helper
# ===========================================================

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

def clean_str(value: Any, default: str = "Unknown") -> str:
    """Safely convert arbitrary values to a trimmed string."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text or default


def load_customer_data(customer_input: str) -> Optional[Dict[str, Any]]:
    """Return a unified customer object based on the three CSV files."""
    try:
        credit_df = pd.read_csv("data/credit_scores.csv")
        account_df = pd.read_csv("data/account_status.csv")
        pr_df = pd.read_csv("data/pr_status.csv")

        for df in (credit_df, account_df, pr_df):
            df["ID"] = df["ID"].astype(str)

        # Pre-compute lowercase names once for safe comparisons
        account_df["__name_lower"] = (
            account_df["Name"].fillna("").str.strip().str.lower()
        )
        query = str(customer_input or "").strip()
        if not query:
            return {"error": "Customer identifier cannot be empty."}

        is_id = query.isdigit()
        account_rows: pd.DataFrame

        if is_id:
            account_rows = account_df[account_df["ID"] == query]
        else:
            lower = query.lower()
            account_rows = account_df[account_df["__name_lower"] == lower]
            if len(account_rows) > 1:
                return {
                    "error": (
                        "Multiple customers share that name. Please provide a unique ID."
                    )
                }

        if account_rows.empty:
            return None

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

    except FileNotFoundError as err:
        logger.exception("CSV file not found")
        return {"error": f"Required data file missing: {err}"}
    except Exception as err:
        logger.exception("Customer load error")
        return {"error": str(err)}


# ===========================================================
# Policy RAG utilities
# ===========================================================

def setup_rag() -> Optional[FAISS]:
    """Build a FAISS index for the policy PDF documents."""
    try:
        policy_dir = "policies"
        docs = []

        if not os.path.isdir(policy_dir):
            raise FileNotFoundError(
                f"Policy directory '{policy_dir}' not found. "
                "Ensure your policy PDFs are placed there."
            )

        for fname in os.listdir(policy_dir):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(policy_dir, fname)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())

        if not docs:
            raise RuntimeError("No policy PDF documents found in 'policies/'.")

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
    """Return a cached FAISS store so we do not rebuild on every request."""
    global _POLICY_DB
    if _POLICY_DB is None:
        _POLICY_DB = setup_rag()
    return _POLICY_DB


def build_policy_tool(policy_db):
    """Wrap the FAISS policy index into a CrewAI tool."""
    if policy_db is None:
        return None

    @tool("PolicyRetriever")
    def search_policies(query: Any) -> str:
        """Return policy passages relevant to the given free-text query."""
        try:
            text = str(query or "").strip()
            if not text:
                return "Empty policy query."

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

    if not isinstance(payload, dict):
        return {
            "type": "error",
            "error": "llm_payload_invalid",
            "message": "Loan assistant returned an invalid response.",
        }

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

        if not str(assessment_summary["policy_notes"]).strip():
            return {
                "type": "error",
                "error": "policy_evidence_missing",
                "message": "Policy-backed risk and interest details are required for loan applications.",
            }

        if str(assessment_summary["interest_rate"]).strip().lower() in {"", "unknown", "n/a"}:
            return {
                "type": "error",
                "error": "interest_rate_missing",
                "message": "Interest rate derived from policy guidance is missing. Please rerun the request.",
            }

        if str(assessment_summary["risk"]).strip().lower() in {"", "unknown"}:
            return {
                "type": "error",
                "error": "risk_missing",
                "message": "Risk classification based on policy guidance is missing. Please rerun the request.",
            }
        return {
            "type": "loan_application",
            "customer": customer_summary,
            "ai_assessment": assessment_summary,
            "letter": letter,
        }

    if resp_type == "error":
        return {
            "type": "error",
            "error": payload.get("error", "llm_error"),
            "message": payload.get("message", "An unknown error occurred."),
        }

    # Default to a general QA style response
    answer = payload.get("answer") or FALLBACK
    return {
        "type": "qa",
        "answer": answer.strip() or FALLBACK,
    }


def run_unified_pipeline(user_text: str, tools: List[Any]) -> Dict[str, Any]:
    prompt = load_prompt("unified_agent.md").strip()
    if not prompt:
        prompt = "You are a cautious loan assistant. Always return valid JSON as instructed."

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
        return {
            "type": "error",
            "error": "llm_output_parse_error",
            "message": "Unable to parse the model response. Please try again.",
        }

    return normalize_loan_response(parsed)


# ===========================================================
# Public entry point for Streamlit
# ===========================================================

def handle_user_input(user_text: str) -> Dict[str, Any]:
    try:
        text = (user_text or "").strip()
        if not text:
            return {
                "type": "error",
                "error": "empty_input",
                "message": "Please provide a non-empty loan question or request.",
            }

        policy_db = get_policy_db()
        if policy_db is None:
            return {
                "type": "error",
                "error": "policy_unavailable",
                "message": (
                    "Policy database is unavailable. Ensure the policy PDFs exist and try again."
                ),
            }

        tools = build_customer_and_policy_tools(policy_db)
        response = run_unified_pipeline(text, tools)

        if not isinstance(response, dict) or "type" not in response:
            return {
                "type": "error",
                "error": "unexpected_response",
                "message": "Loan assistant returned an unexpected response.",
            }

        return response

    except Exception as err:
        logger.exception("handle_user_input failed")
        return {
            "type": "error",
            "error": "internal_error",
            "message": f"An unexpected error occurred: {err}",
        }
