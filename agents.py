import logging
import os
import json
from datetime import datetime

import pandas as pd
from crewai import Agent, Task, Crew
from crewai.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

import os
from dotenv import load_dotenv

# ---------------------------------------------------------
# Environment & Logging Setup
# ---------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Decision tracking
# ---------------------------------------------------------
DECISION_FILE = "data/loan_decisions.json"


def record_decision(customer, decision, risk, rate, pr_status):
    """Save loan decisions permanently."""
    try:
        os.makedirs("data", exist_ok=True)

        if os.path.exists(DECISION_FILE):
            with open(DECISION_FILE, "r") as f:
                data = json.load(f)
        else:
            data = {}

        entry = {
            "name": customer.get("Name", "Unknown"),
            "nationality": customer.get("Nationality", "Unknown"),
            "pr_status": pr_status,
            "decision": decision,
            "risk": risk,
            "rate": rate,
            "timestamp": datetime.now().isoformat()
        }

        data[customer["ID"]] = entry

        with open(DECISION_FILE, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Decision saved for customer ID {customer['ID']}")

    except Exception as e:
        logger.error(f"Decision save error: {e}")


def clean_str(value, default="Unknown"):
    """Utility: safe string cleaning."""
    if pd.isna(value):
        return default
    s = str(value).strip()
    return s if s else default


# ---------------------------------------------------------
# Customer Data Loading
# ---------------------------------------------------------
def load_customer_data(customer_input: str):
    """
    Load customer info from 3 independent CSV files:
    - credit_scores.csv       (ID,Name,Email,CreditScore)
    - account_status.csv      (ID,Name,Nationality,Email,AccountStatus)
    - pr_status.csv           (ID,Name,Email,PRStatus)

    Search by ID or Name.
    """
    try:
        # Load CSVs
        credit_df = pd.read_csv("data/credit_scores.csv")
        account_df = pd.read_csv("data/account_status.csv")
        pr_df = pd.read_csv("data/pr_status.csv")

        # Ensure ID is string
        credit_df["ID"] = credit_df["ID"].astype(str)
        account_df["ID"] = account_df["ID"].astype(str)
        pr_df["ID"] = pr_df["ID"].astype(str)

        # Normalize input
        customer_input = str(customer_input).strip()

        # -----------------------
        # First: try ID match
        # -----------------------
        is_id = customer_input.isdigit()
        if is_id:
            # Use ID for credit lookup
            row_credit = credit_df[credit_df["ID"] == customer_input]
        else:
            # Search by exact name for credit
            name_lower = customer_input.lower()
            matches = credit_df[credit_df["Name"].str.lower() == name_lower]

            if len(matches) > 1:
                return {
                    "error": (
                        f"Multiple customers found for name '{customer_input}'. "
                        "Please provide a unique ID."
                    )
                }
            row_credit = matches

        if row_credit.empty:
            return None

        row_credit = row_credit.iloc[0]
        customer_id = clean_str(row_credit.get("ID"))
        name = clean_str(row_credit.get("Name"))
        email = clean_str(row_credit.get("Email"))
        credit_score = row_credit.get("CreditScore")

        # -----------------------
        # Account Status Lookup
        # -----------------------
        row_account = account_df[account_df["ID"] == customer_id]
        if row_account.empty:
            return {"error": f"Account status missing for ID {customer_id}"}
        row_account = row_account.iloc[0]

        nationality_value = clean_str(row_account.get("Nationality"))

        # -----------------------
        # PR Lookup Logic
        # -----------------------
        if nationality_value.lower() != "singaporean":
            # Non-Singaporean → PR REQUIRED
            row_pr = pr_df[pr_df["ID"] == customer_id]
            if row_pr.empty:
                return {
                    "error": (
                        f"PR status is required for non-Singaporean ID {customer_id}, "
                        "but PR record is missing."
                    )
                }
            row_pr = row_pr.iloc[0]
            pr_status = clean_str(row_pr.get("PRStatus"))
        else:
            # Singaporean → PR NOT NEEDED
            pr_status = "Not Required"

        # -----------------------
        # Build Final Customer Object
        # -----------------------
        customer = {
            "ID": customer_id,
            "Name": name,
            "Email": email,
            "CreditScore": credit_score,
            "Nationality": nationality_value,
            "AccountStatus": clean_str(row_account.get("AccountStatus")),
            "PRStatus": pr_status,
        }

        return customer

    except Exception as e:
        logger.error(f"Customer load error: {e}")
        return None


# ---------------------------------------------------------
# Setup RAG
# ---------------------------------------------------------
def setup_rag():
    """Load PDF policies and build FAISS index."""
    try:
        docs = []
        policy_dir = "policies"

        for file in os.listdir(policy_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(policy_dir, file))
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embedding)

    except Exception as e:
        logger.error(f"RAG setup failed: {e}")
        return None


# ---------------------------------------------------------
# Policy Tool
# ---------------------------------------------------------
def build_policy_tool(policy_db):
    """Build a robust PolicyRetriever tool around the FAISS policy DB."""

    if policy_db is None:
        # Explicitly signal that the tool cannot be built
        logging.error("Policy DB is None. PolicyRetriever tool will not be available.")
        return None

    @tool("PolicyRetriever")
    def search_policies(query : str) -> str:
        """Retrieve relevant loan policy sections from the internal PDF library.

        The input query may come in any format from the LLM (string, dict, list, etc.).
        This wrapper makes sure it is converted to a string before searching.
        """
        try:
            # Make the tool robust to non-string inputs
            if not isinstance(query, str):
                query = str(query)

            query = query.strip()
            if not query:
                return "Empty policy query. Please provide a valid question."

            matches = policy_db.similarity_search(query, k=5)
            if not matches:
                return "No relevant policy sections found."

            return "\n\n".join(
                f"[Match {i+1}]\n{doc.page_content.strip()}"
                for i, doc in enumerate(matches)
            )

        except Exception as e:
            logging.exception("PolicyRetriever tool failed")
            return f"Policy search failed: {e}"

    return search_policies


# ---------------------------------------------------------
# Main Loan Processing
# ---------------------------------------------------------
def process_customer(customer_input: str):
    """Main processing entry point."""
    try:
        customer = load_customer_data(customer_input)
        if customer is None:
            return {"output": "Customer not found."}

        # Extract fields
        name = customer.get("Name", "Unknown")
        nationality = customer.get("Nationality", "Unknown")
        raw_pr = str(customer.get("PR_Status", "")).lower().strip()

        # HIGH RISK RULE (NEW)
        # If NOT Singaporean and NOT PR → HIGH RISK + AUTO REJECT
        high_risk_override = (
            nationality.lower() != "singaporean"
            and raw_pr in ("no", "false", "0", "", "none", "nan")
        )

        # Setup RAG + tool
        policy_db = setup_rag()
        policy_tool = build_policy_tool(policy_db)

        # ---------------- Agents ----------------
        data_agent = Agent(
            role="Data Retrieval Agent",
            goal="Summarize customer information.",
            backstory="You are responsible for accurately extracting and explaining customer attributes.",
            allow_delegation=False,
        )

        policy_agent = Agent(
            role="Policy Analyst",
            goal="Interpret and summarize relevant loan policies.",
            backstory=(
                "You are an expert in financial regulations and this bank's "
                "internal credit & loan policies. You always base your analysis "
                "on the provided policy documents and never guess."
            ),
            tools=[t for t in [policy_tool] if t is not None],
            allow_delegation=False,
        )

        decision_agent = Agent(
            role="Loan Decision Agent",
            goal="Make a final loan decision and draft a formal letter.",
            backstory=(
                "You are a senior loan officer who balances risk and opportunity "
                "while strictly following the bank's credit rules. "
                "REMINDER: If a Non Singaporean with NO PR Status, High risk."
            ),
            allow_delegation=False,
        )

        # ---------------- Tasks ----------------
        data_task = Task(
            name="fetch_customer",
            description=f"Return customer data: {customer}",
            agent=data_agent,
            expected_output="JSON summary of customer attributes.",
        )

        policy_task = Task(
            name="extract_policy",
            description=(
                "Using the policy tool, extract the sections relevant to this customer's "
                "credit score, nationality, PR status and high risk."
            ),
            agent=policy_agent,
            expected_output="Summary of key policy rules that apply.",
        )

        decision_task = Task(
            name="make_decision",
            description=(
                "Given the customer data and policy summary, decide:\n"
                "- risk level\n"
                "- whether to approve or reject the loan\n"
                "- applicable interest rate (if any)\n"
                "Return a JSON object with structure:\n"
                "{\n"
                " 'risk': '<Low/Medium/High>',\n"
                " 'decision': '<approved/rejected>',\n"
                " 'interest_rate': '<rate or null>',\n"
                " 'formal_letter': '<letter to customer>',\n"
                " 'pr_status': '<PR status used>'\n"
                "}\n"
                "REMINDER: If a Non Singaporean with NO PR Status, High risk."
            ),
            agent=decision_agent,
            expected_output="A JSON dict with the decision fields.",
        )

        # Run crew
        crew = Crew(
            agents=[data_agent, policy_agent, decision_agent],
            tasks=[data_task, policy_task, decision_task],
            verbose=True,
        )

        results = crew.kickoff()
        outputs = {o.name: o.raw for o in results.tasks_output}

        # Parse JSON from decision task
        raw_json = outputs.get("make_decision", "").replace("'", '"')
        try:
            decision_data = json.loads(raw_json)
        except Exception:
            # If parsing fails, fallback to a minimal structure
            decision_data = {}

        # Extract fields with defaults
        risk = decision_data.get("risk", "Unknown")
        decision = decision_data.get("decision", "pending").capitalize()
        rate = decision_data.get("interest_rate", "Unknown")
        letter = decision_data.get("formal_letter", "")
        out_pr = decision_data.get("pr_status", raw_pr)

        # ENFORCE HIGH RISK AUTO REJECTION
        if high_risk_override:
            risk = "High Risk"
            decision = "Rejected"

        # Save permanent record
        record_decision(customer, decision, risk, rate, out_pr)

        return {
            "output": letter,
            "decision": decision,
            "risk": risk,
            "rate": rate,
            "customer_name": name,
            "nationality": nationality,
            "pr_status": out_pr
        }

    except Exception as e:
        return {"output": f"Error: {e}"}


# ---------------------------------------------------------
# Generic Loan Q&A
# ---------------------------------------------------------
def answer_loan_question(question: str) -> dict:
    """
    Generic loan Q&A entry point.

    The question can be:
    - About general loan policies (interest rates by risk, eligibility rules, etc.)
    - About a specific customer (their risk, decision, credit score, etc.)

    This function:
    - Uses PDF-based RAG over loan policies (if available)
    - Uses CSV-based customer data and/or the existing process_customer pipeline
    - Delegates reasoning to a dedicated QA Agent
    - Returns a friendly natural-language answer and optional metadata context.
    """
    FALLBACK_MSG = "I don't have the necessary information to answer that."

    try:
        if not isinstance(question, str) or not question.strip():
            return {
                "answer": (
                    "Please ask a loan-related question, for example: "
                    "'What is the latest interest rate for high-risk customers?' "
                    "or 'What is the risk level for customer 1111?'."
                ),
                "context": {},
            }

        user_question = question.strip()

        # -------------------------------------------------
        # Build tools: Policy RAG + CSV / evaluation access
        # -------------------------------------------------
        try:
            policy_db = setup_rag()
        except Exception as e:
            logging.exception("Failed to set up policy RAG index.")
            policy_db = None

        policy_tool = build_policy_tool(policy_db) if policy_db is not None else None

        @tool("CustomerDataLookup")
        def customer_data_lookup(query: str) -> str:
            """Look up raw customer information from CSVs by ID or exact name.

            Use this when the user asks about a specific customer's credit score,
            nationality, PR status, or account status.
            """
            try:
                data = load_customer_data(query)
                if data is None:
                    return "Customer not found."
                if isinstance(data, dict):
                    return json.dumps(data)
                return str(data)
            except Exception as e:
                logging.exception("CustomerDataLookup failed")
                return f"Customer lookup failed: {e}"

        @tool("CustomerEvaluation")
        def customer_evaluation(query: str) -> str:
            """Run the full loan evaluation pipeline for a customer.

            Use this when the user asks about a specific customer's risk level,
            loan decision or applicable interest rate.
            """
            try:
                result = process_customer(query)
                if result is None:
                    return "Customer not found."
                if isinstance(result, dict):
                    return json.dumps(result)
                return str(result)
            except Exception as e:
                logging.exception("CustomerEvaluation failed")
                return f"Customer evaluation failed: {e}"

        tools = [customer_data_lookup, customer_evaluation]
        # Only append policy tool if it was successfully built
        if policy_tool is not None:
            try:
                _ = policy_tool.name  # minimal sanity check
                tools.append(policy_tool)
            except Exception:
                logging.error("Invalid policy tool detected; skipping it.")

        # --------------------------------
        # Guardrail: loan-domain questions
        # --------------------------------
        loan_keywords = [
            "loan", "credit", "interest", "rate", "risk",
            "mortgage", "installment", "emi", "overdraft",
            "principal", "repayment", "customer", "borrower",
            "lending", "collateral", "credit score",
        ]
        lowered = user_question.lower()
        if not any(k in lowered for k in loan_keywords):
            # Clearly out-of-domain → friendly fallback
            return {
                "answer": FALLBACK_MSG,
                "context": {},
            }

        # -------------------------------
        # QA Agent that uses the tools
        # -------------------------------
        qa_agent = Agent(
            role="Loan Knowledge Q&A Agent",
            goal=(
                "Answer questions about loan policies, interest rates, credit risk "
                "and this bank's customers using ONLY the provided tools and data."
            ),
            backstory=(
                "You are a senior loan officer. You have access to:\n"
                "- A tool that searches internal loan policies from PDF files.\n"
                "- Tools that look up and evaluate specific customers from CSV data.\n"
                "You are careful, conservative, and never guess."
            ),
            tools=tools,
            allow_delegation=False,
        )

        qa_task = Task(
            name="loan_qa",
            description=(
                "User question:\n"
                f"{user_question}\n\n"
                "Important instructions:\n"
                "- You may call the available tools to search policies or "
                "look up/evaluate customers.\n"
                "- ONLY use information returned by these tools. Do NOT use any "
                "outside or general world knowledge.\n"
                "- If the question cannot be answered from the tools and data "
                "you have, you MUST respond exactly with this sentence:\n"
                f'"{FALLBACK_MSG}"\n'
                "- If the question is not related to loans, credit, interest "
                "rates, loan risk, or bank customers, you MUST also respond with "
                f'"{FALLBACK_MSG}"\n'
                "- When you can answer, be concise and clear (2–5 sentences)."
            ),
            agent=qa_agent,
            expected_output=(
                "A concise natural-language answer to the user's question. "
                f"If you lack information, return exactly: '{FALLBACK_MSG}'"
            ),
        )

        crew = Crew(
            agents=[qa_agent],
            tasks=[qa_task],
            verbose=False,
        )

        results = crew.kickoff()
        try:
            answer_text = results.tasks_output[0].raw
        except Exception:
            logging.exception("Failed to read QA task output")
            answer_text = FALLBACK_MSG

        # Option B: natural language + metadata (when applicable).
        # We don't track per-tool usage here yet, so context is left minimal,
        # but this can be extended later if needed.
        return {
            "answer": answer_text,
            "context": {},
        }

    except Exception as e:
        logging.exception("Error in answer_loan_question")
        return {
            "answer": FALLBACK_MSG,
            "context": {"error": str(e)},
        }
