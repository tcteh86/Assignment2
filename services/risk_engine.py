"""Business logic for normalizing loan applications and generating risk outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from services.llm import LLMClient
from services.rag import format_policy_snippets, retrieve_policy_rules


@dataclass
class RiskAssessment:
    """Container for standardized risk outputs."""

    risk_level: str
    recommendation: str
    rationale: str
    policy_citations: List[str]
    follow_up_questions: List[str]
    missing_documents: List[str]
    confidence: float
    human_review_required: bool
    highlighted_risks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "policy_citations": self.policy_citations,
            "follow_up_questions": self.follow_up_questions,
            "missing_documents": self.missing_documents,
            "confidence": self.confidence,
            "human_review_required": self.human_review_required,
            "highlighted_risks": self.highlighted_risks,
        }


def normalize_application_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce and enrich form data with derived metrics."""
    income = float(form_data.get("monthly_income") or 0.0)
    liabilities = float(form_data.get("monthly_liabilities") or 0.0)
    loan_amount = float(form_data.get("loan_amount") or 0.0)
    credit_score = int(form_data.get("credit_score") or 0)

    debt_to_income = round(liabilities / income, 2) if income else 0.0
    loan_to_income = round(loan_amount / income, 2) if income else 0.0

    normalized = {
        "name": form_data.get("name", "").strip(),
        "national_id": form_data.get("national_id", "").strip(),
        "age": int(form_data.get("age") or 0),
        "employment_status": form_data.get("employment_status", "Unknown"),
        "monthly_income": income,
        "monthly_liabilities": liabilities,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_purpose": form_data.get("loan_purpose", ""),
        "existing_customer": bool(form_data.get("existing_customer")),
        "documents_provided": form_data.get("documents_provided") or [],
        "debt_to_income": debt_to_income,
        "loan_to_income": loan_to_income,
    }
    return normalized


def derive_risk_factors(application: Dict[str, Any]) -> List[str]:
    """Produce quick, deterministic risk hints to guide the model."""
    factors: List[str] = []
    dti = float(application.get("debt_to_income") or 0.0)
    lti = float(application.get("loan_to_income") or 0.0)
    credit_score = int(application.get("credit_score") or 0)
    documents = application.get("documents_provided") or []

    if dti > 0.45:
        factors.append("High debt-to-income ratio (>45%).")
    if lti > 8:
        factors.append("Loan amount exceeds 8x monthly income.")
    if credit_score and credit_score < 640:
        factors.append("Credit score below 640 threshold.")
    if not documents:
        factors.append("No supporting documents uploaded.")
    return factors


def _fallback_assessment(application: Dict[str, Any], risk_factors: List[str]) -> RiskAssessment:
    """Local heuristic when the LLM is unavailable."""
    risk_level = "Low"
    recommendation = "Approve"
    docs = application.get("documents_provided") or []

    credit_score = int(application.get("credit_score") or 0)
    dti = float(application.get("debt_to_income") or 0.0)
    lti = float(application.get("loan_to_income") or 0.0)

    if credit_score < 640 or dti > 0.5:
        risk_level = "High"
        recommendation = "Refer"
    elif dti > 0.35:
        risk_level = "Medium"
        recommendation = "Refer"

    rationale_parts = [
        f"Debt-to-income: {dti:.2f}",
        f"Loan-to-income: {lti:.2f}",
        f"Credit score: {credit_score}",
    ]
    return RiskAssessment(
        risk_level=risk_level,
        recommendation=recommendation,
        rationale="; ".join(rationale_parts),
        policy_citations=[],
        follow_up_questions=["Provide income verification", "Clarify current liabilities"],
        missing_documents=["Proof of income"] if not docs else [],
        confidence=0.5,
        human_review_required=risk_level != "Low",
        highlighted_risks=risk_factors,
    )


def evaluate_application(
    form_data: Dict[str, Any],
    llm_client: LLMClient,
    policy_path: str = "data/policy_rules.md",
) -> RiskAssessment:
    """Orchestrate normalization, retrieval, and LLM generation."""
    normalized = normalize_application_form(form_data)
    risk_factors = derive_risk_factors(normalized)

    query = f"{normalized['loan_purpose']} {normalized['employment_status']} {normalized['loan_amount']}"
    policies = retrieve_policy_rules(query=query, top_k=4, path=policy_path)
    policy_context = format_policy_snippets(policies)

    try:
        model_payload = llm_client.generate_assessment(
            normalized_application=normalized,
            policy_context=policy_context,
            risk_factors=risk_factors,
        )
    except Exception:
        return _fallback_assessment(normalized, risk_factors)

    return RiskAssessment(
        risk_level=model_payload.get("risk_level", "Medium"),
        recommendation=model_payload.get("recommendation", "Refer"),
        rationale=model_payload.get(
            "rationale", "Model did not provide a detailed rationale."
        ),
        policy_citations=model_payload.get("policy_citations", []),
        follow_up_questions=model_payload.get("follow_up_questions", []),
        missing_documents=model_payload.get("missing_documents", []),
        confidence=float(model_payload.get("confidence", 0.5)),
        human_review_required=bool(model_payload.get("human_review_required", False)),
        highlighted_risks=model_payload.get("highlighted_risks", risk_factors),
    )
