import json
from pathlib import Path

from services.llm import LLMClient
from services.rag import load_policy_rules, retrieve_policy_rules
from services.risk_engine import derive_risk_factors, evaluate_application, normalize_application_form


class _FakeChat:
    def __init__(self, outer):
        self.outer = outer
        self.completions = self

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        payload = {
            "risk_level": "Low",
            "recommendation": "Approve",
            "rationale": "Stable income and low DTI.",
            "policy_citations": ["SEC-02", "SEC-03"],
            "follow_up_questions": ["Confirm employment start date"],
            "missing_documents": [],
            "confidence": 0.84,
            "human_review_required": False,
            "highlighted_risks": ["None"],
        }

        class _Response:
            def __init__(self, content):
                self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})})]

        return _Response(json.dumps(payload))


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = _FakeChat(self)


def test_policy_rules_parse_and_retrieve(tmp_path):
    sample_policy = tmp_path / "policy_rules.md"
    sample_policy.write_text("# Section A\nRule A\n# Section B\nRule B", encoding="utf-8")
    rules = load_policy_rules(str(sample_policy))
    assert rules[0]["title"] == "Section A"
    results = retrieve_policy_rules("Rule B", top_k=1, path=str(sample_policy))
    assert results[0]["id"].startswith("SEC-")
    assert results[0]["title"] == "Section B"


def test_normalize_application_form_computes_ratios():
    normalized = normalize_application_form(
        {
            "monthly_income": 8000,
            "monthly_liabilities": 2400,
            "loan_amount": 40000,
            "credit_score": 690,
        }
    )
    assert normalized["debt_to_income"] == 0.3
    assert normalized["loan_to_income"] == 5.0


def test_evaluate_application_uses_fake_llm(monkeypatch):
    fake_client = _FakeClient()
    llm = LLMClient(api_key="test-key", client=fake_client)
    assessment = evaluate_application(
        {
            "monthly_income": 10000,
            "monthly_liabilities": 2000,
            "loan_amount": 30000,
            "credit_score": 750,
            "loan_purpose": "debt consolidation",
            "employment_status": "Salaried",
        },
        llm_client=llm,
        policy_path="data/policy_rules.md",
    )
    assert assessment.risk_level == "Low"
    assert fake_client.calls, "LLM client should be invoked"


def test_risk_factors_highlight_thresholds():
    factors = derive_risk_factors(
        {
            "debt_to_income": 0.6,
            "loan_to_income": 9,
            "credit_score": 500,
            "documents_provided": [],
        }
    )
    assert any("debt-to-income" in f for f in factors)
    assert any("credit score" in f.lower() for f in factors)
