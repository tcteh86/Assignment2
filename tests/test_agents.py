import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure the module import passes even when the test environment lacks a real key.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import agents  # noqa: E402  (import after sys.path/env setup)


logger = logging.getLogger("tests.test_agents")


def _make_payload(**overrides: Any) -> dict:
    """Helper to build a well-formed loan_application payload."""
    base = {
        "type": "loan_application",
        "customer": {
            "id": "123",
            "name": "Test User",
            "nationality": "Singaporean",
            "pr_status": "Not Required",
            "account_status": "Active",
            "credit_score": 780,
        },
        "ai_assessment": {
            "risk": "Low",
            "ai_recommendation": "Recommended",
            "interest_rate": "3.5%",
            "policy_notes": "Section 2.1 sets a 3.5% base rate.",
            "pr_status_used": "Not Required",
        },
        "letter": "Draft letter",
    }
    for key, value in overrides.items():
        base[key] = value
    return base


@pytest.fixture(autouse=True)
def reset_policy_cache(monkeypatch):
    """Ensure each test starts with a clean FAISS cache reference."""
    monkeypatch.setattr(agents, "_POLICY_DB", None, raising=False)


@pytest.fixture(autouse=True)
def log_test_start(request):
    """Emit structured logs before and after every test for HTML reporting."""
    logger.info("START %s", request.node.nodeid)
    yield
    logger.info("END %s", request.node.nodeid)


def test_safe_json_loads_handles_code_fence():
    raw = """```json
    {"type": "qa", "answer": "hello"}
    ```"""
    parsed = agents.safe_json_loads(raw)
    assert parsed == {"type": "qa", "answer": "hello"}


def test_safe_json_loads_returns_none_for_invalid_text():
    assert agents.safe_json_loads("not-json") is None
    assert agents.safe_json_loads(123) is None  # type: ignore[arg-type]


def test_normalize_loan_response_returns_valid_payload():
    payload = _make_payload()
    result = agents.normalize_loan_response(payload)
    assert result["type"] == "loan_application"
    assert result["ai_assessment"]["interest_rate"] == "3.5%"
    assert result["customer"]["name"] == "Test User"


@pytest.mark.parametrize(
    "field, value, expected_error",
    [
        ("policy_notes", "", "policy_evidence_missing"),
        ("interest_rate", "unknown", "interest_rate_missing"),
        ("risk", "Unknown", "risk_missing"),
    ],
)
def test_normalize_loan_response_guardrails(field, value, expected_error):
    assessment = _make_payload()["ai_assessment"]
    assessment[field] = value
    payload = _make_payload(ai_assessment=assessment)
    result = agents.normalize_loan_response(payload)
    assert result["type"] == "error"
    assert result["error"] == expected_error


def test_normalize_loan_response_handles_qa_with_fallback():
    result = agents.normalize_loan_response({"type": "qa", "answer": ""})
    assert result == {
        "type": "qa",
        "answer": "I don't have the necessary information to answer that.",
    }


def test_warm_policy_cache_reuses_existing_store(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(agents, "_POLICY_DB", sentinel, raising=False)

    def boom():
        raise AssertionError("setup_rag should not be called when cache exists")

    monkeypatch.setattr(agents, "setup_rag", boom, raising=False)

    assert agents.warm_policy_cache() is True


def test_warm_policy_cache_force_rebuild_invokes_setup(monkeypatch):
    sentinel = object()

    def fake_setup():
        return sentinel

    monkeypatch.setattr(agents, "_POLICY_DB", sentinel, raising=False)
    monkeypatch.setattr(agents, "setup_rag", fake_setup, raising=False)

    assert agents.warm_policy_cache(force_rebuild=True) is True
    assert agents._POLICY_DB is sentinel


def test_build_customer_and_policy_tools_without_policy_db(monkeypatch):
    monkeypatch.setattr(agents, "load_customer_data", lambda q: {"ID": "1"}, raising=False)
    tools = agents.build_customer_and_policy_tools(policy_db=None)
    assert len(tools) == 1
    tool = tools[0]
    output = tool.func("1")
    assert json.loads(output)["ID"] == "1"


def test_build_policy_tool_returns_matches(monkeypatch):
    class DummyDoc:
        def __init__(self, text):
            self.page_content = text

    class DummyPolicy:
        def __init__(self):
            self.queries = []

        def similarity_search(self, query, k=5):
            self.queries.append(query)
            return [DummyDoc("Policy snippet about 3.5% interest.")]

    policy_db = DummyPolicy()
    policy_tool = agents.build_policy_tool(policy_db)
    result = policy_tool.func("interest rate guidance")
    assert "Match 1" in result
    assert "3.5% interest" in result
    assert policy_db.queries == ["interest rate guidance"]
