"""Wrapper around the OpenAI Chat Completions API for risk explanations."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON content from a model response."""
    try:
        return json.loads(text)
    except Exception:
        return {}


class LLMClient:
    """Thin OpenAI client with a deterministic prompt for loan risk reasoning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = client or self._build_client()

    def _build_client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.api_key)

    def generate_assessment(
        self,
        normalized_application: Dict[str, Any],
        policy_context: str,
        risk_factors: List[str],
    ) -> Dict[str, Any]:
        """Call the Chat Completions API to produce a structured assessment."""
        if not self.api_key and not self.client:
            raise ValueError("OPENAI_API_KEY is required to run the LLM-backed assessment.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior credit risk underwriter. "
                    "Given an applicant profile and policy snippets, return JSON with "
                    "the following keys: risk_level (Low/Medium/High), "
                    "recommendation (Approve/Decline/Refer), rationale (concise paragraph), "
                    "policy_citations (list of policy IDs/titles referenced), "
                    "follow_up_questions (list), missing_documents (list), "
                    "confidence (0-1 float), human_review_required (boolean), "
                    "highlighted_risks (list of bullet strings). "
                    "Be specific, cite policy IDs, and avoid hallucinating missing data."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Applicant data:\n{json.dumps(normalized_application, indent=2)}\n\n"
                    f"Known risk factors:\n- " + "\n- ".join(risk_factors or ["None observed"]) + "\n\n"
                    f"Policy excerpts:\n{policy_context}\n\n"
                    "Respond ONLY with valid JSON."
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content or "{}"
        return _safe_json_loads(content)
