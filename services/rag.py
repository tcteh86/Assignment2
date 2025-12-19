"""Lightweight retrieval over local policy rules for the Streamlit prototype."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


def _parse_policy_markdown(markdown_text: str) -> List[Dict]:
    """Split a markdown policy file into labeled sections."""
    sections: List[Dict] = []
    current_title = "General"
    current_lines: List[str] = []

    def _flush_section() -> None:
        if not current_lines:
            return
        sections.append(
            {
                "id": f"SEC-{len(sections) + 1:02d}",
                "title": current_title.strip(),
                "text": "\n".join(current_lines).strip(),
            }
        )

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            _flush_section()
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    _flush_section()
    return sections


def _tokenize(text: str) -> List[str]:
    """Very small tokenizer for TF-IDF style scoring."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def _compute_idf(docs: List[List[str]]) -> Dict[str, float]:
    doc_freq: defaultdict[str, int] = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            doc_freq[term] += 1
    total_docs = len(docs)
    return {term: math.log((1 + total_docs) / (1 + freq)) + 1 for term, freq in doc_freq.items()}


@lru_cache
def load_policy_rules(path: str = "data/policy_rules.md") -> List[Dict]:
    """Load and parse the policy rules file once per process."""
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy rulebook missing at {policy_path}")
    return _parse_policy_markdown(policy_path.read_text(encoding="utf-8"))


@lru_cache
def _build_policy_index(path: str = "data/policy_rules.md"):
    """Create a lightweight TF-IDF index for semantic retrieval."""
    rules = load_policy_rules(path)
    tokenized_docs = [_tokenize(rule["text"]) for rule in rules]
    idf = _compute_idf(tokenized_docs)
    tfidf_vectors: List[Dict[str, float]] = []
    for tokens in tokenized_docs:
        tf = Counter(tokens)
        tfidf_vectors.append({term: (tf[term] / len(tokens)) * idf.get(term, 0.0) for term in tf})
    return idf, tfidf_vectors, rules


def retrieve_policy_rules(query: str, top_k: int = 3, path: str = "data/policy_rules.md") -> List[Dict]:
    """Return the top-k policy sections relevant to a query."""
    idf, tfidf_vectors, rules = _build_policy_index(path)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return rules[:top_k]

    query_tf = Counter(query_tokens)
    query_vector = {term: (query_tf[term] / len(query_tokens)) * idf.get(term, 0.0) for term in query_tf}

    def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        shared_terms = set(vec_a) & set(vec_b)
        dot_product = sum(vec_a[t] * vec_b[t] for t in shared_terms)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in tfidf_vectors]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

    return [
        {
            "id": rules[idx]["id"],
            "title": rules[idx]["title"],
            "text": rules[idx]["text"],
            "score": float(similarities[idx]),
        }
        for idx in ranked_indices
    ]


def format_policy_snippets(rules: List[Dict]) -> str:
    """Render policy snippets for model prompting."""
    rendered = []
    for rule in rules:
        rendered.append(f"[{rule['id']}] {rule['title']}\n{rule['text']}")
    return "\n\n".join(rendered)
