# Operational Workflows

## 1. User Interaction Flow
```mermaid
sequenceDiagram
    participant U as Loan Officer
    participant UI as Streamlit UI
    participant AG as handle_user_input
    participant FAISS as PolicyRetriever
    participant CSV as CustomerDataLookup
    participant LLM as OpenAI Chat API

    U->>UI: Enter question / loan request
    UI->>AG: handle_user_input(text)
    AG->>CSV: load_customer_data (as tool call)
    CSV-->>AG: JSON customer profile / error
    AG->>FAISS: PolicyRetriever(query)
    FAISS-->>AG: Policy excerpts
    AG->>LLM: CrewAI agent prompt + tool traces
    LLM-->>AG: JSON response (qa / loan_application / error)
    AG-->>UI: Structured dict
    UI-->>U: Rendered cards + approval controls
    U->>UI: Approve / Reject + reason
    UI->>U: Confirmation + stats update
```

## 2. Loan Application Decision Flow
```mermaid
graph LR
    A[Loan text received] --> B{Customer identifier?}
    B -- No --> E[Return missing_customer error]
    B -- Yes --> C[CustomerDataLookup]
    C -->|Error| E
    C -->|Success| D{Nationality = Singaporean?}
    D -- No --> F{PR status present?}
    F -- No --> G[Return PR error]
    F -- Yes --> H[Invoke PolicyRetriever]
    D -- Yes --> H
    H --> I[Unified Agent composes summary]
    I --> J{policy_notes & risk & rate present?}
    J -- No --> K[Return policy_evidence_missing]
    J -- Yes --> L[Streamlit renders approval UI]
    L --> M[Officer decision recorded + stats]
```

## 3. Streamlit Session State
| Key | Description | Initialization |
|-----|-------------|----------------|
| `pending_application` | Cached AI evaluation awaiting officer action | `None` until loan response | 
| `officer_decision` | Radio selection (`Approve` / `Reject`) | Pre-populated with AI suggestion |
| `officer_reason` | Free-text justification | Blank for each new evaluation |
| `decision_stats` | Dict with aggregated approvals/rejections | `{approved: 0, rejected: 0}` |

## 4. Officer Approval Checklist
1. Review **Customer Snapshot** card.
2. Review **AI Assessment** card with risk, rate, policy evidence.
3. Expand **AI Draft Letter** if more context is needed.
4. Select `Approve` or `Reject` in the decision radio buttons.
5. Provide a justification in the textarea (required for compliance).
6. Click **Record Final Decision** to log outcome and reset UI.

## 5. Error Handling Signals
| Error Code | Trigger | Surface |
|------------|---------|---------|
| `empty_input` | User submits blank text | Streamlit warning |
| `policy_unavailable` | FAISS index failure or missing PDFs | Streamlit error |
| `llm_output_parse_error` | Agent returns malformed JSON | Streamlit error |
| `policy_evidence_missing` | LLM omitted policy-based risk/interest | Streamlit error prompting re-run |

