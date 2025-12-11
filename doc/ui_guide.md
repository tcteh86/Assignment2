# UI & UX Guide

## 1. Layout
- **Sidebar** (left): decision metrics, approval rate progress bar.
- **Main Column**:
  1. Hero title + description.
  2. Input row (`textarea + submit button`).
  3. Dynamic content zone:
     - Answer card for Q&A.
     - Evaluation cards, policy notes, memo expander, officer form for loan applications.

## 2. Visual System
- CSS injected via `st.markdown` defines `card`, `card accent`, and `memo-box` classes.
- Cards use translucent backgrounds with subtle drop shadows for a modern look.
- Accent cards employ a gradient to highlight key numbers or warnings.

## 3. Components
| Component | Description | Source |
|-----------|-------------|--------|
| Customer Snapshot | Key-value pairs (ID, Name, Nationality, etc.) | `render_info_card` |
| AI Assessment | Risk, recommendation, interest rate, PR status used | `render_info_card(..., accent=True)` |
| Policy Evidence | Text block summarizing what RAG returned | `render_text_card` |
| Memo | Markdown-safe block inside an expander | Streamlit expander + `memo-box` |
| Decision Panel | Radio buttons, justification textarea, submit button | Main column bottom |

## 4. Accessibility
- Textarea placeholder guides officers on expected input.
- Buttons use `use_container_width=True` to enlarge click targets.
- Decision justification is required; Streamlit raises `st.warning` if blank.
- Colors maintain sufficient contrast; accent gradients overlay white text with strong shadow.

## 5. Responsive Behavior
- Streamlit auto-stacks columns on narrow viewports.
- The textarea remains full width; button column drops underneath on small screens.
- Cards remain legible due to padding and border radius regardless of width.

## 6. Officer Workflow Summary
1. Review system output (customer + assessment + policy notes).
2. Expand memo for long-form rationale if needed.
3. Select decision (defaults to AI recommendation).
4. Provide justification text.
5. Click `Record Final Decision` to log outcome and update sidebar metrics.

