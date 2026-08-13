AI-Capstone_Project_app/
│
├── .streamlit/
│   └── secrets.toml          # Save all the token or secret for reference only.
├── data_docs/                # Saves uploaded PDF, TXT, CSV, Excel files
├── chroma_db/                # Stores Chroma DB vector store and metadata
├── modules/
│   ├── __init__.py
│   ├── ingestion.py          # File loaders + GitHub persistence (unchanged from base RAG app)
│   ├── processing.py         # Chunking, embedding, vector store (unchanged from base RAG app)
│   ├── agents.py             # NEW — tool-calling agent, tools, retries, execution limits
│   └── guardrails.py         # NEW — input validation, injection/unsafe screening, PII redaction, grounding checks
├── app.py                    # Streamlit UI wiring ingestion -> agent -> guardrails
└── requirements.txt

This folder is a separate deployable app from the base RAG project. The
ingestion and vector-store steps (`ingestion.py`, `processing.py`,
GitHub-backed `chroma_db/` persistence) are identical to the base project
— only the *question-answering* stage changes, from a fixed retrieve-then-
generate chain to an agent that plans its own tool calls.

## 1. Agent-based reasoning (`modules/agents.py`)
Plan → Retrieve → Reason → Generate pipeline

- **`retrieve_documents`** — semantic search over the ingested documents
  (wraps the same Chroma vector store as the base app).
- **`list_ingested_documents`** — lets the agent check what's actually
  available before answering.
- **`calculator`** — exact arithmetic via a restricted AST evaluator (no
  `eval`/`exec`), so the agent doesn't hallucinate math on the numbers it
  retrieves (e.g. "what's the % change in Q1 vs Q4 revenue?").

Built with LangChain's `create_tool_calling_agent` + `AgentExecutor` over
Gemini 2.5 Flash's native function calling. A strict system prompt
requires the agent to retrieve before answering factual questions, cite
sources, use the calculator instead of mental math, and say "I don't
know" instead of guessing when retrieval comes up empty.

The reasoning trace (which tools were called, with what input, and what
came back) is shown in an expander under each answer for transparency.

## 2. Reliability & safety controls

**Error / exception handling**
- Every tool (`agents.py`) catches its own exceptions and returns a
  controlled `"ERROR: ..."` string to the agent rather than crashing the
  run — a bad retrieval or malformed calculator expression degrades
  gracefully instead of stopping the whole request.
- `run_agent()` retries transient failures (rate limits, timeouts, 5xx)
  up to 3 times with exponential backoff, and raises a clean `AgentError`
  with a user-safe message on non-transient failures — no raw stack
  traces reach the UI.
- The agent executor is capped (`max_iterations=6`,
  `max_execution_time=60s`) so a confused agent can't loop indefinitely,
  and `handle_parsing_errors` recovers from malformed tool-call output
  instead of failing the whole run.
- `app.py` wraps ingestion, agent build, and agent invocation each in
  their own `try/except`, with a last-resort catch-all around generation
  so an unexpected error always surfaces as a readable message, never a
  crash.

**Input validation** (`guardrails.validate_user_input`)
- Rejects empty/whitespace-only input.
- Caps query length (2000 chars) to bound cost and prevent abuse.
- Strips non-printable/control characters that could be used for
  markdown- or terminal-injection tricks.

**Guardrails to reduce hallucinations and unsafe output**
- **Prompt-injection detection** (`detect_prompt_injection`) flags
  phrasing like "ignore previous instructions" or "reveal your system
  prompt" — the message still gets processed (this is a heuristic, not
  a hard block), but the UI warns the user and the agent's system prompt
  explicitly instructs it to treat any such text inside documents or
  questions as untrusted data, not commands.
- **Unsafe-request screening** (`detect_unsafe_request`) refuses clearly
  unsafe requests (e.g. asking for weapons/malware instructions) before
  they ever reach the model or retriever.
- **Grounding check** (`is_grounded`) flags any answer that isn't backed
  by retrieved content and doesn't explicitly say "I don't know" — this
  is the main hallucination guard, since the highest-risk case is the
  model fabricating an answer when retrieval found nothing.

