import streamlit as st

from modules.agents import AgentError, build_agent_executor, run_agent
from modules.guardrails import (
    InputValidationError,
    apply_output_guardrails,
    detect_prompt_injection,
    detect_unsafe_request,
    REFUSAL_MESSAGE,
    validate_user_input,
)
from modules.ingestion import (
    save_and_load_document,
    push_file_to_github,
    push_folder_to_github,
    pull_folder_from_github,
)
from modules.processing import (
    CHROMA_DIR,
    load_vectorstore,
    process_documents_to_vectorstore,
)

st.set_page_config(
    page_title="Agentic Enterprise RAG with Gemini 2.5", layout="wide"
)

st.title("🤖 Agentic Enterprise RAG — Reasoning, Tools & Guardrails")
st.write(
    "Upload enterprise documents, then ask questions. Unlike a simple "
    "retrieve-then-answer chain, this app uses an **AI agent** that plans "
    "which tools to call (document search, calculator, document listing), "
    "reasons over the results, and is wrapped in input/output guardrails "
    "to reduce hallucinations and unsafe outputs."
)

# --------------------------------------------------------------------------
# API key
# --------------------------------------------------------------------------
try:
  GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
  st.error(
      "API Key missing! Please add GEMINI_API_KEY in your Streamlit Cloud"
      " app's Settings → Secrets (or in .streamlit/secrets.toml for local"
      " dev), like this:\n\n"
      "GEMINI_API_KEY = \"your_actual_key_here\""
  )
  st.stop()

# --------------------------------------------------------------------------
# Optional GitHub persistence (same mechanism as the base RAG app — see
# SECURITY WARNING in modules/ingestion.py before enabling on a public app)
# --------------------------------------------------------------------------
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")

if GITHUB_TOKEN and GITHUB_REPO and "chroma_restored" not in st.session_state:
  with st.spinner("Restoring vector database from GitHub..."):
    try:
      pull_folder_from_github(CHROMA_DIR, GITHUB_TOKEN, GITHUB_REPO)
    except Exception as restore_err:
      st.warning(f"Could not restore chroma_db from GitHub: {restore_err}")
  st.session_state["chroma_restored"] = True

# --------------------------------------------------------------------------
# Sidebar: ingestion (same pipeline as the base RAG app)
# --------------------------------------------------------------------------
with st.sidebar:
  st.header("📂 Document Ingestion")
  uploaded_file = st.file_uploader(
      "Upload Enterprise Document", type=["pdf", "txt", "csv", "xlsx", "xls"]
  )

  if uploaded_file is not None:
    if st.button("Process & Ingest Document"):
      # Visible, persistent progress — this stays on screen (does not
      # collapse or disappear like st.spinner does) so token/chunking/
      # embedding detail is actually readable, not hidden behind a spinner.
      progress_bar = st.progress(0, text="Starting ingestion...")
      log_box = st.empty()
      log_lines: list[str] = []

      def log(line: str):
        log_lines.append(line)
        log_box.markdown("\n".join(f"- {entry}" for entry in log_lines))

      def on_progress(stage: str, **info):
        if stage == "chunking_start":
          progress_bar.progress(0.05, text="Chunking document...")
          log("📄 Splitting document into chunks...")
        elif stage == "chunking_done":
          progress_bar.progress(0.15, text="Chunking complete")
          log(
              f"✂️ **Chunking complete** — {info['chunk_count']} chunks, "
              f"{info['total_chars']:,} characters "
              f"(≈{info['approx_tokens']:,} tokens, estimated)"
          )
        elif stage == "embedding_progress":
          done, total = info["done"], info["total"]
          pct = 0.15 + 0.85 * (done / total)
          progress_bar.progress(pct, text=f"Embedding chunk {done}/{total}")
          log(f"🧬 Embedded chunk **{done}/{total}**")
        elif stage == "embedding_done":
          progress_bar.progress(1.0, text="Embedding complete")
          log(f"✅ **Embedding complete** — {info['total']} chunks stored in ChromaDB")

      try:
        docs, saved_path = save_and_load_document(uploaded_file)
        log(f"💾 Saved file to `{saved_path}`")

        vector_store, chunk_count = process_documents_to_vectorstore(
            docs, GEMINI_API_KEY, progress_callback=on_progress
        )

        github_url = None
        chroma_files_pushed = 0
        if GITHUB_TOKEN and GITHUB_REPO:
          try:
            log("☁️ Backing up document to GitHub...")
            github_url = push_file_to_github(
                saved_path, GITHUB_TOKEN, GITHUB_REPO
            )
            log("☁️ Syncing `chroma_db/` to GitHub...")
            chroma_files_pushed = push_folder_to_github(
                CHROMA_DIR, GITHUB_TOKEN, GITHUB_REPO
            )
            log(f"☁️ Synced **{chroma_files_pushed}** vector store files to GitHub")
          except Exception as gh_err:
            st.warning(f"Saved locally, but GitHub backup failed: {gh_err}")

        # A new/changed vector store invalidates any cached agent below.
        st.session_state.pop("agent_executor", None)

        st.success(f"Successfully processed {uploaded_file.name}!")
        st.info(
            f"Saved to: `{saved_path}`\n\nGenerated **{chunk_count}** chunks"
            " inside `chroma_db/`"
            + (f"\n\nBacked up to GitHub: {github_url}" if github_url else "")
        )
      except Exception as e:
        progress_bar.progress(1.0, text="Failed")
        log(f"❌ **Error:** {e}")
        st.error(f"Error processing document: {e}")

  st.markdown("---")
  st.caption(
      "🛡️ Guardrails active: input validation, prompt-injection "
      "detection, unsafe-request screening, PII redaction, and "
      "grounding checks on every response."
  )

# --------------------------------------------------------------------------
# Main: agentic Q&A
# --------------------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Ask Questions — Answered by an AI Agent")

vector_store = load_vectorstore(GEMINI_API_KEY)

if vector_store is None:
  st.warning(
      "⚠️ No vector database found. Please upload and process a document via"
      " the sidebar first."
  )
else:

  @st.cache_resource(show_spinner=False)
  def _build_executor(_vector_store, api_key):
    return build_agent_executor(_vector_store, api_key)

  try:
    agent_executor = _build_executor(vector_store, GEMINI_API_KEY)
  except AgentError as e:
    st.error(f"Could not initialize the agent: {e}")
    st.stop()

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
      for warning in message.get("warnings", []):
        st.caption(f"⚠️ {warning}")

  if user_query := st.chat_input(
      "Type your question regarding the uploaded documents..."
  ):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      # ---- 1. Input validation --------------------------------------
      try:
        clean_query = validate_user_input(user_query)
      except InputValidationError as e:
        st.error(str(e))
        st.session_state.messages.append(
            {"role": "assistant", "content": str(e), "warnings": []}
        )
        st.stop()

      # ---- 2. Pre-generation guardrails -------------------------------
      if detect_unsafe_request(clean_query):
        st.warning(REFUSAL_MESSAGE)
        st.session_state.messages.append(
            {"role": "assistant", "content": REFUSAL_MESSAGE, "warnings": []}
        )
        st.stop()

      injection_flagged = detect_prompt_injection(clean_query)
      if injection_flagged:
        st.caption(
            "🛡️ This message resembles a prompt-injection attempt — it "
            "will be treated as untrusted input, and the agent's system "
            "instructions take precedence."
        )

      # ---- 3. Agent reasoning (with retries/error handling) -----------
      with st.spinner("Agent is planning, retrieving, and reasoning..."):
        try:
          result = run_agent(agent_executor, clean_query)
        except AgentError as e:
          error_msg = f"The agent couldn't complete this request: {e}"
          st.error(error_msg)
          st.session_state.messages.append(
              {"role": "assistant", "content": error_msg, "warnings": []}
          )
          st.stop()
        except Exception as e:  # noqa: BLE001 - last-resort safety net
          error_msg = (
              "An unexpected error occurred while generating the answer. "
              "Please try rephrasing your question."
          )
          st.error(error_msg)
          st.caption(f"Debug detail: {e}")
          st.session_state.messages.append(
              {"role": "assistant", "content": error_msg, "warnings": []}
          )
          st.stop()

      # ---- 4. Output guardrails ---------------------------------------
      guarded = apply_output_guardrails(result.answer, result.retrieved_docs)

      st.markdown(guarded.text)
      for warning in guarded.warnings:
        st.caption(f"⚠️ {warning}")

      with st.expander("🔍 Agent reasoning trace"):
        st.caption(f"Completed in {result.attempts} attempt(s).")
        if not result.steps:
          st.caption("The agent answered directly without calling a tool.")
        for i, (action, observation) in enumerate(result.steps, start=1):
          st.markdown(f"**Step {i}: `{action.tool}`**")
          st.code(str(action.tool_input), language="text")
          st.text(
              str(observation)[:800]
              + ("..." if len(str(observation)) > 800 else "")
          )

      st.session_state.messages.append({
          "role": "assistant",
          "content": guarded.text,
          "warnings": guarded.warnings,
      })
