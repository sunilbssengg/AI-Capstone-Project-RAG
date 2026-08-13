import streamlit as st
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
from modules.qa_pipeline import get_qa_chain

st.set_page_config(
    page_title="Enterprise RAG with Gemini 2.5", layout="wide"
)

st.title("📄 Enterprise RAG Document Pipeline & Q&A")
st.write(
    "Upload enterprise documents (PDF, TXT, CSV, Excel), chunk them, store"
    " metadata locally in ChromaDB, and query via Gemini 2.5."
)

# Retrieve API Key from Streamlit Secrets
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

# Optional: GitHub persistence for uploaded files. Only active if both
# secrets are set — silently skipped otherwise (local-disk-only behavior,
# same as before). See SECURITY WARNING in modules/ingestion.py before
# enabling this on a public app.
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")  # e.g. "username/reponame"

# On a fresh session (e.g. after a Streamlit Cloud restart/redeploy wipes
# local disk), restore chroma_db from GitHub before anything tries to load
# it. Only runs once per session and only if GitHub secrets are set.
if GITHUB_TOKEN and GITHUB_REPO and "chroma_restored" not in st.session_state:
  with st.spinner("Restoring vector database from GitHub..."):
    try:
      pull_folder_from_github(CHROMA_DIR, GITHUB_TOKEN, GITHUB_REPO)
    except Exception as restore_err:
      st.warning(f"Could not restore chroma_db from GitHub: {restore_err}")
  st.session_state["chroma_restored"] = True

# Layout: Sidebar for Uploads, Main Area for Chat
with st.sidebar:
  st.header("📂 Document Ingestion")
  uploaded_file = st.file_uploader(
      "Upload Enterprise Document", type=["pdf", "txt", "csv", "xlsx", "xls"]
  )

  if uploaded_file is not None:
    if st.button("Process & Ingest Document"):
      with st.spinner("Saving file, chunking text, and embedding to Chroma..."):
        try:
          # Step 1: Save and load file
          docs, saved_path = save_and_load_document(uploaded_file)

          # Step 2: Process, Chunk, Embed & Save to Chroma DB
          vector_store, chunk_count = process_documents_to_vectorstore(
              docs, GEMINI_API_KEY
          )

          # Step 3 (optional): Persist the raw file AND the updated
          # chroma_db/ vector store to GitHub so both survive Streamlit
          # Cloud restarts. Skipped silently if GITHUB_TOKEN / GITHUB_REPO
          # secrets aren't configured.
          github_url = None
          chroma_files_pushed = 0
          if GITHUB_TOKEN and GITHUB_REPO:
            try:
              github_url = push_file_to_github(
                  saved_path, GITHUB_TOKEN, GITHUB_REPO
              )
              chroma_files_pushed = push_folder_to_github(
                  CHROMA_DIR, GITHUB_TOKEN, GITHUB_REPO
              )
            except Exception as gh_err:
              st.warning(f"Saved locally, but GitHub backup failed: {gh_err}")

          st.success(f"Successfully processed {uploaded_file.name}!")
          st.info(
              f"Saved to: `{saved_path}`\n\nGenerated **{chunk_count}** chunks"
              " inside `chroma_db/`"
              + (f"\n\nBacked up to GitHub: {github_url}" if github_url else "")
              + (
                  f"\n\nSynced **{chroma_files_pushed}** vector store files"
                  " to GitHub"
                  if chroma_files_pushed
                  else ""
              )
          )
        except Exception as e:
          st.error(f"Error processing document: {e}")

# Main Chat Interface for Q&A
st.markdown("---")
st.subheader("💬 Ask Questions about your Documents")

vector_store = load_vectorstore(GEMINI_API_KEY)

if vector_store is None:
  st.warning(
      "⚠️ No vector database found. Please upload and process a document via"
      " the sidebar first."
  )
else:
  # Cache the chain so it isn't rebuilt (and the LLM client re-created) on
  # every single chat message / script rerun.
  @st.cache_resource(show_spinner=False)
  def _build_chain(_vector_store, api_key):
    return get_qa_chain(_vector_store, api_key)

  rag_chain = _build_chain(vector_store, GEMINI_API_KEY)

  # Initialize chat history state
  if "messages" not in st.session_state:
    st.session_state.messages = []

  # Display prior chat messages
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # Accept user prompt
  if user_query := st.chat_input(
      "Type your question regarding the uploaded documents..."
  ):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      with st.spinner("Thinking with Gemini 2.5..."):
        try:
          response = rag_chain.invoke({"input": user_query})
          answer = response["answer"]
          st.markdown(answer)
          st.session_state.messages.append(
              {"role": "assistant", "content": answer}
          )
        except Exception as e:
          error_msg = f"An error occurred during generation: {e}"
          st.error(error_msg)
