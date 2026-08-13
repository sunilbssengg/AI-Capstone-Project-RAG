import streamlit as st
from modules.ingestion import save_and_load_document
from modules.processing import load_vectorstore, process_documents_to_vectorstore
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
  GEMINI_API_KEY = st.secrets["secrets"]["GEMINI_API_KEY"]
except Exception:
  st.error(
      "API Key missing! Please set GEMINI_API_KEY inside"
      " .streamlit/secrets.toml"
  )
  st.stop()

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
          vector_store, chunk_count = process_documents_to_vectorstore(docs)

          st.success(f"Successfully processed {uploaded_file.name}!")
          st.info(
              f"Saved to: `{saved_path}`\n\nGenerated **{chunk_count}** chunks"
              " inside `chroma_db/`"
          )
        except Exception as e:
          st.error(f"Error processing document: {e}")

# Main Chat Interface for Q&A
st.markdown("---")
st.subheader("💬 Ask Questions about your Documents")

vector_store = load_vectorstore()

if vector_store is None:
  st.warning(
      "⚠️ No vector database found. Please upload and process a document via"
      " the sidebar first."
  )
else:
  rag_chain = get_qa_chain(vector_store, GEMINI_API_KEY)

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