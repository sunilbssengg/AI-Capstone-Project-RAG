import os

# Suppress ChromaDB's telemetry warning (harmless version-mismatch noise,
# not a functional issue) — must be set before chromadb is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_db"

# Gemini's embedding model — lightweight (API-based), no local model
# download, no PyTorch. Reuses the same GEMINI_API_KEY already required
# for the LLM, so no extra secret is needed.
# NOTE: "text-embedding-004" was retired by Google in early 2026 (returns
# 404). The current model is "gemini-embedding-001".
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


# Chars-per-token is model-dependent; Gemini doesn't expose a public local
# tokenizer, so this is a standard, commonly-used approximation (~4 chars
# per token for English text) — good enough to show the user a ballpark
# figure, not an exact billing count.
CHARS_PER_TOKEN_ESTIMATE = 4

EMBED_BATCH_SIZE = 10


def process_documents_to_vectorstore(
    documents, api_key: str, progress_callback=None, batch_size: int = EMBED_BATCH_SIZE
):
  """
  Applies chunking/tokenization concepts and stores data in ChromaDB.

  progress_callback(stage: str, **info) is called at each step so a UI
  (e.g. Streamlit) can render live token/chunking/embedding detail
  instead of a single opaque spinner. Stages emitted, in order:
    - "chunking_start"
    - "chunking_done"   -> chunk_count, total_chars, approx_tokens
    - "embedding_progress" -> done, total (called once per batch)
    - "embedding_done"  -> total
  progress_callback is optional; pass None to run silently as before.
  """

  def emit(stage, **info):
    if progress_callback is not None:
      progress_callback(stage, **info)

  # Chunking Strategy: Recursive text splitting tracking structural token/character boundaries
  # chunk_size defines token/character limits; chunk_overlap maintains context continuity between chunks.
  emit("chunking_start")
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
      separators=["\n\n", "\n", " ", ""],
  )

  chunks = text_splitter.split_documents(documents)
  total_chars = sum(len(c.page_content) for c in chunks)
  approx_tokens = total_chars // CHARS_PER_TOKEN_ESTIMATE
  emit(
      "chunking_done",
      chunk_count=len(chunks),
      total_chars=total_chars,
      approx_tokens=approx_tokens,
  )

  # Initialize Embedding Model (Gemini API — no local model, no PyTorch)
  embeddings = _get_embeddings(api_key)

  # Create the (possibly-empty, possibly-existing) persistent Chroma store
  # up front, then add chunks in small batches instead of one single
  # from_documents() call — this is what lets us report real progress
  # instead of blocking silently on one giant embedding request.
  vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

  total = len(chunks)
  for start in range(0, total, batch_size):
    batch = chunks[start : start + batch_size]
    vector_store.add_documents(batch)
    emit("embedding_progress", done=min(start + batch_size, total), total=total)

  emit("embedding_done", total=total)

  return vector_store, len(chunks)


def load_vectorstore(api_key: str):
  """Loads existing Chroma vector store from local storage."""
  if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    embeddings = _get_embeddings(api_key)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
  return None
