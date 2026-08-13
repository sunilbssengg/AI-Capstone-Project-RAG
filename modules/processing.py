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


def process_documents_to_vectorstore(documents, api_key: str):
  """Applies chunking/tokenization concepts and stores data in ChromaDB."""

  # Chunking Strategy: Recursive text splitting tracking structural token/character boundaries
  # chunk_size defines token/character limits; chunk_overlap maintains context continuity between chunks.
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
      separators=["\n\n", "\n", " ", ""],
  )

  chunks = text_splitter.split_documents(documents)

  # Initialize Embedding Model (Gemini API — no local model, no PyTorch)
  embeddings = _get_embeddings(api_key)

  # Store vectors and metadata inside the local chroma_db folder
  vector_store = Chroma.from_documents(
      documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR
  )

  return vector_store, len(chunks)


def load_vectorstore(api_key: str):
  """Loads existing Chroma vector store from local storage."""
  if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    embeddings = _get_embeddings(api_key)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
  return None
