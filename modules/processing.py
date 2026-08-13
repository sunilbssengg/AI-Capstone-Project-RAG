import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_db"


def process_documents_to_vectorstore(documents):
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

  # Initialize Embedding Model
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

  # Store vectors and metadata inside the local chroma_db folder
  vector_store = Chroma.from_documents(
      documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR
  )

  return vector_store, len(chunks)


def load_vectorstore():
  """Loads existing Chroma vector store from local storage."""
  if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
  return None