import os
import pandas as pd
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

UPLOAD_DIR = "data_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_and_load_document(uploaded_file):
  """Saves uploaded enterprise document locally and loads it via LangChain."""
  file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

  # Save file to local directory
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  ext = uploaded_file.name.split(".")[-1].lower()
  documents = []

  # Route loader based on file extension
  if ext == "pdf":
    loader = PyPDFLoader(file_path)
    documents = loader.load()
  elif ext == "txt":
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
  elif ext == "csv":
    loader = CSVLoader(file_path)
    documents = loader.load()
  elif ext in ["xls", "xlsx"]:
    # Convert Excel rows to structured document text entries
    excel_data = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in excel_data.items():
      for index, row in df.iterrows():
        row_str = f"Sheet: {sheet_name}, " + ", ".join(
            [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        )
        from langchain_core.documents import Document

        documents.append(
            Document(
                page_content=row_str,
                metadata={"source": uploaded_file.name, "sheet": sheet_name},
            )
        )
  else:
    raise ValueError(f"Unsupported file format: {ext}")

  return documents, file_path
