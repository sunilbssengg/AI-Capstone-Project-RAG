import os
import base64
import pandas as pd
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

UPLOAD_DIR = "data_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def push_file_to_github(file_path: str, github_token: str, github_repo: str,
                         repo_path_prefix: str = "data_docs") -> str | None:
  """
  Optionally commits an uploaded file to a GitHub repo for persistence
  across Streamlit Cloud restarts (local disk is wiped on reboot/redeploy).

  SECURITY WARNING: this requires a GitHub Personal Access Token with
  WRITE access to the target repo, stored as a Streamlit secret. Because
  the app is publicly accessible, anyone who uploads a file through the
  UI will trigger a commit to your repo using this token. Only enable
  this on a repo you're comfortable having written to by anonymous users,
  or add your own access control in front of the app first.

  Returns the GitHub file URL on success, or None if GITHUB_TOKEN /
  GITHUB_REPO aren't configured (silently skipped, not an error).
  """
  if not github_token or not github_repo:
    return None

  from github import Github, GithubException

  with open(file_path, "rb") as f:
    content_bytes = f.read()

  repo_path = f"{repo_path_prefix}/{os.path.basename(file_path)}"
  gh = Github(github_token)
  repo = gh.get_repo(github_repo)

  try:
    # If the file already exists at this path, update it; otherwise create it.
    existing = repo.get_contents(repo_path)
    result = repo.update_file(
        path=repo_path,
        message=f"Update uploaded document: {os.path.basename(file_path)}",
        content=content_bytes,
        sha=existing.sha,
    )
  except GithubException as exc:
    if exc.status == 404:
      result = repo.create_file(
          path=repo_path,
          message=f"Add uploaded document: {os.path.basename(file_path)}",
          content=content_bytes,
      )
    else:
      raise

  return result["content"].html_url


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
