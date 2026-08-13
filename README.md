enterprise-rag-app/
│
├── .streamlit/
│   └── secrets.toml
├── data_docs/                # Saves uploaded PDF, TXT, CSV, Excel files
├── chroma_db/                # Stores Chroma DB vector store and metadata
├── modules/
│   ├── __init__.py
│   ├── ingestion.py          # Handles file loaders for PDF, TXT, CSV, Excel
│   ├── processing.py         # Handles tokenization, chunking, and embedding
│   └── qa_pipeline.py        # Handles vector store retrieval and Gemini 2.5 LLM chain
├── app.py                    # Main Streamlit Interface
└── requirements.txt




https://sunil-ai-capstone-project-rag.streamlit.app/
