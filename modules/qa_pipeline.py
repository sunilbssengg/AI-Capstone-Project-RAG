from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def get_qa_chain(vector_store, api_key):
  """Builds a RAG chain utilizing Gemini 2.5."""
  # Initialize Gemini 2.5 Model via LangChain
  llm = ChatGoogleGenerativeAI(
      model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2
  )

  retriever = vector_store.as_retriever(
      search_type="similarity", search_kwargs={"k": 4}
  )

  system_prompt = (
      "You are an AI enterprise assistant for question-answering tasks. "
      "Use the following pieces of retrieved context to answer "
      "the question. If you don't know the answer, say that you "
      "don't know.\n\n"
      "{context}"
  )

  prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{input}"),
  ])

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  return rag_chain
