
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
import os
from dotenv import load_dotenv
from app.retriever import get_retriever

load_dotenv()


retriever = get_retriever()

def pdf_search(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found in the PDF database."
    return "\n\n".join([doc.page_content for doc in docs])

pdf_search_tool = Tool(
    name="pdf_search",
    func=pdf_search,
    description="Search travel documents to answer user questions."
)


duckduckgo = DuckDuckGoSearchRun()

def web_search(query: str) -> str:
    result = duckduckgo.run(query)
    return result if result else "No web results found."

web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Use this if the answer is not found in the travel documents."
)
