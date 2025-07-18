
# 🧳 Agentic Travel Assistant

A smart travel assistant that searches through your documents and the web to answer travel questions.

A multi-agent AI assistant built with **LangChain + LangGraph**, capable of:
- Extracting answers from travel-related **PDF documents**
- Using **web search (DuckDuckGo)** only if answer is not found in PDFs
- Fully autonomous decision-making using **OpenAI GPT model (e.g. GPT-4 / GPT-4o)**

- Served via **FastAPI backend**
- With a sleek **Streamlit frontend**

---

## Features

- PDF-based RAG search (with BGE embeddings + Chroma)  
- Web fallback using Tavily (only when needed)  
- Tool-routing agent powered by LangGraph  
-  Agent stops when satisfied — no wasted web calls  
-  Logs which tools were used and why  
- FastAPI backend + Streamlit chat UI

---

## Project Structure
```
LANGGRAPH PROJECT/
├── __pycache__/
├── api/
├── app/
│   ├── __pycache__/
│   ├── agents.py
│   ├── config.py
│   ├── models.py
│   ├── retriever.py
│   └── tools.py
├── chroma_store/
├── data/
├── ui/
│   └── interface.py
├── venv/
├── .env
├── graph.ipynb
├── README.md
├── requirements.txt
└── run.py

```

### Web Interface
Just open the web app and start chatting!

## Dependencies

- Python 3.8+
- OpenAI API key
- LangChain, LangGraph, FastAPI, Streamlit

## Tech Stack

- **LLM**: OpenAI GPT model
- **Framework**: LangChain + LangGraph  
- **Vector Store**: ChromaDB
- **Web Search**: DuckDuckGo
- **UI**: Streamlit + FastAPI

## License

MIT License