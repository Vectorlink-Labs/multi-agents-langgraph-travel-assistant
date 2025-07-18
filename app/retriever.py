
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR, DOCS_PATH


def load_documents(path: str = DOCS_PATH):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_load_vectorstore():
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Vectorstore already exists. Loading from disk...")
        embedding_model = get_embedding_model()
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
        )
    else:
        print("Loading documents...")
        docs = load_documents()
        print(f"Loaded {len(docs)} documents.")

        print("Splitting into chunks...")
        chunks = split_documents(docs)

        print("Creating embeddings and vector store...")
        embedding_model = get_embedding_model()
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DIR
        )

        print("Vectorstore built and saved.")

    return vectordb


def get_retriever():
    vectordb = build_or_load_vectorstore()
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
