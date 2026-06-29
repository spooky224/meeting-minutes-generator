# src/rag.py
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.splitter import split_transcript

def build_faiss_index(transcript: str) -> FAISS:
    """
    Split transcript into chunks, embed them, and store in FAISS index.
    Returns the FAISS vectorstore.
    """
    # Step 1: split transcript into chunks
    chunks = split_transcript(transcript)

    # Step 2: create embeddings (HuggingFace local model)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: build FAISS index
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    return vectorstore

def query_faiss(vectorstore: FAISS, query: str, k: int = 5) -> List[str]:
    """
    Query the FAISS index for top-k relevant transcript chunks.
    Returns the raw text chunks.
    """
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
