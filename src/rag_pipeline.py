"""
src/rag_pipeline.py
====================
Configurable, isolated RAG pipeline for benchmarking different chunking /
embedding / reranking strategies against the SAME document and the SAME
fixed set of probe questions.

CRITICAL ISOLATION GUARANTEE:
This module never touches the shared production FAISS index in src/rag.py
(data/faiss_index) and never writes anything to disk. Every RAGPipeline
instance builds its own throwaway, in-memory FAISS index scoped to a single
benchmark run. Running benchmarks can never corrupt or slow down the live
chat system, and vice versa — they don't share any state.

Generation is held constant across configs on purpose: every run uses the
SAME qa_agent the live app uses, so only retrieval quality (chunking /
embedding / reranking) varies between benchmark configs, not the model doing
the answering. That's what makes comparisons between configs meaningful.

Usage:
    config = RAGConfig(
        chunking_method="recursive_char",
        chunk_size=800,
        chunk_overlap=100,
        embedding_model="all-MiniLM-L6-v2",
        use_reranker=True,
        top_k=5,
    )
    pipeline = RAGPipeline(config)
    result = pipeline.run(transcript, queries, qa_agent)   # -> BenchmarkResult
"""

import time
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
)


# ── Config ──────────────────────────────────────────────────────────────────

ChunkingMethod = Literal["fixed_char", "recursive_char", "token"]

EmbeddingModel = Literal[
    "all-MiniLM-L6-v2",           # local, fast, small — current production default
    "all-mpnet-base-v2",          # local, slower, generally more accurate
    "models/text-embedding-004",  # cloud (Gemini) — costs API calls
]

RerankerModel = Literal["cross-encoder/ms-marco-MiniLM-L-6-v2"]


class RAGConfig(BaseModel):
    chunking_method: ChunkingMethod = "recursive_char"
    chunk_size: int = Field(800, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=1000)
    embedding_model: EmbeddingModel = "all-MiniLM-L6-v2"
    use_reranker: bool = False
    reranker_model: RerankerModel = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(5, ge=1, le=20)
    # How many candidates to retrieve before reranking down to top_k.
    # Only relevant when use_reranker=True.
    rerank_candidate_pool: int = Field(15, ge=1, le=50)

    def label(self) -> str:
        """Short human-readable summary for the results table/UI."""
        parts = [
            self.chunking_method,
            f"size={self.chunk_size}",
            f"ov={self.chunk_overlap}",
            self.embedding_model.split("/")[-1],
        ]
        if self.use_reranker:
            parts.append("+rerank")
        return " · ".join(parts)


# ── Results ───────────────────────────────────────────────────────────────────

class QueryResult(BaseModel):
    query: str
    retrieved_chunks: List[str]
    answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    # Filled in afterward by src.rag_judge — left None until scored, so this
    # class stays usable even before Phase 2 (the judge) exists.
    context_relevance: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None


class BenchmarkResult(BaseModel):
    config: RAGConfig
    chunk_count: int
    avg_chunk_size: float
    indexing_latency_ms: float
    query_results: List[QueryResult]


# ── Chunking strategies ─────────────────────────────────────────────────────
# Deliberately separate from src/splitter.py: that module auto-picks ONE
# adaptive strategy based on transcript length for production use. Here the
# whole point is letting the user pick the strategy explicitly to compare them.

def _chunk(text: str, config: RAGConfig) -> List[str]:
    if config.chunking_method == "fixed_char":
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        return splitter.split_text(text)

    if config.chunking_method == "recursive_char":
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        return splitter.split_text(text)

    if config.chunking_method == "token":
        try:
            from langchain_text_splitters import TokenTextSplitter
            splitter = TokenTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                encoding_name="gpt2",
            )
            return splitter.split_text(text)
        except ImportError:
            print("[WARN] tiktoken not available, falling back to recursive_char splitter")
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " "],
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            return splitter.split_text(text)


    raise ValueError(f"Unknown chunking_method: {config.chunking_method}")


# ── Embedding factory (cached so repeat benchmark runs don't reload weights) ──

_embedding_cache: dict = {}


def _get_embedding_fn(model_name: str):
    if model_name in _embedding_cache:
        return _embedding_cache[model_name]

    if model_name == "models/text-embedding-004":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        emb = GoogleGenerativeAIEmbeddings(model=model_name)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")

    _embedding_cache[model_name] = emb
    return emb


# ── Reranker factory ─────────────────────────────────────────────────────────
# Uses sentence-transformers' CrossEncoder, already in requirements.txt via
# the sentence-transformers dependency — no new package needed.

_reranker_cache: dict = {}


def _get_reranker(model_name: str):
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(model_name)
    _reranker_cache[model_name] = reranker
    return reranker


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    One instance = one config, scoped to a single benchmark run. Builds its
    own in-memory FAISS index — never touches the shared production index in
    src/rag.py, and never writes anything to disk.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._vectorstore: Optional[FAISS] = None
        self._chunks: List[str] = []

    def index(self, transcript: str) -> float:
        """Chunk + embed the transcript into a throwaway in-memory index.
        Also warms up the reranker here (if enabled) so its one-time model
        load cost is absorbed into indexing_latency_ms — a fair one-time
        setup cost — rather than contaminating the first query's
        retrieval_latency_ms with a cold-start spike.
        Returns indexing latency in ms."""
        start = time.perf_counter()
        self._chunks = _chunk(transcript, self.config)
        if not self._chunks:
            raise ValueError("Chunking produced zero chunks — check chunk_size/overlap.")

        embedding_fn = _get_embedding_fn(self.config.embedding_model)
        self._vectorstore = FAISS.from_texts(self._chunks, embedding=embedding_fn)

        if self.config.use_reranker:
            _get_reranker(self.config.reranker_model)   # forces load/cache now, not on first query

        return (time.perf_counter() - start) * 1000

    def retrieve(self, query: str) -> List[str]:
        """Retrieve top_k chunks for a query, optionally reranked."""
        if self._vectorstore is None:
            raise RuntimeError("Call index() before retrieve().")

        pool_size = self.config.rerank_candidate_pool if self.config.use_reranker else self.config.top_k
        docs = self._vectorstore.similarity_search(query, k=pool_size)
        candidates = [d.page_content for d in docs]

        if not self.config.use_reranker or not candidates:
            return candidates[: self.config.top_k]

        reranker = _get_reranker(self.config.reranker_model)
        pairs = [(query, c) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[: self.config.top_k]]

    def generate_answer(self, query: str, context_chunks: List[str], qa_agent) -> str:
        """
        Uses the SAME qa_agent the live app uses, so generation is held
        constant across benchmark configs — only retrieval quality varies.

        A fresh random thread_id is used on every call (not per-pipeline)
        so the agent's checkpointer memory never carries context between
        probe queries in the same run — each query must be scored on its
        own, uncontaminated by earlier answers in the same benchmark.
        """
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
        result = qa_agent.invoke(
            {"messages": [{"role": "user", "content": f"Q: {query}\n\nContext:\n{context}"}]},
            {"configurable": {"thread_id": f"benchmark-{uuid.uuid4()}"}},
        )
        return result["messages"][-1].content

    def run(self, transcript: str, queries: List[str], qa_agent) -> BenchmarkResult:
        """Full end-to-end benchmark: index once, then run every probe query
        through retrieve -> generate, timing each stage independently."""
        indexing_latency_ms = self.index(transcript)

        query_results: List[QueryResult] = []
        for q in queries:
            t0 = time.perf_counter()
            chunks = self.retrieve(q)
            retrieval_latency_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            answer = self.generate_answer(q, chunks, qa_agent)
            generation_latency_ms = (time.perf_counter() - t1) * 1000

            query_results.append(QueryResult(
                query=q,
                retrieved_chunks=chunks,
                answer=answer,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
            ))

        avg_chunk_size = sum(len(c) for c in self._chunks) / len(self._chunks)

        return BenchmarkResult(
            config=self.config,
            chunk_count=len(self._chunks),
            avg_chunk_size=avg_chunk_size,
            indexing_latency_ms=indexing_latency_ms,
            query_results=query_results,
        )