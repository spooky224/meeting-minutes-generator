"""
src/rag.py
==========
A single, persistent, shared FAISS index (on disk under data/faiss_index)
that holds chunks from EVERY uploaded artifact, each tagged with metadata:

    {"artifact_id": ..., "file_name": ...}

This lets chat search either:
  - across all uploaded files at once (no filter), or
  - a single file independently (filter={"artifact_id": ...})

The index is built/extended incrementally as each artifact finishes
processing (embed_artifact), and vectors are removed when an artifact is
deleted (delete_artifact_vectors). A full rebuild fallback exists in case
the installed langchain_community/faiss version doesn't support targeted
id-based deletion for this index type.

Thread-safety: embed_artifact() runs on a background thread while the main
Streamlit thread may be querying at the same time, so all read-modify-write
access to the shared index is guarded by a lock.
"""

import os
import shutil
import threading
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.splitter import split_transcript

INDEX_DIR = os.path.join("data", "faiss_index")

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None
_lock = threading.Lock()


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily load the embedding model once and reuse it everywhere —
    loading it repeatedly on every Streamlit rerun would be expensive."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def _load_index_locked() -> Optional[FAISS]:
    """
    Return the shared index, loading it from disk on first access.
    Must be called while holding _lock.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    index_file = os.path.join(INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        _vectorstore = FAISS.load_local(
            INDEX_DIR,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def embed_artifact(artifact_id: str, file_name: str, transcript: str) -> List[str]:
    """
    Split the transcript, embed it, and add it to the shared persistent
    index, tagged with this artifact's id. Returns the chunk ids that were
    added — the caller (pipeline) stores these on the artifact document so
    they can be targeted for deletion later.

    Safe to call from a background thread.
    """
    chunks = split_transcript(transcript)
    if not chunks:
        return []

    ids = [f"{artifact_id}::{i}" for i in range(len(chunks))]
    metadatas = [{"artifact_id": artifact_id, "file_name": file_name} for _ in chunks]

    with _lock:
        global _vectorstore
        vs = _load_index_locked()
        if vs is None:
            vs = FAISS.from_texts(
                chunks, embedding=_get_embeddings(), metadatas=metadatas, ids=ids
            )
            _vectorstore = vs
        else:
            vs.add_texts(chunks, metadatas=metadatas, ids=ids)
        os.makedirs(INDEX_DIR, exist_ok=True)
        vs.save_local(INDEX_DIR)

    return ids


def delete_artifact_vectors(chunk_ids: List[str]) -> None:
    """
    Remove one artifact's vectors from the shared index in place.

    Raises if the installed FAISS/langchain_community version doesn't
    support id-based deletion for this index type — callers should catch
    this and fall back to rebuild_index_from_artifacts().
    """
    if not chunk_ids:
        return
    with _lock:
        vs = _load_index_locked()
        if vs is None:
            return
        vs.delete(ids=chunk_ids)   # may raise NotImplementedError on some versions
        vs.save_local(INDEX_DIR)


def rebuild_index_from_artifacts(remaining_artifacts: List[dict]) -> None:
    """
    Fallback: wipe and rebuild the whole index from scratch using only the
    artifacts that should still exist.

    remaining_artifacts: list of dicts with "_id", "file_name", "transcript"
    (i.e. full artifact documents, as returned by db.get_artifact).
    """
    global _vectorstore
    with _lock:
        if not remaining_artifacts:
            _vectorstore = None
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
            return

        all_chunks, all_ids, all_meta = [], [], []
        for art in remaining_artifacts:
            chunks = split_transcript(art["transcript"])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{art['_id']}::{i}")
                all_meta.append({"artifact_id": art["_id"], "file_name": art["file_name"]})

        vs = FAISS.from_texts(
            all_chunks, embedding=_get_embeddings(), metadatas=all_meta, ids=all_ids
        )
        _vectorstore = vs
        os.makedirs(INDEX_DIR, exist_ok=True)
        vs.save_local(INDEX_DIR)


def query_faiss(query: str, k: int = 5, artifact_id: Optional[str] = None) -> List[str]:
    """
    Query the shared index. Pass artifact_id to restrict the search to a
    single file's chunks; omit it to search across every uploaded file.
    Returns an empty list if nothing has been embedded yet.
    """
    with _lock:
        vs = _load_index_locked()
    if vs is None:
        return []
    filter_dict = {"artifact_id": artifact_id} if artifact_id else None
    docs = vs.similarity_search(query, k=k, filter=filter_dict)
    return [doc.page_content for doc in docs]