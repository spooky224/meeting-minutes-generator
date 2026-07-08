"""
src/db.py
=========
All MongoDB persistence for the app. Two collections:

  artifacts     — one document per uploaded file. Tracks the background
                   pipeline's lifecycle (processing → ready/failed), stores
                   the raw transcript, the generated MeetingMinutes, and the
                   FAISS chunk ids that belong to this file (so we can cleanly
                   delete its embeddings later).

  conversations — one document per chat. Scoped to a user, NOT to a file.
                   A conversation is completely independent from artifacts:
                   deleting a file never touches conversations, and a
                   conversation is only ever removed by an explicit delete.

Environment variables:
    MONGO_URI=mongodb://localhost:27017   (or Atlas URI)
    MONGO_DB=meeting_minutes             (optional, defaults to "meeting_minutes")

NOTE on auth: there is no login system yet. Every read/write is scoped to
DEFAULT_USER_ID below. Once real auth exists, swap the default value for the
authenticated user's id at the call sites in app.py — nothing else here needs
to change, since every function already takes user_id as a parameter.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

from src.schema import MeetingMinutes


# ── Static user (replace with real auth later) ─────────────────────────────

DEFAULT_USER_ID = "default_user"


# ── Connection ────────────────────────────────────────────────────────────────
# A single MongoClient is safe to share across threads (pymongo pools
# connections internally), which matters since artifacts are processed on a
# background thread while the main Streamlit thread reads/writes too.

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _client = MongoClient(uri, serverSelectionTimeoutMS=4000)
    return _client


def _database():
    db_name = os.getenv("MONGO_DB", "meeting_minutes")
    return _get_client()[db_name]


def _artifacts_col() -> Collection:
    col = _database()["artifacts"]
    col.create_index("user_id")
    col.create_index([("created_at", DESCENDING)])
    return col


def _conversations_col() -> Collection:
    col = _database()["conversations"]
    col.create_index("user_id")
    col.create_index([("updated_at", DESCENDING)])
    return col


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Artifacts ─────────────────────────────────────────────────────────────────

def create_artifact(file_name: str, transcript: str, user_id: str = DEFAULT_USER_ID) -> str:
    """
    Register a newly uploaded file. Status starts as 'processing'.
    Called synchronously (it's just a fast Mongo insert) right before the
    background pipeline thread is started.
    """
    col = _artifacts_col()
    artifact_id = str(uuid.uuid4())
    now = _now()
    col.insert_one({
        "_id":         artifact_id,
        "user_id":     user_id,
        "file_name":   file_name,
        "transcript":  transcript,
        "status":      "processing",   # processing | ready | failed
        "minutes":     None,
        "chunk_ids":   [],
        "error":       None,
        "created_at":  now,
        "updated_at":  now,
    })
    return artifact_id


def save_artifact_result(artifact_id: str, minutes: MeetingMinutes, chunk_ids: List[str]) -> None:
    """Called by the background pipeline once processing + embedding succeed."""
    _artifacts_col().update_one(
        {"_id": artifact_id},
        {"$set": {
            "status":     "ready",
            "minutes":    minutes.model_dump(),
            "chunk_ids":  chunk_ids,
            "updated_at": _now(),
        }},
    )


def mark_artifact_failed(artifact_id: str, error: str) -> None:
    """Called by the background pipeline if anything raises."""
    _artifacts_col().update_one(
        {"_id": artifact_id},
        {"$set": {"status": "failed", "error": error[:500], "updated_at": _now()}},
    )


def list_artifacts(user_id: str = DEFAULT_USER_ID) -> list:
    """
    Lightweight list for the sidebar — excludes the (potentially large)
    transcript field. Newest first.
    """
    docs = _artifacts_col().find(
        {"user_id": user_id},
        {"transcript": 0},
        sort=[("created_at", DESCENDING)],
    )
    return list(docs)


def get_artifact(artifact_id: str) -> Optional[dict]:
    """Full document, including transcript — needed for the minutes modal
    and for index-rebuild fallbacks."""
    return _artifacts_col().find_one({"_id": artifact_id})


def delete_artifact(artifact_id: str) -> Optional[dict]:
    """
    Delete the artifact document and return what it was, so the caller
    (app.py) can clean up its FAISS vectors afterward using chunk_ids.
    Does NOT touch conversations.
    """
    doc = _artifacts_col().find_one({"_id": artifact_id})
    if doc is None:
        return None
    _artifacts_col().delete_one({"_id": artifact_id})
    return doc


# ── Conversations (user-scoped, independent of any artifact) ──────────────────

def create_conversation(user_id: str = DEFAULT_USER_ID) -> str:
    """Create a blank conversation belonging to a user. No transcript link."""
    col = _conversations_col()
    conv_id = str(uuid.uuid4())
    now = _now()
    col.insert_one({
        "_id":        conv_id,
        "user_id":    user_id,
        "title":      "New conversation",
        "created_at": now,
        "updated_at": now,
        "messages":   [],
    })
    return conv_id


def list_conversations(user_id: str = DEFAULT_USER_ID) -> list:
    """All of a user's conversations, newest first — regardless of which
    files were discussed in them."""
    docs = _conversations_col().find(
        {"user_id": user_id},
        {"_id": 1, "title": 1, "updated_at": 1, "messages": 1},
        sort=[("updated_at", DESCENDING)],
    )
    return [
        {
            "id":            d["_id"],
            "title":         d["title"],
            "updated_at":    d["updated_at"],
            "message_count": len(d.get("messages", [])),
        }
        for d in docs
    ]


def load_conversation(conversation_id: str) -> Optional[dict]:
    return _conversations_col().find_one({"_id": conversation_id})


def _generate_title(first_question: str) -> str:
    q = first_question.strip().rstrip("?").strip()
    if len(q) > 55:
        q = q[:52] + "…"
    return (q[0].upper() + q[1:]) if q else "Untitled conversation"


def append_messages(
    conversation_id: str,
    user_msg: str,
    assistant_msg: str,
    scope_artifact_id: Optional[str] = None,
) -> None:
    """
    Append a user+assistant pair and update timestamp.
    Auto-generates the title from the first user message.
    scope_artifact_id records which file (if any) the question was scoped
    to, purely for transparency/history — it's never required for the
    conversation to keep working.
    """
    col = _conversations_col()
    doc = col.find_one({"_id": conversation_id}, {"messages": 1, "title": 1})
    if doc is None:
        return

    is_first = len(doc.get("messages", [])) == 0
    update = {
        "$push": {
            "messages": {
                "$each": [
                    {"role": "user",      "content": user_msg,      "scope_artifact_id": scope_artifact_id},
                    {"role": "assistant", "content": assistant_msg, "scope_artifact_id": scope_artifact_id},
                ]
            }
        },
        "$set": {"updated_at": _now()},
    }
    if is_first:
        update["$set"]["title"] = _generate_title(user_msg)

    col.update_one({"_id": conversation_id}, update)


def delete_conversation(conversation_id: str) -> None:
    """Permanently delete a conversation. This is the ONLY way conversations
    are ever removed — nothing else (including artifact deletion) cascades here."""
    _conversations_col().delete_one({"_id": conversation_id})