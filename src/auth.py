"""
auth.py — User authentication utilities
---------------------------------------
Handles:
  - User signup (with bcrypt password hashing)
  - User login (verify credentials)
  - JWT issuance and validation
Environment variables:
  JWT_SECRET=supersecret   (set your own strong secret in .env)
"""
import re
import os
import uuid
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional
from pymongo.collection import Collection
from src.db import _database, _now

# ── JWT Config ───────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET", "supersecret")
ALGORITHM = "HS256"
TOKEN_EXP_HOURS = 12


# ── Users Collection ─────────────────────────────────────────────────────────
def _users_col() -> Collection:
    col = _database()["users"]
    col.create_index("email", unique=True)
    col.create_index("username", unique=True)
    return col


# ── User Management ──────────────────────────────────────────────────────────
def create_user(username, email, password):
    users = _users_col()

    # Check if email or username already exists
    if users.find_one({"email": email}):
        raise ValueError("Email already taken")
    if users.find_one({"username": username}):
        raise ValueError("Username already taken")

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    user_doc = {"username": username, "email": email, "password": hashed_pw}
    result = users.insert_one(user_doc)
    return str(result.inserted_id)
    



def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return "Password must contain an uppercase letter"
    if not re.search(r"[a-z]", password):
        return "Password must contain a lowercase letter"
    if not re.search(r"\d", password):
        return "Password must contain a number"
    if not re.search(r"[!@#$%^&*]", password):
        return "Password must contain a special character"
    return None

def find_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email."""
    return _users_col().find_one({"email": email})

def find_user_by_id(user_id: str) -> Optional[dict]:
    """Look up a user by id."""
    return _users_col().find_one({"_id": user_id})

def verify_user(email: str, password: str) -> Optional[str]:
    """Check credentials. Return user_id if valid, else None."""
    user = find_user_by_email(email)
    if not user:
        return None
    if bcrypt.checkpw(password.encode("utf-8"), user["hashed_password"].encode("utf-8")):
        return user["_id"]
    return None


# ── JWT Helpers ──────────────────────────────────────────────────────────────
def issue_jwt(user_id: str) -> str:
    """Create a JWT for a given user_id."""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXP_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_jwt(token: str) -> Optional[str]:
    """Decode JWT and return user_id if valid, else None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("user_id")
    except Exception:
        return None

def is_valid_email(email: str) -> bool:
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None