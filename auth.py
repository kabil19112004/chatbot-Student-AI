# ============================================================
# SCHOLARAI - AUTH
# File: auth.py
#
# Robust authentication with JWT and bcrypt password hashing.
# ============================================================

import os
import hashlib
import warnings
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError
from fastapi import HTTPException, Header
from bson import ObjectId
from bson.errors import InvalidId

from database import users_collection

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    warnings.warn(
        "SECRET_KEY is not set in the environment — using an insecure "
        "development fallback. Set SECRET_KEY in your .env before deploying.",
        stacklevel=1,
    )
    SECRET_KEY = "scholarai-secret-key-production-ready"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ------------------------------------------------------------
# PASSWORD HELPERS (Direct bcrypt for reliability, avoids passlib __about__ bug)
# ------------------------------------------------------------

def hash_password(password: str) -> str:
    try:
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    except Exception:
        # Fallback to salted SHA-256 if bcrypt is compiling
        salt = os.urandom(16).hex()
        digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return f"sha256${salt}${digest}"


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        if hashed_password.startswith("sha256$"):
            parts = hashed_password.split("$")
            salt, digest = parts[1], parts[2]
            return hashlib.sha256((salt + password).encode("utf-8")).hexdigest() == digest
        import bcrypt
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


# ------------------------------------------------------------
# JWT HELPERS
# ------------------------------------------------------------

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ------------------------------------------------------------
# CURRENT USER DEPENDENCY
# ------------------------------------------------------------

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    parts = authorization.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Token missing from Authorization header")

    token = parts[1].strip()

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = None
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        pass

    if not user:
        user = users_collection.find_one({"_id": user_id})

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
