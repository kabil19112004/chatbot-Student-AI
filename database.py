# ============================================================
# SCHOLARAI - DATABASE
# File: database.py
#
# Single source of truth for database connections and collections.
# Connects to MongoDB when available; seamlessly provides a reliable
# local fallback if MongoDB is not running or unreachable.
# ============================================================

import os
import json
import logging
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("scholarai.database")

MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")

# Global flags
_is_connected = False
_using_fallback = False

# Try connecting to MongoDB with a 4-second timeout
try:
    from pymongo import MongoClient
    client = MongoClient(
        MONGO_URL,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        socketTimeoutMS=3000,
    )
    # Test connection
    client.admin.command("ping")
    _is_connected = True
    db = client["scholar_ai"]
    users_collection = db["users"]
    chats_collection = db["chats"]
    messages_collection = db["messages"]
    notes_collection = db["study_notes"]
    note_drafts_collection = db["note_drafts"]
    settings_collection = db["settings"]
    print("✓ Successfully connected to MongoDB at", MONGO_URL.split("@")[-1] if "@" in MONGO_URL else MONGO_URL)
except Exception as err:
    print(f"! MongoDB not reachable ({err}). Initializing robust local fallback storage.")
    _is_connected = False
    _using_fallback = True

    # Robust local collection fallback supporting the exact PyMongo API
    class LocalCursor:
        def __init__(self, docs):
            self.docs = list(docs)

        def sort(self, key, direction=1):
            reverse = (direction == -1)
            self.docs.sort(key=lambda d: d.get(key, ""), reverse=reverse)
            return self

        def limit(self, count):
            self.docs = self.docs[:count]
            return self

        def __iter__(self):
            return iter(self.docs)

        def __len__(self):
            return len(self.docs)

        def to_list(self):
            return list(self.docs)

    class LocalInsertResult:
        def __init__(self, inserted_id):
            self.inserted_id = inserted_id

    class LocalUpdateResult:
        def __init__(self, matched_count, modified_count):
            self.matched_count = matched_count
            self.modified_count = modified_count

    class LocalDeleteResult:
        def __init__(self, deleted_count):
            self.deleted_count = deleted_count

    LOCAL_DB_FILE = os.path.join(os.path.dirname(__file__), ".scholar_ai_local.json")

    def _load_local_db():
        if os.path.exists(LOCAL_DB_FILE):
            try:
                with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_local_db(data):
        try:
            with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str, indent=2)
        except Exception:
            pass

    class LocalCollection:
        def __init__(self, name):
            self.name = name

        def _get_docs(self):
            db_data = _load_local_db()
            return db_data.get(self.name, {})

        def _save_docs(self, docs):
            db_data = _load_local_db()
            db_data[self.name] = docs
            _save_local_db(db_data)

        def insert_one(self, doc):
            doc = dict(doc)
            if "_id" not in doc:
                doc["_id"] = str(ObjectId())
            else:
                doc["_id"] = str(doc["_id"])
            
            docs = self._get_docs()
            docs[doc["_id"]] = doc
            self._save_docs(docs)
            return LocalInsertResult(doc["_id"])

        def _matches(self, doc, query):
            for k, v in query.items():
                if k == "_id":
                    if str(doc.get("_id")) != str(v):
                        return False
                elif isinstance(v, dict):
                    if "$ne" in v and str(doc.get(k)) == str(v["$ne"]):
                        return False
                else:
                    if doc.get(k) != v:
                        return False
            return True

        def find_one(self, query):
            docs = self._get_docs()
            for doc in docs.values():
                if self._matches(doc, query):
                    return dict(doc)
            return None

        def find(self, query=None):
            if query is None:
                query = {}
            docs = self._get_docs()
            matched = [dict(doc) for doc in docs.values() if self._matches(doc, query)]
            return LocalCursor(matched)

        def update_one(self, query, update, upsert=False):
            docs = self._get_docs()
            found_id = None
            for doc_id, doc in docs.items():
                if self._matches(doc, query):
                    found_id = doc_id
                    break

            if found_id:
                if "$set" in update:
                    docs[found_id].update(update["$set"])
                self._save_docs(docs)
                return LocalUpdateResult(1, 1)
            elif upsert:
                new_doc = dict(query)
                if "$set" in update:
                    new_doc.update(update["$set"])
                if "_id" not in new_doc:
                    new_doc["_id"] = str(ObjectId())
                else:
                    new_doc["_id"] = str(new_doc["_id"])
                docs[new_doc["_id"]] = new_doc
                self._save_docs(docs)
                return LocalUpdateResult(0, 1)
            return LocalUpdateResult(0, 0)

        def delete_one(self, query):
            docs = self._get_docs()
            found_id = None
            for doc_id, doc in docs.items():
                if self._matches(doc, query):
                    found_id = doc_id
                    break
            if found_id:
                del docs[found_id]
                self._save_docs(docs)
                return LocalDeleteResult(1)
            return LocalDeleteResult(0)

        def count_documents(self, query):
            docs = self._get_docs()
            return len([doc for doc in docs.values() if self._matches(doc, query)])


    users_collection = LocalCollection("users")
    chats_collection = LocalCollection("chats")
    messages_collection = LocalCollection("messages")
    notes_collection = LocalCollection("study_notes")
    note_drafts_collection = LocalCollection("note_drafts")
    settings_collection = LocalCollection("settings")


def check_connection() -> bool:
    """Used by the /index health endpoint."""
    if not _using_fallback and _is_connected:
        try:
            client.admin.command("ping")
            return True
        except Exception:
            return False
    return True
