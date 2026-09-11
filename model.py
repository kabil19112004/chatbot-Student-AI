# ============================================================
# SCHOLARAI - MODELS
# File: model.py
#
# All Pydantic request/response models.
# ============================================================

from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator


# ------------------------------------------------------------
# AUTH
# ------------------------------------------------------------

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    course: Optional[str] = None
    college: Optional[str] = None
    avatar: Optional[str] = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip() if v is not None else v


# ------------------------------------------------------------
# CHAT
# ------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


# ------------------------------------------------------------
# NOTES
# ------------------------------------------------------------

class NoteRequest(BaseModel):
    title: str
    subject: str = "General"
    content: str

    @field_validator("title", "subject", "content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class FavoriteRequest(BaseModel):
    favorite: bool


class DraftNoteRequest(BaseModel):
    title: Optional[str] = ""
    subject: Optional[str] = "General"
    content: Optional[str] = ""


VALID_AI_TOOLS = {
    "Summarize Note",
    "Explain Concept",
    "Generate Questions",
    "Generate Flashcards",
    "Train & Fine-Tune Guide",
    "Deep Research & Synthesis",
}


class AIToolRequest(BaseModel):
    note_id: str
    tool: str

    @field_validator("tool")
    @classmethod
    def tool_is_valid(cls, v: str) -> str:
        if v not in VALID_AI_TOOLS:
            raise ValueError(f"Invalid AI tool. Must be one of {sorted(VALID_AI_TOOLS)}")
        return v


class DeepSearchRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    target_engine: Optional[str] = "llama3.2:latest"

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class ModelSwitchRequest(BaseModel):
    model_name: str

    @field_validator("model_name")
    @classmethod
    def model_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

VALID_THEMES = {"light", "dark", "system"}
VALID_STYLES = {"Balanced Academic", "Concise", "Detailed", "Friendly Tutor"}
VALID_LANGUAGES = {"English (US)", "English (UK)", "Tamil", "Hindi"}


class SettingsRequest(BaseModel):
    theme: str = "light"
    language: str = "English (US)"
    response_style: str = "Balanced Academic"
    show_sources: bool = True
    ai_suggestions: bool = True
    notif_study: bool = True
    notif_quiz: bool = True
    notif_digest: bool = False

    @field_validator("theme")
    @classmethod
    def theme_is_valid(cls, v: str) -> str:
        if v not in VALID_THEMES:
            return "light"
        return v

    @field_validator("response_style")
    @classmethod
    def style_is_valid(cls, v: str) -> str:
        if v not in VALID_STYLES:
            return "Balanced Academic"
        return v

    @field_validator("language")
    @classmethod
    def language_is_valid(cls, v: str) -> str:
        if v not in VALID_LANGUAGES:
            return "English (US)"
        return v
