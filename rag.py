# ============================================================
# SCHOLARAI - RAG
# File: rag.py
#
# Retrieves a user's own study notes, ranks with TF-IDF / keyword similarity,
# and builds context for RAG.
# ============================================================

import re
from database import notes_collection

MAX_CONTEXT_CHARS_PER_NOTE = 6000


def _simple_keyword_score(query: str, doc: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower()))
    doc_words = set(re.findall(r"\w+", doc.lower()))
    if not query_words or not doc_words:
        return 0.0
    common = query_words.intersection(doc_words)
    return len(common) / len(query_words)


def retrieve_relevant_notes(user_id: str, question: str, top_k: int = 3) -> list[dict]:
    """Returns the user's top_k most relevant notes for `question`."""
    cursor = notes_collection.find({"user_id": str(user_id)})
    notes = list(cursor)
    if not notes:
        return []

    documents = []
    valid_notes = []

    for note in notes:
        content = note.get("content", "")
        if content and content.strip():
            documents.append(content)
            valid_notes.append(note)

    if not documents:
        return []

    # Attempt TF-IDF with sklearn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        document_vectors = vectorizer.fit_transform(documents)
        question_vector = vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, document_vectors)[0]
        ranked_indexes = similarities.argsort()[::-1]

        results = []
        for index in ranked_indexes[:top_k]:
            score = float(similarities[index])
            if score <= 0:
                continue
            results.append({
                "title": valid_notes[index].get("title", "Untitled"),
                "subject": valid_notes[index].get("subject", ""),
                "content": valid_notes[index].get("content", ""),
                "score": score,
            })
        if results:
            return results
    except Exception:
        pass

    # Keyword overlap fallback
    scored = []
    for note in valid_notes:
        text = f"{note.get('title', '')} {note.get('subject', '')} {note.get('content', '')}"
        score = _simple_keyword_score(question, text)
        if score > 0:
            scored.append((score, note))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, note in scored[:top_k]:
        results.append({
            "title": note.get("title", "Untitled"),
            "subject": note.get("subject", ""),
            "content": note.get("content", ""),
            "score": round(score, 3),
        })

    return results


def build_context(retrieved_notes: list[dict]) -> str:
    if not retrieved_notes:
        return ""

    context = []
    for note in retrieved_notes:
        content_snippet = note.get("content", "")[:MAX_CONTEXT_CHARS_PER_NOTE]
        context.append(
            f"SOURCE: {note.get('title', 'Untitled')}\n"
            f"SUBJECT: {note.get('subject', 'General')}\n\n"
            f"{content_snippet}"
        )

    return "\n\n".join(context)
