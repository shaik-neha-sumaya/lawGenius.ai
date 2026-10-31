"""LawGenius chat, IPC retrieval and per-user chat storage."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = BASE_DIR / "data" / "lawgenius.sqlite3"
LAWS_PATH = BASE_DIR / "laws_raw.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite").strip()
ENABLE_SEMANTIC_RAG = os.getenv("ENABLE_SEMANTIC_RAG", "false").lower() == "true"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "law-ai/InLegalBERT").strip()
_vectorstore = None
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = """You are LawGenius AI, an educational assistant focused on Indian law.
Answer clearly and distinguish statutory text from general explanation. Use the
retrieved IPC material only when relevant. Do not invent statutes, section numbers,
case citations, quotations, or legal outcomes. Flag when the law may have changed
or when a lawyer should verify current law. This is educational information, not
legal advice. The Indian Penal Code has been replaced for many matters by the
Bharatiya Nyaya Sanhita from 1 July 2024; explain the date/context carefully and
never assume IPC applies to a current incident without checking transitional rules.
"""


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS chats (
        chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
        history TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at DESC)")
    con.commit()
    return con


def _empty_chat():
    return {"generated": [], "past": [], "source": []}


def create_new_chat(user_id: str) -> str:
    chat_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute("INSERT INTO chats(chat_id,user_id,title,history,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (chat_id, str(user_id), "New chat", json.dumps(_empty_chat()), now, now))
    return chat_id


def get_chat_list(user_id: str) -> list:
    with _connect() as con:
        rows = con.execute("SELECT chat_id,title FROM chats WHERE user_id=? ORDER BY updated_at DESC", (str(user_id),)).fetchall()
    return [{"name": row["chat_id"], "title": row["title"]} for row in rows]


def load_chat(chat_name: str, user_id: str) -> dict:
    with _connect() as con:
        row = con.execute("SELECT history FROM chats WHERE chat_id=? AND user_id=?", (str(chat_name), str(user_id))).fetchone()
    if not row:
        raise KeyError("Chat not found")
    try:
        data = json.loads(row["history"])
        return {"generated": data.get("generated", []), "past": data.get("past", []), "source": data.get("source", [])}
    except (TypeError, json.JSONDecodeError):
        return _empty_chat()


def _save_chat(chat_name: str, user_id: str, chat_data: dict, first_message: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    title = (first_message.strip().replace("\n", " ")[:55] or "New chat")
    with _connect() as con:
        row = con.execute("SELECT title FROM chats WHERE chat_id=? AND user_id=?", (str(chat_name), str(user_id))).fetchone()
        if not row:
            raise KeyError("Chat not found")
        if row["title"] == "New chat" and first_message:
            con.execute("UPDATE chats SET title=?,history=?,updated_at=? WHERE chat_id=? AND user_id=?",
                        (title, json.dumps(chat_data, ensure_ascii=False), now, str(chat_name), str(user_id)))
        else:
            con.execute("UPDATE chats SET history=?,updated_at=? WHERE chat_id=? AND user_id=?",
                        (json.dumps(chat_data, ensure_ascii=False), now, str(chat_name), str(user_id)))


@lru_cache(maxsize=1)
def _ipc_data():
    if not LAWS_PATH.exists():
        return {}
    with LAWS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("IPC", {}) if isinstance(payload, dict) else {}


def _section_entry(section_number: str):
    ipc = _ipc_data()
    return ipc.get(f"section{section_number}") or ipc.get(section_number)


def hybrid_retrieve(query: str, k: int = 5):
    """Retrieve exact IPC sections and lexical matches from the supplied IPC JSON."""
    ipc = _ipc_data()
    if not ipc:
        return "", "GEN"
    found, seen = [], set()
    for number in re.findall(r"\b(?:ipc\s*)?section\s*(\d+[A-Za-z]?)\b", query, re.I):
        details = _section_entry(number)
        if details:
            key = f"section{number}"
            seen.add(key)
            found.append(f"{key}: {details.get('title','')}\n{details.get('content','')[:5000]}")
    terms = {w for w in re.findall(r"[a-zA-Z]{3,}", query.lower()) if w not in {
        "what", "which", "where", "when", "does", "this", "that", "with", "from", "under", "about", "explain", "tell", "please"
    }}
    scored = []
    for key, details in ipc.items():
        if key in seen:
            continue
        title = str(details.get("title", ""))
        content = str(details.get("content", ""))
        haystack = (title + " " + content[:2500]).lower()
        score = sum(haystack.count(term) * (3 if term in title.lower() else 1) for term in terms)
        if score:
            scored.append((score, key, details))
    scored.sort(key=lambda item: item[0], reverse=True)
    for _, key, details in scored[:max(0, k-len(found))]:
        found.append(f"{key}: {details.get('title','')}\n{str(details.get('content',''))[:3500]}")

    semantic_found = []
    if ENABLE_SEMANTIC_RAG:
        try:
            semantic_found = _semantic_retrieve(query, k)
            existing = set(found)
            for passage in semantic_found:
                if passage not in existing:
                    found.append(passage)
        except Exception as exc:
            # Keyword/section retrieval remains available if model download or FAISS fails.
            print(f"Optional semantic retrieval unavailable: {exc}")

    if semantic_found and scored:
        source = "HYBRID"
    elif semantic_found:
        source = "RAG"
    else:
        source = "IPC_JSON" if found else "GEN"
    return "\n\n".join(found[:k + 1]), source


def _semantic_retrieve(query: str, k: int = 5):
    """Optional InLegalBERT + FAISS retrieval; enabled via .env."""
    global _vectorstore
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("Install the optional FAISS/LangChain packages from requirements.txt") from exc

    index_dir = BASE_DIR / "data" / "ipc_embed_db_inlegalbert"
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"cache_folder": str(BASE_DIR / ".model_cache")},
        )
        if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
            _vectorstore = FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )
        else:
            docs = []
            for key, details in _ipc_data().items():
                docs.append(Document(
                    page_content=f"{key}: {details.get('title','')}\n{details.get('content','')}",
                    metadata={"section": key, "title": details.get("title", "")},
                ))
            if not docs:
                return []
            _vectorstore = FAISS.from_documents(docs, embeddings)
            index_dir.mkdir(parents=True, exist_ok=True)
            _vectorstore.save_local(str(index_dir))
    docs = _vectorstore.similarity_search(query, k=k)
    return [doc.page_content[:4000] for doc in docs]


def _generate(prompt: str) -> str:
    if client is None:
        raise RuntimeError("Gemini is not configured. Add GEMINI_API_KEY to the project .env file.")
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 1200},
    )
    answer = (getattr(response, "text", None) or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty response. Please retry.")
    return answer


def process_input(chat_name: str, user_input: str, user_id: str, return_source: bool = False):
    question = (user_input or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    if len(question) > 8000:
        raise ValueError("Please keep your question under 8,000 characters.")
    history = load_chat(chat_name, user_id)
    context, source = hybrid_retrieve(question)
    recent = list(zip(history["past"][-6:], history["generated"][-6:]))
    conversation = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in recent)
    prompt = f"Conversation so far:\n{conversation or '(none)'}\n\nRetrieved IPC context (may be incomplete/outdated):\n{context or '(no matching IPC passage retrieved)'}\n\nUser question: {question}\n\nAnswer the question. If context is insufficient, say so rather than guessing. Mention relevant dates and current-law caveats where appropriate."
    answer = _generate(prompt)
    history["past"].append(question)
    history["generated"].append(answer)
    history["source"].append(source)
    _save_chat(chat_name, user_id, history, first_message=question)
    return (answer, source) if return_source else answer
