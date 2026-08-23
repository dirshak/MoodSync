"""SQLite persistence layer. Every operation opens and closes its own short-lived
connection so this is safe to call from both Flask request threads and the
background generation thread."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "emotionground.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    text TEXT NOT NULL,
    audio_path TEXT,
    valence REAL,
    arousal REAL,
    status TEXT NOT NULL DEFAULT 'done',
    created_at TEXT NOT NULL,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with closing(get_conn()) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with closing(get_conn()) as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    """Runs an INSERT/UPDATE/DELETE and returns lastrowid."""
    with closing(get_conn()) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


# --- Chats -----------------------------------------------------------------

def create_chat(title: str) -> int:
    return execute(
        "INSERT INTO chats (title, created_at) VALUES (?, ?)",
        (title, now_iso()),
    )


def list_chats() -> list[dict]:
    rows = query(
        """
        SELECT c.id, c.title, c.created_at,
               (SELECT m.text FROM messages m
                WHERE m.chat_id = c.id ORDER BY m.id DESC LIMIT 1) AS preview
        FROM chats c
        ORDER BY c.id DESC
        """
    )
    return [dict(r) for r in rows]


def get_chat(chat_id: int) -> dict | None:
    row = query_one("SELECT id, title, created_at FROM chats WHERE id = ?", (chat_id,))
    return dict(row) if row else None


def update_chat_title(chat_id: int, title: str) -> None:
    execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))


def delete_chat(chat_id: int) -> None:
    execute("DELETE FROM chats WHERE id = ?", (chat_id,))


# --- Messages ----------------------------------------------------------------

def list_messages(chat_id: int) -> list[dict]:
    rows = query(
        """
        SELECT id, chat_id, role, text, audio_path, valence, arousal, status, created_at
        FROM messages WHERE chat_id = ? ORDER BY id ASC
        """,
        (chat_id,),
    )
    return [dict(r) for r in rows]


def get_message(message_id: int) -> dict | None:
    row = query_one(
        """
        SELECT id, chat_id, role, text, audio_path, valence, arousal, status, created_at
        FROM messages WHERE id = ?
        """,
        (message_id,),
    )
    return dict(row) if row else None


def create_message(
    chat_id: int,
    role: str,
    text: str,
    audio_path: str | None = None,
    valence: float | None = None,
    arousal: float | None = None,
    status: str = "done",
) -> int:
    return execute(
        """
        INSERT INTO messages (chat_id, role, text, audio_path, valence, arousal, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (chat_id, role, text, audio_path, valence, arousal, status, now_iso()),
    )


def update_message(
    message_id: int,
    text: str | None = None,
    audio_path: str | None = None,
    status: str | None = None,
) -> None:
    fields, params = [], []
    if text is not None:
        fields.append("text = ?")
        params.append(text)
    if audio_path is not None:
        fields.append("audio_path = ?")
        params.append(audio_path)
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if not fields:
        return
    params.append(message_id)
    execute(f"UPDATE messages SET {', '.join(fields)} WHERE id = ?", tuple(params))
