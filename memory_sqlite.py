from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sqlite3

from memory_system import MemoryStore


SCHEMA_VERSION = 1


def _connect(path: str | Path) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            position INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_position
            ON memories(position);
        """
    )


def save_memory_store_sqlite(path: str | Path, store: MemoryStore) -> None:
    data = store.to_dict()
    target = Path(path)
    with _connect(target) as conn:
        _ensure_schema(conn)
        with conn:
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM metadata")
            conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    (
                        "embedding_config",
                        json.dumps(data["embedding_config"], ensure_ascii=False, separators=(",", ":")),
                    ),
                ],
            )
            conn.executemany(
                "INSERT INTO memories(memory_id, position, payload_json) VALUES (?, ?, ?)",
                [
                    (
                        memory["memory_id"],
                        position,
                        json.dumps(memory, ensure_ascii=False, separators=(",", ":")),
                    )
                    for position, memory in enumerate(data["memories"])
                ],
            )


def load_memory_store_sqlite(path: str | Path) -> MemoryStore:
    target = Path(path)
    with _connect(target) as conn:
        _ensure_schema(conn)
        metadata = {
            key: value
            for key, value in conn.execute("SELECT key, value FROM metadata")
        }
        raw_config = metadata.get("embedding_config", "{}")
        memories = [
            json.loads(payload_json)
            for (payload_json,) in conn.execute(
                "SELECT payload_json FROM memories ORDER BY position"
            )
        ]

    data: dict[str, Any] = {
        "embedding_config": json.loads(raw_config),
        "memories": memories,
    }
    return MemoryStore.from_dict(data)


def sqlite_store_status(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {
            "exists": False,
            "memory_count": None,
            "backend": None,
            "model_name": None,
            "schema_version": None,
        }

    with _connect(target) as conn:
        _ensure_schema(conn)
        metadata = {
            key: value
            for key, value in conn.execute("SELECT key, value FROM metadata")
        }
        (memory_count,) = conn.execute("SELECT COUNT(*) FROM memories").fetchone()

    raw_config = json.loads(metadata.get("embedding_config", "{}"))
    return {
        "exists": True,
        "memory_count": memory_count,
        "backend": raw_config.get("backend"),
        "model_name": raw_config.get("model_name"),
        "schema_version": metadata.get("schema_version"),
    }
