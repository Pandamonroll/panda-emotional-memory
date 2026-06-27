from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import sqlite3

from memory_inference import AffectShadow, clamp
from memory_system import EmbeddingConfig, MemoryItem, MemoryStore


DEFAULT_DB_PATH = Path(os.environ.get("PANDA_MEMORY_DB", "panda_memory.sqlite"))
DEFAULT_MODEL_PATH = Path(".models/sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_OUTPUT_PATH = Path("shadow_panda_memories.json")


@dataclass
class PandaRow:
    memory_id: str
    text: str
    category: str
    tags: list[str]
    source: str
    importance: float
    valence: float
    emotion_scores: dict[str, float]
    metadata: dict[str, object]
    created_at: str
    updated_at: str


def load_rows(db_path: Path) -> list[PandaRow]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        rows = []
        for record in cur.execute(
            """
            SELECT
                memory_id,
                text,
                category,
                tags_json,
                source,
                importance,
                valence,
                emotion_scores_json,
                metadata_json,
                created_at,
                updated_at
            FROM memories
            ORDER BY created_at
            """
        ):
            rows.append(
                PandaRow(
                    memory_id=record[0],
                    text=record[1],
                    category=record[2],
                    tags=json.loads(record[3]),
                    source=record[4],
                    importance=float(record[5]),
                    valence=float(record[6]),
                    emotion_scores=json.loads(record[7]),
                    metadata=json.loads(record[8]),
                    created_at=record[9],
                    updated_at=record[10],
                )
            )
        return rows
    finally:
        conn.close()


def infer_affect(row: PandaRow) -> AffectShadow:
    emotions = row.emotion_scores
    joy = float(emotions.get("joy", 0.0))
    sadness = float(emotions.get("sadness", 0.0))
    anger = float(emotions.get("anger", 0.0))
    fear = float(emotions.get("fear", 0.0))
    trust = float(emotions.get("trust", 0.0))
    anticipation = float(emotions.get("anticipation", 0.0))
    surprise = float(emotions.get("surprise", 0.0))
    disgust = float(emotions.get("disgust", 0.0))

    arousal = clamp(0.50 * fear + 0.42 * anger + 0.36 * anticipation + 0.34 * surprise + 0.22 * joy, -1.0, 1.0)
    tenderness = clamp(0.58 * trust + 0.28 * joy - 0.30 * anger - 0.20 * disgust, -1.0, 1.0)
    tension = clamp(0.64 * fear + 0.50 * anger + 0.28 * sadness + 0.20 * surprise - 0.22 * trust, -1.0, 1.0)

    relationship_bonus = 0.22 if row.category == "relationship" else 0.0
    intimacy = clamp(0.44 * trust + 0.18 * joy - 0.18 * disgust + relationship_bonus, -1.0, 1.0)

    dominant_emotion = None
    if emotions:
        best_name, best_score = max(emotions.items(), key=lambda item: item[1])
        if best_score > 0.08:
            dominant_emotion = best_name

    return AffectShadow(
        valence=clamp(row.valence, -1.0, 1.0),
        arousal=arousal,
        tenderness=tenderness,
        tension=tension,
        intimacy=intimacy,
        dominant_emotion=dominant_emotion,
    )


def infer_relationship_weight(row: PandaRow) -> float:
    base_by_category = {
        "relationship": 0.82,
        "identity": 0.60,
        "preference": 0.56,
        "goal": 0.46,
        "mood": 0.52,
        "general": 0.40,
    }
    base = base_by_category.get(row.category, 0.42)
    tag_bonus = 0.0
    if "affection" in row.tags or "shared-history" in row.tags:
        tag_bonus += 0.08
    if "memory-system" in row.tags:
        tag_bonus += 0.04
    return clamp(base + tag_bonus)


def infer_abstraction_level(row: PandaRow, affect: AffectShadow) -> float:
    by_category = {
        "identity": 0.86,
        "preference": 0.76,
        "goal": 0.66,
        "relationship": 0.52,
        "mood": 0.44,
        "general": 0.50,
    }
    base = by_category.get(row.category, 0.50)
    if max(affect.tension, affect.arousal) > 0.30:
        base -= 0.10
    if affect.intimacy > 0.30:
        base -= 0.04
    return clamp(base)


def to_memory_item(row: PandaRow) -> MemoryItem:
    affect = infer_affect(row)
    tags = sorted(set(row.tags + [row.category]))
    return MemoryItem(
        summary=row.text,
        original_text=row.text,
        tags=tags,
        imprint_strength=clamp(row.importance),
        relationship_weight=infer_relationship_weight(row),
        abstraction_level=infer_abstraction_level(row, affect),
        affect_shadow=affect,
        created_at=row.created_at,
        source=f"panda-import:{row.source}",
        memory_id=row.memory_id,
    )


def choose_embedding_config(backend: str, model_name: str | None) -> EmbeddingConfig:
    if backend == "transformers":
        chosen_model = model_name or str(DEFAULT_MODEL_PATH)
        return EmbeddingConfig(
            backend="transformers",
            model_name=chosen_model,
            model_cache_dir=".model-cache",
        )
    return EmbeddingConfig(
        backend="lexical",
        model_name=model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_cache_dir=".model-cache",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import PandaMemory rows into the new memory store format.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to panda_memory.sqlite")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Where to save the imported memory store JSON")
    parser.add_argument(
        "--backend",
        choices=["lexical", "transformers"],
        default="transformers" if DEFAULT_MODEL_PATH.exists() else "lexical",
        help="Embedding backend to use for the imported store",
    )
    parser.add_argument("--model-name", default=None, help="Optional model name or local model path")
    parser.add_argument("--limit", type=int, default=None, help="Import only the first N memories")
    args = parser.parse_args()

    rows = load_rows(Path(args.db))
    if args.limit is not None:
        rows = rows[: args.limit]

    config = choose_embedding_config(args.backend, args.model_name)
    store = MemoryStore(config)
    for row in rows:
        store.add_memory(to_memory_item(row))

    output_path = Path(args.output)
    store.save(output_path)
    print(f"Imported {len(rows)} memories to {output_path}")
    print(f"backend={store.embedding_backend.name}")


if __name__ == "__main__":
    main()
