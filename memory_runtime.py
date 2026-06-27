from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os

from memory_system import EmbeddingConfig, ExchangeObservation, MemoryStore, SearchResult
from memory_sqlite import load_memory_store_sqlite, save_memory_store_sqlite, sqlite_store_status


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / ".models" / "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SHADOW_STORE_PATH = PROJECT_ROOT / "shadow_panda_memories.json"
DEFAULT_LEGACY_LIVE_STORE_PATH = PROJECT_ROOT / "live_shadow_memories.json"
DEFAULT_LIVE_STORE_PATH = PROJECT_ROOT / "live_memory.sqlite"


@dataclass
class RuntimeObservation:
    store_path: Path
    memory_count: int
    observation: ExchangeObservation


def choose_embedding_config() -> EmbeddingConfig:
    if DEFAULT_MODEL_PATH.exists():
        return EmbeddingConfig(
            backend="transformers",
            model_name=str(DEFAULT_MODEL_PATH),
            model_cache_dir=str(PROJECT_ROOT / ".model-cache"),
        )
    return EmbeddingConfig(
        backend="lexical",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_cache_dir=str(PROJECT_ROOT / ".model-cache"),
    )


class MemoryRuntime:
    """
    Thin persistence layer for live observation.

    PandaMemory and the imported shadow snapshot stay untouched. This runtime
    evolves a separate live store that can safely observe real exchanges.
    """

    def __init__(
        self,
        *,
        live_store_path: str | Path = DEFAULT_LIVE_STORE_PATH,
        shadow_store_path: str | Path = DEFAULT_SHADOW_STORE_PATH,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self.live_store_path = Path(live_store_path)
        self.shadow_store_path = Path(shadow_store_path)
        self.embedding_config = embedding_config or choose_embedding_config()
        self._store: MemoryStore | None = None

    def _load_shadow_base(self) -> MemoryStore:
        if self.shadow_store_path.exists():
            return MemoryStore.load(self.shadow_store_path)
        return MemoryStore(self.embedding_config)

    def _load_store_path(self, path: Path) -> MemoryStore:
        if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            return load_memory_store_sqlite(path)
        return MemoryStore.load(path)

    def _save_store_path(self, path: Path, store: MemoryStore) -> None:
        if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            save_memory_store_sqlite(path, store)
            return

        temp_path = path.with_name(f".{path.name}.tmp")
        store.save(temp_path)
        os.replace(temp_path, path)

    def _load_initial_store(self) -> MemoryStore:
        if self.live_store_path == DEFAULT_LIVE_STORE_PATH and DEFAULT_LEGACY_LIVE_STORE_PATH.exists():
            return MemoryStore.load(DEFAULT_LEGACY_LIVE_STORE_PATH)
        return self._load_shadow_base()

    def _retarget_store(self, store: MemoryStore) -> MemoryStore:
        current_model = store.embedding_config.model_name or ""
        desired_model = self.embedding_config.model_name or ""
        if (
            store.embedding_config.backend == self.embedding_config.backend
            and current_model == desired_model
        ):
            return store

        runtime_store = MemoryStore(self.embedding_config)
        store.embedding_config = self.embedding_config
        store.embedding_backend = runtime_store.embedding_backend
        store.affect_inferrer = runtime_store.affect_inferrer
        store.reflection_inferrer = runtime_store.reflection_inferrer
        store.reflection = runtime_store.reflection
        store.reindex_memories()
        return store

    def load_store(self) -> MemoryStore:
        if self._store is not None:
            return self._store

        if self.live_store_path.exists():
            self._store = self._retarget_store(self._load_store_path(self.live_store_path))
            return self._store

        store = self._retarget_store(self._load_initial_store())
        self._save_store_path(self.live_store_path, store)
        self._store = store
        return store

    def save_store(self, store: MemoryStore) -> None:
        self._save_store_path(self.live_store_path, store)
        self._store = store

    def observe_exchange(
        self,
        event_text: str,
        assistant_response: str,
        *,
        source: str = "conversation",
        extra_tags: list[str] | None = None,
    ) -> RuntimeObservation:
        store = self.load_store()
        observation = store.observe_exchange(
            event_text,
            assistant_response,
            source=source,
            extra_tags=extra_tags,
        )
        self.save_store(store)
        return RuntimeObservation(
            store_path=self.live_store_path,
            memory_count=len(store.memories),
            observation=observation,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        store = self.load_store()
        return store.search_memories(query, limit=limit)

    def status(self) -> dict[str, object]:
        live_exists = self.live_store_path.exists()
        shadow_exists = self.shadow_store_path.exists()
        memory_count: int | None = None
        backend_name = self.embedding_config.backend
        model_name = self.embedding_config.model_name
        schema_version: object = None

        legacy_json_exists = DEFAULT_LEGACY_LIVE_STORE_PATH.exists()
        if live_exists and self.live_store_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            status = sqlite_store_status(self.live_store_path)
            memory_count = status["memory_count"] if isinstance(status["memory_count"], int) else None
            backend_name = status["backend"] or backend_name
            model_name = status["model_name"] or model_name
            schema_version = status["schema_version"]
        elif self.live_store_path == DEFAULT_LIVE_STORE_PATH and legacy_json_exists:
            data = json.loads(DEFAULT_LEGACY_LIVE_STORE_PATH.read_text(encoding="utf-8"))
            memory_count = len(data.get("memories", [])) + len(data.get("notes", []))
            raw_config = data.get("embedding_config", {})
            backend_name = raw_config.get("backend", backend_name)
            model_name = raw_config.get("model_name", model_name)
        else:
            status_path = self.live_store_path if live_exists else self.shadow_store_path
            if status_path.exists():
                data = json.loads(status_path.read_text(encoding="utf-8"))
                memory_count = len(data.get("memories", [])) + len(data.get("notes", []))
                raw_config = data.get("embedding_config", {})
                backend_name = raw_config.get("backend", backend_name)
                model_name = raw_config.get("model_name", model_name)

        return {
            "live_store_path": str(self.live_store_path),
            "store_format": "sqlite" if self.live_store_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"} else "json",
            "shadow_store_path": str(self.shadow_store_path),
            "live_store_exists": live_exists,
            "legacy_json_store_path": str(DEFAULT_LEGACY_LIVE_STORE_PATH),
            "legacy_json_store_exists": legacy_json_exists,
            "shadow_store_exists": shadow_exists,
            "memory_count": memory_count,
            "backend": backend_name,
            "model_name": model_name,
            "schema_version": schema_version,
        }


def _print_observation(result: RuntimeObservation) -> None:
    observation = result.observation
    payload = {
        "store_path": str(result.store_path),
        "memory_count": result.memory_count,
        "kind": observation.reflection.kind,
        "summary": observation.reflection.summary,
        "reflection_text": observation.reflection.reflection_text,
        "response_dominant": observation.reflection.response_affect.dominant_emotion,
        "recalled": [
            {
                "score": round(match.score, 4),
                "summary": match.memory.summary,
                "response_dominant": match.memory.response_affect.dominant_emotion,
            }
            for match in observation.recalled
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Observe real exchanges into the live emotional memory store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    observe_parser = subparsers.add_parser("observe", help="Observe one exchange and persist it to the live store.")
    observe_parser.add_argument("--event", required=True, help="The user-side or event-side text.")
    observe_parser.add_argument("--response", required=True, help="The assistant response text.")
    observe_parser.add_argument("--tag", action="append", default=[], help="Optional extra tag. Repeatable.")

    search_parser = subparsers.add_parser("search", help="Search the live store.")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="How many memories to return")

    subparsers.add_parser("status", help="Show live runtime status.")

    export_parser = subparsers.add_parser("export-json", help="Export the live store to JSON.")
    export_parser.add_argument("--output", required=True, help="Output JSON path")

    args = parser.parse_args()
    runtime = MemoryRuntime()

    if args.command == "observe":
        result = runtime.observe_exchange(
            args.event,
            args.response,
            extra_tags=args.tag,
        )
        _print_observation(result)
        return

    if args.command == "search":
        matches = runtime.search(args.query, limit=args.limit)
        payload = [
            {
                "score": round(match.score, 4),
                "summary": match.memory.summary,
                "reflection_text": match.memory.reflection_text,
                "response_dominant": match.memory.response_affect.dominant_emotion,
            }
            for match in matches
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "status":
        print(json.dumps(runtime.status(), ensure_ascii=False, indent=2))
        return

    if args.command == "export-json":
        store = runtime.load_store()
        store.save(args.output)
        print(json.dumps({"output": args.output, "memory_count": len(store.memories)}, ensure_ascii=False, indent=2))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
