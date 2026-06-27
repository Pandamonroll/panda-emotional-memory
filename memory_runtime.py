from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os

from memory_system import EmbeddingConfig, ExchangeObservation, MemoryStore, SearchResult


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / ".models" / "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SHADOW_STORE_PATH = PROJECT_ROOT / "shadow_panda_memories.json"
DEFAULT_LIVE_STORE_PATH = PROJECT_ROOT / "live_shadow_memories.json"


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
            self._store = self._retarget_store(MemoryStore.load(self.live_store_path))
            return self._store

        store = self._retarget_store(self._load_shadow_base())
        store.save(self.live_store_path)
        self._store = store
        return store

    def save_store(self, store: MemoryStore) -> None:
        target = self.live_store_path
        temp_path = target.with_name(f".{target.name}.tmp")
        store.save(temp_path)
        os.replace(temp_path, target)
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

        status_path = self.live_store_path if live_exists else self.shadow_store_path
        if status_path.exists():
            data = json.loads(status_path.read_text(encoding="utf-8"))
            memory_count = len(data.get("memories", [])) + len(data.get("notes", []))
            raw_config = data.get("embedding_config", {})
            backend_name = raw_config.get("backend", backend_name)
            model_name = raw_config.get("model_name", model_name)

        return {
            "live_store_path": str(self.live_store_path),
            "shadow_store_path": str(self.shadow_store_path),
            "live_store_exists": live_exists,
            "shadow_store_exists": shadow_exists,
            "memory_count": memory_count,
            "backend": backend_name,
            "model_name": model_name,
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

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
