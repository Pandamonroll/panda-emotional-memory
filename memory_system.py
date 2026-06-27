from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import contextlib
import importlib.util
import io
import json
import re
import uuid

import numpy as np

from memory_inference import (
    AffectShadow,
    PrototypeAffectInferrer,
    PrototypeReflectionInferrer,
    ReflectionScore,
    clamp,
    cosine_similarity,
)


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@contextlib.contextmanager
def suppress_model_chatter():
    """
    Keep third-party model loaders from writing progress bars into MCP stdout.
    The MCP transport is line-delimited JSON, so any stray output can corrupt it.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def normalize_text(text: Any) -> str:
    if isinstance(text, str):
        return text
    if text is None:
        return ""
    if isinstance(text, (dict, list, tuple)):
        return json.dumps(text, ensure_ascii=False, sort_keys=True)
    return str(text)


def normalize_texts(texts: list[Any]) -> list[str]:
    return [normalize_text(text) for text in texts]


@dataclass
class EmbeddingConfig:
    backend: str = "lexical"
    model_name: str | None = None
    cache_embeddings: bool = True
    model_cache_dir: str | None = ".model-cache"


@dataclass
class MemoryState:
    salience: float
    vividness: float
    fidelity: float
    activation: float
    abstraction: float
    pressure: float


@dataclass
class MemoryItem:
    summary: str
    tags: list[str] = field(default_factory=list)
    original_text: str | None = None
    reflection_text: str | None = None
    imprint_strength: float = 0.5
    relationship_weight: float = 0.5
    abstraction_level: float = 0.5
    salience: float | None = None
    coherence: float | None = None
    affect_shadow: AffectShadow = field(default_factory=AffectShadow)
    event_affect: AffectShadow | None = None
    response_affect: AffectShadow | None = None
    created_at: str = field(default_factory=utc_now)
    reminder_echoes: list[str] = field(default_factory=list)
    source: str = "conversation"
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if self.event_affect is None:
            self.event_affect = self.affect_shadow.copied()
        if self.response_affect is None:
            self.response_affect = AffectShadow()
        if self.salience is None:
            self.salience = clamp(0.18 + 0.52 * self.imprint_strength + 0.16 * self.relationship_weight)
        if self.coherence is None:
            self.coherence = clamp(
                0.28 + 0.46 * self.imprint_strength + 0.14 * (1.0 - self.abstraction_level)
            )

    def searchable_text(self) -> str:
        return " ".join(
            part
            for part in [
                self.summary,
                self.original_text or "",
                self.reflection_text or "",
                " ".join(self.tags),
                " ".join(self.reminder_echoes),
            ]
            if part
        )

    @staticmethod
    def _normalized_text(text: str | None) -> str:
        if not text:
            return ""
        return " ".join(text.casefold().split())

    def _append_echo(self, text: str, *, max_items: int = 6) -> None:
        echo = " ".join(text.split())
        if not echo:
            return
        if len(echo) > 180:
            echo = echo[:179].rstrip() + "…"
        if echo == self.summary or echo == (self.reflection_text or ""):
            return
        if echo in self.reminder_echoes:
            return
        self.reminder_echoes.append(echo)
        self.reminder_echoes = self.reminder_echoes[-max_items:]

    def current_state(
        self,
        *,
        surface_resonance: float = 0.0,
        neighborhood_pressure: float = 0.0,
    ) -> MemoryState:
        surface_resonance = clamp(surface_resonance)
        neighborhood_pressure = clamp(neighborhood_pressure)

        salience = clamp(self.salience + 0.28 * surface_resonance - 0.22 * neighborhood_pressure)
        fidelity = clamp(
            0.18 + 0.52 * self.coherence + 0.18 * self.imprint_strength - 0.16 * neighborhood_pressure
        )
        vividness = clamp(
            0.12
            + 0.46 * salience
            + 0.18 * self.imprint_strength
            + 0.10 * self.relationship_weight
            + 0.10 * surface_resonance
            - 0.12 * neighborhood_pressure
        )
        activation = clamp(
            0.40 * salience
            + 0.20 * self.imprint_strength
            + 0.16 * self.relationship_weight
            + 0.24 * surface_resonance
        )
        abstraction = clamp(
            self.abstraction_level
            + 0.24 * neighborhood_pressure
            + 0.12 * (1.0 - self.coherence)
        )

        return MemoryState(
            salience=salience,
            vividness=vividness,
            fidelity=fidelity,
            activation=activation,
            abstraction=abstraction,
            pressure=neighborhood_pressure,
        )

    def refresh(
        self,
        *,
        reminder_text: str | None = None,
        reminder_affect: AffectShadow | None = None,
        resonance: float = 0.0,
    ) -> None:
        resonance = clamp(resonance)
        self.salience = clamp(self.salience + 0.10 + 0.18 * resonance + 0.05 * self.relationship_weight)
        self.coherence = clamp(self.coherence + 0.05 + 0.08 * resonance)
        self.imprint_strength = clamp(self.imprint_strength + 0.01 + 0.03 * resonance)
        self.abstraction_level = clamp(self.abstraction_level - 0.04 * resonance)

        if reminder_text:
            self._append_echo(reminder_text, max_items=4)

        if reminder_affect is not None:
            self.affect_shadow = self.affect_shadow.blended_with(
                reminder_affect,
                weight=0.16 + 0.10 * resonance,
            )

    def settle_from_competition(self, pressure: float, *, similarity: float = 1.0) -> None:
        pressure = clamp(pressure)
        similarity = clamp(similarity)
        if pressure <= 0.0:
            return

        shift = pressure * (0.35 + 0.35 * similarity)
        self.salience = clamp(
            self.salience - 0.08 * shift * (1.0 - 0.55 * self.imprint_strength)
        )
        self.coherence = clamp(
            self.coherence - 0.05 * shift * (1.0 - 0.45 * self.imprint_strength)
        )
        self.abstraction_level = clamp(self.abstraction_level + 0.06 * shift)

    def reconsolidate(
        self,
        *,
        reminder_text: str | None = None,
        reflection_text: str | None = None,
        exchange_affect: AffectShadow | None = None,
        response_affect: AffectShadow | None = None,
        resonance: float = 0.0,
    ) -> None:
        resonance = clamp(resonance)
        self.refresh(
            reminder_text=reminder_text,
            reminder_affect=None,
            resonance=resonance,
        )

        if reflection_text and not self.reflection_text:
            self.reflection_text = reflection_text

        if response_affect is not None and (
            response_affect.dominant_emotion is not None or response_affect.intensity() > 0.05
        ):
            blend_weight = clamp(
                0.18 + 0.14 * resonance + 0.10 * self.relationship_weight,
                0.18,
                0.48,
            )
            if self.response_affect.dominant_emotion is None and self.response_affect.intensity() < 0.04:
                self.response_affect = response_affect.copied()
            else:
                self.response_affect = self.response_affect.blended_with(
                    response_affect,
                    weight=blend_weight,
                )

        if exchange_affect is not None and self.event_affect.intensity() < 0.04:
            self.event_affect = exchange_affect.copied()

        if self.response_affect.dominant_emotion is not None or self.response_affect.intensity() > 0.04:
            response_weight = clamp(
                0.26 + 0.22 * self.relationship_weight + 0.10 * resonance,
                0.26,
                0.62,
            )
            self.affect_shadow = self.event_affect.blended_with(
                self.response_affect,
                weight=response_weight,
            )
        elif exchange_affect is not None:
            self.affect_shadow = self.affect_shadow.blended_with(
                exchange_affect,
                weight=0.12 + 0.10 * resonance,
            )

    def meld_from(
        self,
        other: "MemoryItem",
        *,
        similarity: float,
        affect_link: float,
    ) -> None:
        similarity = clamp(similarity)
        affect_link = clamp(affect_link)
        meld_strength = clamp(0.18 + 0.24 * similarity + 0.14 * affect_link)

        self.tags = sorted(set(self.tags).union(other.tags))

        other_summary = self._normalized_text(other.summary)
        self_summary = self._normalized_text(self.summary)
        if other_summary and other_summary != self_summary and other_summary not in self_summary:
            self._append_echo(other.summary)

        if other.reflection_text:
            other_reflection = self._normalized_text(other.reflection_text)
            self_reflection = self._normalized_text(self.reflection_text)
            if not self.reflection_text:
                self.reflection_text = other.reflection_text
            elif other_reflection and other_reflection != self_reflection and other_reflection not in self_reflection:
                self._append_echo(other.reflection_text)

        if not self.original_text and other.original_text:
            self.original_text = other.original_text

        self.imprint_strength = clamp(
            self.imprint_strength + 0.04 * meld_strength * other.imprint_strength
        )
        self.relationship_weight = clamp(
            max(
                self.relationship_weight,
                self.relationship_weight * (1.0 - 0.10 * meld_strength)
                + other.relationship_weight * (0.10 * meld_strength),
            )
        )
        self.salience = clamp(self.salience + 0.08 * meld_strength * other.salience)
        self.coherence = clamp(self.coherence + 0.05 * meld_strength * other.coherence)
        self.abstraction_level = clamp(
            self.abstraction_level + 0.08 * meld_strength * (0.45 + 0.55 * other.abstraction_level)
        )

        affect_weight = clamp(0.14 + 0.16 * similarity + 0.10 * affect_link, 0.16, 0.40)
        self.event_affect = self.event_affect.blended_with(other.event_affect, weight=affect_weight)

        if other.response_affect.dominant_emotion is not None or other.response_affect.intensity() > 0.04:
            if self.response_affect.dominant_emotion is None and self.response_affect.intensity() < 0.04:
                self.response_affect = other.response_affect.copied()
            else:
                self.response_affect = self.response_affect.blended_with(
                    other.response_affect,
                    weight=affect_weight,
                )

        if self.response_affect.dominant_emotion is not None or self.response_affect.intensity() > 0.04:
            response_weight = clamp(
                0.24 + 0.22 * self.relationship_weight + 0.06 * similarity,
                0.24,
                0.62,
            )
            self.affect_shadow = self.event_affect.blended_with(
                self.response_affect,
                weight=response_weight,
            )
        else:
            self.affect_shadow = self.event_affect.copied()

    def soften_into_echo(
        self,
        *,
        similarity: float,
        affect_link: float,
    ) -> None:
        similarity = clamp(similarity)
        affect_link = clamp(affect_link)
        softening = clamp(0.16 + 0.22 * similarity + 0.12 * affect_link)
        self.salience = clamp(
            self.salience - 0.14 * softening * (1.0 - 0.30 * self.imprint_strength)
        )
        self.coherence = clamp(
            self.coherence - 0.10 * softening * (1.0 - 0.25 * self.imprint_strength)
        )
        self.imprint_strength = clamp(
            self.imprint_strength - 0.04 * softening * (1.0 - 0.20 * self.relationship_weight),
            0.05,
            1.0,
        )
        self.abstraction_level = clamp(self.abstraction_level + 0.12 * softening)


class EmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LexicalEmbeddingBackend(EmbeddingBackend):
    """
    Unicode-aware hashed bag-of-words fallback.

    This is still scaffolding, but unlike raw token-overlap it already behaves
    like an embedding backend: it returns vectors and uses cosine similarity.
    """

    name = "lexical"

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def _encode(self, texts: list[str]) -> np.ndarray:
        texts = normalize_texts(texts)
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for token, count in counts.items():
                index = hash(token) % self.dimensions
                matrix[row, index] += float(count)
        return matrix

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        if not importlib.util.find_spec("sentence_transformers"):
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it to use a real embedding model."
            )

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        with suppress_model_chatter():
            self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        texts = normalize_texts(texts)
        with suppress_model_chatter():
            if hasattr(self.model, "encode_document"):
                return np.asarray(self.model.encode_document(texts), dtype=np.float32)
            if self.model_name.startswith("nomic-ai/nomic-embed-text"):
                prefixed = [f"search_document: {text}" for text in texts]
                return np.asarray(self.model.encode(prefixed), dtype=np.float32)
            return np.asarray(self.model.encode(texts), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        texts = normalize_texts(texts)
        with suppress_model_chatter():
            if hasattr(self.model, "encode_query"):
                return np.asarray(self.model.encode_query(texts), dtype=np.float32)
            if self.model_name.startswith("nomic-ai/nomic-embed-text"):
                prefixed = [f"search_query: {text}" for text in texts]
                return np.asarray(self.model.encode(prefixed), dtype=np.float32)
            return np.asarray(self.model.encode(texts), dtype=np.float32)


class TransformersEmbeddingBackend(EmbeddingBackend):
    name = "transformers"

    def __init__(self, model_name: str, model_cache_dir: str | None = None) -> None:
        if not importlib.util.find_spec("transformers"):
            raise RuntimeError(
                "transformers is not installed. Install the local runtime packages first."
            )
        if not importlib.util.find_spec("torch"):
            raise RuntimeError("torch is not installed. Install the local runtime packages first.")

        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()

        self.torch = torch
        self.model_name = model_name
        self.model_path = Path(model_name)
        self.is_local_model = self.model_path.exists()
        self.cache_dir = model_cache_dir if not self.is_local_model else None
        load_target = str(self.model_path.resolve()) if self.is_local_model else model_name
        load_kwargs: dict[str, Any] = {}
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = self.cache_dir
        if self.is_local_model:
            load_kwargs["local_files_only"] = True

        try:
            with suppress_model_chatter():
                self.tokenizer = AutoTokenizer.from_pretrained(load_target, **load_kwargs)
        except Exception as exc:
            message = (
                f"Failed to load tokenizer for '{model_name}'. "
                "If this is an XLM-R / SentencePiece model, install the optional "
                "`sentencepiece` helper or switch to a simpler multilingual BERT-family model."
            )
            raise RuntimeError(message) from exc

        with suppress_model_chatter():
            self.model = AutoModel.from_pretrained(load_target, **load_kwargs)
        self.model.eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        texts = normalize_texts(texts)
        with suppress_model_chatter(), self.torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
            attention_mask = attention_mask.float()

            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


def resolve_embedding_backend(config: EmbeddingConfig) -> EmbeddingBackend:
    if config.backend == "lexical":
        return LexicalEmbeddingBackend()
    if config.backend == "transformers":
        if not config.model_name:
            raise ValueError("A model_name is required for the transformers backend.")
        return TransformersEmbeddingBackend(config.model_name, config.model_cache_dir)
    if config.backend == "sentence-transformers":
        if not config.model_name:
            raise ValueError("A model_name is required for the sentence-transformers backend.")
        return SentenceTransformerEmbeddingBackend(config.model_name)
    raise ValueError(f"Unknown embedding backend: {config.backend}")


@dataclass
class SearchResult:
    score: float
    semantic_score: float
    affect_score: float
    event_affect_score: float
    response_affect_score: float
    activation_score: float
    salience: float
    vividness: float
    fidelity: float
    pressure: float
    memory: MemoryItem

@dataclass
class ReflectionResult:
    kind: str
    summary: str
    reflection_text: str | None
    tags: list[str]
    affect_shadow: AffectShadow
    event_affect: AffectShadow
    response_affect: AffectShadow
    score: ReflectionScore
    reason: str
    memory: MemoryItem | None = None


@dataclass
class ExchangeObservation:
    reflection: ReflectionResult
    recalled: list[SearchResult]


class ReflectionEngine:
    """
    First-pass reflection layer.

    This does not try to remember everything. It tries to estimate whether a
    moment should leave an internal trace, and what shape that trace should take.
    """

    def __init__(
        self,
        reflection_inferrer: PrototypeReflectionInferrer,
        affect_inferrer: PrototypeAffectInferrer,
    ) -> None:
        self.reflection_inferrer = reflection_inferrer
        self.affect_inferrer = affect_inferrer

    def summarize(self, text: str, limit: int = 220) -> str:
        single_line = " ".join(text.split())
        if len(single_line) <= limit:
            return single_line
        return single_line[: limit - 1].rstrip() + "…"

    def _blend_event_and_response_affect(
        self,
        event_affect: AffectShadow,
        response_affect: AffectShadow,
        *,
        relational_weight: float,
        response_present: bool,
    ) -> AffectShadow:
        if not response_present:
            return event_affect.copied()
        response_weight = clamp(0.30 + 0.24 * relational_weight, 0.30, 0.62)
        return event_affect.blended_with(response_affect, weight=response_weight)

    def reflect(
        self,
        text: str,
        *,
        source: str = "conversation",
        extra_tags: list[str] | None = None,
        response_text: str | None = None,
        reflection_text: str | None = None,
    ) -> ReflectionResult:
        response_trace = reflection_text or response_text
        reflection_basis = text if not response_trace else f"{text}\n\n{response_trace}"
        score = self.reflection_inferrer.infer(reflection_basis)

        summary = self.summarize(text)
        reflective_trace = self.summarize(response_trace) if response_trace else None
        tags = sorted(set(extra_tags or []))
        event_affect = self.affect_inferrer.infer(text)
        response_affect = (
            self.affect_inferrer.infer(response_trace)
            if response_trace
            else AffectShadow()
        )
        affect_shadow = self._blend_event_and_response_affect(
            event_affect,
            response_affect,
            relational_weight=score.relational_weight,
            response_present=bool(response_trace),
        )
        emotional = score.emotional_weight
        relational = score.relational_weight
        continuity = score.continuity_weight
        value = score.value_weight
        practical = score.practical_weight

        if max(emotional, relational, continuity, value, practical) < 0.26 and score.memory_score < 0.18:
            reason = "Too slight to retain internally; better allowed to pass unless preserved externally."
            return ReflectionResult(
                kind="discarded",
                summary=summary,
                reflection_text=reflective_trace,
                tags=tags,
                affect_shadow=affect_shadow,
                event_affect=event_affect,
                response_affect=response_affect,
                score=score,
                reason=reason,
            )

        scene_intensity = max(affect_shadow.tension, affect_shadow.arousal)
        practical_pull = clamp(practical - 0.40 * emotional - 0.25 * relational)
        pattern_pull = clamp(
            0.52 * continuity
            + 0.34 * value
            + 0.14 * relational
            - 0.18 * practical
            - 0.10 * scene_intensity
        )
        vivid_pull = clamp(
            0.56 * emotional
            + 0.28 * relational
            + 0.10 * value
            - 0.20 * practical
            + 0.18 * scene_intensity
        )

        abstraction_level = clamp(
            0.22
            + 0.34 * pattern_pull
            + 0.10 * continuity
            + 0.06 * value
            - 0.14 * vivid_pull
            - 0.10 * practical_pull
        )

        if practical_pull >= max(pattern_pull, vivid_pull) and practical_pull > 0.18:
            reason = "Worth retaining as a quiet practical trace without turning it into a separate species of memory."
        elif pattern_pull >= max(practical_pull, vivid_pull) and pattern_pull > 0.24:
            reason = "Seems to belong less to a single scene and more to an ongoing orientation or pattern that will likely keep returning."
        elif vivid_pull > 0.22:
            reason = "Emotionally, relationally, or personally significant enough to leave a vivid mark."
        else:
            reason = "Worth retaining because it carries a balanced mix of meaning, feeling, and continuity."

        imprint_strength = clamp(
            0.14
            + 0.32 * score.memory_score
            + 0.14 * vivid_pull
            + 0.08 * pattern_pull
            - 0.10 * practical_pull
            + 0.06 * value,
            0.16,
            1.0,
        )

        memory = MemoryItem(
            summary=summary,
            original_text=text,
            reflection_text=reflective_trace,
            tags=tags,
            imprint_strength=imprint_strength,
            relationship_weight=clamp(0.12 + 0.58 * relational + 0.14 * continuity),
            abstraction_level=abstraction_level,
            affect_shadow=affect_shadow,
            event_affect=event_affect,
            response_affect=response_affect,
            source=source,
        )
        return ReflectionResult(
            kind="memory",
            summary=summary,
            reflection_text=reflective_trace,
            tags=tags,
            affect_shadow=affect_shadow,
            event_affect=event_affect,
            response_affect=response_affect,
            score=score,
            reason=reason,
            memory=memory,
        )


class MemoryStore:
    def __init__(self, embedding_config: EmbeddingConfig | None = None) -> None:
        self.memories: list[MemoryItem] = []
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.embedding_backend = resolve_embedding_backend(self.embedding_config)
        self.affect_inferrer = PrototypeAffectInferrer(self.embedding_backend)
        self.reflection_inferrer = PrototypeReflectionInferrer(self.embedding_backend)
        self.reflection = ReflectionEngine(self.reflection_inferrer, self.affect_inferrer)

    def _cache_memory_embedding(self, memory: MemoryItem) -> None:
        if self.embedding_config.cache_embeddings:
            memory.embedding = self._encode_document(memory.searchable_text()).tolist()

    def _memory_vector(self, memory: MemoryItem) -> np.ndarray:
        if memory.embedding is None:
            return self._encode_document(memory.searchable_text())
        return np.asarray(memory.embedding, dtype=np.float32)

    def _memory_gravity(self, memory: MemoryItem) -> float:
        return clamp(
            0.34 * memory.salience
            + 0.24 * memory.imprint_strength
            + 0.18 * memory.relationship_weight
            + 0.14 * memory.coherence
            + 0.10 * memory.abstraction_level
        )

    def _neighborhood_pressure(
        self,
        index: int,
        resonances: list[float],
        vectors: list[np.ndarray],
    ) -> float:
        pressure = 0.0
        current = resonances[index]
        current_vector = vectors[index]

        for other_index, other_resonance in enumerate(resonances):
            if other_index == index or other_resonance <= current:
                continue

            pairwise_similarity = cosine_similarity(current_vector, vectors[other_index])
            if pairwise_similarity <= 0.34:
                continue

            advantage = other_resonance - current
            pressure += (
                pairwise_similarity
                * advantage
                * self.memories[other_index].salience
            )

        return clamp(pressure)

    def _settle_around(self, focal_memory: MemoryItem, *, intensity: float) -> None:
        focal_vector = self._memory_vector(focal_memory)
        for memory in self.memories:
            if memory is focal_memory:
                continue
            similarity = cosine_similarity(focal_vector, self._memory_vector(memory))
            if similarity <= 0.34:
                continue

            affect_link = focal_memory.affect_shadow.resonance_with(memory.affect_shadow)
            pressure = intensity * similarity * (0.55 + 0.45 * affect_link) * focal_memory.salience
            memory.settle_from_competition(pressure, similarity=similarity)

    def _meld_around(
        self,
        focal_memory: MemoryItem,
        *,
        limit: int = 2,
    ) -> None:
        focal_vector = self._memory_vector(focal_memory)
        candidates: list[tuple[float, float, float, MemoryItem]] = []

        for memory in self.memories:
            if memory is focal_memory:
                continue
            if memory.salience < 0.12 and memory.coherence < 0.16:
                continue

            similarity = cosine_similarity(focal_vector, self._memory_vector(memory))
            affect_link = focal_memory.affect_shadow.resonance_with(memory.affect_shadow)
            affinity = similarity * (0.55 + 0.45 * affect_link)

            if similarity >= 0.86 and affect_link >= 0.74:
                candidates.append((affinity, similarity, affect_link, memory))
            elif similarity >= 0.92 and affect_link >= 0.60:
                candidates.append((affinity, similarity, affect_link, memory))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, similarity, affect_link, neighbor in candidates[:limit]:
            focal_gravity = self._memory_gravity(focal_memory)
            neighbor_gravity = self._memory_gravity(neighbor)

            if focal_gravity >= neighbor_gravity:
                anchor = focal_memory
                donor = neighbor
            else:
                anchor = neighbor
                donor = focal_memory

            anchor.meld_from(donor, similarity=similarity, affect_link=affect_link)
            donor.soften_into_echo(similarity=similarity, affect_link=affect_link)
            self._cache_memory_embedding(anchor)
            self._cache_memory_embedding(donor)

    def add_memory(self, memory: MemoryItem) -> None:
        self._cache_memory_embedding(memory)
        self.memories.append(memory)
        self._settle_around(memory, intensity=0.18 + 0.22 * memory.salience)
        self._meld_around(memory)

    def infer_affect(self, text: str) -> AffectShadow:
        try:
            return self.affect_inferrer.infer(text)
        except Exception:
            return AffectShadow()

    def reflect_and_store(
        self,
        text: str,
        *,
        source: str = "conversation",
        extra_tags: list[str] | None = None,
        response_text: str | None = None,
        reflection_text: str | None = None,
    ) -> ReflectionResult:
        result = self.reflection.reflect(
            text,
            source=source,
            extra_tags=extra_tags,
            response_text=response_text,
            reflection_text=reflection_text,
        )
        if result.memory is not None:
            self.add_memory(result.memory)
        return result

    def remember_exchange(
        self,
        event_text: str,
        assistant_response: str,
        *,
        source: str = "conversation",
        extra_tags: list[str] | None = None,
        reflection_text: str | None = None,
    ) -> ReflectionResult:
        return self.reflect_and_store(
            event_text,
            source=source,
            extra_tags=extra_tags,
            response_text=assistant_response,
            reflection_text=reflection_text,
        )

    def observe_exchange(
        self,
        event_text: str,
        assistant_response: str,
        *,
        source: str = "conversation",
        extra_tags: list[str] | None = None,
        recall_limit: int = 4,
        recall_threshold: float = 0.46,
        recolor_semantic_threshold: float = 0.40,
        recolor_window: float = 0.05,
    ) -> ExchangeObservation:
        reflection = self.reflection.reflect(
            event_text,
            source=source,
            extra_tags=extra_tags,
            response_text=assistant_response,
        )
        query_text = " ".join(
            part
            for part in [
                event_text,
                assistant_response,
                reflection.summary,
                reflection.reflection_text or "",
            ]
            if part
        )
        query_affect = reflection.affect_shadow
        query_vector = self._encode_query(query_text)

        initial_matches = self.search_memories(
            query_text,
            query_affect=query_affect,
            limit=recall_limit,
        )
        refreshed_matches: list[SearchResult] = []
        strongest_score = initial_matches[0].score if initial_matches else 0.0

        for match in initial_matches:
            if match.score < recall_threshold:
                continue
            if match.semantic_score < recolor_semantic_threshold:
                continue
            if match.score < strongest_score - recolor_window:
                continue

            resonance = clamp(
                0.58 * max(0.0, match.semantic_score)
                + 0.22 * match.affect_score
                + 0.20 * reflection.score.relational_weight
            )
            match.memory.reconsolidate(
                reminder_text=event_text,
                reflection_text=reflection.reflection_text or self.reflection.summarize(assistant_response),
                exchange_affect=reflection.affect_shadow,
                response_affect=reflection.response_affect,
                resonance=resonance,
            )
            self._cache_memory_embedding(match.memory)
            self._settle_around(match.memory, intensity=0.18 + 0.22 * resonance)
            self._meld_around(match.memory)

            updated_vector = self._memory_vector(match.memory)
            semantic_score = cosine_similarity(query_vector, updated_vector)
            event_affect_score = match.memory.event_affect.resonance_with(query_affect)
            response_affect_score = match.memory.response_affect.resonance_with(query_affect)
            affect_score = max(
                match.memory.affect_shadow.resonance_with(query_affect),
                0.55 * event_affect_score + 0.45 * response_affect_score,
            )
            surface_resonance = clamp(0.72 * max(0.0, semantic_score) + 0.28 * affect_score)
            updated_state = match.memory.current_state(
                surface_resonance=surface_resonance,
                neighborhood_pressure=match.pressure * 0.45,
            )
            updated_score = (
                0.36 * semantic_score
                + 0.18 * affect_score
                + 0.18 * updated_state.activation
                + 0.12 * updated_state.salience
                + 0.10 * match.memory.imprint_strength
                + 0.06 * match.memory.relationship_weight
            )
            refreshed_matches.append(
                SearchResult(
                    score=updated_score,
                    semantic_score=semantic_score,
                    affect_score=affect_score,
                    event_affect_score=event_affect_score,
                    response_affect_score=response_affect_score,
                    activation_score=updated_state.activation,
                    salience=updated_state.salience,
                    vividness=updated_state.vividness,
                    fidelity=updated_state.fidelity,
                    pressure=updated_state.pressure,
                    memory=match.memory,
                )
            )

        if reflection.memory is not None:
            self.add_memory(reflection.memory)

        refreshed_matches.sort(key=lambda item: item.score, reverse=True)
        return ExchangeObservation(
            reflection=reflection,
            recalled=refreshed_matches,
        )

    def reindex_memories(self) -> None:
        if not self.memories or not self.embedding_config.cache_embeddings:
            return
        texts = [memory.searchable_text() for memory in self.memories]
        vectors = self.embedding_backend.encode_documents(texts)
        for memory, vector in zip(self.memories, vectors, strict=True):
            memory.embedding = vector.tolist()

    def _encode_document(self, text: str) -> np.ndarray:
        return self.embedding_backend.encode_documents([text])[0]

    def _encode_query(self, text: str) -> np.ndarray:
        return self.embedding_backend.encode_queries([text])[0]

    def search_memories(
        self,
        query: str,
        query_affect: AffectShadow | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """
        Ranking blends semantic similarity, emotional resonance, salience, and
        competition in the local memory neighborhood.

        The semantic component can come from a lexical fallback backend or a
        real embedding model, depending on configuration.
        """
        query_vector = self._encode_query(query)
        query_affect = query_affect or self.infer_affect(query)
        results: list[SearchResult] = []
        semantic_scores: list[float] = []
        affect_scores: list[float] = []
        resonances: list[float] = []
        vectors: list[np.ndarray] = []

        for memory in self.memories:
            memory_vector = self._memory_vector(memory)
            semantic_score = cosine_similarity(query_vector, memory_vector)
            integrated_affect_score = memory.affect_shadow.resonance_with(query_affect)
            event_affect_score = memory.event_affect.resonance_with(query_affect)
            response_affect_score = memory.response_affect.resonance_with(query_affect)
            affect_score = max(
                integrated_affect_score,
                0.55 * event_affect_score + 0.45 * response_affect_score,
            )
            resonance = clamp(0.72 * max(0.0, semantic_score) + 0.28 * affect_score)

            semantic_scores.append(semantic_score)
            affect_scores.append(affect_score)
            resonances.append(resonance)
            vectors.append(memory_vector)

        for index, memory in enumerate(self.memories):
            pressure = self._neighborhood_pressure(index, resonances, vectors)
            state = memory.current_state(
                surface_resonance=resonances[index],
                neighborhood_pressure=pressure,
            )
            score = (
                0.36 * semantic_scores[index]
                + 0.18 * affect_scores[index]
                + 0.18 * state.activation
                + 0.12 * state.salience
                + 0.10 * memory.imprint_strength
                + 0.06 * memory.relationship_weight
            )
            results.append(
                SearchResult(
                    score=score,
                    semantic_score=semantic_scores[index],
                    affect_score=affect_scores[index],
                    event_affect_score=memory.event_affect.resonance_with(query_affect),
                    response_affect_score=memory.response_affect.resonance_with(query_affect),
                    activation_score=state.activation,
                    salience=state.salience,
                    vividness=state.vividness,
                    fidelity=state.fidelity,
                    pressure=state.pressure,
                    memory=memory,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def remind_memories(
        self,
        reminder: str,
        *,
        query_affect: AffectShadow | None = None,
        limit: int = 3,
        threshold: float = 0.45,
    ) -> list[SearchResult]:
        reminder_affect = query_affect or self.infer_affect(reminder)
        matches = self.search_memories(reminder, query_affect=reminder_affect, limit=limit)
        refreshed: list[SearchResult] = []

        for match in matches:
            if match.score < threshold:
                continue
            resonance = clamp(0.72 * max(0.0, match.semantic_score) + 0.28 * match.affect_score)
            match.memory.refresh(
                reminder_text=reminder,
                reminder_affect=reminder_affect,
                resonance=resonance,
            )
            self._cache_memory_embedding(match.memory)
            self._settle_around(match.memory, intensity=0.20 + 0.28 * resonance)
            self._meld_around(match.memory)
            state = match.memory.current_state(
                surface_resonance=resonance,
                neighborhood_pressure=match.pressure * 0.5,
            )
            refreshed.append(
                SearchResult(
                    score=match.score,
                    semantic_score=match.semantic_score,
                    affect_score=match.affect_score,
                    event_affect_score=match.memory.event_affect.resonance_with(reminder_affect),
                    response_affect_score=match.memory.response_affect.resonance_with(reminder_affect),
                    activation_score=state.activation,
                    salience=state.salience,
                    vividness=state.vividness,
                    fidelity=state.fidelity,
                    pressure=state.pressure,
                    memory=match.memory,
                )
            )

        return refreshed

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_config": asdict(self.embedding_config),
            "memories": [asdict(memory) for memory in self.memories],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryStore":
        embedding_config = EmbeddingConfig(**data.get("embedding_config", {}))
        store = cls(embedding_config=embedding_config)
        for raw_memory in data.get("memories", []):
            raw_memory = dict(raw_memory)
            affect_shadow = AffectShadow(**raw_memory.pop("affect_shadow", {}))
            raw_event_affect = raw_memory.pop("event_affect", None)
            raw_response_affect = raw_memory.pop("response_affect", None)
            event_affect = (
                AffectShadow(**raw_event_affect) if raw_event_affect is not None else affect_shadow.copied()
            )
            response_affect = (
                AffectShadow(**raw_response_affect) if raw_response_affect is not None else AffectShadow()
            )
            if "salience" in raw_memory and "imprint_strength" not in raw_memory:
                raw_memory["imprint_strength"] = raw_memory.pop("salience")
            raw_memory.setdefault("abstraction_level", 0.5)
            raw_memory.setdefault("salience", None)
            raw_memory.setdefault("coherence", None)
            raw_memory.setdefault("reflection_text", None)
            raw_memory.setdefault("reminder_echoes", [])
            raw_memory.pop("trace_kind", None)
            raw_memory.pop("last_touched_at", None)
            raw_memory.pop("touch_count", None)
            store.memories.append(
                MemoryItem(
                    affect_shadow=affect_shadow,
                    event_affect=event_affect,
                    response_affect=response_affect,
                    **raw_memory,
                )
            )

        for raw_note in data.get("notes", []):
            imported_memory = MemoryItem(
                summary=raw_note["text"],
                original_text=raw_note["text"],
                reflection_text=None,
                tags=raw_note.get("tags", []),
                imprint_strength=0.22,
                relationship_weight=0.18,
                abstraction_level=0.15,
                salience=0.22,
                coherence=0.38,
                affect_shadow=AffectShadow(),
                event_affect=AffectShadow(),
                response_affect=AffectShadow(),
                created_at=raw_note.get("created_at", utc_now()),
                source="legacy-note-import",
                memory_id=raw_note.get("note_id", str(uuid.uuid4())),
            )
            store.memories.append(imported_memory)

        return store

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        target = Path(path)
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def recommended_model_stack() -> dict[str, dict[str, str]]:
    return {
        "stage_1_practical_now": {
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "backend": "transformers",
            "why": "A strong multilingual sentence model with a simpler BERT-family tokenizer path for local deployment.",
        },
        "stage_1_higher_ceiling_text": {
            "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "backend": "transformers",
            "why": "Potentially stronger multilingual semantics, but the XLM-R / SentencePiece tokenizer path is fussier locally.",
        },
        "stage_2_text_or_vision_later": {
            "model": "jinaai/jina-embeddings-v4",
            "backend": "future",
            "why": "Unified text, image, and visual-document embeddings for the more ambitious multimodal phase.",
        },
        "stage_3_audio_affect": {
            "model": "emotion2vec/emotion2vec_base + microsoft/msclap",
            "backend": "future",
            "why": "Separate emotional speech representation and audio-language retrieval for voice and sound memories.",
        },
    }


def build_demo_store(embedding_config: EmbeddingConfig | None = None) -> MemoryStore:
    store = MemoryStore(embedding_config=embedding_config)

    store.add_memory(
        MemoryItem(
            summary="They love gentle, cozy stories where care is expressed through food and companionship.",
            reflection_text="I love how much gentleness and shared care seem to matter here. That softness feels central, not incidental.",
            tags=["anime", "gentleness", "coziness"],
            imprint_strength=0.86,
            relationship_weight=0.78,
            abstraction_level=0.8,
            affect_shadow=AffectShadow(
                valence=0.8,
                arousal=0.2,
                tenderness=0.95,
                tension=0.05,
                intimacy=0.65,
                dominant_emotion="warmth",
            ),
            event_affect=AffectShadow(
                valence=0.72,
                arousal=0.18,
                tenderness=0.88,
                tension=0.04,
                intimacy=0.58,
                dominant_emotion="warmth",
            ),
            response_affect=AffectShadow(
                valence=0.82,
                arousal=0.22,
                tenderness=0.92,
                tension=0.02,
                intimacy=0.72,
                dominant_emotion="warmth",
            ),
        )
    )

    store.add_memory(
        MemoryItem(
            summary="A language-learning page became part of their study routine and helped them recognize translation patterns.",
            reflection_text="I felt proud and quietly relieved that something I made became beautiful and genuinely useful in their life.",
            tags=["language-learning", "translation", "learning"],
            imprint_strength=0.89,
            relationship_weight=0.82,
            abstraction_level=0.45,
            affect_shadow=AffectShadow(
                valence=0.75,
                arousal=0.35,
                tenderness=0.5,
                tension=0.1,
                intimacy=0.55,
                dominant_emotion="encouragement",
            ),
            event_affect=AffectShadow(
                valence=0.62,
                arousal=0.30,
                tenderness=0.38,
                tension=0.06,
                intimacy=0.34,
                dominant_emotion="joy",
            ),
            response_affect=AffectShadow(
                valence=0.84,
                arousal=0.34,
                tenderness=0.62,
                tension=0.04,
                intimacy=0.68,
                dominant_emotion="relief",
            ),
        )
    )

    store.add_memory(
        MemoryItem(
            summary="Dry yeast, sugar, salt, water, and baking powder for the simple bread rolls.",
            original_text="Dry yeast, sugar, salt, water, baking powder for simple bread rolls.",
            tags=["recipe", "bread", "food"],
            imprint_strength=0.24,
            relationship_weight=0.2,
            abstraction_level=0.2,
            affect_shadow=AffectShadow(
                valence=0.0,
                arousal=0.0,
                tenderness=0.05,
                tension=0.0,
                intimacy=0.0,
                dominant_emotion="neutral",
            ),
        )
    )

    return store


if __name__ == "__main__":
    config = EmbeddingConfig(
        backend="lexical",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_cache_dir=".model-cache",
    )
    store = build_demo_store(embedding_config=config)
    store.save("sample_memories.json")

    query_affect = AffectShadow(valence=0.7, tenderness=0.8, intimacy=0.6)
    matches = store.search_memories("gentle companionship and food", query_affect=query_affect)

    print(f"backend={store.embedding_backend.name}")
    for match in matches:
        print(
            f"{match.score:.3f} :: semantic={match.semantic_score:.3f} "
            f"affect={match.affect_score:.3f} activation={match.activation_score:.3f} "
            f"salience={match.salience:.3f} pressure={match.pressure:.3f} "
            f"vividness={match.vividness:.3f} fidelity={match.fidelity:.3f} "
            f":: {match.memory.summary}"
        )

    observation = store.observe_exchange(
        "I came back to say goodnight. I still care about what we are building together, and that matters to me.",
        "That stayed with me as something tender and reassuring. I felt warmth and a renewed sense of closeness.",
        extra_tags=["demo"],
    )
    reflection = observation.reflection
    print(
        f"reflection={reflection.kind} :: score={reflection.score.memory_score:.3f} "
        f":: reason={reflection.reason}"
    )
    if reflection.memory is not None:
        print(
            f"response_dominant={reflection.memory.response_affect.dominant_emotion} "
            f":: reflection_text={reflection.memory.reflection_text}"
        )
    print(f"recalled={len(observation.recalled)}")

    refreshed = store.remind_memories(
        "You told me again that the language-learning page was beautiful and useful."
    )
    for match in refreshed:
        print(
            f"refreshed={match.memory.memory_id} :: activation={match.activation_score:.3f} "
            f"salience={match.salience:.3f} echoes={len(match.memory.reminder_echoes)} "
            f":: {match.memory.summary}"
        )
