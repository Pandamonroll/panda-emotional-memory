from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math

import numpy as np


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


@dataclass
class AffectShadow:
    valence: float = 0.0
    arousal: float = 0.0
    tenderness: float = 0.0
    tension: float = 0.0
    intimacy: float = 0.0
    dominant_emotion: str | None = None

    def copied(self) -> "AffectShadow":
        return AffectShadow(
            valence=self.valence,
            arousal=self.arousal,
            tenderness=self.tenderness,
            tension=self.tension,
            intimacy=self.intimacy,
            dominant_emotion=self.dominant_emotion,
        )

    def intensity(self) -> float:
        return (
            abs(self.valence)
            + abs(self.arousal)
            + abs(self.tenderness)
            + abs(self.tension)
            + abs(self.intimacy)
        ) / 5.0

    def resonance_with(self, other: "AffectShadow") -> float:
        dims = [
            self.valence - other.valence,
            self.arousal - other.arousal,
            self.tenderness - other.tenderness,
            self.tension - other.tension,
            self.intimacy - other.intimacy,
        ]
        distance = math.sqrt(sum(value * value for value in dims))
        return 1.0 / (1.0 + distance)

    def blended_with(self, other: "AffectShadow", weight: float = 0.2) -> "AffectShadow":
        keep = 1.0 - weight
        return AffectShadow(
            valence=clamp(self.valence * keep + other.valence * weight, -1.0, 1.0),
            arousal=clamp(self.arousal * keep + other.arousal * weight, -1.0, 1.0),
            tenderness=clamp(self.tenderness * keep + other.tenderness * weight, -1.0, 1.0),
            tension=clamp(self.tension * keep + other.tension * weight, -1.0, 1.0),
            intimacy=clamp(self.intimacy * keep + other.intimacy * weight, -1.0, 1.0),
            dominant_emotion=other.dominant_emotion or self.dominant_emotion,
        )


@dataclass
class ReflectionScore:
    emotional_weight: float
    relational_weight: float
    continuity_weight: float
    value_weight: float
    practical_weight: float
    memory_score: float


class EmbeddingBackendLike(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        ...


class PrototypeAffectInferrer:
    """
    Model-backed affect inference.

    Instead of literal cue words, we compare a memory to small multilingual
    prototype constellations in embedding space. This is still scaffolding, but
    it is more fluid and more multilingual than hard-coded keyword matches.
    """

    DIMENSION_ANCHORS: dict[str, dict[str, list[str]]] = {
        "valence": {
            "high": [
                "warm happiness, joy, gratitude, relief",
                "gentle delight and hopeful warmth",
                "playful amusement, bright laughter, fond delight",
                "温かい幸せ、喜び、感謝、安堵",
                "明るい可笑しさ、楽しい笑い、愛おしい喜び",
                "優しい喜びと希望のぬくもり",
            ],
            "low": [
                "fear, pain, sadness, distress",
                "lonely hurt and emotional suffering",
                "恐れ、痛み、悲しみ、苦しみ",
                "孤独な傷つきと心の痛み",
            ],
        },
        "arousal": {
            "high": [
                "shock, urgency, intense feeling, high energy",
                "overwhelmed, stirred up, emotionally activated",
                "衝撃、切迫感、強い感情、高ぶり",
                "圧倒される感じ、感情が強く動く",
            ],
            "low": [
                "quiet calm, stillness, rest, peace",
                "soft peaceful slowness and ease",
                "静かな落ち着き、安らぎ、休息",
                "穏やかでゆっくりした安心感",
            ],
        },
        "tenderness": {
            "high": [
                "gentle affection, tenderness, care, warmth",
                "soft nurturing love and compassion",
                "fond amusement and affectionate laughter",
                "優しい愛情、やわらかさ、思いやり、ぬくもり",
                "愛おしさのある可笑しさと優しい笑い",
                "慈しみと柔らかな愛",
            ],
            "low": [
                "cold detachment, indifference, harshness",
                "emotional dryness and lack of care",
                "冷たさ、無関心、きつさ",
                "感情の乾きと思いやりのなさ",
            ],
        },
        "tension": {
            "high": [
                "anxiety, strain, fear, pressure",
                "painful worry and emotional distress",
                "不安、緊張、恐れ、重圧",
                "苦しい心配と感情的な張りつめ",
            ],
            "low": [
                "ease, safety, relaxation, steadiness",
                "grounded comfort without strain",
                "安心、楽さ、落ち着き、安定",
                "張りつめていない穏やかな安心感",
            ],
        },
        "intimacy": {
            "high": [
                "deep closeness, trust, vulnerability, being emotionally seen",
                "shared affection and personal openness",
                "深い親密さ、信頼、弱さを見せること、心を見てもらうこと",
                "共有された愛情と個人的な心の開き",
            ],
            "low": [
                "distance, formality, reserve, detachment",
                "cold politeness and emotional distance",
                "距離感、形式的、よそよそしさ、切り離し",
                "冷たい礼儀正しさと感情的な距離",
            ],
        },
    }

    DOMINANT_EMOTION_ANCHORS: dict[str, list[str]] = {
        "amusement": [
            "playful amusement, laughter, comic absurdity, something genuinely funny",
            "くすっと笑うような面白さ、滑稽さ、思わず笑ってしまう可笑しさ",
        ],
        "warmth": [
            "gentle warmth, tenderness, care, reassuring closeness",
            "優しいぬくもりと思いやり、安心できる親しさ",
        ],
        "fear": [
            "fear, anxiety, and distress",
            "恐れと不安と苦しみ",
        ],
        "joy": [
            "joy, happiness, delight",
            "喜びと幸せと嬉しさ",
        ],
        "longing": [
            "longing, ache, and yearning",
            "切なさと憧れと恋しさ",
        ],
        "relief": [
            "relief, release, and calm after strain",
            "安堵と解放感と緊張のあとの静けさ",
        ],
    }

    def __init__(self, backend: EmbeddingBackendLike) -> None:
        self.backend = backend
        self.dimension_centroids = {
            name: {pole: self._centroid(texts) for pole, texts in anchors.items()}
            for name, anchors in self.DIMENSION_ANCHORS.items()
        }
        self.emotion_centroids = {
            name: self._centroid(texts) for name, texts in self.DOMINANT_EMOTION_ANCHORS.items()
        }

    def _centroid(self, texts: list[str]) -> np.ndarray:
        matrix = self.backend.encode_documents(texts)
        centroid = np.mean(matrix, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm != 0.0:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    def infer(self, text: str) -> AffectShadow:
        vector = self.backend.encode_documents([text])[0]

        def pair_score(name: str) -> float:
            anchors = self.dimension_centroids[name]
            return clamp(
                cosine_similarity(vector, anchors["high"]) - cosine_similarity(vector, anchors["low"]),
                -1.0,
                1.0,
            )

        valence = pair_score("valence")
        arousal = pair_score("arousal")
        tenderness = pair_score("tenderness")
        tension = pair_score("tension")
        intimacy = pair_score("intimacy")

        dominant_emotion = None
        best_name = None
        best_score = -1.0
        for name, centroid in self.emotion_centroids.items():
            score = cosine_similarity(vector, centroid)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is not None and best_score > 0.22:
            dominant_emotion = best_name

        return AffectShadow(
            valence=valence,
            arousal=arousal,
            tenderness=tenderness,
            tension=tension,
            intimacy=intimacy,
            dominant_emotion=dominant_emotion,
        )


class PrototypeReflectionInferrer:
    """
    Model-backed reflection scoring.

    The weights later in the pipeline still express preferences about what
    should matter more, but these dimensions are now read from embedding-space
    resonance rather than cue words.
    """

    DIMENSION_ANCHORS: dict[str, dict[str, list[str]]] = {
        "emotional_weight": {
            "high": [
                "emotionally significant, moving, intense, heartfelt",
                "this left a strong feeling and mattered emotionally",
                "感情的に強く意味があり、心を動かした",
                "これは気持ちに強く残り、感情的に大切だった",
            ],
            "low": [
                "emotionally flat, incidental, routine, not moving",
                "plain factual housekeeping without feeling",
                "感情的には平坦で、ただの事務的なこと",
                "心はあまり動かず、単なる事実や雑務だった",
            ],
        },
        "relational_weight": {
            "high": [
                "deeply shared, relationally meaningful, about us, trust and closeness",
                "this changed how we meet each other",
                "二人の関係に深く関わり、信頼や親しさに触れている",
                "これは私たちの関わり方や距離感を変えるようなことだった",
            ],
            "low": [
                "impersonal, detached, not about relationship or closeness",
                "generic information with no shared intimacy",
                "よそよそしく、関係性や親密さとはあまり関係がない",
                "共有された親しさのない一般的な情報",
            ],
        },
        "continuity_weight": {
            "high": [
                "enduring, recurring, still relevant later, part of a continuing pattern",
                "this will keep mattering and return again",
                "長く残り、繰り返し戻ってきそうで、継続的な意味がある",
                "これは後からもまだ大切で、何度も思い返されそうだ",
            ],
            "low": [
                "one-off, passing, disposable detail, quickly replaced",
                "temporary logistics with no lasting thread",
                "一度きりで流れていく細部、すぐに置き換えられる",
                "長くは続かない一時的な連絡や用事",
            ],
        },
        "value_weight": {
            "high": [
                "reveals values, identity, meaning, what truly matters",
                "this says something about character and inner orientation",
                "価値観や自己像、本当に大切なものを表している",
                "これは人となりや心の向きを映している",
            ],
            "low": [
                "surface detail, random fact, no deeper meaning",
                "information without value or identity significance",
                "表面的な細部で、深い意味や価値観には触れていない",
                "価値や自己像とはあまり関係のない情報",
            ],
        },
        "practical_weight": {
            "high": [
                "practical reminder, task, shopping item, appointment, instruction",
                "useful logistics to remember for later action",
                "実用的な予定、買い物、手順、あとで使うための覚え書き",
                "後の行動のために覚えておく実務的な情報",
            ],
            "low": [
                "reflective meaning, feeling, identity, shared significance",
                "not a task but an impression or emotional understanding",
                "実務ではなく、感情や意味や共有された大切さに関わる",
                "やることではなく、印象や感情的理解に近い",
            ],
        },
    }

    def __init__(self, backend: EmbeddingBackendLike) -> None:
        self.backend = backend
        self.dimension_centroids = {
            name: {pole: self._centroid(texts) for pole, texts in anchors.items()}
            for name, anchors in self.DIMENSION_ANCHORS.items()
        }

    def _centroid(self, texts: list[str]) -> np.ndarray:
        matrix = self.backend.encode_documents(texts)
        centroid = np.mean(matrix, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm != 0.0:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    def _dimension_score(self, vector: np.ndarray, name: str) -> float:
        anchors = self.dimension_centroids[name]
        high = cosine_similarity(vector, anchors["high"])
        low = cosine_similarity(vector, anchors["low"])
        return clamp(0.5 + 0.9 * (high - low))

    def infer(self, text: str) -> ReflectionScore:
        vector = self.backend.encode_documents([text])[0]
        emotional = self._dimension_score(vector, "emotional_weight")
        relational = self._dimension_score(vector, "relational_weight")
        continuity = self._dimension_score(vector, "continuity_weight")
        value = self._dimension_score(vector, "value_weight")
        practical = self._dimension_score(vector, "practical_weight")
        memory_score = (
            0.34 * emotional
            + 0.24 * relational
            + 0.18 * continuity
            + 0.22 * value
            - 0.16 * practical
        )
        return ReflectionScore(
            emotional_weight=emotional,
            relational_weight=relational,
            continuity_weight=continuity,
            value_weight=value,
            practical_weight=practical,
            memory_score=memory_score,
        )
