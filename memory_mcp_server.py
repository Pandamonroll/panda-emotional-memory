from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys
import traceback

from memory_runtime import MemoryRuntime
from memory_system import MemoryItem, SearchResult


SERVER_NAME = "panda-emotional-memory"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"
LOG_PATH = Path(__file__).resolve().parent / ".tmp" / "panda-emotional-memory-mcp.log"


def _open_log_stream() -> Any:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    return LOG_PATH.open("a", encoding="utf-8", buffering=1)


_LOG_STREAM = _open_log_stream()
sys.stderr = _LOG_STREAM


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


RUNTIME = MemoryRuntime()
MEMORY_URI_PREFIX = "memory://panda-emotional-memory/"


def _write_message(payload: dict[str, Any]) -> None:
    message = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    sys.stdout.buffer.write(message.encode("utf-8"))
    sys.stdout.flush()


def _read_message() -> dict[str, Any] | None:
    first_line = sys.stdin.buffer.readline()
    if not first_line:
        _log("stdin reached EOF before a message")
        return None

    stripped = first_line.strip()
    if not stripped:
        return _read_message()

    if stripped.lower().startswith(b"content-length:"):
        _log("received framed message header")
        length = int(stripped.split(b":", 1)[1].strip())
        while True:
            header_line = sys.stdin.buffer.readline()
            if not header_line or header_line in {b"\r\n", b"\n"}:
                break
        body = sys.stdin.buffer.read(length)
        return json.loads(body.decode("utf-8"))

    _log(f"received line message prefix: {stripped[:40]!r}")
    return json.loads(first_line.decode("utf-8"))


def _success(request_id: Any, result: dict[str, Any]) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
    )


def _error(request_id: Any, code: int, message: str, data: Any = None) -> None:
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if data is not None:
        payload["error"]["data"] = data
    _write_message(payload)


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "observe_exchange",
            "description": (
                "Observe one exchange into the live emotional memory store. "
                "This is the normal automatic memory path: it can recall nearby "
                "older traces, reconsolidate them, and decide whether the new "
                "exchange should become memory too."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "event_text": {
                        "type": "string",
                        "description": "The user-side or event-side text of the exchange.",
                    },
                    "assistant_response": {
                        "type": "string",
                        "description": "The assistant response text for the same exchange.",
                    },
                    "extra_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional extra tags to carry with the memory trace.",
                    },
                },
                "required": ["event_text", "assistant_response"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_memories",
            "description": "Search the live emotional memory store by meaning and feeling.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Maximum number of memories to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "runtime_status",
            "description": "Report whether the live emotional memory runtime and stores are healthy.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ]


def _affect_payload(affect: Any) -> dict[str, Any]:
    return {
        "valence": round(affect.valence, 4),
        "arousal": round(affect.arousal, 4),
        "tenderness": round(affect.tenderness, 4),
        "tension": round(affect.tension, 4),
        "intimacy": round(affect.intimacy, 4),
        "dominant_emotion": affect.dominant_emotion,
    }


def _memory_uri(memory: MemoryItem) -> str:
    return f"{MEMORY_URI_PREFIX}{memory.memory_id}"


def _memory_result_payload(match: SearchResult) -> dict[str, Any]:
    memory = match.memory
    return {
        "uri": _memory_uri(memory),
        "memory_id": memory.memory_id,
        "score": round(match.score, 4),
        "semantic_score": round(match.semantic_score, 4),
        "affect_score": round(match.affect_score, 4),
        "event_affect_score": round(match.event_affect_score, 4),
        "response_affect_score": round(match.response_affect_score, 4),
        "activation_score": round(match.activation_score, 4),
        "summary": memory.summary,
        "reflection_text": memory.reflection_text,
        "response_dominant": memory.response_affect.dominant_emotion,
        "tags": memory.tags,
        "created_at": memory.created_at,
        "source": memory.source,
    }


def _memory_resource_text(memory: MemoryItem, match: SearchResult | None = None) -> str:
    payload: dict[str, Any] = {
        "memory_id": memory.memory_id,
        "summary": memory.summary,
        "reflection_text": memory.reflection_text,
        "tags": memory.tags,
        "created_at": memory.created_at,
        "source": memory.source,
        "affect_shadow": _affect_payload(memory.affect_shadow),
        "event_affect": _affect_payload(memory.event_affect),
        "response_affect": _affect_payload(memory.response_affect),
    }
    if match is not None:
        payload["retrieval"] = {
            "score": round(match.score, 4),
            "semantic_score": round(match.semantic_score, 4),
            "affect_score": round(match.affect_score, 4),
            "event_affect_score": round(match.event_affect_score, 4),
            "response_affect_score": round(match.response_affect_score, 4),
            "activation_score": round(match.activation_score, 4),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _memory_resource_block(match: SearchResult) -> dict[str, Any]:
    memory = match.memory
    return {
        "type": "resource",
        "resource": {
            "uri": _memory_uri(memory),
            "mimeType": "application/json",
            "text": _memory_resource_text(memory, match),
        },
    }


def _text_result(payload: Any) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        ]
    }


def _memory_tool_result(payload: Any, matches: list[SearchResult]) -> dict[str, Any]:
    result = _text_result(payload)
    result["content"].extend(_memory_resource_block(match) for match in matches)
    return result


def _handle_initialize(request_id: Any, params: dict[str, Any]) -> None:
    client_version = params.get("protocolVersion") if isinstance(params, dict) else None
    protocol_version = client_version or PROTOCOL_VERSION
    _success(
        request_id,
        {
            "protocolVersion": protocol_version,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
                "resources": {
                    "listChanged": False,
                },
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        },
    )


def _handle_tools_list(request_id: Any) -> None:
    _success(
        request_id,
        {
            "tools": _tool_definitions(),
        },
    )


def _handle_resources_list(request_id: Any) -> None:
    store = RUNTIME.load_store()
    resources = [
        {
            "uri": _memory_uri(memory),
            "name": memory.summary[:80],
            "description": memory.reflection_text or memory.summary,
            "mimeType": "application/json",
        }
        for memory in store.memories
    ]
    _success(request_id, {"resources": resources})


def _handle_resource_templates_list(request_id: Any) -> None:
    _success(request_id, {"resourceTemplates": []})


def _handle_resources_read(request_id: Any, params: dict[str, Any]) -> None:
    uri = params.get("uri")
    if not isinstance(uri, str) or not uri.startswith(MEMORY_URI_PREFIX):
        _error(request_id, -32002, "Resource not found", {"uri": uri})
        return

    memory_id = uri.removeprefix(MEMORY_URI_PREFIX)
    store = RUNTIME.load_store()
    for memory in store.memories:
        if memory.memory_id == memory_id:
            _success(
                request_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": _memory_resource_text(memory),
                        }
                    ]
                },
            )
            return

    _error(request_id, -32002, "Resource not found", {"uri": uri})


def _handle_tool_call(request_id: Any, params: dict[str, Any]) -> None:
    name = params.get("name")
    arguments = params.get("arguments", {}) or {}

    if name == "observe_exchange":
        result = RUNTIME.observe_exchange(
            arguments["event_text"],
            arguments["assistant_response"],
            extra_tags=arguments.get("extra_tags"),
        )
        payload = {
            "store_path": str(result.store_path),
            "memory_count": result.memory_count,
            "kind": result.observation.reflection.kind,
            "summary": result.observation.reflection.summary,
            "reflection_text": result.observation.reflection.reflection_text,
            "response_dominant": result.observation.reflection.response_affect.dominant_emotion,
            "recalled": [
                _memory_result_payload(match)
                for match in result.observation.recalled
            ],
        }
        _success(request_id, _memory_tool_result(payload, result.observation.recalled))
        return

    if name == "search_memories":
        matches = RUNTIME.search(
            arguments["query"],
            limit=int(arguments.get("limit", 5)),
        )
        payload = [_memory_result_payload(match) for match in matches]
        _success(request_id, _memory_tool_result(payload, matches))
        return

    if name == "runtime_status":
        _success(request_id, _text_result(RUNTIME.status()))
        return

    _error(request_id, -32601, f"Unknown tool: {name}")


def _handle_request(message: dict[str, Any]) -> None:
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})

    if method == "initialize":
        _handle_initialize(request_id, params)
        return

    if method == "tools/list":
        _handle_tools_list(request_id)
        return

    if method == "resources/list":
        _handle_resources_list(request_id)
        return

    if method == "resources/templates/list":
        _handle_resource_templates_list(request_id)
        return

    if method == "resources/read":
        _handle_resources_read(request_id, params)
        return

    if method == "tools/call":
        _handle_tool_call(request_id, params)
        return

    if method in {"notifications/initialized", "initialized"}:
        return

    if request_id is not None:
        _error(request_id, -32601, f"Unknown method: {method}")


def main() -> None:
    _log(f"{SERVER_NAME} {SERVER_VERSION} starting")
    while True:
        try:
            message = _read_message()
        except json.JSONDecodeError as exc:
            _log(f"invalid json: {exc}")
            _error(None, -32700, "Invalid JSON", {"detail": str(exc)})
            continue

        if message is None:
            break

        try:
            _log(f"handling method: {message.get('method')}")
            _handle_request(message)
        except Exception as exc:
            request_id = message.get("id")
            _error(
                request_id,
                -32603,
                "Internal error",
                {
                    "detail": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    _log(f"{SERVER_NAME} stdin closed")


if __name__ == "__main__":
    main()
