"""Named OpenAI-compatible endpoints stored in an encrypted setting."""

import json
import re
from typing import Any

_CONNECTION_ID = re.compile(r"custom-[0-9a-f]{32}\Z")


def connection_provider(value: str) -> bool:
    return bool(_CONNECTION_ID.fullmatch(value or ""))


def parse_connections(raw: str | None) -> list[dict[str, str]]:
    try:
        entries: Any = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(entries, list):
        return []
    result = []
    seen = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        if not all(isinstance(item.get(k), str) and item[k].strip() for k in ("id", "name", "base_url", "model", "api_key")):
            continue
        if not connection_provider(item["id"]) or item["id"] in seen:
            continue
        seen.add(item["id"])
        result.append({k: item[k] for k in ("id", "name", "base_url", "model", "api_key")})
    return result


def model_target(connection: dict[str, str]) -> str:
    return f'{connection["id"]}:{connection["model"]}'
