from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from analyse_stocks.pipeline.core import extract_json


@dataclass(frozen=True)
class OllamaRuntimeConfig:
    base_url: str
    model: str
    timeout_seconds: int
    ready_timeout_seconds: int


def ensure_model_ready(config: OllamaRuntimeConfig) -> None:
    started_at = time.perf_counter()
    deadline = started_at + config.ready_timeout_seconds
    last_error: Exception | None = None

    while time.perf_counter() < deadline:
        try:
            with httpx.Client(base_url=config.base_url, timeout=config.timeout_seconds) as client:
                tags_response = client.get("/api/tags")
                tags_response.raise_for_status()
                payload = tags_response.json()
                model_names = [item.get("name", "") for item in payload.get("models", [])]
                if config.model in model_names:
                    logging.info(
                        "Ollama is ready with model=%s available after %.2fs",
                        config.model,
                        time.perf_counter() - started_at,
                    )
                    return
                raise RuntimeError(f"Model {config.model} is not available. Found: {model_names}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.info("Waiting for Ollama model %s: %s", config.model, exc)
            time.sleep(3)

    raise RuntimeError(
        f"Ollama model {config.model} did not become ready in {config.ready_timeout_seconds}s"
    ) from last_error


def run_chat_json(
    config: OllamaRuntimeConfig,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    request_payload = {
        "model": config.model,
        "think": False,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
        },
    }

    started_at = time.perf_counter()
    with httpx.Client(base_url=config.base_url, timeout=config.timeout_seconds) as client:
        response = client.post("/api/chat", json=request_payload)
        response.raise_for_status()
        payload = response.json()

    message = payload.get("message", {})
    raw_response = str(message.get("content", "") or "").strip()
    thinking = str(message.get("thinking", "") or "").strip()
    if not raw_response and thinking:
        raw_response = thinking
    parsed = extract_json(raw_response)

    logging.info(
        "Ollama chat completed model=%s parsed=%s duration=%.2fs prompt_chars=%s response_chars=%s",
        config.model,
        parsed is not None,
        time.perf_counter() - started_at,
        len(prompt),
        len(raw_response),
    )
    if parsed is None:
        logging.warning(
            "Ollama returned invalid JSON. Request=%s RawResponse=%s",
            json.dumps(request_payload, ensure_ascii=False)[:3000],
            raw_response[:5000],
        )

    return {
        "prompt": prompt,
        "request_payload": request_payload,
        "raw_response": raw_response,
        "parsed": parsed,
    }
