from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class ServiceConfig:
    service_name: str
    grpc_host: str
    grpc_port: int

    @property
    def bind_address(self) -> str:
        return f"{self.grpc_host}:{self.grpc_port}"


@dataclass(frozen=True)
class ClientApiConfig(ServiceConfig):
    parsing_service_address: str
    kpi_extraction_service_address: str
    redis_url: str
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_username: str
    clickhouse_password: str
    clickhouse_database: str
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int


@dataclass(frozen=True)
class KpiExtractionConfig(ServiceConfig):
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    ollama_ready_timeout_seconds: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_service_config(service_name: str, default_port: int) -> ServiceConfig:
    return ServiceConfig(
        service_name=service_name,
        grpc_host=_env("GRPC_HOST", "0.0.0.0"),
        grpc_port=_env_int("GRPC_PORT", default_port),
    )


def load_client_api_config() -> ClientApiConfig:
    base = load_service_config("client-api", 50051)
    return ClientApiConfig(
        service_name=base.service_name,
        grpc_host=base.grpc_host,
        grpc_port=base.grpc_port,
        parsing_service_address=_env("PARSING_SERVICE_ADDRESS", "parsing-service:50052"),
        kpi_extraction_service_address=_env("KPI_EXTRACTION_SERVICE_ADDRESS", "kpi-extraction-service:50053"),
        redis_url=_env("REDIS_URL", "redis://redis:6379/0"),
        clickhouse_host=_env("CLICKHOUSE_HOST", "clickhouse"),
        clickhouse_port=_env_int("CLICKHOUSE_PORT", 8123),
        clickhouse_username=_env("CLICKHOUSE_USERNAME", "default"),
        clickhouse_password=_env("CLICKHOUSE_PASSWORD", ""),
        clickhouse_database=_env("CLICKHOUSE_DATABASE", "analysis"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://ollama:11434"),
        ollama_model=_env("OLLAMA_MODEL", "qwen3:4b"),
        ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 120),
    )


def load_kpi_extraction_config() -> KpiExtractionConfig:
    base = load_service_config("kpi-extraction-service", 50053)
    return KpiExtractionConfig(
        service_name=base.service_name,
        grpc_host=base.grpc_host,
        grpc_port=base.grpc_port,
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://ollama:11434"),
        ollama_model=_env("OLLAMA_MODEL", "qwen3:4b"),
        ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 180),
        ollama_ready_timeout_seconds=_env_int("OLLAMA_READY_TIMEOUT_SECONDS", 180),
    )
