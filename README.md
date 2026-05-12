# AnalyseStocksViaFinancialPapers

## Architecture

The project is designed as a Docker Compose based microservice system.

- `client-api`: external gRPC entrypoint and orchestrator.
- `parsing-service`: document parsing over gRPC.
- `kpi-extraction-service`: KPI extraction over gRPC with Ollama REST calls.
- `ollama`: local LLM runtime with `qwen2.5:1.5b`.
- `redis`: transient state and job metadata.
- `clickhouse`: analytical storage for extracted KPIs.

All service-to-service communication is gRPC, except `kpi-extraction-service -> ollama`, which uses REST.

## Run Everything

1. Copy `.env.example` to `.env` if you want to override defaults.
2. Start the whole stack:

```bash
docker compose up -d --build
```

On the first start, the model `qwen2.5:1.5b` is downloaded into the persistent Docker volume `ollama_data`.
Next starts reuse the same volume, so the model is not downloaded again.

## gRPC Contracts

Shared protobuf definitions are stored in [proto/analysis.proto](/C:/Users/SS/Documents/AnalyseStocksViaFinancialPapers/proto/analysis.proto:1).

## Persistence

- Redis data is stored in the Docker volume `redis_data`.
- ClickHouse data is stored in the Docker volume `clickhouse_data`.
- Ollama models are stored in the Docker volume `ollama_data`.
