# AnalyseStocksViaFinancialPapers

## Architecture

The project is designed as a Docker Compose based microservice system.

- `client-api`: external gRPC entrypoint and orchestrator.
- `parsing-service`: document parsing over gRPC.
- `kpi-extraction-service`: KPI extraction over gRPC with Ollama REST calls.
- `ollama`: local LLM runtime with `qwen3:4b`.
- `redis`: transient state and job metadata.
- `clickhouse`: analytical storage for extracted KPIs.

All service-to-service communication is gRPC, except `kpi-extraction-service -> ollama`, which uses REST.

## Run Everything

1. Copy `.env.example` to `.env` if you want to override defaults.
2. Start the whole stack:

```bash
docker compose up -d --build
```

On the first start, the model `qwen3:4b` is downloaded into the persistent Docker volume `ollama_data`.
Next starts reuse the same volume, so the model is not downloaded again.

## External Interfaces

- Grafana for logs: `http://localhost:3000`
- Loki API: `http://localhost:3100`
- gRPC UI for `client-api`: `http://localhost:8080`

Default Grafana credentials are controlled by `.env`:

- username: `admin`
- password: `admin`

ClickHouse credentials are also defined in `.env` and default to:

- username: `analysis_app`
- password: `analysis_app_password`

## gRPC Contracts

Shared protobuf definitions are stored in [proto/analysis.proto](/C:/Users/SS/Documents/AnalyseStocksViaFinancialPapers/proto/analysis.proto:1).

gRPC reflection is enabled for all internal services, so `grpcui` can discover the `client-api` contract automatically.

## Persistence

- Redis data is stored in the Docker volume `redis_data`.
- ClickHouse data is stored in the Docker volume `clickhouse_data`.
- Ollama models are stored in the Docker volume `ollama_data`.

## Logging

All Python services log structured JSON to stdout. `promtail` reads Docker container logs and pushes them to `loki`, and Grafana is preconfigured with Loki as the default data source.

Suggested Grafana LogQL query:

```text
{service="client-api"}
```

## Running The User Client

The easiest user-facing client is `grpcui`.

1. Start the stack with `docker compose up -d --build`
2. Open `http://localhost:8080`
3. Select `analysestocks.v1.ClientApiService`
4. Call `AnalyzeDocument`

Example request body:

```json
{
  "document": {
    "documentId": "demo-report-001",
    "companyTicker": "AAPL",
    "mimeType": "text/plain",
    "rawText": "Revenue for the quarter was 100 billion USD. Net income was 20 billion USD."
  }
}
```
