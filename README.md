# AnalyseStocksViaFinancialPapers

## Docker Compose

The project is bootstrapped around `docker compose`.

### Start Ollama with Qwen2.5 1.5B

1. Copy `.env.example` to `.env` if you want to override defaults.
2. Run:

```bash
docker compose up -d
```

After the first start, the model `qwen2.5:1.5b` will be downloaded into the persistent Docker volume `ollama_data`.
On the next starts, the init container will detect the model in that volume and skip the download.

### Check that the model is available

```bash
docker compose exec ollama ollama list
```
