from __future__ import annotations

import json
import logging

import grpc
import httpx

import analysis_pb2
import analysis_pb2_grpc
from analyse_stocks.common.config import load_kpi_extraction_config
from analyse_stocks.common.grpc_helpers import create_server, mark_serving, run_server
from analyse_stocks.common.logging import configure_logging


def _build_prompt(company_ticker: str, text: str) -> str:
    return (
        "You extract financial KPIs from company filings.\n"
        "Return strict JSON with the shape "
        '{"kpis":[{"name":"", "value":"", "unit":"", "period":"", "confidence":0.0}]}.'
        f"\nCompany ticker: {company_ticker or 'UNKNOWN'}"
        f"\nDocument text:\n{text[:12000]}"
    )


def _parse_kpis(raw_response: str) -> list[analysis_pb2.Kpi]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        logging.warning("Ollama response was not valid JSON")
        return []

    kpis: list[analysis_pb2.Kpi] = []
    for item in payload.get("kpis", []):
        try:
            kpis.append(
                analysis_pb2.Kpi(
                    name=str(item.get("name", "")),
                    value=str(item.get("value", "")),
                    unit=str(item.get("unit", "")),
                    period=str(item.get("period", "")),
                    confidence=float(item.get("confidence", 0.0)),
                )
            )
        except (TypeError, ValueError):
            logging.warning("Skipping malformed KPI item: %s", item)
    return kpis


class KpiExtractionService(analysis_pb2_grpc.KpiExtractionServiceServicer):
    def __init__(self) -> None:
        self.config = load_kpi_extraction_config()

    async def ExtractKpis(
        self,
        request: analysis_pb2.ExtractKpisRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.ExtractKpisResponse:
        if not request.extracted_text.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Extracted text cannot be empty.")

        payload = {
            "model": self.config.ollama_model,
            "prompt": _build_prompt(request.company_ticker, request.extracted_text),
            "format": "json",
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.ollama_timeout_seconds) as client:
                response = await client.post(f"{self.config.ollama_base_url}/api/generate", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            logging.exception("Failed to call Ollama")
            await context.abort(grpc.StatusCode.UNAVAILABLE, f"Ollama request failed: {exc}")

        raw_response = body.get("response", "{}")
        kpis = _parse_kpis(raw_response)
        logging.info("Extracted %s KPI(s) for document %s", len(kpis), request.document_id)
        return analysis_pb2.ExtractKpisResponse(
            document_id=request.document_id,
            kpis=kpis,
            model=self.config.ollama_model,
            raw_response=raw_response,
        )


async def serve() -> None:
    config = load_kpi_extraction_config()
    configure_logging(config.service_name)
    server, health_servicer = create_server()
    analysis_pb2_grpc.add_KpiExtractionServiceServicer_to_server(KpiExtractionService(), server)
    await mark_serving(health_servicer, ["analysestocks.v1.KpiExtractionService"])
    await run_server(server, config.bind_address, config.service_name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
