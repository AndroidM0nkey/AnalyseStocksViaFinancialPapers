from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

import clickhouse_connect
import grpc
import httpx
from google.protobuf.struct_pb2 import Struct
from redis.asyncio import from_url as redis_from_url
from clickhouse_connect.driver.exceptions import ClickHouseError

import analysis_pb2
import analysis_pb2_grpc
from analyse_stocks.common.config import load_client_api_config
from analyse_stocks.common.grpc_helpers import create_server, enable_reflection, mark_serving, run_server
from analyse_stocks.common.logging import configure_logging


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_float(value: str) -> float | None:
    if not value:
        return None
    normalized = value.replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _build_kpi_dict(kpis: list[analysis_pb2.NormalizedKpi]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in kpis:
        if item.canonical_kpi and item.canonical_kpi not in result:
            result[item.canonical_kpi] = item.value
    return result


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _compute_derived_metrics(kpis: dict[str, float]) -> dict[str, float]:
    revenue = kpis.get("revenue")
    ebitda = kpis.get("ebitda")
    net_income = kpis.get("net_income")
    debt = kpis.get("debt")
    equity = kpis.get("equity")
    cash = kpis.get("cash")
    operating_income = kpis.get("operating_income")

    derived = {
        "ebitda_margin": _safe_div(ebitda, revenue),
        "net_margin": _safe_div(net_income, revenue),
        "operating_margin": _safe_div(operating_income, revenue),
        "debt_to_equity": _safe_div(debt, equity),
        "cash_to_debt": _safe_div(cash, debt),
    }
    return {key: value for key, value in derived.items() if value is not None}


def _build_analytical_report_prompt(report_input: dict[str, Any]) -> str:
    return f"""
You are a senior equity research analyst.

Your task is to prepare a concise analytical report based ONLY on the structured KPI data below.
Do not use outside knowledge.
Do not invent missing facts.
If the data is mixed or insufficient, say so explicitly.
Write the analytical report in Russian.

Company: {report_input.get("company")}
Ticker: {report_input.get("ticker")}
Event date: {report_input.get("event_date")}
Filing type: {report_input.get("filing_type")}
Fiscal period: {report_input.get("fiscal_period")}
Currency: {report_input.get("currency")}

Normalized KPIs:
{json.dumps(report_input.get("normalized_kpis", {}), ensure_ascii=False, indent=2)}

Derived metrics:
{json.dumps(report_input.get("derived_metrics", {}), ensure_ascii=False, indent=2)}

Write the output as valid JSON only with the following schema:
{{
  "executive_summary": "string",
  "positive_factors": ["string", "string"],
  "risk_factors": ["string", "string"],
  "financial_health_assessment": "string",
  "profitability_assessment": "string",
  "leverage_assessment": "string",
  "investment_view": {{
    "signal": "strong_buy | buy | hold | sell | strong_sell",
    "confidence": 0.0,
    "expected_short_term_reaction": "string",
    "rationale": "string"
  }},
  "key_kpi_interpretation": [
    {{
      "kpi": "string",
      "interpretation": "string"
    }}
  ]
}}
""".strip()


def _build_market_decision_prompt(report_input: dict[str, Any]) -> str:
    return f"""
You are a senior equity research analyst.

Your task is to produce ONLY a compact market decision based ONLY on the KPI data below.
Do not use outside knowledge.
Do not invent missing facts.
If the data is mixed or insufficient, prefer a neutral signal.
Write the output in Russian.

Company: {report_input.get("company")}
Ticker: {report_input.get("ticker")}
Event date: {report_input.get("event_date")}
Filing type: {report_input.get("filing_type")}
Fiscal period: {report_input.get("fiscal_period")}
Currency: {report_input.get("currency")}

Normalized KPIs:
{json.dumps(report_input.get("normalized_kpis", {}), ensure_ascii=False, indent=2)}

Derived metrics:
{json.dumps(report_input.get("derived_metrics", {}), ensure_ascii=False, indent=2)}

Return valid JSON only with this schema:
{{
  "signal": "strong_buy | buy | hold | sell | strong_sell",
  "confidence": 0.0,
  "expected_move": "string",
  "rationale": "string"
}}
""".strip()


def _render_report_markdown(report: dict[str, Any]) -> str:
    if not report:
        return "Не удалось распарсить JSON-ответ модели."

    investment_view = report.get("investment_view", {})
    lines = [
        "# Аналитический отчет",
        "",
        "## Executive Summary",
        report.get("executive_summary", ""),
        "",
        "## Positive Factors",
    ]
    lines.extend(f"- {item}" for item in report.get("positive_factors", []))
    lines.extend(
        [
            "",
            "## Risk Factors",
        ]
    )
    lines.extend(f"- {item}" for item in report.get("risk_factors", []))
    lines.extend(
        [
            "",
            "## Financial Health Assessment",
            report.get("financial_health_assessment", ""),
            "",
            "## Profitability Assessment",
            report.get("profitability_assessment", ""),
            "",
            "## Leverage Assessment",
            report.get("leverage_assessment", ""),
            "",
            "## Investment View",
            f"- Signal: **{investment_view.get('signal', '')}**",
            f"- Confidence: **{investment_view.get('confidence', '')}**",
            f"- Expected short-term reaction: **{investment_view.get('expected_short_term_reaction', '')}**",
            f"- Rationale: {investment_view.get('rationale', '')}",
            "",
            "## KPI Interpretation",
        ]
    )
    lines.extend(
        f"- **{item.get('kpi', '')}**: {item.get('interpretation', '')}"
        for item in report.get("key_kpi_interpretation", [])
    )
    return "\n".join(lines)


def _to_struct(payload: dict[str, Any] | None) -> Struct:
    struct = Struct()
    if payload:
        struct.update(payload)
    return struct


class ClientApiService(analysis_pb2_grpc.ClientApiServiceServicer):
    def __init__(self) -> None:
        self.config = load_client_api_config()
        self.redis = None
        self.clickhouse = None

    def _get_redis(self) -> Any:
        if self.redis is None:
            self.redis = redis_from_url(self.config.redis_url, decode_responses=True)
        return self.redis

    def _get_clickhouse(self) -> Any:
        if self.clickhouse is None:
            try:
                self.clickhouse = clickhouse_connect.get_client(
                    host=self.config.clickhouse_host,
                    port=self.config.clickhouse_port,
                    username=self.config.clickhouse_username,
                    password=self.config.clickhouse_password,
                    database=self.config.clickhouse_database,
                )
            except ClickHouseError:
                logging.exception("ClickHouse connection is unavailable")
                self.clickhouse = False
        return self.clickhouse

    async def _run_ollama_json(self, prompt: str) -> tuple[dict[str, Any] | None, str]:
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.config.ollama_timeout_seconds) as client:
            response = await client.post(f"{self.config.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
            body = response.json()
        raw_response = body.get("response", "{}")
        return _extract_json(raw_response), raw_response

    async def AnalyzeDocument(
        self,
        request: analysis_pb2.AnalyzeDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.AnalyzeDocumentResponse:
        document = request.document
        document_id = document.document_id or str(uuid.uuid4())
        redis = self._get_redis()
        clickhouse = self._get_clickhouse()
        await redis.hset(f"analysis:{document_id}", mapping={"status": "processing"})

        async with grpc.aio.insecure_channel(self.config.parsing_service_address) as parsing_channel:
            parsing_stub = analysis_pb2_grpc.ParsingServiceStub(parsing_channel)
            parsing_response = await parsing_stub.ParseDocument(
                analysis_pb2.ParseDocumentRequest(
                    document=analysis_pb2.DocumentReference(
                        document_id=document_id,
                        company_ticker=document.company_ticker,
                        source_uri=document.source_uri,
                        mime_type=document.mime_type,
                        raw_text=document.raw_text,
                    )
                )
            )

        async with grpc.aio.insecure_channel(self.config.kpi_extraction_service_address) as kpi_channel:
            kpi_stub = analysis_pb2_grpc.KpiExtractionServiceStub(kpi_channel)
            kpi_response = await kpi_stub.ExtractKpis(
                analysis_pb2.ExtractKpisRequest(
                    document_id=document_id,
                    company_ticker=document.company_ticker,
                    extracted_text=parsing_response.extracted_text,
                )
            )

        rows = [
            [
                document_id,
                document.company_ticker,
                kpi.name,
                kpi.value,
                kpi.unit,
                kpi.period,
                kpi.confidence,
                kpi_response.model,
            ]
            for kpi in kpi_response.kpis
        ]
        if rows and clickhouse:
            try:
                clickhouse.insert(
                    "kpi_extractions",
                    rows,
                    column_names=[
                        "document_id",
                        "company_ticker",
                        "kpi_name",
                        "kpi_value",
                        "unit",
                        "period",
                        "confidence",
                        "model",
                    ],
                )
            except ClickHouseError:
                logging.exception("Failed to persist KPI rows to ClickHouse")

        await redis.hset(
            f"analysis:{document_id}",
            mapping={
                "status": "completed",
                "kpi_count": len(kpi_response.kpis),
                "clickhouse_persisted": "true" if rows and clickhouse else "false",
            },
        )
        logging.info("Completed analysis for %s", document_id)
        return analysis_pb2.AnalyzeDocumentResponse(
            document_id=document_id,
            parsed_text=parsing_response.extracted_text,
            kpis=kpi_response.kpis,
            status="completed",
        )

    async def RunPipeline(
        self,
        request: analysis_pb2.RunPipelineRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.RunPipelineResponse:
        if not request.raw_text.strip() and not request.source_uri.strip() and not request.pdf_path.strip():
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "RunPipeline requires raw_text, source_uri, or pdf_path.",
            )

        document_id = request.document_id or str(uuid.uuid4())
        redis = self._get_redis()
        clickhouse = self._get_clickhouse()
        await redis.hset(f"analysis:{document_id}", mapping={"status": "processing"})

        source_uri = request.source_uri or request.pdf_path
        async with grpc.aio.insecure_channel(self.config.parsing_service_address) as parsing_channel:
            parsing_stub = analysis_pb2_grpc.ParsingServiceStub(parsing_channel)
            parsing_response = await parsing_stub.ParseDocument(
                analysis_pb2.ParseDocumentRequest(
                    document=analysis_pb2.DocumentReference(
                        document_id=document_id,
                        company_ticker=request.ticker,
                        source_uri=source_uri,
                        mime_type=request.mime_type or "application/pdf",
                        raw_text=request.raw_text,
                    )
                )
            )

        async with grpc.aio.insecure_channel(self.config.kpi_extraction_service_address) as kpi_channel:
            kpi_stub = analysis_pb2_grpc.KpiExtractionServiceStub(kpi_channel)
            kpi_response = await kpi_stub.ExtractKpis(
                analysis_pb2.ExtractKpisRequest(
                    document_id=document_id,
                    company_ticker=request.ticker,
                    extracted_text=parsing_response.extracted_text,
                )
            )

        normalized_kpis = [
            analysis_pb2.NormalizedKpi(
                canonical_kpi=kpi.name,
                value=_parse_float(kpi.value) or 0.0,
                unit=kpi.unit,
                period=kpi.period,
                confidence=kpi.confidence,
            )
            for kpi in kpi_response.kpis
            if _parse_float(kpi.value) is not None
        ]
        normalized_kpi_dict = _build_kpi_dict(normalized_kpis)
        derived_metrics = _compute_derived_metrics(normalized_kpi_dict)

        report_input = {
            "company": request.company,
            "ticker": request.ticker,
            "event_date": request.event_date,
            "filing_type": request.filing_type,
            "fiscal_period": request.fiscal_period,
            "currency": request.currency or "RUB",
            "normalized_kpis": normalized_kpi_dict,
            "derived_metrics": derived_metrics,
        }

        try:
            analytical_report, analytical_raw = await self._run_ollama_json(
                _build_analytical_report_prompt(report_input)
            )
            market_decision, decision_raw = await self._run_ollama_json(
                _build_market_decision_prompt(report_input)
            )
        except httpx.HTTPError as exc:
            await redis.hset(f"analysis:{document_id}", mapping={"status": "failed"})
            logging.exception("Failed to generate analytical outputs")
            await context.abort(grpc.StatusCode.UNAVAILABLE, f"Ollama request failed: {exc}")

        rows = [
            [
                document_id,
                request.ticker,
                kpi.canonical_kpi,
                str(kpi.value),
                kpi.unit,
                kpi.period,
                kpi.confidence,
                kpi_response.model,
            ]
            for kpi in normalized_kpis
        ]
        clickhouse_persisted = False
        if rows and clickhouse:
            try:
                clickhouse.insert(
                    "kpi_extractions",
                    rows,
                    column_names=[
                        "document_id",
                        "company_ticker",
                        "kpi_name",
                        "kpi_value",
                        "unit",
                        "period",
                        "confidence",
                        "model",
                    ],
                )
                clickhouse_persisted = True
            except ClickHouseError:
                logging.exception("Failed to persist KPI rows to ClickHouse")

        await redis.hset(
            f"analysis:{document_id}",
            mapping={
                "status": "completed",
                "kpi_count": len(normalized_kpis),
                "clickhouse_persisted": "true" if clickhouse_persisted else "false",
            },
        )

        metadata = analysis_pb2.PipelineMetadata(
            pdf_path=request.pdf_path,
            company=request.company,
            ticker=request.ticker,
            event_date=request.event_date,
            filing_type=request.filing_type,
            fiscal_period=request.fiscal_period,
            currency=request.currency or "RUB",
            num_pages=int(parsing_response.metadata.get("num_pages", "0")),
            num_candidates=len(kpi_response.kpis),
            dictionary_hits=0,
            llm_calls_for_normalization=0,
        )
        logging.info("Completed pipeline for %s", document_id)
        return analysis_pb2.RunPipelineResponse(
            document_id=document_id,
            status="completed",
            parsed_text=parsing_response.extracted_text,
            metadata=metadata,
            normalized_kpis_list=normalized_kpis,
            normalized_kpis_dict=normalized_kpi_dict,
            derived_metrics=derived_metrics,
            analytical_report=_to_struct(analytical_report),
            market_decision=_to_struct(market_decision),
            analytical_report_markdown=_render_report_markdown(analytical_report or {}),
            raw_analytical_response=analytical_raw,
            raw_decision_response=decision_raw,
        )


async def serve() -> None:
    config = load_client_api_config()
    configure_logging(config.service_name)
    server, health_servicer = create_server()
    grpc_service_name = "analysestocks.v1.ClientApiService"
    analysis_pb2_grpc.add_ClientApiServiceServicer_to_server(ClientApiService(), server)
    enable_reflection(server, [grpc_service_name])
    await mark_serving(health_servicer, [grpc_service_name])
    await run_server(server, config.bind_address, config.service_name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
