from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import clickhouse_connect
import grpc
from clickhouse_connect.driver.exceptions import ClickHouseError
from google.protobuf.json_format import MessageToDict, ParseDict
from redis.asyncio import from_url as redis_from_url

import analysis_pb2
import analysis_pb2_grpc
from analyse_stocks.common.config import load_client_api_config
from analyse_stocks.common.grpc_helpers import create_server, enable_reflection, mark_serving, run_server
from analyse_stocks.common.logging import configure_logging


def _render_report_markdown(report: dict[str, Any], raw_response: str = "") -> str:
    if not report:
        if raw_response.strip():
            return "\n".join(
                [
                    "Не удалось распарсить JSON-ответ модели.",
                    "",
                    "```text",
                    raw_response.strip(),
                    "```",
                ]
            )
        return "Не удалось распарсить JSON-ответ модели."

    inv = report.get("investment_view", {})
    if not isinstance(inv, dict):
        inv = {}
    kpi_items = report.get("key_kpi_interpretation", [])
    if not isinstance(kpi_items, list):
        kpi_items = []

    lines = []
    lines.append("# Аналитический отчет")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(str(report.get("executive_summary", "")))
    lines.append("")
    lines.append("## Positive Factors")
    for item in report.get("positive_factors", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Risk Factors")
    for item in report.get("risk_factors", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Financial Health Assessment")
    lines.append(str(report.get("financial_health_assessment", "")))
    lines.append("")
    lines.append("## Profitability Assessment")
    lines.append(str(report.get("profitability_assessment", "")))
    lines.append("")
    lines.append("## Leverage Assessment")
    lines.append(str(report.get("leverage_assessment", "")))
    lines.append("")
    lines.append("## Investment View")
    lines.append(f"- Signal: **{inv.get('signal', '')}**")
    lines.append(f"- Confidence: **{inv.get('confidence', '')}**")
    lines.append(f"- Expected short-term reaction: **{inv.get('expected_short_term_reaction', '')}**")
    lines.append(f"- Rationale: {inv.get('rationale', '')}")
    lines.append("")
    lines.append("## KPI Interpretation")
    for item in kpi_items:
        if isinstance(item, dict):
            lines.append(f"- **{item.get('kpi', '')}**: {item.get('interpretation', '')}")
    return "\n".join(lines)


def _struct_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "DESCRIPTOR"):
        converted = MessageToDict(value, preserving_proto_field_name=True)
        return converted if isinstance(converted, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _result_storage_key(document_id: str) -> str:
    return f"analysis-result:{document_id}"


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
                    candidates=parsing_response.candidates,
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
                    candidates=parsing_response.candidates,
                )
            )
            outputs_response = await kpi_stub.GenerateAnalyticalOutputs(
                analysis_pb2.GenerateAnalyticalOutputsRequest(
                    company=request.company,
                    ticker=request.ticker,
                    event_date=request.event_date,
                    filing_type=request.filing_type,
                    fiscal_period=request.fiscal_period,
                    currency=request.currency or "RUB",
                    normalized_kpis_dict=dict(kpi_response.normalized_kpis_dict),
                    derived_metrics=dict(kpi_response.derived_metrics),
                )
            )

        normalized_kpis = list(kpi_response.normalized_kpis_list)
        normalized_kpi_dict = dict(kpi_response.normalized_kpis_dict)
        derived_metrics = dict(kpi_response.derived_metrics)
        analytical_report = _struct_to_dict(outputs_response.analytical_report)
        market_decision = _struct_to_dict(outputs_response.market_decision)
        logging.info("RunPipeline outputs analytical_report keys=%s", sorted(analytical_report.keys()))
        logging.info("RunPipeline outputs market_decision keys=%s", sorted(market_decision.keys()))

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
            num_candidates=len(parsing_response.candidates),
            dictionary_hits=int(kpi_response.dictionary_hits),
            llm_calls_for_normalization=int(kpi_response.llm_calls_for_normalization),
        )

        response = analysis_pb2.RunPipelineResponse(
            document_id=document_id,
            status="completed",
            analytical_report=outputs_response.analytical_report,
            market_decision=outputs_response.market_decision,
            analytical_report_markdown=_render_report_markdown(
                analytical_report,
                outputs_response.raw_analytical_response,
            ),
            metadata=metadata,
            debug_pages=parsing_response.debug_pages,
            debug_candidates_raw=parsing_response.candidates,
            debug_candidates_normalized_before_consolidation=kpi_response.normalized_items,
            debug_failed_candidates=kpi_response.failed_candidates,
            normalized_kpis_list=normalized_kpis,
            normalized_kpis_dict=normalized_kpi_dict,
            derived_metrics=derived_metrics,
            raw_analytical_response=outputs_response.raw_analytical_response,
            raw_decision_response=outputs_response.raw_decision_response,
        )
        await redis.set(
            _result_storage_key(document_id),
            json.dumps(MessageToDict(response, preserving_proto_field_name=True), ensure_ascii=False),
        )

        logging.info("Completed pipeline for %s", document_id)
        return response

    async def GetPipelineResult(
        self,
        request: analysis_pb2.GetPipelineResultRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.RunPipelineResponse:
        document_id = request.document_id.strip()
        if not document_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "document_id is required.")

        redis = self._get_redis()
        stored_payload = await redis.get(_result_storage_key(document_id))
        if not stored_payload:
            legacy_status = await redis.hget(f"analysis:{document_id}", "status")
            if legacy_status:
                return analysis_pb2.RunPipelineResponse(
                    document_id=document_id,
                    status=legacy_status,
                )
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Stored pipeline result for {document_id} was not found.",
            )

        try:
            payload = json.loads(stored_payload)
            response = analysis_pb2.RunPipelineResponse()
            ParseDict(payload, response)
            return response
        except (json.JSONDecodeError, ValueError, TypeError):
            logging.exception("Failed to decode stored pipeline result for %s", document_id)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Stored pipeline result for {document_id} is corrupted.",
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
