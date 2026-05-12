from __future__ import annotations

import logging
import uuid
from typing import Any

import clickhouse_connect
import grpc
from redis.asyncio import from_url as redis_from_url
from clickhouse_connect.driver.exceptions import ClickHouseError

import analysis_pb2
import analysis_pb2_grpc
from analyse_stocks.common.config import load_client_api_config
from analyse_stocks.common.grpc_helpers import create_server, enable_reflection, mark_serving, run_server
from analyse_stocks.common.logging import configure_logging


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
