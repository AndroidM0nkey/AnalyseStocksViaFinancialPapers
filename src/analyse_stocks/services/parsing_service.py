from __future__ import annotations

import logging

import grpc

import analysis_pb2
import analysis_pb2_grpc
from analyse_stocks.common.config import load_service_config
from analyse_stocks.common.grpc_helpers import create_server, mark_serving, run_server
from analyse_stocks.common.logging import configure_logging


class ParsingService(analysis_pb2_grpc.ParsingServiceServicer):
    async def ParseDocument(
        self,
        request: analysis_pb2.ParseDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.ParseDocumentResponse:
        document = request.document

        if document.raw_text.strip():
            extracted_text = document.raw_text.strip()
            metadata = {
                "source": "raw_text",
                "mime_type": document.mime_type or "text/plain",
            }
        elif document.source_uri.strip():
            extracted_text = (
                "Remote parsing is not implemented yet. "
                f"Use raw_text for now. Requested source: {document.source_uri}"
            )
            metadata = {
                "source": "source_uri",
                "mime_type": document.mime_type or "application/octet-stream",
                "status": "stubbed",
            }
        else:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Document must include raw_text or source_uri.",
            )

        logging.info("Parsed document %s", document.document_id or "<generated>")
        return analysis_pb2.ParseDocumentResponse(
            document_id=document.document_id,
            extracted_text=extracted_text,
            metadata=metadata,
        )


async def serve() -> None:
    config = load_service_config("parsing-service", 50052)
    configure_logging(config.service_name)
    server, health_servicer = create_server()
    analysis_pb2_grpc.add_ParsingServiceServicer_to_server(ParsingService(), server)
    await mark_serving(health_servicer, ["analysestocks.v1.ParsingService"])
    await run_server(server, config.bind_address, config.service_name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
