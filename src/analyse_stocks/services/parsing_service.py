from __future__ import annotations

from dataclasses import asdict
import logging

import grpc

import analysis_pb2
import analysis_pb2_grpc
from analyse_stocks.common.config import load_service_config
from analyse_stocks.common.grpc_helpers import create_server, enable_reflection, mark_serving, run_server
from analyse_stocks.common.logging import configure_logging
from analyse_stocks.pipeline.core import extract_pages, generate_candidates, pages_from_raw_text, resolve_pdf_path


class ParsingService(analysis_pb2_grpc.ParsingServiceServicer):
    async def ParseDocument(
        self,
        request: analysis_pb2.ParseDocumentRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.ParseDocumentResponse:
        document = request.document
        if document.raw_text.strip():
            pages = pages_from_raw_text(document.raw_text.strip())
            source = "raw_text"
        elif document.source_uri.strip():
            try:
                pdf_path = resolve_pdf_path(document.source_uri)
            except FileNotFoundError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            pages = extract_pages(pdf_path)
            source = "pdf_path"
        else:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Document must include raw_text or source_uri.",
            )
        extracted_text = "\n\n".join(page["text"] for page in pages if page.get("text")).strip()
        candidates = generate_candidates(pages)
        metadata = {
            "source": source,
            "mime_type": document.mime_type or ("application/pdf" if source == "pdf_path" else "text/plain"),
            "num_pages": str(len(pages)),
            "num_candidates": str(len(candidates)),
        }

        logging.info("Parsed document %s", document.document_id or "<generated>")
        return analysis_pb2.ParseDocumentResponse(
            document_id=document.document_id,
            extracted_text=extracted_text,
            metadata=metadata,
            debug_pages=[
                analysis_pb2.PageDebug(
                    page_num=page["page_num"],
                    text=page.get("text", ""),
                    section_hint=page.get("section_hint") or "",
                    section_type=page.get("section_type") or "",
                    section_score=float(page.get("section_score", 0.0)),
                )
                for page in pages
            ],
            candidates=[
                analysis_pb2.CandidateData(
                    source_type=candidate.source_type,
                    page_num=candidate.page_num,
                    section_hint=candidate.section_hint or "",
                    section_type=candidate.section_type or "",
                    section_score=candidate.section_score,
                    label_text=candidate.label_text,
                    value_text=candidate.value_text,
                    raw_text=candidate.raw_text,
                    normalized_value_text=candidate.normalized_value_text or "",
                    extracted_period=candidate.extracted_period or "",
                    pre_mapped_kpi=candidate.pre_mapped_kpi or "",
                    pre_map_confidence=candidate.pre_map_confidence,
                )
                for candidate in candidates
            ],
        )


async def serve() -> None:
    config = load_service_config("parsing-service", 50052)
    configure_logging(config.service_name)
    server, health_servicer = create_server()
    grpc_service_name = "analysestocks.v1.ParsingService"
    analysis_pb2_grpc.add_ParsingServiceServicer_to_server(ParsingService(), server)
    enable_reflection(server, [grpc_service_name])
    await mark_serving(health_servicer, [grpc_service_name])
    await run_server(server, config.bind_address, config.service_name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
