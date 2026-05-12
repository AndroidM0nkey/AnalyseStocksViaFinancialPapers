from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
from typing import Any

import grpc
from google.protobuf.struct_pb2 import Struct

import analysis_pb2
import analysis_pb2_grpc
from analyse_stocks.common.config import load_kpi_extraction_config
from analyse_stocks.common.grpc_helpers import create_server, enable_reflection, mark_serving, run_server
from analyse_stocks.common.logging import configure_logging
from analyse_stocks.common.ollama_runtime import OllamaRuntimeConfig, ensure_model_ready, run_chat_json
from analyse_stocks.pipeline.core import (
    CANONICAL_KPIS,
    Candidate,
    NormalizedKPI,
    build_kpi_dict,
    compute_derived_metrics,
    consolidate_kpis,
    extract_json,
    normalize_candidate_with_dictionary,
)


def _to_struct(payload: dict[str, Any] | None) -> Struct:
    struct = Struct()
    if payload:
        struct.update(payload)
    return struct


def _build_prompt(company_ticker: str, text: str) -> str:
    return (
        "You extract financial KPIs from company filings.\n"
        "Return strict JSON with the shape "
        '{"kpis":[{"name":"", "value":"", "unit":"", "period":"", "confidence":0.0}]}.'
        f"\nCompany ticker: {company_ticker or 'UNKNOWN'}"
        f"\nDocument text:\n{text[:12000]}"
    )


NORMALIZATION_PROMPT = """
You are a financial information extraction system.

Task:
Given one candidate extracted from a Russian corporate report, determine whether it is a valid KPI.
If yes, map it to one canonical KPI from the allowed list and normalize the numeric value.

Allowed canonical KPI list:
{canonical_kpis}

Rules:
1. Return valid JSON only.
2. If the candidate is not a KPI, set "is_kpi": false and "canonical_kpi": null.
3. Use candidate.normalized_value_text when it is reasonable.
4. If period is available in candidate.extracted_period, use it unless clearly wrong.
5. If unit is rubles, use "RUB". If percent, use "PERCENT". If unknown, null.
6. Ignore boilerplate, page numbers, regulatory metadata, and non-financial counters.
7. Prefer the simplest correct mapping.

Candidate:
{candidate_json}

Return one valid JSON object only.
Allowed values for "canonical_kpi":
{canonical_kpis}
Allowed values for "unit": "RUB", "USD", "EUR", "PERCENT", or null.

Use exactly this JSON shape:
{{
  "canonical_kpi": null,
  "value": null,
  "unit": null,
  "period": null,
  "is_kpi": false,
  "confidence": 0.0,
  "reason": ""
}}
""".strip()


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

Keep the response concise.
Do not repeat KPI values more than once.
Do not include markdown.
Return one valid JSON object only with exactly these keys:
{{
  "executive_summary": "",
  "positive_factors": ["", ""],
  "risk_factors": ["", ""],
  "financial_health_assessment": "",
  "profitability_assessment": "",
  "leverage_assessment": "",
  "investment_view": {{
    "signal": "hold",
    "confidence": 0.0,
    "expected_short_term_reaction": "",
    "rationale": ""
  }},
  "key_kpi_interpretation": [
    {{
      "kpi": "",
      "interpretation": ""
    }}
  ]
}}

Limit "key_kpi_interpretation" to at most 5 items.
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

Keep the response concise.
Return one valid JSON object only with exactly these keys:
{{
  "signal": "hold",
  "confidence": 0.0,
  "expected_move": "",
  "rationale": ""
}}
""".strip()


def _runtime_config(config) -> OllamaRuntimeConfig:
    return OllamaRuntimeConfig(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        timeout_seconds=config.ollama_timeout_seconds,
        ready_timeout_seconds=config.ollama_ready_timeout_seconds,
    )


def _run_ollama_json(config, prompt: str, max_new_tokens: int) -> tuple[dict[str, Any] | None, str]:
    result = run_chat_json(_runtime_config(config), prompt, max_new_tokens=max_new_tokens)
    raw_response = result["raw_response"]
    return extract_json(raw_response), raw_response


def _payload_keys(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return sorted(str(key) for key in payload.keys())


def _is_valid_analytical_report(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {
        "executive_summary",
        "positive_factors",
        "risk_factors",
        "financial_health_assessment",
        "profitability_assessment",
        "leverage_assessment",
        "investment_view",
        "key_kpi_interpretation",
    }
    if not required.issubset(payload.keys()):
        return False
    investment_view = payload.get("investment_view")
    return isinstance(investment_view, dict) and "signal" in investment_view and "rationale" in investment_view


def _is_valid_market_decision(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"signal", "confidence", "expected_move", "rationale"}
    return required.issubset(payload.keys())


def _repair_invalid_json(
    config,
    raw_response: str,
    target: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    repair_prompt = f"""
You are repairing an invalid JSON response produced by another model run.

Task: convert the draft below into one valid JSON object for `{target}`.
Do not add markdown.
Do not explain anything.
If a field is incomplete, finish it briefly and consistently with the visible text.

Draft response:
{raw_response}
    """.strip()
    return _run_ollama_json(config, repair_prompt, max_new_tokens=max_new_tokens)


def _format_number(value: Any, currency: str | None = None) -> str:
    if not isinstance(value, (int, float)):
        return "н/д"
    if currency:
        return f"{value:g} {currency}"
    return f"{value:g}"


def _build_fallback_analytical_report(report_input: dict[str, Any]) -> dict[str, Any]:
    normalized = report_input.get("normalized_kpis", {})
    derived = report_input.get("derived_metrics", {})
    currency = report_input.get("currency") or "RUB"
    revenue = normalized.get("revenue")
    net_income = normalized.get("net_income")
    cash = normalized.get("cash")
    debt = normalized.get("debt")
    operating_margin = derived.get("operating_margin")
    net_margin = derived.get("net_margin")

    positives: list[str] = []
    risks: list[str] = []
    interpretations: list[dict[str, str]] = []

    if revenue is not None:
        positives.append(f"Выручка составила {_format_number(revenue, currency)}.")
        interpretations.append({"kpi": "revenue", "interpretation": f"Выручка составила {_format_number(revenue, currency)}."})
    if net_income is not None:
        positives.append(f"Чистая прибыль составила {_format_number(net_income, currency)}.")
        interpretations.append({"kpi": "net_income", "interpretation": f"Чистая прибыль составила {_format_number(net_income, currency)}."})
    if cash is not None:
        positives.append(f"Денежные средства составили {_format_number(cash, currency)}.")
        interpretations.append({"kpi": "cash", "interpretation": f"Объем денежных средств составил {_format_number(cash, currency)}."})
    if operating_margin is not None:
        interpretations.append(
            {
                "kpi": "operating_margin",
                "interpretation": f"Операционная маржа составила {operating_margin:.2%}.",
            }
        )
    if net_margin is not None:
        interpretations.append(
            {
                "kpi": "net_margin",
                "interpretation": f"Чистая маржа составила {net_margin:.2%}.",
            }
        )
    if debt is not None:
        risks.append(f"Долг составил {_format_number(debt, currency)}.")
    if not risks:
        risks.append("Структура долга и обязательств раскрыта неполно, поэтому часть рисков остается неопределенной.")
    if len(positives) < 2:
        positives.append("Данных достаточно для базовой оценки операционных результатов, но не для полного инвестиционного кейса.")

    signal = "hold"
    confidence = 0.55
    if revenue is not None and net_income is not None and cash is not None:
        signal = "buy"
        confidence = 0.7

    return {
        "executive_summary": (
            f"{report_input.get('company') or 'Компания'} показала набор KPI, достаточный для базовой оценки результатов. "
            "Отчет построен fallback-логикой, потому что модель не вернула валидный структурированный JSON."
        ),
        "positive_factors": positives[:3],
        "risk_factors": risks[:3],
        "financial_health_assessment": "Умеренно позитивная, если ориентироваться на доступные KPI и запас ликвидности.",
        "profitability_assessment": "Умеренно позитивная на основе доступных показателей прибыли и маржинальности.",
        "leverage_assessment": "Нейтральная, так как детализация по обязательствам ограничена.",
        "investment_view": {
            "signal": signal,
            "confidence": confidence,
            "expected_short_term_reaction": "Нейтрально-позитивная реакция при отсутствии негативных скрытых факторов.",
            "rationale": "Итог сформирован по fallback-логике из доступных KPI, потому что модель не вернула корректный JSON-отчет.",
        },
        "key_kpi_interpretation": interpretations[:5],
    }


def _build_fallback_market_decision(report_input: dict[str, Any]) -> dict[str, Any]:
    normalized = report_input.get("normalized_kpis", {})
    signal = "hold"
    confidence = 0.55
    if normalized.get("revenue") is not None and normalized.get("net_income") is not None:
        signal = "buy"
        confidence = 0.68
    return {
        "signal": signal,
        "confidence": confidence,
        "expected_move": "Сдержанно позитивная динамика при подтверждении качества отчетности.",
        "rationale": "Решение сформировано по fallback-логике из доступных KPI, потому что модель не вернула корректный JSON.",
    }


def _llm_normalize_candidate(config, candidate: Candidate) -> tuple[NormalizedKPI | None, str]:
    prompt = NORMALIZATION_PROMPT.format(
        canonical_kpis=json.dumps(CANONICAL_KPIS, ensure_ascii=False),
        candidate_json=json.dumps(asdict(candidate), ensure_ascii=False, indent=2),
    )
    data, raw_response = _run_ollama_json(config, prompt, max_new_tokens=250)
    if not data:
        return None, raw_response
    return (
        NormalizedKPI(
            canonical_kpi=data.get("canonical_kpi"),
            value=data.get("value"),
            unit=data.get("unit"),
            period=data.get("period") or candidate.extracted_period,
            is_kpi=bool(data.get("is_kpi", False)),
            confidence=float(data.get("confidence", 0.0)),
            reason=data.get("reason", ""),
            source_type=candidate.source_type,
            page_num=candidate.page_num,
            section_hint=candidate.section_hint,
            section_type=candidate.section_type,
            section_score=candidate.section_score,
            label_text=candidate.label_text,
            value_text=candidate.value_text,
            normalized_value_text=candidate.normalized_value_text,
            extracted_period=candidate.extracted_period,
            raw_text=candidate.raw_text,
            normalization_source="llm",
        ),
        raw_response,
    )


class KpiExtractionService(analysis_pb2_grpc.KpiExtractionServiceServicer):
    def __init__(self) -> None:
        self.config = load_kpi_extraction_config()

    async def ExtractKpis(
        self,
        request: analysis_pb2.ExtractKpisRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.ExtractKpisResponse:
        candidates = [
            Candidate(
                source_type=item.source_type,
                page_num=item.page_num,
                section_hint=item.section_hint or None,
                section_type=item.section_type or None,
                section_score=item.section_score,
                label_text=item.label_text,
                value_text=item.value_text,
                raw_text=item.raw_text,
                normalized_value_text=item.normalized_value_text or None,
                extracted_period=item.extracted_period or None,
                pre_mapped_kpi=item.pre_mapped_kpi or None,
                pre_map_confidence=item.pre_map_confidence,
            )
            for item in request.candidates
        ]
        if not candidates and not request.extracted_text.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Candidates or extracted text must be provided.")

        normalized_items: list[NormalizedKPI | None] = []
        failed_candidates: list[tuple[Candidate, str]] = []
        raw_llm_responses: list[str] = []
        llm_calls = 0
        dictionary_hits = 0

        if candidates:
            for candidate in candidates:
                try:
                    dict_result = normalize_candidate_with_dictionary(candidate)
                    if dict_result is not None:
                        dictionary_hits += 1
                        normalized_items.append(dict_result)
                        continue
                    llm_result, raw_response = _llm_normalize_candidate(self.config, candidate)
                    raw_llm_responses.append(raw_response)
                    if llm_result is not None:
                        llm_calls += 1
                    normalized_items.append(llm_result)
                except Exception as exc:  # noqa: BLE001
                    failed_candidates.append((candidate, str(exc)))
        else:
            payload, raw_response = _run_ollama_json(
                self.config,
                _build_prompt(request.company_ticker, request.extracted_text),
                max_new_tokens=350,
            )
            raw_llm_responses.append(raw_response)
            for item in (payload or {}).get("kpis", []):
                normalized_items.append(
                    NormalizedKPI(
                        canonical_kpi=str(item.get("name", "")),
                        value=float(item.get("value")) if item.get("value") is not None else None,
                        unit=str(item.get("unit", "")) or None,
                        period=str(item.get("period", "")) or None,
                        is_kpi=True,
                        confidence=float(item.get("confidence", 0.0)),
                        reason="fallback_text_only",
                        source_type="text_pair",
                        page_num=1,
                        section_hint=None,
                        section_type=None,
                        section_score=0.0,
                        label_text=str(item.get("name", "")),
                        value_text=str(item.get("value", "")),
                        normalized_value_text=str(item.get("value", "")),
                        extracted_period=str(item.get("period", "")) or None,
                        raw_text=request.extracted_text,
                        normalization_source="llm",
                    )
                )

        normalized_kpis_list_dicts = consolidate_kpis([item for item in normalized_items if item is not None])
        normalized_kpis_dict = build_kpi_dict(normalized_kpis_list_dicts)
        derived_metrics = compute_derived_metrics(normalized_kpis_dict)
        normalized_kpis_list = [
            analysis_pb2.NormalizedKpi(
                canonical_kpi=item.get("canonical_kpi", "") or "",
                value=float(item.get("value", 0.0)),
                unit=item.get("unit", "") or "",
                period=item.get("period", "") or "",
                confidence=float(item.get("confidence", 0.0)),
            )
            for item in normalized_kpis_list_dicts
            if item.get("canonical_kpi") and item.get("value") is not None
        ]
        kpis = [
            analysis_pb2.Kpi(
                name=item.canonical_kpi,
                value=str(item.value),
                unit=item.unit or "",
                period=item.period or "",
                confidence=item.confidence,
            )
            for item in normalized_kpis_list
        ]
        logging.info("Extracted %s KPI(s) for document %s", len(kpis), request.document_id)
        return analysis_pb2.ExtractKpisResponse(
            document_id=request.document_id,
            kpis=kpis,
            model=self.config.ollama_model,
            raw_response="\n\n".join(raw_llm_responses),
            normalized_items=[
                analysis_pb2.NormalizedKpiDetail(
                    canonical_kpi=item.canonical_kpi or "",
                    value=float(item.value) if item.value is not None else 0.0,
                    unit=item.unit or "",
                    period=item.period or "",
                    is_kpi=item.is_kpi,
                    confidence=item.confidence,
                    reason=item.reason,
                    source_type=item.source_type,
                    page_num=item.page_num,
                    section_hint=item.section_hint or "",
                    section_type=item.section_type or "",
                    section_score=item.section_score,
                    label_text=item.label_text,
                    value_text=item.value_text,
                    normalized_value_text=item.normalized_value_text or "",
                    extracted_period=item.extracted_period or "",
                    raw_text=item.raw_text,
                    normalization_source=item.normalization_source,
                )
                for item in normalized_items
                if item is not None
            ],
            normalized_kpis_list=normalized_kpis_list,
            normalized_kpis_dict=normalized_kpis_dict,
            derived_metrics=derived_metrics,
            dictionary_hits=dictionary_hits,
            llm_calls_for_normalization=llm_calls,
            failed_candidates=[
                analysis_pb2.FailedCandidate(
                    candidate=analysis_pb2.CandidateData(
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
                    ),
                    error=error,
                )
                for candidate, error in failed_candidates
            ],
        )

    async def GenerateAnalyticalOutputs(
        self,
        request: analysis_pb2.GenerateAnalyticalOutputsRequest,
        context: grpc.aio.ServicerContext,
    ) -> analysis_pb2.GenerateAnalyticalOutputsResponse:
        report_input = {
            "company": request.company,
            "ticker": request.ticker,
            "event_date": request.event_date,
            "filing_type": request.filing_type,
            "fiscal_period": request.fiscal_period,
            "currency": request.currency,
            "normalized_kpis": dict(request.normalized_kpis_dict),
            "derived_metrics": dict(request.derived_metrics),
        }

        analytical_report, analytical_raw = _run_ollama_json(
            self.config,
            _build_analytical_report_prompt(report_input),
            max_new_tokens=1400,
        )
        logging.info("Analytical report parsed keys: %s", _payload_keys(analytical_report))
        if not _is_valid_analytical_report(analytical_report):
            logging.warning("Analytical report JSON is invalid, attempting repair")
            repaired_report, repaired_raw = _repair_invalid_json(
                self.config,
                analytical_raw,
                "analytical_report",
                max_new_tokens=900,
            )
            logging.info("Analytical report repaired keys: %s", _payload_keys(repaired_report))
            if _is_valid_analytical_report(repaired_report):
                logging.info("Analytical report repair succeeded")
                analytical_report, analytical_raw = repaired_report, repaired_raw
            else:
                logging.warning("Analytical report repair failed")
                analytical_report = _build_fallback_analytical_report(report_input)
                logging.info("Analytical report fallback keys: %s", _payload_keys(analytical_report))

        market_decision, decision_raw = _run_ollama_json(
            self.config,
            _build_market_decision_prompt(report_input),
            max_new_tokens=350,
        )
        logging.info("Market decision parsed keys: %s", _payload_keys(market_decision))
        if not _is_valid_market_decision(market_decision):
            logging.warning("Market decision JSON is invalid, attempting repair")
            repaired_decision, repaired_decision_raw = _repair_invalid_json(
                self.config,
                decision_raw,
                "market_decision",
                max_new_tokens=350,
            )
            logging.info("Market decision repaired keys: %s", _payload_keys(repaired_decision))
            if _is_valid_market_decision(repaired_decision):
                logging.info("Market decision repair succeeded")
                market_decision, decision_raw = repaired_decision, repaired_decision_raw
            else:
                logging.warning("Market decision repair failed")
                market_decision = _build_fallback_market_decision(report_input)
                logging.info("Market decision fallback keys: %s", _payload_keys(market_decision))

        return analysis_pb2.GenerateAnalyticalOutputsResponse(
            analytical_report=_to_struct(analytical_report),
            market_decision=_to_struct(market_decision),
            raw_analytical_response=analytical_raw,
            raw_decision_response=decision_raw,
        )


async def serve() -> None:
    config = load_kpi_extraction_config()
    configure_logging(config.service_name)
    logging.info("Checking Ollama readiness before serving requests")
    await asyncio.to_thread(ensure_model_ready, _runtime_config(config))
    logging.info("Ollama readiness check completed")
    server, health_servicer = create_server()
    grpc_service_name = "analysestocks.v1.KpiExtractionService"
    analysis_pb2_grpc.add_KpiExtractionServiceServicer_to_server(KpiExtractionService(), server)
    enable_reflection(server, [grpc_service_name])
    await mark_serving(health_servicer, [grpc_service_name])
    await run_server(server, config.bind_address, config.service_name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
