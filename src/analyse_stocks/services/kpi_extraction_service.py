from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
import re
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
    find_kpi_consistency_issues,
    normalize_llm_value_by_candidate,
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

Candidate fields:
{candidate_text}
""".strip()


def _build_analytical_report_prompt(report_input: dict[str, Any]) -> str:
    kpi_summary = _build_kpi_summary(report_input)
    constraints = _build_analysis_constraints(report_input)
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

KPI summary:
{kpi_summary}

Constraints:
{constraints}

Keep the response concise.
Do not repeat KPI values more than once.
Do not include markdown.
Do not output KPI dictionary keys as top-level JSON keys.
Do not infer leverage conclusions unless debt is explicitly present in the KPI summary.
Do not call high cash balance a risk by itself.
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
    kpi_summary = _build_kpi_summary(report_input)
    constraints = _build_analysis_constraints(report_input)
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

KPI summary:
{kpi_summary}

Constraints:
{constraints}

Keep the response concise.
Signal must be one of: buy, hold, sell.
Confidence must be between 0.0 and 1.0.
expected_move and rationale must be non-empty.
If the data is incomplete or mixed, prefer hold with medium confidence rather than empty fields.
Return one valid JSON object only with exactly these keys:
{{
  "signal": "hold",
  "confidence": 0.0,
  "expected_move": "",
  "rationale": ""
}}
""".strip()


def _build_analytical_report_retry_prompt(report_input: dict[str, Any]) -> str:
    return (
        _build_analytical_report_prompt(report_input)
        + "\n\nImportant: do not repeat the KPI dictionary in the response. "
        + "Do not use keys such as revenue, cash, net_income, operating_income, total_assets at the top level. "
        + "Use only the required report keys."
    )


def _build_market_decision_retry_prompt(report_input: dict[str, Any]) -> str:
    return (
        _build_market_decision_prompt(report_input)
        + "\n\nImportant: do not repeat the KPI dictionary in the response. "
        + "Top-level keys must be only: signal, confidence, expected_move, rationale."
    )


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


def _candidate_to_prompt_text(candidate: Candidate) -> str:
    return "\n".join(
        [
            f"source_type: {candidate.source_type}",
            f"page_num: {candidate.page_num}",
            f"section_hint: {candidate.section_hint or ''}",
            f"section_type: {candidate.section_type or ''}",
            f"section_score: {candidate.section_score}",
            f"label_text: {candidate.label_text}",
            f"value_text: {candidate.value_text}",
            f"normalized_value_text: {candidate.normalized_value_text or ''}",
            f"extracted_period: {candidate.extracted_period or ''}",
            f"pre_mapped_kpi: {candidate.pre_mapped_kpi or ''}",
            f"pre_map_confidence: {candidate.pre_map_confidence}",
            f"raw_text: {candidate.raw_text}",
        ]
    )


def _candidate_has_meaningful_label(candidate: Candidate) -> bool:
    label = candidate.label_text or ""
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", label)
    digits = re.findall(r"\d", label)
    if len(letters) < 3:
        return False
    if len(letters) <= len(digits):
        return False
    return True


def _should_skip_llm_candidate(candidate: Candidate) -> bool:
    raw_blob = " ".join(
        [
            candidate.label_text or "",
            candidate.value_text or "",
            candidate.raw_text or "",
        ]
    )
    if not _candidate_has_meaningful_label(candidate):
        return True
    if len(raw_blob) > 700:
        return True
    if re.fullmatch(r"[\d\s,.\-%()/]+", candidate.label_text or ""):
        return True
    return False


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
    if not isinstance(investment_view, dict):
        return False
    if "signal" not in investment_view or "rationale" not in investment_view:
        return False
    return isinstance(payload.get("positive_factors"), list) and isinstance(payload.get("risk_factors"), list)


def _is_valid_market_decision(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"signal", "confidence", "expected_move", "rationale"}
    if not required.issubset(payload.keys()):
        return False
    signal = str(payload.get("signal", "")).strip().lower()
    if signal not in {"buy", "hold", "sell"}:
        return False
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    expected_move = str(payload.get("expected_move", "")).strip()
    rationale = str(payload.get("rationale", "")).strip()
    return 0.0 <= confidence <= 1.0 and bool(expected_move) and bool(rationale)


def _clean_text_value(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bмлр:\b", "млрд", text)
    return text


def _trim_list(items: Any, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    cleaned = [_clean_text_value(item) for item in items if _clean_text_value(item)]
    return cleaned[:limit]


def _localize_text_value(value: Any) -> str:
    text = _clean_text_value(value)
    text = text.replace("R.РУБ", "млрд руб.")
    text = text.replace("RUB", "руб.")
    text = text.replace("млрд млрд руб.", "млрд руб.")
    text = text.replace("млрд млрд руб", "млрд руб")
    text = text.replace(" млрд руб..", " млрд руб.")
    text = text.replace(" млн руб..", " млн руб.")
    text = text.replace(" тыс. руб..", " тыс. руб.")
    return text


def _localize_kpi_name(name: str) -> str:
    mapping = {
        "revenue": "Выручка",
        "operating margin": "Операционная маржа",
        "operating_margin": "Операционная маржа",
        "net margin": "Чистая маржа",
        "net_margin": "Чистая маржа",
        "cash": "Кассовый резерв",
        "total assets": "Общие активы",
        "total_assets": "Общие активы",
        "operating income": "Операционная прибыль",
        "operating_income": "Операционная прибыль",
        "net income": "Чистая прибыль",
        "net_income": "Чистая прибыль",
    }
    return mapping.get(name.strip().lower(), name)


def _clean_interpretations(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kpi = _localize_kpi_name(_clean_text_value(item.get("kpi")))
        interpretation = _localize_text_value(item.get("interpretation"))
        if kpi and interpretation:
            cleaned.append({"kpi": kpi, "interpretation": interpretation})
    return cleaned[:5]


def _sanitize_analytical_report(payload: dict[str, Any], report_input: dict[str, Any]) -> dict[str, Any]:
    normalized = report_input.get("normalized_kpis", {})
    has_debt = isinstance(normalized.get("debt"), (int, float))
    has_cash = isinstance(normalized.get("cash"), (int, float))
    currency = report_input.get("currency") or "RUB"

    report = dict(payload)
    report["executive_summary"] = _localize_text_value(report.get("executive_summary"))
    report["financial_health_assessment"] = _localize_text_value(report.get("financial_health_assessment"))
    report["profitability_assessment"] = _localize_text_value(report.get("profitability_assessment"))
    report["leverage_assessment"] = _localize_text_value(report.get("leverage_assessment"))
    report["positive_factors"] = [_localize_text_value(item) for item in _trim_list(report.get("positive_factors"), 3)]
    report["risk_factors"] = [_localize_text_value(item) for item in _trim_list(report.get("risk_factors"), 3)]
    report["key_kpi_interpretation"] = _clean_interpretations(report.get("key_kpi_interpretation"))

    investment_view = report.get("investment_view")
    if not isinstance(investment_view, dict):
        investment_view = {}
    report["investment_view"] = {
        "signal": str(investment_view.get("signal", "hold")).strip().lower() or "hold",
        "confidence": float(investment_view.get("confidence", 0.0) or 0.0),
        "expected_short_term_reaction": _localize_text_value(investment_view.get("expected_short_term_reaction")),
        "rationale": _localize_text_value(investment_view.get("rationale")),
    }

    fallback_mode = "fallback" in report["executive_summary"].lower() or "fallback" in report["investment_view"]["rationale"].lower()
    if fallback_mode:
        positive_factors: list[str] = []
        interpretations: list[dict[str, str]] = []
        revenue = normalized.get("revenue")
        net_income = normalized.get("net_income")
        cash = normalized.get("cash")
        operating_margin = report_input.get("derived_metrics", {}).get("operating_margin")
        if isinstance(revenue, (int, float)):
            text = f"Выручка составила {_format_currency_human(revenue, currency)}."
            positive_factors.append(text)
            interpretations.append({"kpi": "revenue", "interpretation": text})
        if isinstance(net_income, (int, float)):
            text = f"Чистая прибыль составила {_format_currency_human(net_income, currency)}."
            positive_factors.append(text)
            interpretations.append({"kpi": "net_income", "interpretation": text})
        if isinstance(cash, (int, float)):
            positive_factors.append(f"Денежные средства составили {_format_currency_human(cash, currency)}.")
            interpretations.append(
                {
                    "kpi": "cash",
                    "interpretation": f"Объем денежных средств составил {_format_currency_human(cash, currency)}.",
                }
            )
        if isinstance(operating_margin, (int, float)):
            interpretations.append(
                {
                    "kpi": "operating_margin",
                    "interpretation": f"Операционная маржа составила {operating_margin:.2%}.",
                }
            )
        if positive_factors:
            report["positive_factors"] = positive_factors[:3]
        if interpretations:
            report["key_kpi_interpretation"] = interpretations[:5]
        report["executive_summary"] = (
            f"{report_input.get('company') or 'Компания'} показала набор KPI, достаточный для базовой оценки результатов. "
            "Отчет построен fallback-логикой, потому что модель не вернула валидный структурированный JSON."
        )
        report["financial_health_assessment"] = "Умеренно позитивная, если ориентироваться на доступные KPI и запас ликвидности."
        report["profitability_assessment"] = "Умеренно позитивная на основе доступных показателей прибыли и маржинальности."

    if not has_debt:
        leverage_fallback = "Оценка долговой нагрузки ограничена, потому что валидный показатель долга не был извлечен."
        report["leverage_assessment"] = leverage_fallback
        report["risk_factors"] = [
            item
            for item in report["risk_factors"]
            if "заем" not in item.lower() and "долг" not in item.lower() and "левер" not in item.lower()
        ]
        if not report["risk_factors"]:
            report["risk_factors"] = ["Структура долга и обязательств раскрыта неполно, поэтому оценка рисков ограничена."]

    if not has_cash:
        report["risk_factors"] = [
            item
            for item in report["risk_factors"]
            if "ликвид" not in item.lower()
        ] or report["risk_factors"]

    return report


def _sanitize_market_decision(payload: dict[str, Any], report_input: dict[str, Any]) -> dict[str, Any]:
    decision = dict(payload)
    signal = str(decision.get("signal", "hold")).strip().lower()
    if signal not in {"buy", "hold", "sell"}:
        signal = "hold"
    try:
        confidence = float(decision.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    expected_move = _localize_text_value(decision.get("expected_move"))
    rationale = _localize_text_value(decision.get("rationale"))

    if not expected_move or not rationale or confidence <= 0.0:
        return _build_fallback_market_decision(report_input)

    return {
        "signal": signal,
        "confidence": confidence,
        "expected_move": expected_move,
        "rationale": rationale,
    }


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


def _run_validated_json_with_retries(
    config,
    prompt_attempts: list[tuple[str, str, int]],
    validator,
) -> tuple[dict[str, Any] | None, str]:
    last_payload: dict[str, Any] | None = None
    last_raw = ""
    for attempt_name, prompt, max_new_tokens in prompt_attempts:
        payload, raw = _run_ollama_json(config, prompt, max_new_tokens=max_new_tokens)
        logging.info("Validated JSON attempt=%s keys=%s", attempt_name, _payload_keys(payload))
        last_payload, last_raw = payload, raw
        if validator(payload):
            return payload, raw
    return last_payload, last_raw


def _run_validated_json_with_repair(
    config,
    prompt_attempts: list[tuple[str, str, int]],
    validator,
    repair_target: str,
    repair_tokens: int,
    fallback_builder,
    fallback_input: dict[str, Any],
    log_prefix: str,
) -> tuple[dict[str, Any], str]:
    payload, raw = _run_validated_json_with_retries(config, prompt_attempts, validator)
    logging.info("%s parsed keys: %s", log_prefix, _payload_keys(payload))
    if validator(payload):
        return payload, raw

    logging.warning("%s JSON is invalid after prompt retries, attempting repair", log_prefix)
    repaired_payload, repaired_raw = _repair_invalid_json(
        config,
        raw,
        repair_target,
        max_new_tokens=repair_tokens,
    )
    logging.info("%s repaired keys: %s", log_prefix, _payload_keys(repaired_payload))
    if validator(repaired_payload):
        logging.info("%s repair succeeded", log_prefix)
        return repaired_payload, repaired_raw

    logging.warning("%s repair failed, using deterministic fallback", log_prefix)
    fallback_payload = fallback_builder(fallback_input)
    logging.info("%s fallback keys: %s", log_prefix, _payload_keys(fallback_payload))
    return fallback_payload, raw


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
        candidate_text=_candidate_to_prompt_text(candidate),
    )
    strict_retry_prompt = (
        prompt
        + "\n\nImportant: do not copy candidate fields into the response. "
        + "Return only the target JSON object with the seven required keys."
    )
    data, raw_response = _run_validated_json_with_retries(
        config,
        [
            ("normalize_primary", prompt, 250),
            ("normalize_retry", strict_retry_prompt, 220),
        ],
        lambda payload: isinstance(payload, dict)
        and {"canonical_kpi", "value", "unit", "period", "is_kpi", "confidence", "reason"}.issubset(payload.keys()),
    )
    if not data:
        return None, raw_response
    normalized_value = data.get("value")
    if normalized_value is not None:
        try:
            normalized_value = float(normalized_value)
        except (TypeError, ValueError):
            normalized_value = None
    normalized_value = normalize_llm_value_by_candidate(
        candidate,
        data.get("canonical_kpi"),
        normalized_value,
        data.get("unit"),
    )
    return (
        NormalizedKPI(
            canonical_kpi=data.get("canonical_kpi"),
            value=normalized_value,
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
                    if _should_skip_llm_candidate(candidate):
                        logging.info(
                            "Skipping low-signal candidate for LLM label=%s page=%s",
                            candidate.label_text[:120],
                            candidate.page_num,
                        )
                        normalized_items.append(None)
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
        consistency_issues = find_kpi_consistency_issues(normalized_kpis_dict)
        if consistency_issues:
            logging.warning(
                "Detected KPI consistency issues for document %s: %s",
                request.document_id,
                ", ".join(consistency_issues),
            )
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

        analytical_report, analytical_raw = _run_validated_json_with_repair(
            self.config,
            [
                ("report_primary", _build_analytical_report_prompt(report_input), 1400),
                ("report_retry", _build_analytical_report_retry_prompt(report_input), 900),
            ],
            _is_valid_analytical_report,
            "analytical_report",
            900,
            _build_fallback_analytical_report,
            report_input,
            "Analytical report",
        )
        analytical_report = _sanitize_analytical_report(analytical_report, report_input)
        market_decision, decision_raw = _run_validated_json_with_repair(
            self.config,
            [
                ("decision_primary", _build_market_decision_prompt(report_input), 350),
                ("decision_retry", _build_market_decision_retry_prompt(report_input), 280),
            ],
            _is_valid_market_decision,
            "market_decision",
            350,
            _build_fallback_market_decision,
            report_input,
            "Market decision",
        )
        market_decision = _sanitize_market_decision(market_decision, report_input)

        return analysis_pb2.GenerateAnalyticalOutputsResponse(
            analytical_report=_to_struct(analytical_report),
            market_decision=_to_struct(market_decision),
            raw_analytical_response=analytical_raw,
            raw_decision_response=decision_raw,
        )

def _format_decimal_human(value: float) -> str:
    text = f"{value:.1f}" if abs(value) >= 100 else f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def _format_currency_human(value: Any, currency: str | None = None) -> str:
    if not isinstance(value, (int, float)):
        return "н/д"
    currency = currency or "RUB"
    abs_value = abs(float(value))
    sign = "-" if float(value) < 0 else ""
    if abs_value >= 1_000_000_000:
        return f"{sign}{_format_decimal_human(abs_value / 1_000_000_000)} млрд {currency}"
    if abs_value >= 1_000_000:
        return f"{sign}{_format_decimal_human(abs_value / 1_000_000)} млн {currency}"
    if abs_value >= 1_000:
        return f"{sign}{_format_decimal_human(abs_value / 1_000)} тыс. {currency}"
    return f"{sign}{_format_decimal_human(abs_value)} {currency}"


def _format_metric_human(name: str, value: Any, currency: str | None = None) -> str:
    if not isinstance(value, (int, float)):
        return "н/д"
    if name.endswith("_margin"):
        return f"{value:.2%}"
    return _format_currency_human(value, currency)


def _build_kpi_summary(report_input: dict[str, Any]) -> str:
    normalized = report_input.get("normalized_kpis", {})
    derived = report_input.get("derived_metrics", {})
    currency = report_input.get("currency") or "RUB"
    ordered = [
        "revenue",
        "operating_income",
        "ebitda",
        "net_income",
        "cash",
        "debt",
        "equity",
        "total_assets",
        "operating_cash_flow",
        "capex",
    ]
    labels = {
        "revenue": "Revenue",
        "operating_income": "Operating income",
        "ebitda": "EBITDA",
        "net_income": "Net income",
        "cash": "Cash",
        "debt": "Debt",
        "equity": "Equity",
        "total_assets": "Total assets",
        "operating_cash_flow": "Operating cash flow",
        "capex": "Capex",
        "operating_margin": "Operating margin",
        "net_margin": "Net margin",
        "ebitda_margin": "EBITDA margin",
    }
    lines: list[str] = []
    for key in ordered:
        value = normalized.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"- {labels.get(key, key)}: {_format_metric_human(key, value, currency)}")
    for key in ("operating_margin", "net_margin", "ebitda_margin"):
        value = derived.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"- {labels.get(key, key)}: {_format_metric_human(key, value, currency)}")
    return "\n".join(lines) if lines else "- No reliable KPI values were extracted."


def _build_analysis_constraints(report_input: dict[str, Any]) -> str:
    normalized = report_input.get("normalized_kpis", {})
    constraints = [
        "- Use only the KPI summary above.",
        "- Treat revenue, operating income, and net income as flow metrics for the requested reporting period.",
        "- Treat cash and total assets as end-of-period balance sheet metrics.",
        "- Do not treat high cash balance by itself as a risk.",
        "- Do not compare cash directly against total assets as a negative signal.",
        "- Do not make leverage conclusions unless debt is explicitly present.",
        "- Keep all KPI names in the JSON text in Russian when possible.",
    ]
    if not isinstance(normalized.get("debt"), (int, float)):
        constraints.append("- Debt is not reliably extracted in this run, so leverage assessment must stay cautious.")
    return "\n".join(constraints)


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
