from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pdfplumber


CANONICAL_KPIS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "ebitda",
    "net_income",
    "eps",
    "total_assets",
    "equity",
    "cash",
    "debt",
    "operating_cash_flow",
    "capex",
]

SECTION_PATTERNS = {
    "financial_highlights": [
        "\u043a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0444\u0438\u043d\u0430\u043d\u0441\u043e\u0432\u044b\u0435 \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u0438",
        "\u043e\u0441\u043d\u043e\u0432\u043d\u044b\u0435 \u0444\u0438\u043d\u0430\u043d\u0441\u043e\u0432\u044b\u0435 \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u0438",
        "\u0444\u0438\u043d\u0430\u043d\u0441\u043e\u0432\u044b\u0435 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b",
        "financial highlights",
    ],
    "income_statement": [
        "\u043e\u0442\u0447\u0435\u0442 \u043e \u043f\u0440\u0438\u0431\u044b\u043b\u0438",
        "\u043e\u0442\u0447\u0435\u0442 \u043e \u043f\u0440\u0438\u0431\u044b\u043b\u044f\u0445 \u0438 \u0443\u0431\u044b\u0442\u043a\u0430\u0445",
        "\u043a\u043e\u043d\u0441\u043e\u043b\u0438\u0434\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 \u043e\u0442\u0447\u0435\u0442 \u043e \u043f\u0440\u0438\u0431\u044b\u043b\u0438",
        "income statement",
        "statement of profit",
    ],
    "balance_sheet": [
        "\u043e\u0442\u0447\u0435\u0442 \u043e \u0444\u0438\u043d\u0430\u043d\u0441\u043e\u0432\u043e\u043c \u043f\u043e\u043b\u043e\u0436\u0435\u043d\u0438\u0438",
        "\u0431\u0443\u0445\u0433\u0430\u043b\u0442\u0435\u0440\u0441\u043a\u0438\u0439 \u0431\u0430\u043b\u0430\u043d\u0441",
        "statement of financial position",
        "balance sheet",
    ],
    "cash_flow": [
        "\u043e\u0442\u0447\u0435\u0442 \u043e \u0434\u0432\u0438\u0436\u0435\u043d\u0438\u0438 \u0434\u0435\u043d\u0435\u0436\u043d\u044b\u0445 \u0441\u0440\u0435\u0434\u0441\u0442\u0432",
        "cash flow statement",
        "statement of cash flows",
    ],
    "management_commentary": [
        "\u0443\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0447\u0435\u0441\u043a\u0438\u0439 \u043a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u0439",
        "\u043e\u0431\u0437\u043e\u0440 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432",
        "management discussion",
        "md&a",
    ],
}

BAD_PATTERNS = [
    "\u043e\u043c\u0431",
    "omb",
    "\u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430",
    "\u043e\u0433\u043b\u0430\u0432\u043b\u0435\u043d\u0438\u0435",
]

SOURCE_PRIORITY = {"table_row": 2, "text_pair": 1}

KPI_SYNONYMS = {
    "revenue": [
        "\u0432\u044b\u0440\u0443\u0447\u043a\u0430",
        "\u0434\u043e\u0445\u043e\u0434\u044b",
        "\u043e\u0431\u043e\u0440\u043e\u0442",
        "revenue",
        "sales",
        "turnover",
    ],
    "gross_profit": ["\u0432\u0430\u043b\u043e\u0432\u0430\u044f \u043f\u0440\u0438\u0431\u044b\u043b\u044c", "gross profit"],
    "operating_income": [
        "\u043e\u043f\u0435\u0440\u0430\u0446\u0438\u043e\u043d\u043d\u0430\u044f \u043f\u0440\u0438\u0431\u044b\u043b\u044c",
        "\u043f\u0440\u0438\u0431\u044b\u043b\u044c \u043e\u0442 \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u043e\u043d\u043d\u043e\u0439 \u0434\u0435\u044f\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u0438",
        "operating income",
        "operating profit",
    ],
    "ebitda": ["ebitda", "\u0441\u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u0430\u044f ebitda", "adjusted ebitda"],
    "net_income": [
        "\u0447\u0438\u0441\u0442\u0430\u044f \u043f\u0440\u0438\u0431\u044b\u043b\u044c",
        "\u043f\u0440\u0438\u0431\u044b\u043b\u044c \u0437\u0430 \u043f\u0435\u0440\u0438\u043e\u0434",
        "net income",
        "net profit",
        "profit attributable",
    ],
    "eps": ["eps", "\u043f\u0440\u0438\u0431\u044b\u043b\u044c \u043d\u0430 \u0430\u043a\u0446\u0438\u044e", "earnings per share"],
    "total_assets": ["\u0430\u043a\u0442\u0438\u0432\u044b", "\u0438\u0442\u043e\u0433\u043e \u0430\u043a\u0442\u0438\u0432\u044b", "total assets"],
    "equity": ["\u043a\u0430\u043f\u0438\u0442\u0430\u043b", "\u0441\u043e\u0431\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u043a\u0430\u043f\u0438\u0442\u0430\u043b", "equity"],
    "cash": [
        "\u0434\u0435\u043d\u0435\u0436\u043d\u044b\u0435 \u0441\u0440\u0435\u0434\u0441\u0442\u0432\u0430",
        "\u0434\u0435\u043d\u0435\u0436\u043d\u044b\u0435 \u0441\u0440\u0435\u0434\u0441\u0442\u0432\u0430 \u0438 \u0438\u0445 \u044d\u043a\u0432\u0438\u0432\u0430\u043b\u0435\u043d\u0442\u044b",
        "cash",
        "cash and cash equivalents",
    ],
    "debt": [
        "\u0434\u043e\u043b\u0433",
        "\u0447\u0438\u0441\u0442\u044b\u0439 \u0434\u043e\u043b\u0433",
        "\u043e\u0431\u044f\u0437\u0430\u0442\u0435\u043b\u044c\u0441\u0442\u0432\u0430 \u043f\u043e \u043a\u0440\u0435\u0434\u0438\u0442\u0430\u043c \u0438 \u0437\u0430\u0439\u043c\u0430\u043c",
        "debt",
        "borrowings",
    ],
    "operating_cash_flow": [
        "\u0434\u0435\u043d\u0435\u0436\u043d\u044b\u0439 \u043f\u043e\u0442\u043e\u043a \u043e\u0442 \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u043e\u043d\u043d\u043e\u0439 \u0434\u0435\u044f\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u0438",
        "\u043e\u043f\u0435\u0440\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u0439 \u0434\u0435\u043d\u0435\u0436\u043d\u044b\u0439 \u043f\u043e\u0442\u043e\u043a",
        "operating cash flow",
    ],
    "capex": [
        "\u043a\u0430\u043f\u0438\u0442\u0430\u043b\u044c\u043d\u044b\u0435 \u0437\u0430\u0442\u0440\u0430\u0442\u044b",
        "\u043a\u0430\u043f\u0438\u0442\u0430\u043b\u044c\u043d\u044b\u0435 \u0432\u043b\u043e\u0436\u0435\u043d\u0438\u044f",
        "capex",
    ],
}

MULTIPLIERS = {
    "\u0442\u044b\u0441": 1_000,
    "\u0442\u044b\u0441.": 1_000,
    "thousand": 1_000,
    "\u043c\u043b\u043d": 1_000_000,
    "million": 1_000_000,
    "\u043c\u043b\u0440\u0434": 1_000_000_000,
    "billion": 1_000_000_000,
    "\u0442\u0440\u043b\u043d": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
}


@dataclass
class Candidate:
    source_type: str
    page_num: int
    section_hint: str | None
    section_type: str | None
    section_score: float
    label_text: str
    value_text: str
    raw_text: str
    normalized_value_text: str | None = None
    extracted_period: str | None = None
    pre_mapped_kpi: str | None = None
    pre_map_confidence: float = 0.0


@dataclass
class NormalizedKPI:
    canonical_kpi: str | None
    value: float | None
    unit: str | None
    period: str | None
    is_kpi: bool
    confidence: float
    reason: str
    source_type: str
    page_num: int
    section_hint: str | None
    section_type: str | None
    section_score: float
    label_text: str
    value_text: str
    normalized_value_text: str | None
    extracted_period: str | None
    raw_text: str
    normalization_source: str


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def fuzzy_contains_score(text: str, phrase: str) -> float:
    text = text.lower()
    phrase = phrase.lower()
    if phrase in text:
        return 1.0
    return difflib.SequenceMatcher(None, text[: min(len(text), 500)], phrase).ratio()


def detect_section_info(text: str) -> dict[str, Any]:
    if not text:
        return {"section_hint": None, "section_type": None, "section_score": 0.0}
    lines = [line.strip().lower() for line in text.split("\n") if line.strip()]
    header_zone = " ".join(lines[:25])
    best_type = None
    best_hint = None
    best_score = 0.0
    for section_type, phrases in SECTION_PATTERNS.items():
        for phrase in phrases:
            score = fuzzy_contains_score(header_zone, phrase)
            if phrase in header_zone:
                score += 0.5
            if score > best_score:
                best_score = score
                best_type = section_type
                best_hint = phrase
    if best_score < 0.55:
        return {"section_hint": None, "section_type": None, "section_score": 0.0}
    return {
        "section_hint": best_hint,
        "section_type": best_type,
        "section_score": round(min(best_score, 1.5), 3),
    }


def extract_period_from_text(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"(?:1|2|3|4)\s*\u043a\u0432\.?\s*20\d{2}",
        r"(?:i|ii|iii|iv)\s*\u043a\u0432\.?\s*20\d{2}",
        r"(?:6|9|12)\s*\u043c\u0435\u0441\.?\s*20\d{2}",
        r"\u0437\u0430\s+\d+\s+\u043c\u0435\u0441\u044f\u0446\u0435\u0432\s+20\d{2}",
        r"fy\s*20\d{2}",
        r"20\d{2}\s*\u0433\u043e\u0434",
        r"20\d{2}",
    ]
    lower_text = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower_text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def detect_unit(text: str) -> str | None:
    if not text:
        return None
    lower_text = text.lower()
    if "%" in lower_text:
        return "PERCENT"
    if "\u0440\u0443\u0431" in lower_text or "\u20bd" in lower_text:
        return "RUB"
    if "usd" in lower_text or "$" in lower_text or "\u0434\u043e\u043b\u043b" in lower_text:
        return "USD"
    if "eur" in lower_text or "\u20ac" in lower_text:
        return "EUR"
    return None


def normalize_scale_in_value_text(value_text: str) -> tuple[float | None, str | None, str | None]:
    if not value_text:
        return None, None, None
    original = value_text
    normalized = value_text.lower().replace(" ", "").replace(",", ".").replace("\u2212", "-").replace("\u2013", "-")
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None, detect_unit(original), None
    value = float(match.group(0))
    multiplier = 1
    for scale, scale_value in MULTIPLIERS.items():
        if scale in normalized:
            multiplier = scale_value
            break
    final_value = value * multiplier
    if abs(final_value - round(final_value)) < 1e-9:
        final_value = int(round(final_value))
    return final_value, detect_unit(original), str(final_value)


def pre_map_label_to_kpi(label_text: str) -> tuple[str | None, float]:
    if not label_text:
        return None, 0.0
    label = label_text.lower().strip()
    best_kpi = None
    best_score = 0.0
    for canonical_kpi, synonyms in KPI_SYNONYMS.items():
        for synonym in synonyms:
            synonym = synonym.lower()
            if synonym == label or synonym in label:
                score = 0.95 if synonym == label else 0.82
            else:
                score = difflib.SequenceMatcher(None, label, synonym).ratio()
            if score > best_score:
                best_score = score
                best_kpi = canonical_kpi
    if best_score < 0.72:
        return None, 0.0
    return best_kpi, round(float(best_score), 3)


def extract_pages(pdf_path: str) -> list[dict[str, Any]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages.append({"page_num": i, "text": text, "tables": tables, **detect_section_info(text)})
    return pages


def pages_from_raw_text(text: str) -> list[dict[str, Any]]:
    return [{"page_num": 1, "text": text, "tables": [], **detect_section_info(text)}]


LABEL_VALUE_REGEXES = [
    re.compile(
        r"(?P<label>[\u0410-\u042f\u0430-\u044fA-Za-z\u0401\u04510-9\-\s()/%,\.]{3,100})\s*[:\-\u2013]\s*(?P<value>[+\-]?\d[\d\s,\.]*(?:\u043c\u043b\u043d|\u043c\u043b\u0440\u0434|\u0442\u044b\u0441\.|\u0442\u0440\u043b\u043d)?\s*(?:\u0440\u0443\u0431\.|\u0440\u0443\u0431\u043b\u0435\u0439|\u20bd|%|\u0434\u043e\u043b\u043b\.|usd|eur)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<label>\u0412\u044b\u0440\u0443\u0447\u043a\u0430|EBITDA|\u0427\u0438\u0441\u0442\u0430\u044f \u043f\u0440\u0438\u0431\u044b\u043b\u044c|\u041e\u043f\u0435\u0440\u0430\u0446\u0438\u043e\u043d\u043d\u0430\u044f \u043f\u0440\u0438\u0431\u044b\u043b\u044c|\u041f\u0440\u0438\u0431\u044b\u043b\u044c \u0437\u0430 \u043f\u0435\u0440\u0438\u043e\u0434|\u0414\u0435\u043d\u0435\u0436\u043d\u044b\u0435 \u0441\u0440\u0435\u0434\u0441\u0442\u0432\u0430(?: \u0438 \u0438\u0445 \u044d\u043a\u0432\u0438\u0432\u0430\u043b\u0435\u043d\u0442\u044b)?|\u041a\u0430\u043f\u0438\u0442\u0430\u043b\u044c\u043d\u044b\u0435 \u0437\u0430\u0442\u0440\u0430\u0442\u044b|\u041a\u0430\u043f\u0438\u0442\u0430\u043b|\u0421\u043e\u0431\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u043a\u0430\u043f\u0438\u0442\u0430\u043b|\u0410\u043a\u0442\u0438\u0432\u044b|\u041e\u0431\u044f\u0437\u0430\u0442\u0435\u043b\u044c\u0441\u0442\u0432\u0430|\u0414\u043e\u043b\u0433)\s+(?:\u0441\u043e\u0441\u0442\u0430\u0432\u0438\u043b\u0430|\u0441\u043e\u0441\u0442\u0430\u0432\u0438\u043b|\u0434\u043e\u0441\u0442\u0438\u0433\u043b\u0430|\u0434\u043e\u0441\u0442\u0438\u0433|\u0440\u0430\u0432\u043d\u0430|\u0440\u0430\u0432\u0435\u043d|\u0443\u0432\u0435\u043b\u0438\u0447\u0438\u043b\u0430\u0441\u044c \u0434\u043e|\u0441\u043d\u0438\u0437\u0438\u043b\u0430\u0441\u044c \u0434\u043e)\s+(?P<value>[+\-]?\d[\d\s,\.]*(?:\u043c\u043b\u043d|\u043c\u043b\u0440\u0434|\u0442\u044b\u0441\.|\u0442\u0440\u043b\u043d)?\s*(?:\u0440\u0443\u0431\.|\u0440\u0443\u0431\u043b\u0435\u0439|\u20bd|%|\u0434\u043e\u043b\u043b\.|usd|eur)?)",
        re.IGNORECASE,
    ),
]


def looks_like_bad_candidate(label: str, value: str, raw_text: str) -> bool:
    blob = f"{label} {value} {raw_text}".lower()
    return any(pattern in blob for pattern in BAD_PATTERNS)


def make_candidate(page: dict[str, Any], source_type: str, label: str, value: str, raw_text: str) -> Candidate:
    _, _, normalized_value_text = normalize_scale_in_value_text(value)
    extracted_period = (
        extract_period_from_text(raw_text)
        or extract_period_from_text(page.get("text", ""))
        or extract_period_from_text(page.get("section_hint") or "")
    )
    pre_mapped_kpi, pre_map_confidence = pre_map_label_to_kpi(label)
    return Candidate(
        source_type=source_type,
        page_num=page["page_num"],
        section_hint=page.get("section_hint"),
        section_type=page.get("section_type"),
        section_score=float(page.get("section_score", 0.0)),
        label_text=label,
        value_text=value,
        raw_text=raw_text,
        normalized_value_text=normalized_value_text,
        extracted_period=extracted_period,
        pre_mapped_kpi=pre_mapped_kpi,
        pre_map_confidence=pre_map_confidence,
    )


def generate_table_candidates(page: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for table in page.get("tables", []):
        if not table:
            continue
        for row in table:
            if not row:
                continue
            cells = [clean_text(cell) for cell in row if cell and clean_text(cell)]
            if len(cells) < 2:
                continue
            label = cells[0]
            value_candidate = next((value for value in cells[1:] if re.search(r"\d", value)), None)
            if not value_candidate:
                continue
            raw = " | ".join(cells)
            if looks_like_bad_candidate(label, value_candidate, raw):
                continue
            if re.search(r"(\u0432\u044b\u0440\u0443\u0447|ebitda|\u043f\u0440\u0438\u0431\u044b\u043b|\u0430\u043a\u0442\u0438\u0432|\u043a\u0430\u043f\u0438\u0442\u0430\u043b|\u0434\u0435\u043d\u0435\u0436\u043d|\u0434\u043e\u043b\u0433|\u043e\u0431\u044f\u0437\u0430\u0442|eps)", label, re.IGNORECASE):
                out.append(make_candidate(page, "table_row", label, value_candidate, raw))
    return out


def generate_text_candidates(page: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    lines = [clean_text(line) for line in page.get("text", "").split("\n") if clean_text(line)]
    for line in lines:
        for regex in LABEL_VALUE_REGEXES:
            match = regex.search(line)
            if not match:
                continue
            label = clean_text(match.group("label"))
            value = clean_text(match.group("value"))
            if looks_like_bad_candidate(label, value, line):
                continue
            out.append(make_candidate(page, "text_pair", label, value, line))
            break
    return out


def generate_candidates(pages: list[dict[str, Any]]) -> list[Candidate]:
    all_candidates: list[Candidate] = []
    for page in pages:
        all_candidates.extend(generate_table_candidates(page))
        all_candidates.extend(generate_text_candidates(page))
    return all_candidates


def resolve_pdf_path(pdf_path: str) -> str:
    path = Path(pdf_path)
    if path.exists():
        return str(path)
    for base in (Path("/app"), Path("/workspace")):
        candidate = base / pdf_path
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"PDF path is not accessible inside the container: {pdf_path}")


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)

    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(text[first_brace : last_brace + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_candidate_with_dictionary(candidate: Candidate) -> NormalizedKPI | None:
    if not candidate.pre_mapped_kpi or candidate.pre_map_confidence < 0.88:
        return None
    normalized_value, unit, _ = normalize_scale_in_value_text(candidate.value_text)
    if normalized_value is None:
        return None
    return NormalizedKPI(
        canonical_kpi=candidate.pre_mapped_kpi,
        value=float(normalized_value),
        unit=unit,
        period=candidate.extracted_period,
        is_kpi=True,
        confidence=min(0.97, candidate.pre_map_confidence),
        reason=f"Dictionary-mapped from label '{candidate.label_text}'",
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
        normalization_source="dictionary",
    )


def consolidate_kpis(items: list[NormalizedKPI]) -> list[dict[str, Any]]:
    filtered = [
        item
        for item in items
        if item is not None
        and item.is_kpi
        and item.canonical_kpi in CANONICAL_KPIS
        and item.confidence >= 0.60
        and item.value is not None
    ]
    best_by_key: dict[tuple[str | None, str | None, str | None], tuple[tuple[Any, ...], NormalizedKPI]] = {}
    for item in filtered:
        key = (item.canonical_kpi, item.period, item.unit)
        score = (
            item.confidence,
            SOURCE_PRIORITY.get(item.source_type, 0),
            item.section_score,
            1 if item.normalization_source == "dictionary" else 0,
            1 if item.section_hint else 0,
        )
        if key not in best_by_key or score > best_by_key[key][0]:
            best_by_key[key] = (score, item)
    return [asdict(value[1]) for value in best_by_key.values()]


def build_kpi_dict(normalized_kpis: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in normalized_kpis:
        kpi = item.get("canonical_kpi")
        value = item.get("value")
        if kpi and value is not None and kpi not in result:
            result[kpi] = value
    return result


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_derived_metrics(kpis: dict[str, Any]) -> dict[str, Any]:
    revenue = kpis.get("revenue")
    ebitda = kpis.get("ebitda")
    net_income = kpis.get("net_income")
    debt = kpis.get("debt")
    equity = kpis.get("equity")
    cash = kpis.get("cash")
    operating_income = kpis.get("operating_income")
    derived = {
        "ebitda_margin": safe_div(ebitda, revenue),
        "net_margin": safe_div(net_income, revenue),
        "operating_margin": safe_div(operating_income, revenue),
        "debt_to_equity": safe_div(debt, equity),
        "cash_to_debt": safe_div(cash, debt),
    }
    return {key: value for key, value in derived.items() if value is not None}
