CREATE DATABASE IF NOT EXISTS analysis;

CREATE TABLE IF NOT EXISTS analysis.kpi_extractions
(
    document_id String,
    company_ticker String,
    kpi_name String,
    kpi_value String,
    unit String,
    period String,
    confidence Float64,
    model String,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (company_ticker, document_id, kpi_name, created_at);
