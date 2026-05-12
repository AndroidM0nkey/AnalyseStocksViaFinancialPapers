#!/bin/sh
set -eu

CLICKHOUSE_HTTP_URL="${CLICKHOUSE_HTTP_URL:-http://clickhouse:8123}"
CLICKHOUSE_ADMIN_USER="${CLICKHOUSE_ADMIN_USER:-default}"
CLICKHOUSE_ADMIN_PASSWORD="${CLICKHOUSE_ADMIN_PASSWORD:-}"
CLICKHOUSE_DATABASE="${CLICKHOUSE_DATABASE:-analysis}"
CLICKHOUSE_APP_USER="${CLICKHOUSE_APP_USER:-analysis_app}"
CLICKHOUSE_APP_PASSWORD="${CLICKHOUSE_APP_PASSWORD:-analysis_app_password}"

run_clickhouse() {
  if [ -n "${CLICKHOUSE_ADMIN_PASSWORD}" ]; then
    clickhouse-client --host clickhouse --user "${CLICKHOUSE_ADMIN_USER}" --password "${CLICKHOUSE_ADMIN_PASSWORD}" "$@"
  else
    clickhouse-client --host clickhouse --user "${CLICKHOUSE_ADMIN_USER}" "$@"
  fi
}

echo "Waiting for ClickHouse at ${CLICKHOUSE_HTTP_URL}..."
until run_clickhouse --query "SELECT 1" >/dev/null 2>&1; do
  sleep 2
done

run_clickhouse --multiquery <<EOF
CREATE DATABASE IF NOT EXISTS ${CLICKHOUSE_DATABASE};
CREATE USER IF NOT EXISTS ${CLICKHOUSE_APP_USER} IDENTIFIED BY '${CLICKHOUSE_APP_PASSWORD}';
GRANT ALL ON ${CLICKHOUSE_DATABASE}.* TO ${CLICKHOUSE_APP_USER};
CREATE TABLE IF NOT EXISTS ${CLICKHOUSE_DATABASE}.kpi_extractions
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
EOF

echo "ClickHouse is initialized."
