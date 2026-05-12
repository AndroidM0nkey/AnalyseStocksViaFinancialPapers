FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app/generated

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY proto /app/proto
COPY src /app/src

RUN pip install . && \
    mkdir -p /app/generated && \
    python -m grpc_tools.protoc \
      -I/app/proto \
      --python_out=/app/generated \
      --grpc_python_out=/app/generated \
      /app/proto/analysis.proto && \
    touch /app/generated/__init__.py

ARG SERVICE_MODULE
ENV SERVICE_MODULE=${SERVICE_MODULE}

CMD ["sh", "-c", "python -m ${SERVICE_MODULE}"]
