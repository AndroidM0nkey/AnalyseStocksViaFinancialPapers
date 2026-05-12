#!/bin/sh
set -eu

echo "Waiting for Ollama API at ${OLLAMA_HOST}..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

if ollama show "${OLLAMA_MODEL}" >/dev/null 2>&1; then
  echo "Model ${OLLAMA_MODEL} is already available."
  exit 0
fi

echo "Pulling model ${OLLAMA_MODEL} for first use..."
ollama pull "${OLLAMA_MODEL}"

echo "Model ${OLLAMA_MODEL} is ready."
