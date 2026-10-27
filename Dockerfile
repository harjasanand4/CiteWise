# syntax=docker/dockerfile:1

# ---- 1. build the web UI ---------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- 2. Python runtime -----------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models
# git is needed by `citewise fetch` (sparse checkout of the handbooks)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 citewise \
    && mkdir -p /models /app/data && chown citewise /models /app/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[postgres]"

USER citewise
# Bake the embedding model and reranker into the image so a container never downloads
# on its first request. Build with --build-arg PRELOAD_MODELS=false for a faster CI build.
ARG PRELOAD_MODELS=true
RUN if [ "$PRELOAD_MODELS" = "true" ]; then python -c \
    "from citewise.embeddings import FastEmbedEmbedder; from citewise.retrieval.retriever import CrossEncoderReranker; FastEmbedEmbedder(); CrossEncoderReranker()"; \
    fi

COPY --chown=citewise config ./config
COPY --chown=citewise evals ./evals
COPY --from=web --chown=citewise /web/dist ./web/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "citewise.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
