FROM python:3.12-slim-bookworm AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"

COPY requirements.lock ./

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.lock \
    && python -m pip install --no-cache-dir "pyjwt>=2.8.0" "cryptography>=42.0.0"

ENV HF_HOME=/root/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEM0_DIR=/app/.runtime/mem0 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libglib2.0-0 \
        libgomp1 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin kinetix

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=kinetix:kinetix /root/.cache/huggingface /app/.cache/huggingface

COPY --chown=kinetix:kinetix agents agents
COPY --chown=kinetix:kinetix backend backend
COPY --chown=kinetix:kinetix data data
COPY --chown=kinetix:kinetix evaluation evaluation
COPY --chown=kinetix:kinetix infra infra
COPY --chown=kinetix:kinetix ingestion ingestion
COPY --chown=kinetix:kinetix retrieval retrieval
COPY --chown=kinetix:kinetix telemetry telemetry

RUN mkdir -p /app/data/incoming /app/.runtime/mem0 \
    && chown -R kinetix:kinetix /app

USER kinetix

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.environ.get('PORT', '10000'); urllib.request.urlopen('http://127.0.0.1:%s/api/health/live' % p, timeout=3)"

CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
