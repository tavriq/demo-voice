# Base image pinned by digest (python:3.12-slim, the same as demo-rag).
FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VAR_DIR=/var/lib/demo-voice

WORKDIR /app

RUN useradd --system --uid 10001 --user-group --create-home --home-dir /home/app app \
    && mkdir -p /var/lib/demo-voice \
    && chown app:app /var/lib/demo-voice

COPY requirements.lock .
RUN pip install --require-hashes --no-deps -r requirements.lock

COPY app ./app

USER app

# Hourly limits, the daily budget and conversations (30 days, contacts masked) live here. No audio.
VOLUME ["/var/lib/demo-voice"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/voice/health', timeout=4)"

CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", \
     "--workers", "1", "--limit-concurrency", "16"]
