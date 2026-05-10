# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc g++ \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install "poetry==${POETRY_VERSION}"

WORKDIR /build
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --without dev

COPY app ./app

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PORT=8080

RUN groupadd --system --gid 1000 app \
 && useradd  --system --uid 1000 --gid app --home ${APP_HOME} --shell /usr/sbin/nologin app \
 && mkdir -p /data ${APP_HOME} \
 && chown -R app:app /data ${APP_HOME}

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/app ${APP_HOME}/app

WORKDIR ${APP_HOME}
USER app

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=2).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-server-header"]
