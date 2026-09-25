FROM ghcr.io/astral-sh/uv:0.11.33 AS uv

FROM python:3.13-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=uv /uv /uvx /usr/local/bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim-bookworm AS runtime

ENV HOME="/home/fall-detector" \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 fall-detector \
    && useradd --uid 10001 --gid 10001 --create-home \
        --home-dir /home/fall-detector --shell /usr/sbin/nologin fall-detector

WORKDIR /app

COPY --from=builder --chown=fall-detector:fall-detector /app/.venv /app/.venv
COPY --chown=fall-detector:fall-detector main.py settings.py ./
COPY --chown=fall-detector:fall-detector src ./src
RUN mkdir --parents /app/data \
    && chown fall-detector:fall-detector /app/data

USER fall-detector

EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
