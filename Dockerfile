# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY kairo/pyproject.toml ./kairo/pyproject.toml
COPY kairo/kairo ./kairo/kairo

# Install the locked environment, including kairo as a regular package.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime

# Kaleido uses Chromium when Copilot attaches plots to a message.
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium fonts-liberation tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 kairo

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BROWSER_PATH=/usr/bin/chromium \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY dashboard ./dashboard
COPY kairo/kairo.png ./kairo/kairo.png

USER kairo
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

# PM4Py inspects its parent process; tini also forwards signals and reaps Chromium.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "dashboard/app.py"]
