FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN uv sync --no-dev --frozen 2>/dev/null || uv pip install --system .

ENTRYPOINT ["mcp-server-mcsa"]
