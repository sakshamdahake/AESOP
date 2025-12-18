# 1. Builder Stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# [FIX] Tell uv to create the environment at /opt/venv directly
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

COPY pyproject.toml uv.lock ./

# Install dependencies into /opt/venv
RUN uv sync --frozen --no-install-project --no-dev

# --- Runtime Stage ---
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# 1. Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 2. [FIX] Copy the venv from the EXACT same path it was built in
COPY --from=builder /opt/venv /opt/venv

# 3. Add to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy code
COPY . .

# --- Dev Stage ---
FROM runtime AS dev
# Tell uv where the environment is (so it doesn't create a new one in /app/.venv)
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
# Re-sync dev dependencies
RUN uv sync --frozen

# --- Prod Stage ---
FROM runtime AS prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]