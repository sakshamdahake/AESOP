# 1. Builder Stage (Has uv installed to build the venv)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
# Install dependencies to /app/.venv
RUN uv sync --frozen --no-install-project --no-dev

# --- Runtime Stage ---
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# [CORRECTION IS HERE]
# We copy 'uv' from the official tool image, NOT the builder.
# This guarantees we get the binary from '/uv'.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Add venv and uv to PATH
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

# --- Dev Stage (Hot Reloading) ---
FROM runtime AS dev
# Now 'uv' exists in /bin/uv, so this command will work
RUN uv sync --frozen

# --- Prod Stage ---
FROM runtime AS prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]