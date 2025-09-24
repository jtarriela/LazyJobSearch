FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy minimal files first for dependency layer caching
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .[dev] || pip install .

# Copy the rest of the source
COPY . .

# Default environment variables (can be overridden by compose)
ENV DATABASE_URL=postgresql://ljs_user:ljs_password@postgres:5432/lazyjobsearch \
    REDIS_URL=redis://redis:6379/0 \
    MINIO_ENDPOINT=minio:9000 \
    MINIO_ROOT_USER=minioadmin \
    MINIO_ROOT_PASSWORD=minioadmin123

# Idle by default; use `docker compose exec app bash` or override command
CMD ["bash", "-c", "echo 'App container ready'; tail -f /dev/null"]
