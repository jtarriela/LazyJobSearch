FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# System deps (add others like build-essential if needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

# Copy just dependency metadata first (better layer caching)
COPY pyproject.toml README.md ./

# Install project in editable mode for live code reload via volume mount
RUN pip install --upgrade pip && pip install -e .[dev]

# Now copy the rest of the source (overwrites module with live code anyway)
COPY . /app

# Default command (can be overridden by docker-compose)
CMD ["bash", "scripts/init_and_hold.sh"]