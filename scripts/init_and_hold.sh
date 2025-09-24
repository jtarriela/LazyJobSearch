#!/usr/bin/env bash
set -euo pipefail

echo "[init] Waiting for postgres..."
for i in {1..40}; do
  if pg_isready -h postgres -U postgres -d lazyjobsearch >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "[init] Ensuring pgvector extension..."
psql "postgresql://postgres:postgres@postgres:5432/lazyjobsearch" -c "CREATE EXTENSION IF NOT EXISTS vector;" || true

echo "[init] Creating tables if missing..."
python - <<'EOF'
from libs.db.session import engine
from libs.db import models
models.Base.metadata.create_all(bind=engine)
print("[init] Tables present.")
EOF

echo "[init] Ready. Exec into container with: docker compose exec app bash"
tail -f /dev/null