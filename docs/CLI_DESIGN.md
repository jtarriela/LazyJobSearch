# CLI Design & User Interaction Model

While a web frontend is planned (see `ARCHITECTURE.md` Section 20), initial development, testing, and power-user flows benefit from a comprehensive terminal CLI plus declarative YAML configuration.

## Goals
- Enable end-to-end workflow (seed companies → crawl → ingest resume → match → review → iterate resume → auto-apply) without UI.
- Support repeatable runs via config + minimal commands.
- Provide streaming feedback (progress bars, event logs) and structured JSON output (for scripting).
- Keep side-effecting operations explicit (e.g., `--apply`, `--rewrite`).

## High-Level UX Principles
1. Declarative baseline in YAML; imperative overrides via flags.
2. Idempotent operations (safe to re-run; uses UPSERT semantics & fingerprints).
3. Output defaults to human-readable; `--json` toggles machine-readable.
4. Fail fast on misconfiguration with rich diagnostics (config validation schema).
5. No silent network or LLM calls—always log model + token summary.

## Command Tree (Draft)
```
ljs (lazyjobsearch)
  config
    init                 # Write example config to ~/.lazyjobsearch/config.yaml
    validate             # Validate current config file
  user
    show                 # Display user profile (from DB)
    sync --from-config   # Ensure user + application profiles exist
  resume
    ingest <file>        # Parse, chunk, embed a resume file
    list                 # List resume versions
    show <resume_id>
    activate <resume_id> # Mark a version active
  companies
    seed --file seeds.txt
    list
  crawl
    run [--company <id>|--all]
    status
  match
    run [--resume <id>] [--limit 200]
    top [--resume <id>] [--limit 20]
  review
    start <job_id> [--resume <id>]       # iteration=1
    rewrite <review_id> [--mode auto|manual --file new.pdf]
    next <review_id>                     # request next iteration
    satisfy <review_id>
    show <review_id>
  apply
    run <job_id> [--resume <id>] [--profile <name>] [--dry-run]
    bulk --jobs job_ids.txt [--filter score>=80]
    status <application_id>
  events
    tail [--since 10m]
  schema
    validate                             # Run markdown ↔ model validator
  db
    migrate                              # Alembic upgrade head
```

## Example Flows
### 1. First-Time Setup
```
$ ljs config init
$ edit ~/.lazyjobsearch/config.yaml
$ ljs user sync --from-config
$ ljs companies seed --file seeds/companies.txt
```

### 2. Resume Ingest & Match
```
$ ljs resume ingest ./resumes/jane_v1.pdf
$ ljs match run --resume latest
$ ljs match top --resume latest --limit 10 --json > matches.json
```

### 3. Review → Rewrite Loop
```
$ job_id=$(jq -r '.[0].job_id' matches.json)
$ ljs review start $job_id --resume latest
$ ljs review show <review_id>
# Decide to try AI rewrite
$ ljs review rewrite <review_id> --mode auto
$ ljs review next <review_id>   # after accepting new version
$ ljs review satisfy <review_id>
```

### 4. Auto-Apply (Dry Run vs Live)
```
$ ljs apply run $job_id --resume active --profile default --dry-run
# Inspect artifacts
$ ljs apply run $job_id --resume active --profile default --apply
```

## Configuration
Default resolution order (lowest to highest precedence):
1. Built-in defaults
2. Global file: `~/.lazyjobsearch/config.yaml`
3. Project local: `./lazyjobsearch.yaml`
4. Environment variables `LJS_*` (e.g., `LJS_LLM_MODEL`)
5. CLI flags

Numeric / duration flags accept human-friendly forms (`5m`, `250ms`).

### Validation
A JSON Schema will be defined for user config (future: `config/schema.json`). `ljs config validate` runs it.

## Output Conventions
- Human mode: aligned tables (rich text if terminal supports color).
- JSON mode: arrays/objects with stable keys; datetime in ISO8601 UTC.
- Errors: non-zero exit code, stderr message, optional `--trace` for stack.

## Retry / Backoff Policy
Commands that enqueue work (crawl, review start, rewrite) return quickly with an identifier. A `tail` or `status` subcommand streams events (WebSocket or polling fallback).

## Security / Secrets
- API token or DB connection string loaded from environment (`LJS_DB_URL`, `LJS_API_TOKEN`).
- Never write secrets to config YAML (only references or env var names).

## Future Enhancements
- `watch` mode: auto-refresh top matches when new jobs arrive.
- `export cover-letter` once implemented.
- TUI (text UI) wrapper using `textual` or `rich.prompt` for interactive flows.

## Open Questions
- Should we auto-start a review for all top-N matches above a score threshold? (Potential cost explosion; likely flag-gated.)
- Bulk apply guardrails (confirmation prompt on >X submissions).

---
_This document will evolve; align changes with ADRs if user interaction strategy shifts materially._
