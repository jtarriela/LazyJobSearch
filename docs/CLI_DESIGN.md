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

## Command Tree (Current Implementation Status)
```
ljs (lazyjobsearch)
  config
    init ✅               # Write example config to ~/.lazyjobsearch/config.yaml
    show ✅               # Show current config
    validate ✅           # Validate current config file
  user                   # ✅ User management implemented  
    show ✅               # Display user profile (from DB)
    sync ✅               # Ensure user + application profiles exist
  resume
    ingest ✅             # Parse, chunk, embed a resume file
    list ✅               # List resume versions
    show ✅               # Show resume details
    activate ✅           # Mark a resume as active
    parse ➕              # Parse resume file and extract data (NEW)
    chunk ➕              # Chunk resume content (NEW)
  companies
    seed ✅               # Seed companies from CSV/JSON file
    list ✅               # List companies
    add ➕                # Add single company (NEW)
    select ➕             # Select companies by criteria (NEW)  
    show ➕               # Show specific company details (NEW)
  jobs
    add ➕                # Add job manually (NEW)
    list ➕               # List jobs from database (NEW)
  crawl
    run ✅               # Run crawler for companies
    discover ➕          # Discover careers pages (NEW)
    status ✅            # Show crawl status
  match
    run ✅               # Run matching pipeline
    top ✅               # Show top matches
    test-anduril ➕      # Test Anduril-specific matching (NEW)
    test-anduril-enhanced ➕ # Enhanced Anduril test (NEW)
  review
    start ✅             # Start review process (iteration=1)
    rewrite ⚠️           # PLACEHOLDER - Generate rewrite suggestions
    next ⚠️              # PLACEHOLDER - Next iteration
    satisfy ✅           # Mark review as satisfied
    list ➕              # List reviews (NEW, alongside show)
    show ✅              # Show detailed review information (FIXED - was missing)
  apply
    run ✅               # Apply to jobs
    bulk ✅              # Bulk application operations (FIXED)
    status ✅            # Application status checking (FIXED)
  events
    tail ✅              # Stream events
  schema
    validate ✅          # Run markdown ↔ model validator
  db
    migrate ✅           # Alembic upgrade head
    init-db ➕           # Initialize database (NEW)
  generate               # ➕ NEW COMMAND GROUP
    company-template ➕   # Scaffold company portal template
  notifications          # ➕ NEW COMMAND GROUP  
    digest ➕            # Send digest emails
```

**Legend:**
- ✅ Fully implemented
- ⚠️ Partial implementation / placeholder
- ❌ Missing / not implemented
- ➕ New commands not in original design

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
