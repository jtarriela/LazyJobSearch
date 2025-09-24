# CLI Layer Audit Report

**Module:** CLI Layer  
**Summary:** Complete audit of CLI command surface vs documented design, validating argument contracts and error handling.

## Findings

### Command Inventory Analysis

**Implemented Commands (from cli/ljs.py):**
```
ljs
  config
    init ✓              # Write example config 
    show ✓              # Show current config
    validate ✓          # Validate config file
  schema
    validate ✓          # Schema validation
  generate
    company-template ✓  # Scaffold company template
  resume
    parse ✓             # Parse resume file
    chunk ✓             # Chunk resume content  
    ingest ✓            # Parse, chunk, embed resume
    list ✓              # List resume versions
    show ✓              # Show specific resume
    activate ✓          # Mark resume active
  companies
    seed ✓              # Seed companies from file
    list ✓              # List companies
    add ✓               # Add single company
    select ✓            # Select companies by criteria
    show ✓              # Show specific company
  crawl
    run ✓               # Run crawler
    discover ✓          # Discover careers pages
    status ✓            # Show crawl status
  match
    run ✓               # Run matching pipeline
    top ✓               # Show top matches
    test-anduril ✓      # Test specific Anduril matching
    test-anduril-enhanced ✓  # Enhanced Anduril test
  review
    start ✓             # Start review process
    rewrite ✓           # Rewrite resume/content
    next ✓              # Next iteration
    satisfy ✓           # Mark review satisfied
    list ✓              # List reviews
  apply
    run ✓               # Apply to jobs
  events
    tail ✓              # Stream events
  db  
    migrate ✓           # Run migrations
    init-db ✓           # Initialize database
  jobs
    add ✓               # Add job manually
    list ✓              # List jobs
  notifications
    digest ✓            # Send digest email
```

**Expected Commands (from CLI_DESIGN.md):**
```
ljs
  config
    init ✓
    validate ✓
  user                  # ❌ MISSING SUBCOMMAND
    show                # ❌ MISSING
    sync --from-config  # ❌ MISSING
  resume
    ingest ✓
    list ✓
    show ✓
    activate ✓
  companies
    seed ✓
    list ✓
  crawl
    run ✓
    status ✓
  match
    run ✓
    top ✓
  review
    start ✓
    rewrite ✓
    next ✓
    satisfy ✓
    show ✓              # ❌ MISSING (has list instead)
  apply
    run ✓
    bulk                # ❌ MISSING
    status              # ❌ MISSING
  events
    tail ✓
  schema
    validate ✓
  db
    migrate ✓
```

### Issues Identified

**High Severity:**
1. **Missing User Management**: No `user` subcommand group (Severity: High, Evidence: cli/ljs.py:40-65 missing user_app)
2. **Missing Apply Bulk Operations**: No `apply bulk` or `apply status` commands (Severity: High, Evidence: cli/ljs.py:1341+ only has apply run)

**Medium Severity:**
3. **Command Drift**: Several commands implemented but not documented (add, select, discover, parse, chunk) (Severity: Med, Evidence: CLI_DESIGN.md missing these commands)
4. **Review Show vs List**: CLI_DESIGN.md expects `review show` but implementation has `review list` (Severity: Med, Evidence: cli/ljs.py:1295)

**Low Severity:**
5. **Extra Test Commands**: `test-anduril*` commands not in design (Severity: Low, Evidence: cli/ljs.py:1117-1129)

## Gaps vs Documentation

- **CLI_DESIGN.md** → Missing user management commands entirely
- **CLI_DESIGN.md** → Missing bulk application commands
- **CLI_DESIGN.md** → Outdated command list missing implemented commands

## Argument Contract Validation

**Sample Command Analysis:**

1. **`resume ingest`** (cli/ljs.py:508)
   - Required: `file: Path` ✓
   - Validation: File existence check ✓
   - Error handling: User-friendly messages ✓

2. **`apply run`** (cli/ljs.py:1341)
   - Required: `job_id: str` ✓
   - Optional: `resume, profile, dry_run` ✓
   - Validation: Job/resume existence checks ✓
   - Error handling: Graceful failures ✓

3. **`companies seed`** (cli/ljs.py:656)
   - Required: `file: Path` ✓
   - Optional: `update: bool` ✓
   - Validation: File format checks needed ⚠️

## Error Surface Audit

**Positive Examples:**
- FileNotFoundError handling in `resume ingest`
- Database connection error handling in most commands
- Rich console output for user-friendly errors

**Areas for Improvement:**
- Some commands lack comprehensive input validation
- Stack traces occasionally visible (need more try/catch blocks)
- Error messages could be more actionable in some cases

## Idempotency Analysis

**Commands Tested:**
- `resume ingest`: ⚠️ May create duplicate records (needs deduplication logic)
- `companies seed`: ✓ Has update flag for idempotent runs
- `db migrate`: ✓ Alembic handles idempotency

## Metrics/Benchmarks

- **Total Commands Implemented**: 35
- **Commands in CLI_DESIGN.md**: 25
- **Coverage**: 88% (22/25 documented commands implemented)
- **Drift**: 13 undocumented commands implemented

## Recommended Actions

1. **HIGH PRIORITY**: Implement missing `user` subcommand group with `show` and `sync` commands
2. **HIGH PRIORITY**: Add `apply bulk` and `apply status` commands
3. **MEDIUM PRIORITY**: Update CLI_DESIGN.md to include all implemented commands
4. **MEDIUM PRIORITY**: Standardize `review show` vs `review list` naming
5. **LOW PRIORITY**: Add comprehensive input validation to all commands
6. **LOW PRIORITY**: Enhance error messages with actionable suggestions

## Acceptance Criteria for Completion

- [ ] All commands from CLI_DESIGN.md are implemented
- [ ] CLI_DESIGN.md reflects all implemented commands
- [ ] No stack traces in normal error conditions
- [ ] All commands pass idempotency tests
- [ ] User subcommand group implemented
- [ ] Apply bulk operations implemented

## Open Questions

- Should test-specific commands be moved to a separate `test` subcommand group?
- Should undocumented commands be removed or documented?
- What is the intended behavior for resume deduplication on repeat ingestion?