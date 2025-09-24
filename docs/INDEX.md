# Documentation Index

Central design & reference docs for LazyJobSearch.

## Core Design
- [Architecture & System Design](./ARCHITECTURE.md)
- [Architecture Decision Records (ADR Index)](./adrs/README.md)
- [Portal Template DSL Schema](./portal_template_dsl.schema.json)

## Performance & Scalability
- [Performance Optimization Guide](./PERFORMANCE_OPTIMIZATION.md) - Algorithm analysis, benchmarks, cost optimization
- [Scalability Architecture Guide](./SCALABILITY_GUIDE.md) - Multi-region deployment, database sharding, auto-scaling
- [Algorithm Specifications](./ALGORITHM_SPECIFICATIONS.md) - Detailed algorithmic implementations with complexity analysis

## Implementation Guides
- [Technical Review Packet](./TECHNICAL_REVIEW_PACKET.md) - Executive summary, metrics, risk assessment
- [Implementation Backlog](./BACKLOG_IMPLEMENTATION.md) - Concrete follow-up tasks for production readiness
- [Roadmap](./ROADMAP.md) - Major engineering workstreams and milestones

## Data Contracts
- [Schema Catalog](./schema/) (per-table markdown: jobs, resumes, matches, reviews, applications, portals)

## Command Line Interface
- [CLI Design](./CLI_DESIGN.md) - Terminal interface for workflow automation

## Examples
- [Example Portal Template: Greenhouse Basic](./examples/portal_templates/greenhouse_basic.json)

## Future Docs (Planned)
- CONTRIBUTING.md (workflow, coding standards)
- CLI usage / API reference (FastAPI + OpenAPI)
- Deployment guide (Compose + optional cloud)
- Observability runbook & dashboards

---
Last updated: 2025-01-27
