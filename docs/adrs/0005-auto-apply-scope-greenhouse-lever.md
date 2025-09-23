# ADR 0005: Auto-Apply MVP Scope = Greenhouse + Lever

Date: 2025-09-23
Status: Accepted

## Context
Need early auto-apply value while controlling complexity across heterogeneous ATS portals.

## Decision
Limit MVP automation to Greenhouse and Lever portals using a template DSL + adapter layer.

## Rationale
- High prevalence among tech companies
- Relatively stable DOM structures
- Shared conceptual field mapping reduces template divergence

## Consequences
- Users on other portals (Workday, Taleo, SuccessFactors) wait until Phase 2
- Risk of overfitting DSL to first two portals—must keep abstractions generic

## Alternatives
- Attempt broad multi-portal support now: higher failure & maintenance risk
- Defer auto-apply entirely: lose differentiating “wow” feature

## Follow-Up
- Validate DSL expressiveness on 5+ example forms
- Add portal on-boarding checklist (selectors, rate limits, field dictionary coverage)
