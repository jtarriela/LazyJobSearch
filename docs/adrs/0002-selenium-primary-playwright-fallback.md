# ADR 0002: Selenium Primary, Playwright Fallback

Date: 2025-09-23
Status: Accepted

## Context
Need robust scraping of diverse company career portals with differing JS complexity and anti-bot protections.

## Decision
Implement Selenium (undetected-chromedriver) as the primary automation layer. Provide an abstraction so specific hostile portals can optionally route through Playwright.

## Rationale
- Faster initial development (team familiarity + ecosystem)
- Good enough stealth with undetected-chromedriver for common portals
- Playwright retained for difficult sites (shadow DOM, strict detection)

## Consequences
- Dual runtime test coverage complexity
- Slightly larger container images if both libraries present

## Alternatives
- Playwright only: cleaner API, potentially higher initial adaptation cost
- HTTP-only scraping + partial JS execution (requests+pyppeteer): brittle for dynamic portals

## Follow-Up
- Implement adapter interface `PortalDriver` with capability flags
- Add integration tests for both drivers on sample pages
