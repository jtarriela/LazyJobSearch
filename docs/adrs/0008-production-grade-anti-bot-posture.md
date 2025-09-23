## ADR 0008: Production-Grade Anti-Bot & Humanization Posture

Date: 2025-09-23  
Status: Accepted

### Context
Modern ATS platforms deploy sophisticated detection (automation flags, fingerprinting, behavioral analysis, captchas). Basic user-agent rotation & sleep are insufficient for sustained crawling/applying.

### Decision
Layered approach: undetected-chromedriver + Playwright fallback; residential/rotating proxy pool; fingerprint randomization (UA, viewport, timezone, language, canvas noise); human behavior simulation (Bezier mouse paths, scroll + dwell); adaptive backoff state machine; optional captcha solving behind feature flag; observability metrics; ethical guardrails (robots, per-domain caps, allowlist, kill switch).

### Rationale
Reduces block rates, increases session longevity, provides measurable signals, avoids adversarial escalation while respecting site policies.

### Consequences
- Complexity & cost overhead (proxies, captcha service).
- More moving parts & secrets management.
- Slight latency increase per page.

### Alternatives
| Option | Drawback |
|--------|----------|
| Minimal hygiene only | Rapid blocking |
| Headless only | Highly detectable |
| API scraping attempt | Largely unavailable |

### Follow-Up
1. Add `scrape_sessions` table.
2. Implement `HumanBehaviorSimulator` utility.
3. Proxy provider interface + config.
4. Metrics: `scrape.block_rate`, `scrape.challenge_rate`, `scrape.captcha_rate`.
5. Feature flags: `enable_captcha_solver`, `enable_behavior_sim`.
6. Security review of secrets storage.

---