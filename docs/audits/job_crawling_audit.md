# Job Crawling Audit Report

**Module:** Job Crawling  
**Summary:** Audit of crawling infrastructure including batch processing, pagination, duplicate detection, rate limiting, and failure handling.

## Findings

### Crawler Architecture Analysis

**Available Scrapers (libs/scraper/):**
- ✅ **AndurilScraper**: Specific adapter for Anduril Industries careers page
- ✅ **CareersDiscoveryService**: Automatic careers page URL discovery 
- ✅ **CrawlWorker**: Main orchestration service
- ✅ **AntiBotPosture**: Fingerprinting and behavior simulation

**Scraper Coverage:**
- ✅ Anduril scraper fully implemented
- ❌ **Generic ATS Scrapers Missing**: No Greenhouse, Lever, or Workday scrapers (Severity: High, Evidence: crawl_worker.py:157 shows only anduril)

### Batch Size & Pagination Analysis

**Current Batch Processing:**
- ❌ **No Batch Size Configuration**: Single-threaded, sequential processing (Severity: Med, Evidence: crawl_worker.py:108-122 processes companies one by one)
- ❌ **No Pagination Support**: Scrapers don't handle multi-page job listings (Severity: High, Evidence: anduril_adapter.py has no pagination logic)
- ❌ **No Off-by-One Protection**: No boundary condition testing (Severity: Med, Evidence: no pagination boundary checks)

**Missing Functionality:**
- No configurable batch sizes for company crawling
- No parallel processing of multiple companies
- No pagination state management between crawler runs
- No resume from partial crawls

### Duplicate Detection Assessment

**Current Deduplication:**
- ✅ **Content Fingerprinting**: SHA256 hash generation for job content (Evidence: crawl_worker.py:222-231)
- ❌ **No Duplicate Prevention**: Always creates new Job records without checking existing ones (Severity: High, Evidence: crawl_worker.py:200-206 session.add() without duplicate check)
- ❌ **No URL-Based Deduplication**: No job URL uniqueness enforcement (Severity: High, Evidence: no URL checking in ingestion logic)

**Database Constraints Analysis:**
- Need to verify if Job model has unique constraints on job_url or content fingerprint
- Current implementation allows duplicate jobs to be inserted

### Rate Limiting Analysis

**Implemented Rate Limiting (libs/scraper/anduril_adapter.py):**
- ✅ **Configurable Rate Limits**: `rate_limit_ppm` parameter (default 10 requests/minute)
- ✅ **Batch Throttling**: Sleeps after every 5 job extractions (Evidence: anduril_adapter.py:131-132)
- ✅ **Human Behavior Simulation**: Random sleep intervals via `HumanBehaviorSimulator`

**Rate Limiting Robustness:**
- ✅ Respect rate limits during normal operation
- ❌ **No Retry Logic**: No exponential backoff for rate limit violations (Severity: Med, Evidence: no retry decorator or circuit breaker)
- ❌ **No Rate Limit Detection**: No detection of 429/rate limit responses (Severity: Med)

### Anti-Bot & Failure Handling

**Anti-Bot Measures (libs/scraper/anti_bot.py):**
- ✅ **Proxy Pool Support**: ProxyPool integration with health scoring
- ✅ **Fingerprint Generation**: Browser fingerprinting for detection avoidance
- ✅ **Human Behavior Simulation**: Mouse movements, scroll patterns, dwell times
- ✅ **Session Management**: Proper session lifecycle with metrics

**Error Handling & Logging:**
- ✅ **Structured Error Logging**: Proper exception handling with context (Evidence: crawl_worker.py:114-121)
- ✅ **Failure Recovery**: Individual job extraction failures don't stop full crawl
- ❌ **No Failure Metrics**: No structured metrics for failure rates by portal/job_id (Severity: Med, Evidence: no metrics emission in error handlers)

### Performance & Scalability

**Current Bottlenecks:**
- **Single-threaded Processing**: No concurrent crawling of multiple companies
- **Synchronous I/O**: Database operations block crawler threads  
- **Memory Usage**: No streaming/pagination for large job listings

**Resource Management:**
- ✅ Database session management with proper cleanup
- ✅ WebDriver lifecycle management
- ❌ **No Resource Limits**: No memory/timeout limits for large responses (Severity: Low)

### Testing Coverage

**Available Tests:**
- Found test files mentioning careers discovery
- Need to verify crawl worker and duplicate detection tests

**Missing Test Categories:**
- No pagination boundary tests
- No rate limiting simulation tests
- No duplicate detection tests
- No failure scenario tests

## Gaps vs Documentation

- **ALGORITHM_SPECIFICATIONS.md**: May reference crawling algorithms not implemented
- **ARCHITECTURE.md**: References multiple ATS scrapers but only Anduril exists
- **ADR 0008**: Anti-bot posture well-implemented but needs integration testing

## Metrics/Benchmarks

- **Scraper Coverage**: 25% (1/4 major ATS types - only Anduril)
- **Duplicate Detection**: 0% (fingerprinting exists but not used for deduplication)
- **Pagination Support**: 0% (no multi-page crawling)
- **Rate Limiting**: 85% (implemented but missing retry logic)
- **Error Handling**: 70% (good logging, missing structured metrics)

## Recommended Actions

1. **CRITICAL**: Implement duplicate detection using fingerprint/URL before database insertion
2. **HIGH**: Add pagination support to handle multi-page job listings
3. **HIGH**: Implement Greenhouse and Lever scrapers for broader coverage
4. **HIGH**: Add retry logic with exponential backoff for rate limiting
5. **MEDIUM**: Add parallel processing for multiple companies
6. **MEDIUM**: Implement structured failure metrics (job_id, portal, error_type)
7. **MEDIUM**: Add configurable batch sizes for large-scale crawling
8. **MEDIUM**: Create comprehensive test suite for crawling scenarios
9. **LOW**: Add memory/response size limits for large job descriptions
10. **LOW**: Implement partial crawl resume functionality

## Acceptance Criteria for Completion

- [ ] Duplicate detection prevents re-insertion of same job postings
- [ ] Pagination handles multi-page job listings correctly
- [ ] At least 3 ATS scrapers implemented (Anduril + Greenhouse + Lever)
- [ ] Rate limiting includes retry logic with exponential backoff  
- [ ] Structured failure metrics emitted for all error conditions
- [ ] Parallel processing option for multiple companies
- [ ] Comprehensive test suite covering edge cases
- [ ] No off-by-one errors in pagination boundary conditions
- [ ] Crawler can resume from partial failures
- [ ] Memory limits prevent runaway resource usage

## Open Questions

- Should duplicate detection be based on job URL, content hash, or both?
- What is the optimal batch size for company crawling?
- How should pagination state be persisted between crawler runs?
- Should there be a global rate limit across all scrapers?
- How should we handle job postings that change content but keep same URL?