# SCRAPE_SESSIONS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| company_id | uuid FK -> COMPANIES.id | Target company |
| started_at | timestamptz | Session start time |
| finished_at | timestamptz | Session end time |
| profile_json | jsonb | Scrape profile configuration |
| proxy_identifier | text | Proxy identifier used |
| outcome | text | Session outcome status |
| metrics_json | jsonb | Performance metrics |