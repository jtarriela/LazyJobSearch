# JOBS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | Primary key |
| company_id | uuid FK -> COMPANIES.id | Owning company |
| url | text UNIQUE | Canonical job URL |
| title | text | Title extracted |
| location | text | Raw location string |
| seniority | text | Parsed seniority level |
| jd_fulltext | text | Raw normalized JD text |
| jd_tsv | tsvector | Full-text search vector |
| jd_file_url | text | Object store compressed artifact |
| jd_skills_csv | text | Extracted skill list CSV |
| scraped_at | timestamptz | First/last scrape time (UPSERT updates) |
| scrape_fingerprint | text | Hash of raw JD to detect changes |

Indexes: GIN on jd_tsv, unique(url).