# COMPANIES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| name | text UNIQUE | Company name |
| website | text | Company website URL |
| careers_url | text | Careers page URL |
| crawler_profile_json | jsonb | Selenium adapter configuration |
| created_at | timestamptz | Creation timestamp |