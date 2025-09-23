# RESUMES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| user_id | uuid nullable | Future: link to user accounts |
| fulltext | text | Parsed resume plain text |
| sections_json | jsonb | Structured sections (education, experience, etc.) |
| skills_csv | text | CSV skills list |
| yoe_raw | float | Raw computed years of experience |
| yoe_adjusted | float | Adjusted for education bonus |
| edu_level | text | Highest degree normalized |
| file_url | text | Original file artifact |
| created_at | timestamptz | Ingest timestamp |
| is_active | bool | For A/B testing multiple resumes |
