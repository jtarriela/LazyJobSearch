# REVIEWS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| resume_id | uuid FK -> RESUMES.id | Resume version evaluated |
| job_id | uuid FK -> JOBS.id | Reviewed job |
| llm_score | int | Optional direct score if reused |
| strengths_md | text | Markdown bullet list |
| weaknesses_md | text | Markdown bullet list |
| improvement_brief | text | Summary of plan |
| redact_note | text | Notes on PII redaction if any |
| created_at | timestamptz | Creation timestamp |