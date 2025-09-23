# REVIEWS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| job_id | uuid FK -> JOBS.id | Reviewed job |
| resume_id | uuid FK -> RESUMES.id | Resume version evaluated |
| iteration | int | Iteration number (starts at 1) |
| parent_review_id | uuid nullable | Prior iteration linkage |
| llm_score | int | Optional direct score if reused |
| strengths_md | text | Markdown bullet list |
| weaknesses_md | text | Markdown bullet list |
| improvement_brief | text | Summary of plan |
| improvement_plan_json | jsonb | Structured directives |
| redact_note | text | Notes on PII redaction if any |
| proposed_new_resume_id | uuid nullable | AI-generated rewrite candidate |
| accepted_new_resume_id | uuid nullable | Adopted next version |
| satisfaction | text | NULL | NEEDS_MORE | SATISFIED |
| created_at | timestamptz | Creation timestamp |
| status | text | PENDING | COMPLETED | SUPERSEDED | CANCELLED |