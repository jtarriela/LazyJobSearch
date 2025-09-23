# MATCHES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| job_id | uuid FK -> JOBS.id | Matched job |
| resume_id | uuid FK -> RESUMES.id | Matched resume |
| vector_score | float | Cosine similarity from vector search |
| fts_rank | float | Full-text search ranking |
| llm_score | int | 0-100 LLM assessment |
| action | text | apply, skip, maybe |
| reasoning | text | LLM explanation |
| skill_gaps | jsonb | Missing required skills |
| llm_model | text | Which model was used |
| prompt_hash | text | For prompt change tracking |
| scored_at | timestamptz | When the match was scored |

Indexes: (job_id,resume_id) unique if enforcing single latest; optional btree on llm_score desc.