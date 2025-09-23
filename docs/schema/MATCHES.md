# MATCHES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| job_id | uuid FK -> JOBS.id | |
| resume_id | uuid FK -> RESUMES.id | |
| vector_score | float | Cosine similarity score |
| llm_score | int | 0–100 LLM evaluation |
| action | text | Suggested action / recommendation |
| reasoning | text | Explanation / JSON reasoning blob |
| llm_model | text | Model identifier |
| prompt_hash | text | Hash of prompt template |
| scored_at | timestamptz | When LLM scoring completed |

Indexes: (job_id,resume_id) unique if enforcing single latest; optional btree on llm_score desc.