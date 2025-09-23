# JOB_CHUNKS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| job_id | uuid FK -> JOBS.id | Parent job |
| chunk_text | text | Segment of JD |
| embedding | vector | Semantic embedding |
| token_count | int | Approx token length |

Index: IVFFLAT on embedding (cosine).