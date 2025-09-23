# JOB_CHUNKS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| job_id | uuid FK -> JOBS.id | Parent job |
| chunk_text | text | Text content of the chunk |
| embedding | vector(1536) | OpenAI text-embedding-3-large |
| token_count | int | Number of tokens in chunk |
| chunk_index | int | Order within the job description |

Index: IVFFLAT on embedding (cosine).