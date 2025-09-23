# RESUME_CHUNKS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| resume_id | uuid FK -> RESUMES.id | Parent resume |
| chunk_text | text | Text content of the chunk |
| embedding | vector(1536) | OpenAI text-embedding-3-large |
| token_count | int | Number of tokens in chunk |
| section_type | text | experience, education, skills, etc. |
| chunk_index | int | Order within the section |

Index: IVFFLAT on embedding (cosine).