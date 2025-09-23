# RESUME_CHUNKS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| resume_id | uuid FK -> RESUMES.id | Owning resume |
| chunk_text | text | Chunk content |
| embedding | vector | Semantic embedding |
| token_count | int | Approx token count |

Index: IVFFLAT on embedding (cosine).