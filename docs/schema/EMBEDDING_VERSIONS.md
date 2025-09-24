# EMBEDDING_VERSIONS Table

| Column | Type | Notes |
|--------|------|-------|
| version_id | text PK | Version identifier |
| model_name | text | Embedding model name |
| dimensions | int | Vector dimensions |
| created_at | timestamptz | Creation timestamp |
| deprecated_at | timestamptz | Deprecation timestamp |
| compatible_with | text[] | Compatible version array |