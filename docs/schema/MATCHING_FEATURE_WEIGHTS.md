# MATCHING_FEATURE_WEIGHTS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| created_at | timestamptz | Creation timestamp |
| model_version | text | Model version identifier |
| weights_json | jsonb | Feature weights JSON |