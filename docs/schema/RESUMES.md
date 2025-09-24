# RESUMES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| fulltext | text | Parsed resume plain text |
| sections_json | jsonb | Structured sections (education, experience, etc.) |
| skills_csv | text | CSV skills list |
| yoe_raw | float | Raw computed years of experience |
| yoe_adjusted | float | Adjusted for education bonus |
| edu_level | text | Highest degree normalized |
| file_url | text | Original file artifact |
| created_at | timestamptz | Ingest timestamp |
| version | int | Version number (default 1) |
| parent_resume_id | uuid nullable | Pointer to prior version |
| metadata_tags | text[] | User-provided tags |
| description | text | User summary of version |
| active | bool | Active default? |
| source_review_id | uuid nullable | Review that generated this version |
| iteration_index | int nullable | Mirrors review iteration |
| content_hash | text unique | SHA256 hash for deduplication |
