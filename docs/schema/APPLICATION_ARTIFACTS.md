# APPLICATION_ARTIFACTS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| application_id | uuid FK -> APPLICATIONS.id | Parent application |
| kind | text | Artifact type (resume, cover_letter, etc.) |
| file_url | text | Storage URL for the artifact |
