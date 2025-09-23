# APPLICATION_ARTIFACTS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| application_id | uuid FK -> APPLICATIONS.id | Parent application |
| kind | text | e.g. RECEIPT_PDF, DOM_SNAPSHOT |
| file_url | text | Object store path |
| created_at | timestamptz | Timestamp |
