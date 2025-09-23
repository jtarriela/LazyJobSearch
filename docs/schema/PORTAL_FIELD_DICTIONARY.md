# PORTAL_FIELD_DICTIONARY Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| field_key | text UNIQUE | Canonical key (e.g. resume, cover_letter, phone) |
| label | text | Human readable |
| data_type | text | string | number | file | enum |
| required | bool | Portal-level default requirement |
| description | text | Usage notes |
| created_at | timestamptz | Timestamp |
