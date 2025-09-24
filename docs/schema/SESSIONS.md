# SESSIONS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| user_id | uuid | User identifier |
| portal_family | text | Portal family identifier |
| cookie_jar | json | Session cookie storage |
| expires_at | timestamptz | Session expiration time |
| created_at | timestamptz | Creation timestamp |