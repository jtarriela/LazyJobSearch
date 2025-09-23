# COMPANY_PORTAL_CONFIGS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| company_id | uuid FK -> COMPANIES.id | Associated company |
| portal_id | uuid FK -> PORTALS.id | Associated portal |
| config_json | text | Configuration as JSON |
