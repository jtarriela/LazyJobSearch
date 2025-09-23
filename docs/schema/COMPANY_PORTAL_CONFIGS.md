# COMPANY_PORTAL_CONFIGS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| company_id | uuid FK -> COMPANIES.id | Company |
| portal_id | uuid FK -> PORTALS.id | Portal used |
| config_json | jsonb | Overrides / selectors |
| active | bool | Active flag |
| created_at | timestamptz | Timestamp |
