# PORTAL_TEMPLATES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| portal_id | uuid FK -> PORTALS.id | Parent portal |
| template_name | text | Friendly label |
| template_json | jsonb | DSL definition |
| version | int | Template version number |
| created_at | timestamptz | Creation time |
