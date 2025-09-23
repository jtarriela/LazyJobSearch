# PORTAL_TEMPLATES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| portal_id | uuid FK -> PORTALS.id | Associated portal |
| template_json | text | Template definition as JSON |
