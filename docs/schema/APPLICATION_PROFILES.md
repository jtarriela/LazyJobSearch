# APPLICATION_PROFILES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| user_id | uuid FK -> USERS.id | Owner |
| profile_name | text | Friendly name |
| answers_json | jsonb | Structured field answers |
| files_map_json | jsonb | Mapping logical -> file ids |
| default_profile | bool | One default per user |
| updated_at | timestamptz | Last modification |
