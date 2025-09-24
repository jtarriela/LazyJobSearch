# MATCH_OUTCOMES Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| match_id | uuid FK -> MATCHES.id | Associated match |
| got_response | bool | Whether application got response |
| response_time_hours | int | Response time in hours |
| got_interview | bool | Whether got interview |
| got_offer | bool | Whether got job offer |
| user_satisfaction | int | User satisfaction rating |
| captured_at | timestamptz | Outcome capture timestamp |