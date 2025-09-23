# APPLICATION_EVENTS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| application_id | uuid FK -> APPLICATIONS.id | Parent application |
| event_type | text | e.g. SUBMITTED, RECEIPT_CAPTURED, ERROR |
| payload_json | jsonb | Event metadata |
| occurred_at | timestamptz | Timestamp |
