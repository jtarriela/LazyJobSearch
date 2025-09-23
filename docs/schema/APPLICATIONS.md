# APPLICATIONS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| user_id | uuid FK -> USERS.id | Applicant |
| job_id | uuid FK -> JOBS.id | Target job |
| resume_id | uuid FK -> RESUMES.id | Resume used |
| application_profile_id | uuid FK -> APPLICATION_PROFILES.id | Profile applied with |
| status | text | Draft | Submitted | Error |
| portal | text | Portal identifier |
| submitted_at | timestamptz | Submission time |
| receipt_url | text | Artifact location |
| error_text | text | Error detail if any |
