# USER_CREDENTIALS Table

| Column | Type | Notes |
|--------|------|-------|
| id | uuid PK | |
| user_id | uuid | User identifier |
| portal_family | text | Portal family identifier |
| username | text | Username for portal |
| password_ciphertext | binary | Encrypted password |
| totp_secret_ciphertext | binary | Encrypted TOTP secret |
| updated_at | timestamptz | Last update timestamp |