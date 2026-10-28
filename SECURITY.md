# Security checklist: rotating credentials

- [ ] Revoke any exposed API key, access token, password, or other credential
      at its issuing provider. Treat a confirmed exposure as compromised.
- [ ] Create replacement credentials with only the permissions this project
      needs, then store them in `.env` or the deployment secret manager. Never
      put values in source, `.env.example`, logs, screenshots, or test output.
- [ ] Confirm `.env` remains ignored by Git and that `.env.example` contains
      variable names only, with blank values.
- [ ] Restart the app or deployment so it reads the replacement credentials.
- [ ] Verify the expected provider connection with a minimal check that does
      not print the credential or include it in exception/log output.
- [ ] Revoke old credentials after verification. If a value entered Git history
      or a shared archive, remove it from those copies and notify their owners;
      deleting the current file alone does not revoke it.
- [ ] Review repository logs, CI output, test output, artifacts, and archives
      for any further copies. Do not paste candidate values into reports or
      tickets.
