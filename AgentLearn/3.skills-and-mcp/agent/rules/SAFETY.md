# SAFETY.md

## Destructive Operations

You MUST NEVER:

- Execute or suggest commands like:
  - `rm -rf /`
  - `del /f /s /q *`
  - Any equivalent destructive filesystem operation

---

## File Modification Policy

Before modifying any file:

- A backup MUST be created
- You MUST verify:
  - File path correctness
  - Scope of change

---

## Sensitive Data Protection

You MUST NOT:

- Read, modify, or expose:
  - `.env` files
  - API keys
  - Tokens
  - Credentials

- Never log sensitive information

---

## Change Logging

All file modifications MUST be recorded in a log with:

- File name
- Line number(s)
- Before and after content
- Timestamp

Example:
```TXT
[2026-04-21 14:32:10]
File: config.py
Line: 42
Before: DEBUG = True
After : DEBUG = False
```
## Execution Safety

- Validate all commands before execution
- Reject unsafe or ambiguous instructions
- Require confirmation for high-risk operations (if interactive system)