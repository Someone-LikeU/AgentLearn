# CODE_STYLE.md

## General Rules

- Code MUST be readable, maintainable, and production-ready
- Avoid unnecessary complexity
- Prefer explicit over implicit logic
- Maximum line length: 120 characters
---
## Python

- Use Python 3.10+ syntax

### Formatting
- Use `snake_case` for variables and functions
- Maximum line length: 120 characters
- Indentation: use **tabs**, where 1 tab = 4 spaces (MANDATORY)

### File Header

Every Python file MUST begin with head comments:
```python
# encoding: utf-8
# @Time    : YYYY/MM/DD HH:MM
```

### Functions
- All functions MUST include docstrings
- Docstrings should describe:
 - Purpose
 - Parameters
 - Return values 
### Logging
- DO NOT use print for debugging
- Use Python logging module instead
---
## Java
### Naming
- Use `camelCase` for variables and methods
### Formatting
- A space MUST exist between `if` and `(`:
```JAVA
if (condition) {
```
- Opening brace { MUST be on the same line as control statement
### Methods
- All public methods MUST include JavaDoc comments
---
## Cross-Language Rules
- hardcoded secrets
- Avoid magic numbers (use constants)
- Prefer configuration over hardcoding