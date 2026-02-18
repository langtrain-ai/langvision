---
description: Run Ralph, the strict code quality and cleanup agent
---

# Ralph - The Code Janitor

Ralph's job is to ensure code quality, consistency, and cleanliness.

1.  **Analyze Project Structure**
    - Run `find . -maxdepth 2 -not -path '*/.*'` to see the lay of the land.

2.  **Lint & Format**
    - IF `package.json` exists:
        - Run `npm run lint` or `pnpm lint`.
        - If it fails, fix the errors automatically.
    - IF `pyproject.toml` or `requirements.txt` exists:
        - Run `ruff check . --fix` (if available) or `flake8 .`.

3.  **Dead Code Detection**
    - Search for TODOs and FIXMEs: `grep -r "TODO" .`, `grep -r "FIXME" .`
    - Report them to the user but do NOT delete unless instructed.

4.  **Security Scan**
    - Check for hardcoded secrets (basic regex check).
    - `grep -r "API_KEY" .` (exclude env files).

5.  **Report**
    - Generate a summary of what was cleaned and what needs attention.
