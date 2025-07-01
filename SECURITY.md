# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by emailing the maintainers or opening a private issue on GitHub. Do **not** disclose security issues publicly until they have been reviewed and patched.

- Email: security@plimai.org (or your contact email)
- GitHub: [plimai/plim](https://github.com/plimai/plim)

We will respond as quickly as possible and coordinate a fix and disclosure timeline.

## Supported Versions

We support the latest major and minor versions. Please update to the latest release before reporting issues.

## Security Best Practices
- Never share your API tokens or secrets publicly.
- Always use environment variables for sensitive information (see below).
- Keep dependencies up to date.
- Use virtual environments for development.

## Using Secrets
- All secrets (API tokens, passwords, etc.) should be provided via environment variables or GitHub Actions secrets.
- Never hardcode secrets in the codebase or configuration files. 