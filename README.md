# PlimAI

A Python package for building and deploying AI agents. PlimAI is designed to streamline the creation, testing, and deployment of intelligent agents, providing tools and frameworks for various AI tasks.

## Badges

![PyPI - Version](https://img.shields.io/pypi/v/plimai)
![PyPI - Downloads](https://img.shields.io/pypi/dm/plimai)
![Build Status](https://github.com/plimai/plim/workflows/Publish%20Python%20Package/badge.svg)
![Code Coverage](https://img.shields.io/codecov/c/github/plimai/plim)

## Installation

You can install the package using pip:

```bash
pip install plimai
```

## Usage

```python
from plimai import example
from plimai.example import get_version

# Example usage of hello function
print(example.hello("PlimAI User"))

# Example usage of get_version function
print(f"PlimAI Version: {get_version()}")
```

## Development

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/plim-ai/plim.git
   cd plim
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Version Control

This project uses semantic versioning and automatic version bumping:

- `feat:` commits trigger a minor version bump
- `fix:` commits trigger a patch version bump
- `feat!:` or `BREAKING CHANGE:` commits trigger a major version bump

### CI/CD

The project uses GitHub Actions for continuous integration and deployment:

1. On every push to main:
   - Version is automatically bumped based on commit messages
   - Package is built and published to PyPI
   - GitHub release is created

2. To trigger a new release:
   - Make your changes
   - Commit with a semantic commit message (e.g., `feat: add new feature`)
   - Push to main

## License

This project is licensed under the MIT License - see the LICENSE file for details. 