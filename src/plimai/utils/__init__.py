"""
plimai.utils

Utility functions for the plimai project.
"""

import toml
import os
import re

def get_project_version():
    """Return the current project version from pyproject.toml as a string."""
    pyproject_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../pyproject.toml")
    )
    data = toml.load(pyproject_path)
    return data["project"]["version"]

def parse_version(version_str):
    """Parse a version string (e.g., '0.1.2a1') into (major, minor, patch, suffix)."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)([a-zA-Z0-9]*)$", version_str)
    if not match:
        raise ValueError(f"Version '{version_str}' is not in a recognized format (X.Y.Z or X.Y.Z<suffix>)")
    major, minor, patch, suffix = match.groups()
    return int(major), int(minor), int(patch), suffix 