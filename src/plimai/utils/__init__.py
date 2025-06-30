"""
plimai.utils

Utility functions for the plimai project.
"""

import toml
import os

def get_project_version():
    pyproject_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pyproject.toml")
    data = toml.load(pyproject_path)
    return data["project"]["version"] 