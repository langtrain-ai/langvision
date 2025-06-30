"""
plimai.filesystem

This package contains filesystem utility functions and classes.
"""

def list_files(path):
    """List files in a directory."""
    import os
    return os.listdir(path)

# Example placeholder for a filesystem utility
def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True) 