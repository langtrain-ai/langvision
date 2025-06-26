"""Filesystem module for file management utilities."""
 
def list_files(path):
    """List files in a directory."""
    import os
    return os.listdir(path) 