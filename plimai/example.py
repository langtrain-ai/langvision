"""
Example module demonstrating basic functionality.
"""

def hello(name: str = "World") -> str:
    """
    A simple greeting function.
    
    Args:
        name (str): Name to greet. Defaults to "World".
        
    Returns:
        str: Greeting message
    """
    return f"Hello, {name}!"

def get_version() -> str:
    """
    Get the current version of the package.
    
    Returns:
        str: Current version number
    """
    from plimai import __version__
    return __version__ 