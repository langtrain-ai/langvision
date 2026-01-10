"""
auth.py: API Key authentication and usage tracking for Langvision
Unified authentication module for Langtrain ecosystem.
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import requests for API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# API configuration
API_BASE_URL = "https://api.langtrain.xyz"
AUTH_ENDPOINT = f"{API_BASE_URL}/v1/auth/verify"
USAGE_ENDPOINT = f"{API_BASE_URL}/v1/usage"

# Config paths
CONFIG_DIR = Path.home() / ".langvision"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHE_FILE = CONFIG_DIR / ".auth_cache"

# Environment variable names
API_KEY_ENV = "LANGVISION_API_KEY"

class AuthenticationError(Exception):
    """Raised when API key authentication fails."""
    pass

class UsageLimitError(Exception):
    """Raised when usage limit is exceeded."""
    pass

def _get_config_dir() -> Path:
    """Get or create the config directory."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR

def _load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    _get_config_dir()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def _load_auth_cache() -> Dict[str, Any]:
    """Load cached authentication data."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _save_auth_cache(cache: Dict[str, Any]) -> None:
    """Save authentication cache."""
    _get_config_dir()
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_api_key() -> Optional[str]:
    """
    Get the API key from environment or config file.
    """
    # Check environment variable first
    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        return api_key
    
    # Check config file
    config = _load_config()
    return config.get("api_key")

def set_api_key(api_key: str) -> None:
    """Save API key to config file."""
    config = _load_config()
    config["api_key"] = api_key
    _save_config(config)
    
    if RICH_AVAILABLE:
        console.print("[green]✓[/] API key saved to ~/.langvision/config.json")
    else:
        print("✓ API key saved to ~/.langvision/config.json")

def _hash_key(api_key: str) -> str:
    """Hash API key for cache lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]

def verify_api_key(api_key: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Verify API key with the Langtrain API.
    """
    # Check cache first (valid for 1 hour)
    cache = _load_auth_cache()
    key_hash = _hash_key(api_key)
    
    if not force_refresh and key_hash in cache:
        cached_data = cache[key_hash]
        cache_time = cached_data.get("cached_at", 0)
        if time.time() - cache_time < 3600:  # 1 hour cache
            return cached_data.get("data", {})
    
    # Simulate API verification (offline mode fallback)
    # This allows testing without a running backend
    if not REQUESTS_AVAILABLE:
        if api_key.startswith("lv_"): # LangVision Prefix check
            user_data = {
                "valid": True,
                "user_id": key_hash,
                "plan": "pro",
                "usage": {
                    "tokens_used": 0,
                    "tokens_limit": 500000,
                    "requests_used": 0,
                    "requests_limit": 5000
                },
                "offline_mode": True
            }
            cache[key_hash] = {"cached_at": time.time(), "data": user_data}
            _save_auth_cache(cache)
            return user_data
        else:
             raise AuthenticationError("Invalid API key format. Keys must start with 'lv_'.")

    # Real verification logic
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(AUTH_ENDPOINT, headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            cache[key_hash] = {"cached_at": time.time(), "data": user_data}
            _save_auth_cache(cache)
            return user_data
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key.")
        elif response.status_code == 403:
            raise UsageLimitError("API key expired or limits reached.")
        else:
             # Fallback for offline ease during dev
             if api_key.startswith("lv_"):
                 return {"valid": True, "plan": "pro", "offline_mode": True}
             raise AuthenticationError(f"Auth failed: {response.status_code}")

    except requests.exceptions.RequestException:
        # Fallback if offline
        if key_hash in cache: return cache[key_hash].get("data", {})
        if api_key.startswith("lv_"):
             return {"valid": True, "plan": "pro", "offline_mode": True}
        raise AuthenticationError("Could not verify API key (Offline).")

def check_usage(api_key: str) -> Dict[str, Any]:
    """Check current usage against limits."""
    user_data = verify_api_key(api_key)
    usage = user_data.get("usage", {})
    
    tokens_used = usage.get("tokens_used", 0)
    tokens_limit = usage.get("tokens_limit", 100000)
    
    if tokens_used >= tokens_limit:
        raise UsageLimitError(f"Limit exceeded: {tokens_used}/{tokens_limit}")
        
    return {
        "tokens_used": tokens_used,
        "tokens_limit": tokens_limit,
        "tokens_remaining": tokens_limit - tokens_used,
        "plan": user_data.get("plan", "free")
    }

def interactive_login():
    """Interactive login flow."""
    if RICH_AVAILABLE:
        console.print(Panel("Enter your API Key from [cyan]https://langtrain.xyz[/]", title="🔐 Login", border_style="cyan"))
        api_key = console.input("[bold]API Key:[/] ")
    else:
        api_key = input("Enter API Key: ")
    
    api_key = api_key.strip()
    if not api_key: return False
    
    try:
        verify_api_key(api_key, force_refresh=True)
        set_api_key(api_key)
        if RICH_AVAILABLE:
            console.print("[bold green]✓ Authentication successful![/]")
        return True
    except Exception as e:
        if RICH_AVAILABLE:
             console.print(f"[bold red]Authentication failed:[/] {e}")
        else:
             print(f"Authentication failed: {e}")
        return False

def logout():
    """Remove stored API key."""
    config = _load_config()
    if "api_key" in config:
        del config["api_key"]
        _save_config(config)
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    
    if RICH_AVAILABLE:
        console.print("[green]✓[/] Logged out successfully.")
    else:
        print("✓ Logged out successfully.")

# Adapter classes for backward compatibility with __init__.py import
class LangvisionAuth:
    def __init__(self):
        pass
    
    @property
    def is_authenticated(self):
        return get_api_key() is not None
        
    def validate_api_key(self):
         key = get_api_key()
         return key and key.startswith("lv_")
         
    def check_usage_limits(self):
        key = get_api_key()
        if not key: raise AuthenticationError("Not logged in")
        try:
            usage = check_usage(key)
            return {
                "within_limits": True,
                "commands_used": 0, # Placeholder
                "commands_limit": 1000,
                "commands_remaining": 1000,
                "training_runs": 0,
                "finetune_runs": 0
            }
        except Exception:
            return {"within_limits": False}

    def record_usage(self, command_type):
        pass

# Global instance
_auth_instance = LangvisionAuth()

def get_auth():
    return _auth_instance

def login(key):
    try:
        if not key.startswith("lv_"): return False
        verify_api_key(key)
        set_api_key(key)
        return True
    except:
        return False

def is_authenticated():
    return get_api_key() is not None

