"""
Langvision Authentication Module

Handles API key validation, storage, and usage tracking.
Similar to Claude Code's authentication flow.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Configuration
CONFIG_DIR = Path.home() / ".langvision"
CONFIG_FILE = CONFIG_DIR / "config.json"
USAGE_FILE = CONFIG_DIR / "usage.json"

# API endpoint for validation (can be configured)
API_BASE_URL = "https://api.langtrain.xyz"


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class UsageLimitError(Exception):
    """Raised when usage limits are exceeded."""
    pass


class LangvisionAuth:
    """Handles API key authentication and usage tracking for Langvision."""
    
    def __init__(self):
        self._ensure_config_dir()
        self._config = self._load_config()
        self._usage = self._load_usage()
    
    def _ensure_config_dir(self) -> None:
        """Create configuration directory if it doesn't exist."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def _load_usage(self) -> Dict[str, Any]:
        """Load usage data from file."""
        if USAGE_FILE.exists():
            try:
                with open(USAGE_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._default_usage()
        return self._default_usage()
    
    def _default_usage(self) -> Dict[str, Any]:
        """Return default usage structure."""
        return {
            "total_commands": 0,
            "commands_this_month": 0,
            "last_reset": datetime.now().isoformat(),
            "training_runs": 0,
            "finetune_runs": 0,
            "evaluate_runs": 0,
        }
    
    def _save_usage(self) -> None:
        """Save usage data to file."""
        with open(USAGE_FILE, 'w') as f:
            json.dump(self._usage, f, indent=2)
    
    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    @property
    def api_key(self) -> Optional[str]:
        """Get stored API key."""
        # First check environment variable
        env_key = os.environ.get("LANGVISION_API_KEY")
        if env_key:
            return env_key
        # Then check config file
        return self._config.get("api_key")
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user has a valid API key stored."""
        return self.api_key is not None
    
    def set_api_key(self, api_key: str) -> None:
        """Store API key in config file."""
        self._config["api_key"] = api_key
        self._config["key_hash"] = self._hash_key(api_key)
        self._config["authenticated_at"] = datetime.now().isoformat()
        self._save_config()
    
    def clear_api_key(self) -> None:
        """Remove stored API key."""
        self._config.pop("api_key", None)
        self._config.pop("key_hash", None)
        self._save_config()
    
    def validate_api_key(self, api_key: Optional[str] = None) -> bool:
        """
        Validate API key format.
        
        API keys should follow format: lv-xxxx-xxxx-xxxx-xxxx
        For now, we do local validation. In production, this would
        call the langtrain.xyz API.
        """
        key = api_key or self.api_key
        if not key:
            return False
        
        # Basic format validation
        if not key.startswith("lv-"):
            return False
        
        # Check minimum length
        if len(key) < 20:
            return False
        
        return True
    
    def check_usage_limits(self) -> Dict[str, Any]:
        """
        Check if usage is within limits.
        
        Returns usage info and whether limits are exceeded.
        """
        # Reset monthly counter if needed
        last_reset = datetime.fromisoformat(self._usage.get("last_reset", datetime.now().isoformat()))
        now = datetime.now()
        
        if now.month != last_reset.month or now.year != last_reset.year:
            self._usage["commands_this_month"] = 0
            self._usage["last_reset"] = now.isoformat()
            self._save_usage()
        
        # Free tier limits (can be configured based on plan)
        limits = {
            "monthly_commands": 1000,
            "training_runs": 10,
            "finetune_runs": 50,
        }
        
        usage_info = {
            "commands_used": self._usage.get("commands_this_month", 0),
            "commands_limit": limits["monthly_commands"],
            "commands_remaining": limits["monthly_commands"] - self._usage.get("commands_this_month", 0),
            "training_runs": self._usage.get("training_runs", 0),
            "finetune_runs": self._usage.get("finetune_runs", 0),
            "within_limits": True,
        }
        
        if usage_info["commands_remaining"] <= 0:
            usage_info["within_limits"] = False
        
        return usage_info
    
    def record_usage(self, command_type: str = "general") -> None:
        """Record a command usage."""
        self._usage["total_commands"] = self._usage.get("total_commands", 0) + 1
        self._usage["commands_this_month"] = self._usage.get("commands_this_month", 0) + 1
        
        if command_type == "train":
            self._usage["training_runs"] = self._usage.get("training_runs", 0) + 1
        elif command_type == "finetune":
            self._usage["finetune_runs"] = self._usage.get("finetune_runs", 0) + 1
        elif command_type == "evaluate":
            self._usage["evaluate_runs"] = self._usage.get("evaluate_runs", 0) + 1
        
        self._save_usage()
    
    def get_usage_summary(self) -> str:
        """Get a formatted usage summary."""
        usage = self.check_usage_limits()
        return f"""
Usage Summary:
  Commands this month: {usage['commands_used']} / {usage['commands_limit']}
  Remaining: {usage['commands_remaining']}
  Training runs: {usage['training_runs']}
  Fine-tune runs: {usage['finetune_runs']}
"""


def require_auth(func):
    """Decorator to require authentication for a function."""
    def wrapper(*args, **kwargs):
        auth = LangvisionAuth()
        if not auth.is_authenticated:
            raise AuthenticationError(
                "Authentication required. Please run 'langvision auth login' to authenticate."
            )
        if not auth.validate_api_key():
            raise AuthenticationError(
                "Invalid API key. Please run 'langvision auth login' with a valid key."
            )
        usage = auth.check_usage_limits()
        if not usage["within_limits"]:
            raise UsageLimitError(
                f"Usage limit exceeded. You've used {usage['commands_used']} commands this month.\n"
                f"Upgrade your plan at https://billing.langtrain.xyz"
            )
        return func(*args, **kwargs)
    return wrapper


# Convenience functions
def login(api_key: str) -> bool:
    """Authenticate with API key."""
    auth = LangvisionAuth()
    if auth.validate_api_key(api_key):
        auth.set_api_key(api_key)
        return True
    return False


def logout() -> None:
    """Clear stored authentication."""
    auth = LangvisionAuth()
    auth.clear_api_key()


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    auth = LangvisionAuth()
    return auth.is_authenticated and auth.validate_api_key()


def get_auth() -> LangvisionAuth:
    """Get authentication instance."""
    return LangvisionAuth()
