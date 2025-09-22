#!/usr/bin/env python3
"""
Utility functions for CLI operations.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging_with_progress(log_level: str = 'info', log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging with progress bar support."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create logger
    logger = logging.getLogger('langvision')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_progress_bar(total: int, desc: str = "Processing") -> Any:
    """Create a progress bar using tqdm."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="it", ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    except ImportError:
        # Fallback to simple progress indicator
        return SimpleProgressBar(total, desc)


class SimpleProgressBar:
    """Simple progress bar fallback when tqdm is not available."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.width = 50
        
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        self._display()
        
    def _display(self):
        """Display progress."""
        if self.total > 0:
            percent = self.current / self.total
            filled = int(self.width * percent)
            bar = '█' * filled + '░' * (self.width - filled)
            print(f'\r{self.desc}: |{bar}| {self.current}/{self.total} ({percent:.1%})', end='', flush=True)
            
    def close(self):
        """Close progress bar."""
        print()  # New line


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """Validate file path."""
    path = Path(file_path)
    
    if must_exist and not path.exists():
        return False
    
    if not must_exist:
        # Check if parent directory exists
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                return False
    
    return True


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def print_success(message: str):
    """Print success message with green color."""
    print(f"\033[1;32m✅ {message}\033[0m")


def print_error(message: str):
    """Print error message with red color."""
    print(f"\033[1;31m❌ {message}\033[0m")


def print_warning(message: str):
    """Print warning message with yellow color."""
    print(f"\033[1;33m⚠️ {message}\033[0m")


def print_info(message: str):
    """Print info message with blue color."""
    print(f"\033[1;34mℹ️ {message}\033[0m")


def print_step(step: int, total: int, message: str):
    """Print step message with progress indicator."""
    print(f"\033[1;36m[STEP {step}/{total}]\033[0m {message}")


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {
        'torch': False,
        'torchvision': False,
        'numpy': False,
        'tqdm': False,
        'pyyaml': False,
        'matplotlib': False,
        'pillow': False,
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


def print_dependency_status():
    """Print dependency status."""
    deps = check_dependencies()
    
    print("\n📦 Dependency Status:")
    print("=" * 30)
    
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"{status} {dep}")
    
    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
    }
    
    # Check for CUDA
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
    except ImportError:
        info['cuda_available'] = False
    
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    
    print("\n🖥️ System Information:")
    print("=" * 30)
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"Memory: {info['memory_gb']} GB")
    
    if info['cuda_available']:
        print(f"CUDA: {info['cuda_version']}")
        print(f"GPU: {info['gpu_name']} ({info['gpu_count']} device(s))")
    else:
        print("CUDA: Not available")


def create_output_directory(output_dir: str) -> bool:
    """Create output directory if it doesn't exist."""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print_error(f"Failed to create output directory '{output_dir}': {e}")
        return False


def save_results(results: Dict[str, Any], output_file: str) -> bool:
    """Save results to file."""
    try:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return True
    except Exception as e:
        print_error(f"Failed to save results to '{output_file}': {e}")
        return False
