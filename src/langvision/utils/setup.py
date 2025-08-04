"""
Setup utilities for Langvision framework initialization and validation.
"""

import torch
import logging
import warnings
from typing import Optional, Dict, Any
import sys
import os
from pathlib import Path


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging for Langvision."""
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ] + ([logging.FileHandler(log_file)] if log_file else [])
    )
    
    # Get langvision logger
    logger = logging.getLogger("langvision")
    logger.info(f"Langvision logging initialized at {level} level")
    
    return logger


def validate_environment() -> Dict[str, Any]:
    """Validate the environment for Langvision usage."""
    
    validation_results = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_count": 0,
        "mps_available": False,
        "warnings": [],
        "errors": []
    }
    
    # Check CUDA
    if torch.cuda.is_available():
        validation_results["cuda_version"] = torch.version.cuda
        validation_results["gpu_count"] = torch.cuda.device_count()
        validation_results["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    else:
        validation_results["warnings"].append("CUDA not available - using CPU")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        validation_results["mps_available"] = True
    
    # Check Python version
    if sys.version_info < (3, 8):
        validation_results["errors"].append(
            f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    
    # Check PyTorch version
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (1, 10):
        validation_results["warnings"].append(
            f"PyTorch 1.10+ recommended, found {torch.__version__}"
        )
    
    return validation_results


def setup_cuda(seed: int = 42, 
               deterministic: bool = False,
               benchmark: bool = True,
               max_split_size_mb: Optional[int] = None) -> None:
    """Setup CUDA environment for optimal performance."""
    
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, skipping CUDA setup")
        return
    
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure CUDA settings
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    # Memory management
    if max_split_size_mb:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb}'
    
    # Clear cache
    torch.cuda.empty_cache()
    
    logger = logging.getLogger("langvision.setup")
    logger.info(f"CUDA setup completed with {torch.cuda.device_count()} GPUs")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_dependencies() -> Dict[str, bool]:
    """Check if optional dependencies are available."""
    
    dependencies = {}
    
    # Core dependencies
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        dependencies["torch"] = False
    
    try:
        import torchvision
        dependencies["torchvision"] = True
    except ImportError:
        dependencies["torchvision"] = False
    
    # Optional dependencies
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        dependencies["transformers"] = False
    
    try:
        import wandb
        dependencies["wandb"] = True
    except ImportError:
        dependencies["wandb"] = False
    
    try:
        import sklearn
        dependencies["sklearn"] = True
    except ImportError:
        dependencies["sklearn"] = False
    
    try:
        import cv2
        dependencies["opencv"] = True
    except ImportError:
        dependencies["opencv"] = False
    
    try:
        import pandas
        dependencies["pandas"] = True
    except ImportError:
        dependencies["pandas"] = False
    
    return dependencies


def initialize_langvision(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize Langvision framework with comprehensive setup."""
    
    if config is None:
        config = {}
    
    # Setup logging
    log_level = config.get("log_level", "INFO")
    log_file = config.get("log_file", None)
    logger = setup_logging(log_level, log_file)
    
    logger.info("Initializing Langvision framework...")
    
    # Validate environment
    validation_results = validate_environment()
    
    # Log validation results
    logger.info(f"Python version: {validation_results['python_version']}")
    logger.info(f"PyTorch version: {validation_results['pytorch_version']}")
    
    if validation_results["cuda_available"]:
        logger.info(f"CUDA version: {validation_results['cuda_version']}")
        logger.info(f"GPU count: {validation_results['gpu_count']}")
        for i, gpu_name in enumerate(validation_results.get("gpu_names", [])):
            logger.info(f"GPU {i}: {gpu_name}")
    
    if validation_results["mps_available"]:
        logger.info("MPS (Apple Silicon) available")
    
    # Log warnings and errors
    for warning in validation_results["warnings"]:
        logger.warning(warning)
    
    for error in validation_results["errors"]:
        logger.error(error)
    
    if validation_results["errors"]:
        raise RuntimeError("Environment validation failed with errors")
    
    # Setup CUDA if requested and available
    if config.get("setup_cuda", True) and validation_results["cuda_available"]:
        setup_cuda(
            seed=config.get("seed", 42),
            deterministic=config.get("deterministic", False),
            benchmark=config.get("benchmark", True),
            max_split_size_mb=config.get("max_split_size_mb", None)
        )
    
    # Set random seed
    if "seed" in config:
        set_seed(config["seed"])
        logger.info(f"Random seed set to {config['seed']}")
    
    # Check dependencies
    dependencies = check_dependencies()
    missing_deps = [dep for dep, available in dependencies.items() if not available]
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {missing_deps}")
        logger.info("Install with: pip install langvision[all] for full functionality")
    
    logger.info("Langvision framework initialization completed successfully!")
    
    return {
        "validation_results": validation_results,
        "dependencies": dependencies,
        "config": config
    }


# Convenience function for quick setup
def quick_setup(seed: int = 42, 
               log_level: str = "INFO",
               use_cuda: bool = True) -> None:
    """Quick setup for Langvision with sensible defaults."""
    
    config = {
        "seed": seed,
        "log_level": log_level,
        "setup_cuda": use_cuda,
        "benchmark": True,
        "deterministic": False
    }
    
    initialize_langvision(config)
