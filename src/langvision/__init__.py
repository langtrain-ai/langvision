"""
langvision - Modular Vision LLMs with Efficient LoRA Fine-Tuning

A research-friendly framework for building and fine-tuning Vision Large Language Models
with efficient Low-Rank Adaptation (LoRA) support.
"""

__version__ = "0.1.0"
__author__ = "Pritesh Raj"
__email__ = "priteshraj10@gmail.com"

# Core imports for easy access
from .models.vision_transformer import VisionTransformer
from .models.lora import LoRALinear, LoRAConfig
from .utils.config import Config, default_config
from .training.trainer import Trainer
from .data.datasets import ImageDataset, CIFAR10Dataset, ImageFolderDataset
from .concepts import RLHF, CoT, CCoT, GRPO, RLVR, DPO, PPO, LIME, SHAP

# Version info
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "VisionTransformer",
    "LoRALinear", 
    "LoRAConfig",
    "Config",
    "default_config",
    "Trainer",
    "ImageDataset",
    "CIFAR10Dataset",
    "ImageFolderDataset",
]

# Optional imports for advanced usage
try:
    from .callbacks import EarlyStoppingCallback, LoggingCallback
    from .utils.device import get_device, to_device
    __all__.extend([
        "EarlyStoppingCallback",
        "LoggingCallback", 
        "get_device",
        "to_device"
    ])
except ImportError:
    pass

# Package metadata
PACKAGE_METADATA = {
    "name": "langvision",
    "version": __version__,
    "description": "Modular Vision LLMs with Efficient LoRA Fine-Tuning",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/langtrain-ai/langtrain",
    "license": "MIT",
    "python_requires": ">=3.8",
} 