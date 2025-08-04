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
from .models.lora import LoRALinear, LoRAConfig, AdaLoRALinear, QLoRALinear
from .models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .models.multimodal import VisionLanguageModel, create_multimodal_model, CLIPLoss
from .utils.config import default_config
from .training.trainer import Trainer
from .training.advanced_trainer import AdvancedTrainer, TrainingConfig
from .data.datasets import get_dataset
from .data.enhanced_datasets import (
    EnhancedImageDataset, MultimodalDataset, DatasetConfig, 
    create_enhanced_dataloaders, SmartAugmentation
)
from .utils.metrics import (
    MetricsTracker, ClassificationMetrics, ContrastiveMetrics, 
    EvaluationSuite, PerformanceMetrics
)
from .callbacks.base import Callback, CallbackManager
from .concepts import RLHF, CoT, CCoT, GRPO, RLVR, DPO, PPO, LIME, SHAP

# Version info
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Core Models
    "VisionTransformer",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "VisionLanguageModel", "create_multimodal_model",
    # LoRA Components
    "LoRALinear", "LoRAConfig", "AdaLoRALinear", "QLoRALinear",
    # Training
    "Trainer", "AdvancedTrainer", "TrainingConfig",
    # Data
    "get_dataset", "EnhancedImageDataset", "MultimodalDataset", 
    "DatasetConfig", "create_enhanced_dataloaders", "SmartAugmentation",
    # Utilities
    "default_config", "MetricsTracker", "ClassificationMetrics", 
    "ContrastiveMetrics", "EvaluationSuite", "PerformanceMetrics",
    # Callbacks
    "Callback", "CallbackManager",
    # Loss Functions
    "CLIPLoss",
    # Concepts
    "RLHF", "CoT", "CCoT", "GRPO", "RLVR", "DPO", "PPO", "LIME", "SHAP",
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