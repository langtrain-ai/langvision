"""
langvision - Modular Vision LLMs with Efficient LoRA Fine-Tuning

A research-friendly framework for building and fine-tuning Vision Large Language Models
with efficient Low-Rank Adaptation (LoRA) support.
"""

# The version CI releases (pyproject.toml), not a copy that goes stale.
try:
    from importlib.metadata import version as _version
    __version__ = _version("langvision")
except Exception:  # running from a source tree that isn't installed
    __version__ = "0.0.0"
__author__ = "Pritesh Raj"
__email__ = "priteshraj10@gmail.com"

# High-level Facades (Quick Start API)
from .facade import LoRATrainer, QLoRATrainer, ChatModel

# Core imports for easy access
from .models.vision_transformer import VisionTransformer
from .models.lora import LoRALinear, LoRAConfig, AdaLoRALinear, QLoRALinear
from .models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .models.multimodal import VisionLanguageModel, create_multimodal_model, CLIPLoss
from .utils.config import default_config

# Model Zoo
from .model_zoo import (
    ModelZoo, 
    get_available_models, 
    get_model_info, 
    download_model,
    get_lora_config,
)

# Training - Standard
from .training.trainer import Trainer
from .training.advanced_trainer import AdvancedTrainer, TrainingConfig

# Training - Fast (Unsloth-inspired)
from .training.fast_lora import (
    FastLoRAConfig,
    FastLoRALinear,
    apply_fast_lora,
    merge_lora_weights,
    get_lora_state_dict,
)
from .training.fast_trainer import (
    FastTrainer,
    FastTrainerConfig,
    create_fast_trainer,
)
from .training.memory_efficient import (
    MemoryConfig,
    MemoryTracker,
    estimate_memory_usage,
)

# Datasets
from .data.datasets import (
    get_dataset,
    CIFAR10Dataset,
    CIFAR100Dataset,
    ImageFolderDataset,
    VQADataset,
    CaptioningDataset,
    PreferenceDataset,
)
from .data.enhanced_datasets import (
    EnhancedImageDataset, MultimodalDataset, DatasetConfig, 
    create_enhanced_dataloaders, SmartAugmentation
)

# Metrics
from .utils.metrics import (
    MetricsTracker, ClassificationMetrics, ContrastiveMetrics, 
    EvaluationSuite, PerformanceMetrics
)

# Hardware Detection
from .utils.hardware import (
    HardwareDetector,
    HardwareConfig,
    AcceleratorType,
    get_device,
    get_optimal_dtype,
    auto_configure_training,
    print_hardware_info,
)

# Callbacks
from .callbacks.base import Callback, CallbackManager

# Advanced Concepts
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
    # Model Zoo
    "ModelZoo", "get_available_models", "get_model_info", 
    "download_model", "get_lora_config",
    # Training - Standard
    "Trainer", "AdvancedTrainer", "TrainingConfig",
    # Training - Fast
    "FastLoRAConfig", "FastLoRALinear", "apply_fast_lora",
    "merge_lora_weights", "get_lora_state_dict",
    "FastTrainer", "FastTrainerConfig", "create_fast_trainer",
    "MemoryConfig", "MemoryTracker", "estimate_memory_usage",
    # Data
    "get_dataset", "CIFAR10Dataset", "CIFAR100Dataset", 
    "ImageFolderDataset", "VQADataset", "CaptioningDataset", 
    "PreferenceDataset", "EnhancedImageDataset", "MultimodalDataset", 
    "DatasetConfig", "create_enhanced_dataloaders", "SmartAugmentation",
    # Utilities
    "default_config", "MetricsTracker", "ClassificationMetrics", 
    "ContrastiveMetrics", "EvaluationSuite", "PerformanceMetrics",
    # Hardware
    "HardwareDetector", "HardwareConfig", "AcceleratorType",
    "get_device", "get_optimal_dtype", "auto_configure_training", "print_hardware_info",
    # Callbacks
    "Callback", "CallbackManager",
    # Loss Functions
    "CLIPLoss",
    # Concepts
    "RLHF", "CoT", "CCoT", "GRPO", "RLVR", "DPO", "PPO", "LIME", "SHAP",
    # Facades
    "LoRATrainer", "QLoRATrainer", "ChatModel",
]

# GPU Detection & Auto-Config
try:
    from .gpu_info import get_gpu_info, auto_config, print_gpu_info, GPUInfo
    __all__.extend(["get_gpu_info", "auto_config", "print_gpu_info", "GPUInfo"])
except ImportError:
    pass

# FastVisionModel — Unsloth-style high-level API
try:
    from .fast_model import (
        FastVisionModel,
        LangvisionServerClient,
        VisionRemoteJob,
        VisionTrainingStep,
    )
    __all__.extend([
        "FastVisionModel",
        "LangvisionServerClient",
        "VisionRemoteJob",
        "VisionTrainingStep",
    ])
except ImportError as _e:
    import warnings
    warnings.warn(f"FastVisionModel not available: {_e}", ImportWarning, stacklevel=2)

# Optional imports for advanced usage
try:
    from .callbacks import EarlyStoppingCallback, LoggingCallback
    __all__.extend(["EarlyStoppingCallback", "LoggingCallback"])
except ImportError:
    pass

# Package metadata
PACKAGE_METADATA = {
    "name": "langvision",
    "version": __version__,
    "description": "Modular Vision LLMs with Efficient LoRA Fine-Tuning",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/langtrain-ai/langtrain_langvision",
    "license": "MIT",
    "python_requires": ">=3.8",
} 