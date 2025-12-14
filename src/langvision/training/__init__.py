"""
Langvision Training Module

Provides efficient training utilities for Vision LLM fine-tuning:
- Fast LoRA with RSLoRA and DoRA support
- Memory-efficient training with gradient checkpointing
- Fast attention implementations
- Complete FastTrainer for end-to-end training
- VisionLLMFineTuner for optimal fine-tuning
"""

# Fast LoRA - Optimized Low-Rank Adaptation
from .fast_lora import (
    FastLoRAConfig,
    FastLoRALinear,
    FastLoRAEmbedding,
    apply_fast_lora,
    merge_lora_weights,
    unmerge_lora_weights,
    get_lora_state_dict,
    load_lora_state_dict,
    count_parameters,
)

# Memory Efficiency
from .memory_efficient import (
    MemoryConfig,
    MemoryTracker,
    GradientCheckpointer,
    CPUOffloader,
    ActivationOptimizer,
    memory_efficient_mode,
    estimate_memory_usage,
    find_optimal_batch_size,
)

# Fast Attention
from .fast_attention import (
    FastAttention,
    SlidingWindowAttention,
    MultiHeadAttention,
    create_attention_mask,
)

# Fast Trainer
from .fast_trainer import (
    FastTrainerConfig,
    FastTrainer,
    SequencePacker,
    create_fast_trainer,
)

# Optimal Fine-Tuner (best-in-class)
from .optimal_finetuner import (
    VisionLLMFineTuner,
    OptimalFineTuneConfig,
    FineTuningMethod,
    TrainingObjective,
    NEFTuneEmbedding,
    PagedAdamW,
    create_optimal_finetuner,
)

# Legacy imports for backward compatibility
from .trainer import Trainer
from .advanced_trainer import AdvancedTrainer, TrainingConfig

__all__ = [
    # Fast LoRA
    "FastLoRAConfig",
    "FastLoRALinear",
    "FastLoRAEmbedding",
    "apply_fast_lora",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "count_parameters",
    # Memory
    "MemoryConfig",
    "MemoryTracker",
    "GradientCheckpointer",
    "CPUOffloader",
    "ActivationOptimizer",
    "memory_efficient_mode",
    "estimate_memory_usage",
    "find_optimal_batch_size",
    # Attention
    "FastAttention",
    "SlidingWindowAttention",
    "MultiHeadAttention",
    "create_attention_mask",
    # Trainer
    "FastTrainerConfig",
    "FastTrainer",
    "SequencePacker",
    "create_fast_trainer",
    # Optimal Fine-Tuner
    "VisionLLMFineTuner",
    "OptimalFineTuneConfig",
    "FineTuningMethod",
    "TrainingObjective",
    "NEFTuneEmbedding",
    "PagedAdamW",
    "create_optimal_finetuner",
    # Legacy
    "Trainer",
    "AdvancedTrainer",
    "TrainingConfig",
]
