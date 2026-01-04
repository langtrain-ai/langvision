"""
Optimal Vision LLM Fine-Tuning Pipeline

This module implements the best mechanisms for fine-tuning Vision LLMs.
It has been refactored into modular components.
"""

# Re-export configuration
from .config import (
    OptimalFineTuneConfig,
    FineTuningMethod,
    TrainingObjective,
)

# Re-export optimizers
from .optimizers import PagedAdamW

# Re-export modules
from .modules import NEFTuneEmbedding

# Re-export main class
from .finetuner import VisionLLMFineTuner

def create_optimal_finetuner(config: OptimalFineTuneConfig) -> VisionLLMFineTuner:
    """Factory function to create an optimal fine-tuner instance."""
    return VisionLLMFineTuner(config)

__all__ = [
    "OptimalFineTuneConfig",
    "FineTuningMethod",
    "TrainingObjective",
    "PagedAdamW",
    "NEFTuneEmbedding",
    "VisionLLMFineTuner",
    "create_optimal_finetuner",
]
