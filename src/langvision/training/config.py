"""
Configuration for Vision LLM Fine-Tuning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class FineTuningMethod(Enum):
    """Available fine-tuning methods."""
    LORA = "lora"            # Standard LoRA
    QLORA = "qlora"          # Quantized LoRA (4-bit)
    RSLORA = "rslora"        # Rank-Stabilized LoRA
    DORA = "dora"            # Weight-Decomposed LoRA
    FULL = "full"            # Full fine-tuning (not recommended for VLMs)


class TrainingObjective(Enum):
    """Training objectives for different tasks."""
    SFT = "sft"              # Supervised Fine-Tuning
    DPO = "dpo"              # Direct Preference Optimization
    RLHF = "rlhf"            # Reinforcement Learning from Human Feedback
    CONTRASTIVE = "contrastive"  # Contrastive learning


@dataclass
class OptimalFineTuneConfig:
    """
    Optimal configuration for Vision LLM fine-tuning.
    
    These defaults are carefully tuned based on research and best practices.
    """
    
    # Model
    model_name: str = "llava-v1.6-7b"
    trust_remote_code: bool = True
    
    # Fine-tuning method
    method: FineTuningMethod = FineTuningMethod.QLORA
    objective: TrainingObjective = TrainingObjective.SFT
    
    # LoRA Configuration (optimized defaults)
    lora_r: int = 64               # Higher rank for VLMs
    lora_alpha: float = 128        # 2x rank is optimal
    lora_dropout: float = 0.05     # Small dropout helps
    use_rslora: bool = True        # RSLoRA scaling
    use_dora: bool = False         # DoRA weight decomposition
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head",  # Include language model head
    ])
    
    # Quantization (for QLoRA)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"  # Normalized float 4-bit
    bnb_4bit_use_double_quant: bool = True  # Nested quantization
    
    # Training
    epochs: int = 3
    max_steps: int = -1            # -1 for epoch-based
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 16  # Auto-calculated
    
    # Optimizer
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning Rate Schedule
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    
    # Layer-wise LR (optimal for VLMs)
    use_layer_wise_lr: bool = True
    layer_lr_decay: float = 0.9    # Each layer gets 0.9x the LR of the layer above
    
    # Memory Optimization
    use_gradient_checkpointing: bool = True
    gradient_checkpointing_ratio: float = 0.5
    use_flash_attention: bool = True
    use_paged_adamw: bool = True   # 8-bit paged optimizer
    
    # NEFTune (Noise Embedding Fine-Tuning)
    use_neftune: bool = True
    neftune_noise_alpha: float = 5.0
    
    # Vision-specific
    freeze_vision_encoder: bool = False  # Usually train vision encoder too
    vision_lr_multiplier: float = 0.1    # Lower LR for vision encoder
    
    # Mixed Precision
    use_bf16: bool = True
    use_fp16: bool = False
    
    # Sequence
    max_seq_length: int = 2048
    pack_sequences: bool = True
    
    # Saving
    output_dir: str = "./outputs"
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    report_to: str = "tensorboard"
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0001
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        self.effective_batch_size = (
            self.per_device_batch_size * self.gradient_accumulation_steps
        )
        
        # Auto-select best method based on available memory
        if self.method == FineTuningMethod.QLORA:
            self.load_in_4bit = True
            self.use_rslora = True
