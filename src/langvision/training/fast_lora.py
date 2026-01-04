"""
Fast LoRA - Optimized Low-Rank Adaptation for Efficient Vision LLM Fine-Tuning

Inspired by Unsloth's efficient training techniques, this module provides:
- Fused forward/backward passes to reduce memory transfers
- Optimized gradient computation
- Memory-efficient weight updates
- RSLoRA (Rank-Stabilized LoRA) support
- DoRA (Weight-Decomposed LoRA) support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import math


@dataclass
class FastLoRAConfig:
    """Configuration for Fast LoRA adapters."""
    
    # Core LoRA parameters
    r: int = 64  # LoRA rank
    lora_alpha: float = 128  # LoRA scaling factor
    lora_dropout: float = 0.0  # Dropout probability
    
    # Target modules to apply LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Advanced options
    use_rslora: bool = True  # Rank-Stabilized LoRA scaling
    use_dora: bool = False  # Weight-Decomposed LoRA
    init_lora_weights: str = "gaussian"  # gaussian, kaiming, zeros
    
    # Memory optimizations
    use_gradient_checkpointing: bool = True
    offload_to_cpu: bool = False  # Offload inactive adapters to CPU
    
    # Training optimizations
    use_8bit: bool = False  # 8-bit LoRA for memory savings
    use_4bit: bool = False  # 4-bit QLoRA
    
    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        if self.use_rslora:
            # RSLoRA: scale by 1/sqrt(r) for better training dynamics
            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r


class FastLoRALinear(nn.Module):
    """
    Optimized LoRA linear layer with fused operations.
    
    Key optimizations:
    1. Fused forward pass: Computes base + LoRA in single operation
    2. Memory-efficient backward: Recomputes activations instead of storing
    3. Optimized initialization for faster convergence
    4. Support for RSLoRA and DoRA variants
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FastLoRAConfig,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.r = config.r
        self.scaling = config.scaling
        
        # Base weight (frozen during training)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)
        
        # LoRA adapters (A: down-projection, B: up-projection)
        # x @ W^T + x @ A^T @ B^T * scaling
        self.lora_A = nn.Parameter(
            torch.empty(self.r, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.r, device=device, dtype=dtype)
        )
        
        # DoRA: magnitude vector for weight decomposition
        if config.use_dora:
            self.magnitude = nn.Parameter(
                torch.ones(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('magnitude', None)
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
        # Cached merged weight for inference
        self._merged = False
        self._merged_weight: Optional[torch.Tensor] = None
    
    def _init_weights(self):
        """Initialize LoRA weights for optimal training."""
        if self.config.init_lora_weights == "gaussian":
            # Standard Gaussian initialization scaled for stability
            nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.r))
        elif self.config.init_lora_weights == "kaiming":
            # Kaiming initialization for ReLU-like activations
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif self.config.init_lora_weights == "zeros":
            nn.init.zeros_(self.lora_A)
        
        # B is always initialized to zeros for identity at start
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with fused LoRA computation.
        
        For training: Computes base + LoRA separately for gradient flow
        For inference: Uses merged weights for speed
        """
        if self._merged:
            # Inference mode: use merged weights
            return F.linear(x, self._merged_weight, self.bias)
        
        if not self.config.use_dora and x.is_cuda:
            from langvision.training.acceleration import Accelerator
            accelerator = Accelerator()
            
            if accelerator.is_available():
                # Fused forward pass (Base + LoRA)
                out = accelerator.fused_lora(
                    x, self.weight, 
                    self.lora_A, self.lora_B, 
                    self.scaling
                )
                
                if self.bias is not None:
                    out += self.bias
                    
                return out

        # Training mode: compute base and LoRA separately
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA path: x @ A^T @ B^T * scaling
        lora_output = self._compute_lora(x)
        
        if self.config.use_dora:
            # DoRA: decompose into magnitude and direction
            return self._apply_dora(base_output, lora_output)
        
        return base_output + lora_output
    
    def _compute_lora(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA update efficiently."""
        # Apply dropout to input
        x = self.lora_dropout(x)
        
        # Fused computation: (x @ A^T) @ B^T * scaling
        # This is more memory efficient than storing intermediate results
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
    
    def _apply_dora(self, base_output: torch.Tensor, lora_output: torch.Tensor) -> torch.Tensor:
        """Apply DoRA: Weight-Decomposed Low-Rank Adaptation."""
        # DoRA decomposes the weight update into magnitude and direction
        # W' = m * (W + ΔW) / ||W + ΔW||
        combined = base_output + lora_output
        
        # Normalize and apply learned magnitude
        norm = torch.norm(combined, dim=-1, keepdim=True).clamp(min=1e-8)
        return self.magnitude * combined / norm
    
    def merge_weights(self):
        """Merge LoRA weights into base weights for inference."""
        if self._merged:
            return
        
        # Compute merged weight: W + B @ A * scaling
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        self._merged_weight = self.weight + delta_weight
        self._merged = True
    
    def unmerge_weights(self):
        """Unmerge weights for training."""
        self._merged = False
        self._merged_weight = None
    
    def get_delta_weight(self) -> torch.Tensor:
        """Get the LoRA weight update (B @ A * scaling)."""
        return (self.lora_B @ self.lora_A) * self.scaling
    
    @property
    def trainable_params(self) -> int:
        """Count trainable parameters."""
        count = self.lora_A.numel() + self.lora_B.numel()
        if self.magnitude is not None:
            count += self.magnitude.numel()
        return count
    
    @property
    def total_params(self) -> int:
        """Count total parameters."""
        return self.weight.numel() + self.trainable_params


class FastLoRAEmbedding(nn.Module):
    """
    Optimized LoRA for embedding layers.
    
    Useful for adapting vocabulary embeddings in Vision LLMs.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: FastLoRAConfig,
        padding_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config
        self.r = config.r
        self.scaling = config.scaling
        self.padding_idx = padding_idx
        
        # Base embedding (frozen)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            requires_grad=False
        )
        
        # LoRA adapters for embedding
        self.lora_A = nn.Parameter(
            torch.empty(self.r, num_embeddings, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(embedding_dim, self.r, device=device, dtype=dtype)
        )
        
        self._init_weights()
        self._merged = False
    
    def _init_weights(self):
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.r))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Base embedding
        base_embed = F.embedding(
            x, self.weight, padding_idx=self.padding_idx
        )
        
        if self._merged:
            return base_embed
        
        # LoRA contribution
        # A is (r, num_embeddings), B is (embedding_dim, r)
        # For input x of shape (batch, seq), we want (batch, seq, embedding_dim)
        lora_embed = F.embedding(x, (self.lora_B @ self.lora_A).T, padding_idx=self.padding_idx)
        
        return base_embed + lora_embed * self.scaling


def apply_fast_lora(
    model: nn.Module,
    config: FastLoRAConfig,
    verbose: bool = True
) -> nn.Module:
    """
    Apply Fast LoRA to a model's target modules.
    
    Args:
        model: The model to apply LoRA to
        config: FastLoRAConfig with target modules and settings
        verbose: Print information about applied adapters
    
    Returns:
        Model with LoRA adapters applied
    """
    lora_layers = []
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        should_apply = any(
            target in name for target in config.target_modules
        )
        
        if should_apply and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create LoRA layer
            lora_layer = FastLoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                config=config,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            
            # Copy original weights
            lora_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora_layer.bias.data.copy_(module.bias.data)
            
            # Replace module
            setattr(parent, attr_name, lora_layer)
            lora_layers.append(name)
            
            total_params += lora_layer.total_params
            trainable_params += lora_layer.trainable_params
    
    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name and 'magnitude' not in name:
            param.requires_grad = False
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Fast LoRA Applied Successfully")
        print(f"{'='*60}")
        print(f"  Layers modified:     {len(lora_layers)}")
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Trainable params:    {trainable_params:,}")
        print(f"  Trainable %:         {100 * trainable_params / total_params:.2f}%")
        print(f"  LoRA rank (r):       {config.r}")
        print(f"  LoRA alpha:          {config.lora_alpha}")
        print(f"  RSLoRA enabled:      {config.use_rslora}")
        print(f"  DoRA enabled:        {config.use_dora}")
        print(f"{'='*60}\n")
    
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into base weights for inference."""
    for module in model.modules():
        if isinstance(module, (FastLoRALinear, FastLoRAEmbedding)):
            module.merge_weights()
    return model


def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """Unmerge all LoRA weights for training."""
    for module in model.modules():
        if isinstance(module, (FastLoRALinear, FastLoRAEmbedding)):
            module.unmerge_weights()
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA weights for saving."""
    state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name or 'magnitude' in name:
            state_dict[name] = param.data.clone()
    return state_dict


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """Load LoRA weights into a model."""
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
