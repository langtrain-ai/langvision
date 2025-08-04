import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    """Configuration class for LoRA parameters."""
    r: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    target_modules: Optional[list] = None
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "FEATURE_EXTRACTION"
    inference_mode: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

class LoRALinear(nn.Module):
    """Enhanced LoRA Linear layer with better initialization and features."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 4, 
                 alpha: float = 1.0, 
                 dropout: float = 0.0,
                 bias: bool = True,
                 fan_in_fan_out: bool = False):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = alpha / r
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0
            self.dropout = nn.Identity()
    
    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform initialization."""
        if hasattr(self, 'lora_A') and self.lora_A is not None:
            # Initialize A with random values and B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Original linear transformation
        result = self.linear(x)
        
        # Add LoRA adaptation if enabled
        if self.r > 0 and self.lora_A is not None:
            # Apply dropout to input
            x_dropped = self.dropout(x)
            
            # LoRA computation: x @ A^T @ B^T
            if self.fan_in_fan_out:
                lora_result = F.linear(x_dropped, self.lora_A.T) @ self.lora_B.T
            else:
                lora_result = F.linear(F.linear(x_dropped, self.lora_A.T), self.lora_B.T)
            
            result = result + lora_result * self.scaling
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into the original linear layer."""
        if self.r > 0 and self.lora_A is not None:
            # Compute LoRA weight update
            delta_w = self.lora_B @ self.lora_A * self.scaling
            
            # Merge with original weights
            if self.fan_in_fan_out:
                self.linear.weight.data += delta_w.T
            else:
                self.linear.weight.data += delta_w
            
            # Reset LoRA parameters
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from the original linear layer."""
        if self.r > 0 and self.lora_A is not None:
            # Compute LoRA weight update
            delta_w = self.lora_B @ self.lora_A * self.scaling
            
            # Remove from original weights
            if self.fan_in_fan_out:
                self.linear.weight.data -= delta_w.T
            else:
                self.linear.weight.data -= delta_w

class AdaLoRALinear(LoRALinear):
    """Adaptive LoRA with dynamic rank adjustment."""
    
    def __init__(self, *args, **kwargs):
        self.target_rank = kwargs.pop('target_rank', None)
        self.rank_pattern = kwargs.pop('rank_pattern', None)
        super().__init__(*args, **kwargs)
        
        if self.target_rank is not None:
            self.rank_scheduler = self._create_rank_scheduler()
    
    def _create_rank_scheduler(self):
        """Create a rank scheduler for adaptive rank adjustment."""
        # Placeholder for rank scheduling logic
        return None
    
    def update_rank(self, new_rank: int):
        """Dynamically update the LoRA rank."""
        if new_rank != self.r and new_rank > 0:
            old_r = self.r
            self.r = new_rank
            self.scaling = self.alpha / new_rank
            
            # Resize parameters
            if old_r > 0:
                # Preserve existing weights up to min(old_r, new_rank)
                min_r = min(old_r, new_rank)
                old_A = self.lora_A.data[:min_r, :]
                old_B = self.lora_B.data[:, :min_r]
            
            # Create new parameters
            self.lora_A = nn.Parameter(torch.zeros(new_rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, new_rank))
            
            if old_r > 0:
                # Copy preserved weights
                self.lora_A.data[:min_r, :] = old_A
                self.lora_B.data[:, :min_r] = old_B
            
            # Initialize new parameters
            if new_rank > old_r:
                nn.init.kaiming_uniform_(self.lora_A.data[old_r:, :], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.data[:, old_r:])

class QLoRALinear(nn.Module):
    """Quantized LoRA implementation for memory efficiency."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 4, 
                 alpha: float = 1.0, 
                 dropout: float = 0.0,
                 compute_dtype: torch.dtype = torch.float16,
                 quant_type: str = "nf4"):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        
        # Quantized base layer (placeholder - would need actual quantization library)
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        
        # LoRA adapters in full precision
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=compute_dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=compute_dtype))
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self, 'lora_A') and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to compute dtype
        x = x.to(self.compute_dtype)
        
        # Base layer forward (quantized)
        result = self.base_layer(x)
        
        # LoRA adaptation
        if self.r > 0:
            x_dropped = self.dropout(x)
            lora_result = F.linear(F.linear(x_dropped, self.lora_A.T), self.lora_B.T)
            result = result + lora_result * self.scaling
        
        return result 