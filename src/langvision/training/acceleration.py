"""
Accelerator module for Langvision.

This module provides an interface to access high-performance custom kernels
from langtrain-server if they are available in the environment.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

class Accelerator:
    """
    Manages access to accelerated kernels.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Accelerator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.available = False
        self.kernels = None
        
        try:
            import langtrain_cuda
            self.kernels = langtrain_cuda
            self.available = True
            logger.info("Langtrain high-performance kernels detected and enabled for Langvision.")
        except ImportError:
            logger.info("Langtrain kernels not found. Using standard PyTorch implementations.")
            
    def is_available(self) -> bool:
        return self.available

    def fused_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
        scale: float = None
    ) -> torch.Tensor:
        """
        Run fused attention forward pass.
        Expects inputs in shape [batch, heads, seq, head_dim]
        """
        if self.available and self.kernels:
            if scale is None:
                scale = query.size(-1) ** -0.5
            return self.kernels.fused_attention_forward(query, key, value, scale, is_causal)
        else:
            return None

    def fused_rmsnorm(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Run fused RMSNorm.
        """
        if self.available and self.kernels:
            return self.kernels.fused_rmsnorm_forward(hidden_states, weight, eps)
        return None

    def fused_mlp(
        self,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Run fused SwiGLU MLP.
        """
        if self.available and self.kernels:
            return self.kernels.fused_mlp_forward(hidden_states, gate_weight, up_weight, down_weight)
        return None

    def fused_lora(
        self,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float
    ) -> torch.Tensor:
        """
        Run fused LoRA forward.
        """
        if self.available and self.kernels:
            return self.kernels.fused_lora_forward(x, base_weight, lora_A, lora_B, scaling)
        return None
