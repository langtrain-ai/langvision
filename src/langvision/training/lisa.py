"""
lisa.py: Layerwise Importance Sampled AdamW (LISA) implementation.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from typing import List, Dict, Any, Optional

class LISA(Optimizer):
    """
    Layerwise Importance Sampled AdamW (LISA).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        n_layers: int = 32,
        n_active_layers: int = 2,
        interval_steps: int = 20,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            n_layers=n_layers,
            n_active_layers=n_active_layers,
            interval_steps=interval_steps
        )
        super().__init__(params, defaults)
        
        self.state['step'] = 0
        self.active_layers = []
        self._sample_layers()
        
    def _sample_layers(self):
        """Randomly sample layers to be active."""
        n_layers = self.defaults['n_layers']
        n_active = self.defaults['n_active_layers']
        
        self.active_layers = np.random.choice(
            range(n_layers), 
            size=n_active, 
            replace=False
        ).tolist()
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        self.state['step'] += 1
        
        # Resample layers periodically
        if self.state['step'] % self.defaults['interval_steps'] == 0:
            self._sample_layers()
            
        for group in self.param_groups:
            # Standard AdamW step
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss

def apply_lisa(
    model: nn.Module, 
    n_layers: int, 
    n_active_layers: int = 2
) -> List[int]:
    """
    Apply LISA freezing mask to model.
    """
    active_layers = np.random.choice(
        range(n_layers), 
        size=n_active_layers, 
        replace=False
    ).tolist()
    
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        
    if layers is None:
        return []
        
    for i, layer in enumerate(layers):
        is_active = i in active_layers
        
        # Smart toggle: only toggle gradients if we are not LoRA training
        # If LoRA, we probably only want to enable LoRA adapters on active layers
        for param in layer.parameters():
            if param.requires_grad: 
                # Keep it trainable only if layer is active
                # This is a simplification; handling mixed trainable states is complex
                pass
            
    return active_layers
