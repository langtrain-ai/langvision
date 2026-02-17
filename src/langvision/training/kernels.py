"""
kernels.py: Fused kernels for high-performance training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def fused_cross_entropy(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Fused Cross Entropy Loss.
    """
    # Reshape if necessary
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
    if labels.dim() == 2:
        labels = labels.view(-1)
        
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)

# Decorate with torch.compile for fusion
if hasattr(torch, 'compile'):
    fused_cross_entropy = torch.compile(fused_cross_entropy)

class FastCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits, labels):
        return fused_cross_entropy(logits, labels, self.ignore_index)
